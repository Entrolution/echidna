//! Arbitrary differential operator evaluation via jet coefficients.
//!
//! Evaluate any mixed partial derivative of a recorded tape by constructing
//! higher-order Taylor jets with carefully chosen input coefficients. A single
//! forward pushforward extracts the derivative from a specific output jet
//! coefficient, scaled by a known prefactor from the multivariate Faa di Bruno
//! formula.
//!
//! # How it works
//!
//! For a function `u: R^n -> R` recorded as a [`BytecodeTape`], we want to
//! compute an arbitrary mixed partial:
//!
//! ```text
//! ∂^Q u / (∂x_{i₁}^{q₁} ... ∂x_{iT}^{qT})
//! ```
//!
//! The method parameterises a curve `g(t) = u(x₀ + v⁽¹⁾t + v⁽²⁾t²/2! + ...)`
//! where each active variable is assigned a distinct polynomial "slot" `j_t`,
//! with `coeffs[j_t] = 1/j_t!` for that variable's input. The output jet
//! coefficient at index `k = Σ j_t · q_t` then equals the target derivative
//! divided by a known prefactor.
//!
//! # Usage
//!
//! ```ignore
//! use echidna::diffop::{JetPlan, MultiIndex};
//!
//! // Record a tape
//! let (tape, _) = echidna::record(|x| x[0] * x[0] * x[1], &[1.0, 2.0]);
//!
//! // Plan: compute ∂²u/∂x₀² and ∂u/∂x₁
//! let indices = vec![
//!     MultiIndex::diagonal(2, 0, 2), // d²/dx₀²
//!     MultiIndex::partial(2, 1),      // d/dx₁
//! ];
//! let plan = JetPlan::plan(2, &indices);
//!
//! // Evaluate
//! let result = echidna::diffop::eval_dyn(&plan, &tape, &[1.0, 2.0]);
//! // result.derivatives[0] = 2*x₁ = 4.0  (∂²(x₀²x₁)/∂x₀²)
//! // result.derivatives[1] = x₀² = 1.0    (∂(x₀²x₁)/∂x₁)
//! ```
//!
//! # Design
//!
//! - **Plan once, evaluate many**: [`JetPlan::plan`] precomputes slot assignments,
//!   jet order, and extraction prefactors. Reuse the plan across evaluation points.
//! - **`TaylorDyn`** for runtime jet order: the required order depends on the
//!   differential operator and cannot be known at compile time.
//! - **Pushforward groups**: Multi-indices that share the same set of active
//!   variables are batched into one forward pass. Multi-indices with different
//!   active variables get separate pushforwards to avoid slot contamination.
//! - **Panics on misuse**: dimension mismatches panic, following existing API
//!   conventions.

use crate::bytecode_tape::BytecodeTape;
use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn, TaylorDynGuard};
use crate::Float;

// ══════════════════════════════════════════════
//  MultiIndex
// ══════════════════════════════════════════════

/// A multi-index specifying which mixed partial derivative to compute.
///
/// `orders[i]` = how many times to differentiate with respect to variable `x_i`.
///
/// # Examples
///
/// - `[2, 0, 1]` represents `∂³u/(∂x₀²∂x₂)` (total order 3).
/// - `[0, 1]` represents `∂u/∂x₁` (first partial).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiIndex {
    orders: Vec<u8>,
}

impl MultiIndex {
    /// Create a multi-index from a slice of per-variable differentiation orders.
    ///
    /// # Panics
    ///
    /// Panics if `orders` is empty.
    pub fn new(orders: &[u8]) -> Self {
        assert!(
            !orders.is_empty(),
            "multi-index must have at least one variable"
        );
        MultiIndex {
            orders: orders.to_vec(),
        }
    }

    /// Multi-index for a single-variable diagonal derivative: `d^order u / dx_var^order`.
    ///
    /// # Panics
    ///
    /// Panics if `var >= num_vars` or `order == 0`.
    pub fn diagonal(num_vars: usize, var: usize, order: u8) -> Self {
        assert!(var < num_vars, "var ({}) >= num_vars ({})", var, num_vars);
        assert!(order > 0, "order must be > 0");
        let mut orders = vec![0u8; num_vars];
        orders[var] = order;
        MultiIndex { orders }
    }

    /// Multi-index for a first partial: `∂u/∂x_var`.
    ///
    /// # Panics
    ///
    /// Panics if `var >= num_vars`.
    pub fn partial(num_vars: usize, var: usize) -> Self {
        Self::diagonal(num_vars, var, 1)
    }

    /// Total differentiation order: `Σ orders[i]`.
    pub fn total_order(&self) -> usize {
        self.orders.iter().map(|&o| o as usize).sum()
    }

    /// Active variables: indices where `orders[i] > 0`, paired with their order.
    pub fn active_vars(&self) -> Vec<(usize, u8)> {
        self.orders
            .iter()
            .enumerate()
            .filter(|(_, &o)| o > 0)
            .map(|(i, &o)| (i, o))
            .collect()
    }

    /// Number of variables in this multi-index.
    pub fn num_vars(&self) -> usize {
        self.orders.len()
    }

    /// The per-variable differentiation orders.
    pub fn orders(&self) -> &[u8] {
        &self.orders
    }

    /// Active variable indices only (sorted).
    fn active_var_set(&self) -> Vec<usize> {
        self.orders
            .iter()
            .enumerate()
            .filter(|(_, &o)| o > 0)
            .map(|(i, _)| i)
            .collect()
    }
}

// ══════════════════════════════════════════════
//  Partition utilities (internal)
// ══════════════════════════════════════════════

/// Enumerate all partitions of integer `k` using only the given slot values as parts.
///
/// Each partition is a list of `(slot, multiplicity)` pairs sorted by slot.
fn partitions_with_support(k: usize, slots: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let mut results = Vec::new();
    let mut current = Vec::new();
    partitions_recurse(k, slots, 0, &mut current, &mut results);
    results
}

fn partitions_recurse(
    remaining: usize,
    slots: &[usize],
    start_idx: usize,
    current: &mut Vec<(usize, usize)>,
    results: &mut Vec<Vec<(usize, usize)>>,
) {
    if remaining == 0 {
        results.push(current.clone());
        return;
    }
    for idx in start_idx..slots.len() {
        let s = slots[idx];
        if s > remaining {
            continue;
        }
        let max_mult = remaining / s;
        for mult in 1..=max_mult {
            current.push((s, mult));
            partitions_recurse(remaining - s * mult, slots, idx + 1, current, results);
            current.pop();
        }
    }
}

/// Compute the extraction prefactor: `Π_t (q_t! · (j_t!)^{q_t})`
fn extraction_prefactor<F: Float>(slot_assignments: &[(usize, u8)]) -> F {
    let mut prefactor = F::one();
    for &(slot, order) in slot_assignments {
        let mut q_fact = F::one();
        for i in 2..=(order as usize) {
            q_fact = q_fact * F::from(i).unwrap();
        }
        let mut j_fact = F::one();
        for i in 2..=slot {
            j_fact = j_fact * F::from(i).unwrap();
        }
        let mut j_fact_pow = F::one();
        for _ in 0..order {
            j_fact_pow = j_fact_pow * j_fact;
        }
        prefactor = prefactor * q_fact * j_fact_pow;
    }
    prefactor
}

// ══════════════════════════════════════════════
//  JetPlan
// ══════════════════════════════════════════════

/// A single extraction from a pushforward's output coefficients.
#[derive(Clone, Debug)]
struct Extraction<F> {
    /// Index into the final derivatives vector.
    result_index: usize,
    /// Which output coefficient to read.
    output_coeff_index: usize,
    /// Multiply `coeffs[k]` by this to get the derivative value.
    prefactor: F,
}

/// A group of multi-indices that share one pushforward.
///
/// All multi-indices in a group must have the same set of active variables
/// (though possibly different orders).
#[derive(Clone, Debug)]
struct PushforwardGroup<F> {
    /// Number of Taylor coefficients for this group.
    jet_order: usize,
    /// Input coefficient assignments: `(var_index, slot, 1/slot!)`.
    input_coeffs: Vec<(usize, usize, F)>,
    /// Extractions from this group's output.
    extractions: Vec<Extraction<F>>,
}

/// Immutable plan for jet evaluation. Constructed once, reused across points.
///
/// Use [`JetPlan::plan`] to create a plan from a set of multi-indices, then
/// pass it to [`eval_dyn`] to evaluate at specific points.
#[derive(Clone, Debug)]
pub struct JetPlan<F> {
    /// Max jet order across all groups.
    max_jet_order: usize,
    /// Pushforward groups.
    groups: Vec<PushforwardGroup<F>>,
    /// The multi-indices, in order.
    multi_indices: Vec<MultiIndex>,
}

/// First primes for slot assignment.
const PRIMES: [usize; 20] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

/// Check whether ALL multi-indices in a group can be cleanly extracted
/// with the given variable-to-slot mapping. Returns `Ok(extractions, max_k)`
/// if collision-free, or `Err(())` if any collision exists.
fn try_slots<F: Float>(
    var_slot: &[(usize, usize)],
    multi_indices_with_idx: &[(usize, &MultiIndex)],
) -> Result<(Vec<Extraction<F>>, usize), ()> {
    let group_slots: Vec<usize> = var_slot.iter().map(|&(_, s)| s).collect();
    let mut extractions = Vec::new();
    let mut max_k = 0usize;

    for &(result_index, mi) in multi_indices_with_idx {
        let active = mi.active_vars();

        if active.is_empty() {
            extractions.push(Extraction {
                result_index,
                output_coeff_index: 0,
                prefactor: F::one(),
            });
            continue;
        }

        let slot_orders: Vec<(usize, u8)> = active
            .iter()
            .map(|&(var, order)| {
                let slot = var_slot.iter().find(|(v, _)| *v == var).unwrap().1;
                (slot, order)
            })
            .collect();

        let k: usize = slot_orders.iter().map(|&(s, q)| s * q as usize).sum();

        let partitions = partitions_with_support(k, &group_slots);

        let mut target_partition: Vec<(usize, usize)> = slot_orders
            .iter()
            .map(|&(slot, order)| (slot, order as usize))
            .collect();
        target_partition.sort_by_key(|&(s, _)| s);

        let collision = partitions.iter().any(|p| {
            let mut sorted = p.clone();
            sorted.sort_by_key(|&(s, _)| s);
            sorted != target_partition
        });

        if collision {
            return Err(());
        }

        let prefactor = extraction_prefactor::<F>(&slot_orders);
        max_k = max_k.max(k);

        extractions.push(Extraction {
            result_index,
            output_coeff_index: k,
            prefactor,
        });
    }

    Ok((extractions, max_k))
}

/// Plan slot assignment for a single group of multi-indices that share
/// the same set of active variables.
fn plan_group<F: Float>(
    active_var_set: &[usize],
    multi_indices_with_idx: &[(usize, &MultiIndex)],
) -> PushforwardGroup<F> {
    let t = active_var_set.len();
    assert!(
        t <= PRIMES.len(),
        "too many active variables ({}) — max supported is {}",
        t,
        PRIMES.len()
    );

    // Sort active variables by max order descending (highest-order gets smallest prime)
    let mut var_max_order: Vec<(usize, u8)> = active_var_set
        .iter()
        .map(|&var| {
            let max_ord = multi_indices_with_idx
                .iter()
                .map(|(_, mi)| mi.orders()[var])
                .max()
                .unwrap_or(0);
            (var, max_ord)
        })
        .collect();
    var_max_order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Try prime windows: PRIMES[offset..offset+t], incrementing offset on collision
    let max_offset = PRIMES.len() - t;
    for offset in 0..=max_offset {
        let var_slot: Vec<(usize, usize)> = var_max_order
            .iter()
            .enumerate()
            .map(|(i, &(var, _))| (var, PRIMES[offset + i]))
            .collect();

        if let Ok((extractions, max_k)) = try_slots::<F>(&var_slot, multi_indices_with_idx) {
            let input_coeffs: Vec<(usize, usize, F)> = var_slot
                .iter()
                .map(|&(var, slot)| {
                    let mut factorial = F::one();
                    for i in 2..=slot {
                        factorial = factorial * F::from(i).unwrap();
                    }
                    (var, slot, F::one() / factorial)
                })
                .collect();

            return PushforwardGroup {
                jet_order: max_k + 1,
                input_coeffs,
                extractions,
            };
        }
    }

    panic!(
        "failed to find collision-free slot assignment for active vars {:?}",
        active_var_set
    );
}

impl<F: Float> JetPlan<F> {
    /// Plan jet evaluation for a set of multi-indices.
    ///
    /// Groups multi-indices by their active variable set, assigns collision-free
    /// slots within each group, and precomputes extraction prefactors.
    ///
    /// # Panics
    ///
    /// Panics if `multi_indices` is empty, if any multi-index has wrong `num_vars`,
    /// or if slot assignment fails.
    pub fn plan(num_vars: usize, multi_indices: &[MultiIndex]) -> Self {
        assert!(
            !multi_indices.is_empty(),
            "must provide at least one multi-index"
        );
        for mi in multi_indices {
            assert_eq!(
                mi.num_vars(),
                num_vars,
                "multi-index num_vars ({}) != expected ({})",
                mi.num_vars(),
                num_vars
            );
        }

        // Group multi-indices by their active variable set
        type GroupEntry<'a> = (Vec<usize>, Vec<(usize, &'a MultiIndex)>);
        let mut group_map: Vec<GroupEntry<'_>> = Vec::new();

        for (i, mi) in multi_indices.iter().enumerate() {
            let active_set = mi.active_var_set();
            if let Some(entry) = group_map.iter_mut().find(|(set, _)| *set == active_set) {
                entry.1.push((i, mi));
            } else {
                group_map.push((active_set, vec![(i, mi)]));
            }
        }

        // Plan each group
        let mut groups = Vec::with_capacity(group_map.len());
        let mut max_jet_order = 1;

        for (active_set, members) in &group_map {
            let group = plan_group::<F>(active_set, members);
            max_jet_order = max_jet_order.max(group.jet_order);
            groups.push(group);
        }

        JetPlan {
            max_jet_order,
            groups,
            multi_indices: multi_indices.to_vec(),
        }
    }

    /// The maximum jet order across all groups.
    pub fn jet_order(&self) -> usize {
        self.max_jet_order
    }

    /// The multi-indices this plan computes, in order.
    pub fn multi_indices(&self) -> Vec<MultiIndex> {
        self.multi_indices.clone()
    }
}

// ══════════════════════════════════════════════
//  Result type
// ══════════════════════════════════════════════

/// Result of evaluating a differential operator via jet coefficients.
#[derive(Clone, Debug)]
pub struct DiffOpResult<F> {
    /// Function value `u(x)`.
    pub value: F,
    /// Computed derivatives, in the same order as the plan's multi-indices.
    pub derivatives: Vec<F>,
    /// The multi-indices that were computed.
    pub multi_indices: Vec<MultiIndex>,
}

// ══════════════════════════════════════════════
//  Evaluation
// ══════════════════════════════════════════════

/// Evaluate a differential operator plan using `TaylorDyn` (runtime jet order).
///
/// Each pushforward group gets its own forward pass with only the relevant
/// slot coefficients set. This ensures clean extraction without slot
/// contamination from non-active variables.
///
/// # Panics
///
/// Panics if `x.len()` does not match `tape.num_inputs()`.
pub fn eval_dyn<F: Float + TaylorArenaLocal>(
    plan: &JetPlan<F>,
    tape: &BytecodeTape<F>,
    x: &[F],
) -> DiffOpResult<F> {
    let n = tape.num_inputs();
    assert_eq!(
        x.len(),
        n,
        "x.len() ({}) must match tape.num_inputs() ({})",
        x.len(),
        n
    );

    let num_results = plan.multi_indices.len();
    let mut derivatives = vec![F::zero(); num_results];
    let mut value = x.iter().copied().fold(F::zero(), |a, b| a + b); // placeholder

    for group in &plan.groups {
        let _guard = TaylorDynGuard::<F>::new(group.jet_order);

        // Build inputs: only set slot coefficients for this group's active variables
        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); group.jet_order];
                coeffs[0] = x[i];
                for &(var, slot, inv_fact) in &group.input_coeffs {
                    if var == i && slot < group.jet_order {
                        coeffs[slot] = inv_fact;
                    }
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        let mut buf = Vec::new();
        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];

        for extraction in &group.extractions {
            derivatives[extraction.result_index] =
                out_coeffs[extraction.output_coeff_index] * extraction.prefactor;
        }
    }

    DiffOpResult {
        value,
        derivatives,
        multi_indices: plan.multi_indices.clone(),
    }
}

// ══════════════════════════════════════════════
//  Convenience functions
// ══════════════════════════════════════════════

/// Compute a single mixed partial derivative (plans + evaluates in one call).
///
/// Returns `(value, derivative)` where `value = u(x)` and `derivative` is the
/// mixed partial specified by `orders`.
///
/// # Panics
///
/// Panics if `orders.len()` does not match `tape.num_inputs()`, or if all
/// orders are zero.
pub fn mixed_partial<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    orders: &[u8],
) -> (F, F) {
    let mi = MultiIndex::new(orders);
    let plan = JetPlan::plan(orders.len(), &[mi]);
    let result = eval_dyn(&plan, tape, x);
    (result.value, result.derivatives[0])
}

/// Compute the full Hessian (all second-order partial derivatives).
///
/// Returns `(value, gradient, hessian)` where:
/// - `gradient[i]` = `∂u/∂x_i`
/// - `hessian[i][j]` = `∂²u/(∂x_i ∂x_j)`
///
/// Each derivative requires its own pushforward group, so this performs
/// `n + n*(n+1)/2` forward passes. For large n, consider using
/// `tape.hessian()` instead.
///
/// # Panics
///
/// Panics if `x.len()` does not match `tape.num_inputs()`.
#[allow(clippy::needless_range_loop)]
pub fn hessian<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
) -> (F, Vec<F>, Vec<Vec<F>>) {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let mut indices = Vec::with_capacity(n + n * (n + 1) / 2);

    // First-order partials
    for i in 0..n {
        indices.push(MultiIndex::partial(n, i));
    }

    // Second-order: diagonal and upper-triangle
    for i in 0..n {
        for j in i..n {
            let mut orders = vec![0u8; n];
            if i == j {
                orders[i] = 2;
            } else {
                orders[i] = 1;
                orders[j] = 1;
            }
            indices.push(MultiIndex::new(&orders));
        }
    }

    let plan = JetPlan::plan(n, &indices);
    let result = eval_dyn(&plan, tape, x);

    let gradient: Vec<F> = result.derivatives[..n].to_vec();

    let mut hess = vec![vec![F::zero(); n]; n];
    let mut idx = n;
    for i in 0..n {
        for j in i..n {
            let val = result.derivatives[idx];
            hess[i][j] = val;
            hess[j][i] = val;
            idx += 1;
        }
    }

    (result.value, gradient, hess)
}

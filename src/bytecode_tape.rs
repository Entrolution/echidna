//! Bytecode tape for re-evaluable reverse-mode AD.
//!
//! Unlike the Adept-style [`Tape`](crate::tape::Tape), this tape stores opcodes
//! rather than precomputed multipliers. This allows the tape to be re-evaluated
//! at different inputs without re-recording, at the cost of opcode dispatch
//! during the reverse sweep.
//!
//! # Limitations
//!
//! The tape records one execution path. If the recorded function contains
//! branches (`if x > 0 { ... } else { ... }`), re-evaluating at inputs that
//! take a different branch produces incorrect results.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

use crate::float::IsAllZero;
use num_traits::Float as NumFloat;

#[cfg(feature = "taylor")]
use crate::taylor::Taylor;

/// Sentinel index for constant entries (not tracked).
pub const CONSTANT: u32 = u32::MAX;

/// Trait for user-registered custom operations on the bytecode tape.
///
/// Operations are defined on `F` (the base float type). The tape automatically
/// handles forward-mode tangent propagation and reverse-mode adjoint accumulation
/// via chain rule using the partials you provide.
///
/// For correct second-order derivatives (Hessian, HVP) through custom ops,
/// override [`eval_dual`](CustomOp::eval_dual) and
/// [`partials_dual`](CustomOp::partials_dual). The defaults use first-order
/// chain rule, which treats partials as constant.
///
/// # Example
///
/// ```ignore
/// struct Softplus;
///
/// impl CustomOp<f64> for Softplus {
///     fn eval(&self, a: f64, _b: f64) -> f64 {
///         (1.0 + a.exp()).ln()
///     }
///     fn partials(&self, a: f64, _b: f64, _r: f64) -> (f64, f64) {
///         let sig = 1.0 / (1.0 + (-a).exp());
///         (sig, 0.0)
///     }
/// }
/// ```
pub trait CustomOp<F: Float>: Send + Sync {
    /// Forward evaluation on the base float type.
    fn eval(&self, a: F, b: F) -> F;
    /// Reverse partials `(∂result/∂a, ∂result/∂b)`.
    fn partials(&self, a: F, b: F, result: F) -> (F, F);

    /// Evaluate in dual-number context for second-order derivatives (HVP, Hessian).
    ///
    /// Override this to propagate tangent information through the custom op.
    /// The default uses first-order chain rule: correct for gradients, but
    /// treats partials as constant for second-order derivatives.
    ///
    /// Primals are embedded in the dual numbers: `a.re` is the primal input.
    fn eval_dual(&self, a: Dual<F>, b: Dual<F>) -> Dual<F> {
        let result = self.eval(a.re, b.re);
        let (da, db) = self.partials(a.re, b.re, result);
        Dual::new(result, da * a.eps + db * b.eps)
    }

    /// Partials in dual-number context for second-order derivatives.
    ///
    /// Override this to return partials whose tangent components carry the
    /// derivative of the partial itself. The default returns constant partials.
    fn partials_dual(&self, a: Dual<F>, b: Dual<F>, result: Dual<F>) -> (Dual<F>, Dual<F>) {
        let (da, db) = self.partials(a.re, b.re, result.re);
        (Dual::constant(da), Dual::constant(db))
    }
}

/// Handle returned by [`BytecodeTape::register_custom`], used to invoke custom ops.
#[derive(Clone, Copy, Debug)]
pub struct CustomOpHandle(pub(crate) u16);

/// A bytecode tape that can be re-evaluated at different inputs.
///
/// Created via [`crate::api::record`]. After recording, call [`forward`](Self::forward)
/// to re-evaluate and [`reverse`](Self::reverse) to compute adjoints.
pub struct BytecodeTape<F: Float> {
    opcodes: Vec<OpCode>,
    arg_indices: Vec<[u32; 2]>,
    values: Vec<F>,
    num_inputs: u32,
    num_variables: u32,
    output_index: u32,
    /// Indices of multiple output variables (empty = single-output mode).
    output_indices: Vec<u32>,
    /// Registered custom operations (callback table).
    custom_ops: Vec<Arc<dyn CustomOp<F>>>,
    /// Second operand index for binary custom ops (sparse side table).
    /// Maps tape index → second operand index.
    custom_second_args: HashMap<u32, u32>,
}

impl<F: Float> BytecodeTape<F> {
    /// Create an empty bytecode tape.
    pub fn new() -> Self {
        BytecodeTape {
            opcodes: Vec::new(),
            arg_indices: Vec::new(),
            values: Vec::new(),
            num_inputs: 0,
            num_variables: 0,
            output_index: 0,
            output_indices: Vec::new(),
            custom_ops: Vec::new(),
            custom_second_args: HashMap::new(),
        }
    }

    /// Create a bytecode tape with pre-allocated capacity.
    pub fn with_capacity(est_ops: usize) -> Self {
        BytecodeTape {
            opcodes: Vec::with_capacity(est_ops),
            arg_indices: Vec::with_capacity(est_ops),
            values: Vec::with_capacity(est_ops),
            num_inputs: 0,
            num_variables: 0,
            output_index: 0,
            output_indices: Vec::new(),
            custom_ops: Vec::new(),
            custom_second_args: HashMap::new(),
        }
    }

    /// Register a new input variable. Returns its index.
    #[inline]
    pub fn new_input(&mut self, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.num_inputs += 1;
        self.opcodes.push(OpCode::Input);
        self.arg_indices.push([UNUSED, UNUSED]);
        self.values.push(value);
        idx
    }

    /// Register a scalar constant. Returns its index.
    #[inline]
    pub fn push_const(&mut self, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Const);
        self.arg_indices.push([UNUSED, UNUSED]);
        self.values.push(value);
        idx
    }

    /// Record an operation. Returns the result index.
    ///
    /// **Constant folding**: if all operands point to `Const` entries (not `Input`),
    /// the operation is replaced by a single `Const` with the already-computed value.
    ///
    /// **Algebraic simplification**: identity patterns (`x + 0 → x`, `x * 1 → x`,
    /// etc.) and absorbing patterns (`x * 0 → 0`, `x - x → 0`, `x / x → 1`) are
    /// detected and short-circuited. Absorbing patterns are guarded by a value check
    /// to handle NaN/Inf edge cases correctly.
    #[inline]
    pub fn push_op(&mut self, op: OpCode, arg0: u32, arg1: u32, value: F) -> u32 {
        // Constant folding: if both args (when present) are Const, emit Const instead.
        let arg0_const = self.opcodes[arg0 as usize] == OpCode::Const;
        let arg1_const = arg1 == UNUSED || self.opcodes[arg1 as usize] == OpCode::Const;
        if arg0_const && arg1_const {
            return self.push_const(value);
        }

        // Algebraic simplification: single-arg-const patterns (binary ops only).
        if (arg0_const || arg1_const) && arg1 != UNUSED {
            if let Some(idx) =
                self.try_algebraic_simplify(op, arg0, arg1, arg0_const, arg1_const, value)
            {
                return idx;
            }
        }

        // Same-index simplification: x - x → 0, x / x → 1.
        if arg0 == arg1 && arg1 != UNUSED {
            if let Some(idx) = self.try_same_index_simplify(op, value) {
                return idx;
            }
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(op);
        self.arg_indices.push([arg0, arg1]);
        self.values.push(value);
        idx
    }

    /// Try to simplify a binary op where exactly one argument is a known constant.
    ///
    /// Identity patterns (`x + 0`, `x * 1`, etc.) are always safe — they return
    /// the original index whose value is correct. Absorbing patterns (`x * 0`)
    /// use `push_const(value)` (not `push_const(F::zero())`) to preserve IEEE 754
    /// signed zero semantics, and are guarded by `value == expected` to handle
    /// NaN/Inf correctly (e.g., `NaN * 0 = NaN`, not `0`).
    #[inline(never)]
    fn try_algebraic_simplify(
        &mut self,
        op: OpCode,
        arg0: u32,
        arg1: u32,
        arg0_const: bool,
        arg1_const: bool,
        value: F,
    ) -> Option<u32> {
        let zero = F::zero();
        let one = F::one();
        match op {
            OpCode::Add => {
                if arg1_const && self.values[arg1 as usize] == zero {
                    return Some(arg0);
                }
                if arg0_const && self.values[arg0 as usize] == zero {
                    return Some(arg1);
                }
            }
            OpCode::Sub => {
                if arg1_const && self.values[arg1 as usize] == zero {
                    return Some(arg0);
                }
            }
            OpCode::Mul => {
                // Identity: x * 1 → x, 1 * x → x
                if arg1_const && self.values[arg1 as usize] == one {
                    return Some(arg0);
                }
                if arg0_const && self.values[arg0 as usize] == one {
                    return Some(arg1);
                }
                // Absorbing: x * 0 → const (guarded: NaN * 0 = NaN, not 0)
                if arg1_const && self.values[arg1 as usize] == zero && value == zero {
                    return Some(self.push_const(value));
                }
                if arg0_const && self.values[arg0 as usize] == zero && value == zero {
                    return Some(self.push_const(value));
                }
            }
            OpCode::Div => {
                if arg1_const && self.values[arg1 as usize] == one {
                    return Some(arg0);
                }
            }
            _ => {}
        }
        None
    }

    /// Try to simplify a binary op where both arguments are the same index.
    ///
    /// `x - x → 0` is guarded (Inf - Inf = NaN, not 0).
    /// `x / x → 1` is guarded (0/0 = NaN, not 1).
    #[inline(never)]
    fn try_same_index_simplify(&mut self, op: OpCode, value: F) -> Option<u32> {
        match op {
            OpCode::Sub if value == F::zero() => Some(self.push_const(value)),
            OpCode::Div if value == F::one() => Some(self.push_const(value)),
            _ => None,
        }
    }

    /// Record a powi operation. The `i32` exponent is stored in `arg_indices[1]`.
    ///
    /// **Constant folding**: if the operand is a `Const`, emit `Const` instead.
    ///
    /// **Algebraic simplification**: `x^0 → 1` (guarded), `x^1 → x`,
    /// `x^(-1) → Recip(x)` (cheaper unary dispatch).
    #[inline]
    pub fn push_powi(&mut self, arg0: u32, exp: i32, value: F) -> u32 {
        if self.opcodes[arg0 as usize] == OpCode::Const {
            return self.push_const(value);
        }

        // x^0 → 1 (guarded: 0^0 edge case — only fold when value is actually 1)
        if exp == 0 && value == F::one() {
            return self.push_const(F::one());
        }
        // x^1 → x
        if exp == 1 {
            return arg0;
        }
        // x^(-1) → Recip(x) (cheaper unary opcode dispatch)
        if exp == -1 {
            return self.push_op(OpCode::Recip, arg0, UNUSED, value);
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Powi);
        self.arg_indices.push([arg0, opcode::powi_exp_encode(exp)]);
        self.values.push(value);
        idx
    }

    /// Register a custom operation. Returns a handle for use with
    /// [`crate::BReverse::custom_unary`] and [`crate::BReverse::custom_binary`].
    pub fn register_custom(&mut self, op: Arc<dyn CustomOp<F>>) -> CustomOpHandle {
        let idx = self.custom_ops.len();
        assert!(idx <= u16::MAX as usize, "too many custom ops");
        self.custom_ops.push(op);
        CustomOpHandle(idx as u16)
    }

    /// Record a unary custom op. `arg_indices = [arg0, callback_idx]`.
    #[inline]
    pub fn push_custom_unary(&mut self, arg0: u32, handle: CustomOpHandle, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Custom);
        self.arg_indices.push([arg0, handle.0 as u32]);
        self.values.push(value);
        idx
    }

    /// Record a binary custom op. `arg_indices = [arg0, callback_idx]`,
    /// second operand stored in `custom_second_args`.
    #[inline]
    pub fn push_custom_binary(
        &mut self,
        arg0: u32,
        arg1: u32,
        handle: CustomOpHandle,
        value: F,
    ) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Custom);
        self.arg_indices.push([arg0, handle.0 as u32]);
        self.custom_second_args.insert(idx, arg1);
        self.values.push(value);
        idx
    }

    /// Mark the output variable.
    #[inline]
    pub fn set_output(&mut self, index: u32) {
        self.output_index = index;
    }

    /// Get the output value (available after `forward()` or initial recording).
    #[inline]
    pub fn output_value(&self) -> F {
        self.values[self.output_index as usize]
    }

    /// Index of the (single) output variable.
    ///
    /// Use this with the buffer produced by [`forward_tangent`](Self::forward_tangent)
    /// to read the output: `buf[tape.output_index()]`.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index as usize
    }

    /// Number of input variables.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Number of operations (including inputs and constants).
    #[inline]
    pub fn num_ops(&self) -> usize {
        self.opcodes.len()
    }

    // ── Multi-output support ──

    /// Mark multiple output variables.
    ///
    /// When set, [`num_outputs`](Self::num_outputs), [`output_values`](Self::output_values),
    /// [`jacobian`](Self::jacobian), and [`vjp_multi`](Self::vjp_multi) become available.
    /// Single-output methods (`output_index`, `gradient`, etc.) continue to work using
    /// the first output.
    pub fn set_outputs(&mut self, indices: &[u32]) {
        self.output_indices = indices.to_vec();
        if let Some(&first) = indices.first() {
            self.output_index = first;
        }
    }

    /// Number of output variables. Returns 1 in single-output mode.
    pub fn num_outputs(&self) -> usize {
        if self.output_indices.is_empty() {
            1
        } else {
            self.output_indices.len()
        }
    }

    /// Get all output values (available after `forward()` or initial recording).
    ///
    /// In single-output mode, returns a single-element vector.
    pub fn output_values(&self) -> Vec<F> {
        if self.output_indices.is_empty() {
            vec![self.values[self.output_index as usize]]
        } else {
            self.output_indices
                .iter()
                .map(|&idx| self.values[idx as usize])
                .collect()
        }
    }

    /// Indices of all output entries in the tape buffer.
    ///
    /// For multi-output tapes, returns all registered output indices.
    /// For single-output tapes, returns a single-element slice.
    pub fn all_output_indices(&self) -> &[u32] {
        if self.output_indices.is_empty() {
            std::slice::from_ref(&self.output_index)
        } else {
            &self.output_indices
        }
    }

    // ── GPU accessor methods ──

    /// Slice view of all opcodes in the tape.
    #[inline]
    pub fn opcodes_slice(&self) -> &[OpCode] {
        &self.opcodes
    }

    /// Slice view of all argument index pairs `[arg0, arg1]`.
    #[inline]
    pub fn arg_indices_slice(&self) -> &[[u32; 2]] {
        &self.arg_indices
    }

    /// Slice view of all primal values in the tape.
    #[inline]
    pub fn values_slice(&self) -> &[F] {
        &self.values
    }

    /// Total number of tape entries (inputs + constants + operations).
    #[inline]
    pub fn num_variables_count(&self) -> usize {
        self.num_variables as usize
    }

    /// Returns `true` if the tape contains any custom operations.
    #[inline]
    pub fn has_custom_ops(&self) -> bool {
        !self.custom_ops.is_empty()
    }

    /// Reverse sweep with weighted seeds for multiple outputs.
    ///
    /// Computes `∑_i weights[i] * ∂output_i/∂x` — a vector-Jacobian product.
    ///
    /// Returns the gradient with respect to all inputs (length [`num_inputs`](Self::num_inputs)).
    pub fn reverse_seeded(&self, seeds: &[F]) -> Vec<F> {
        let out_indices = self.all_output_indices();

        assert_eq!(
            seeds.len(),
            out_indices.len(),
            "seeds length must match number of outputs"
        );

        let ni = self.num_inputs as usize;
        let adjoints = self.reverse_seeded_full(seeds, out_indices);
        adjoints[..ni].to_vec()
    }

    /// Compute the full Jacobian of a multi-output tape via reverse mode.
    ///
    /// Performs `m` reverse sweeps (one per output). Returns `J[i][j] = ∂f_i/∂x_j`.
    pub fn jacobian(&mut self, inputs: &[F]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let out_indices = self.all_output_indices();

        let ni = self.num_inputs as usize;
        let mut jac = Vec::with_capacity(out_indices.len());

        for &out_idx in out_indices {
            let adjoints = self.reverse(out_idx);
            jac.push(adjoints[..ni].to_vec());
        }

        jac
    }

    /// Vector-Jacobian product for a multi-output tape.
    ///
    /// Computes `wᵀ · J` where `J` is the Jacobian. More efficient than
    /// computing the full Jacobian when only the weighted combination is needed.
    pub fn vjp_multi(&mut self, inputs: &[F], weights: &[F]) -> Vec<F> {
        self.forward(inputs);
        self.reverse_seeded(weights)
    }

    /// Compute a Jacobian with forced branch choices at specified tape indices.
    ///
    /// For each `(tape_index, sign)` in `forced_signs`, the reverse sweep uses
    /// [`forced_reverse_partials`](opcode::forced_reverse_partials) instead of the
    /// standard partials at that index.
    ///
    /// This is the building block for Clarke subdifferential enumeration.
    pub fn jacobian_limiting(&mut self, inputs: &[F], forced_signs: &[(u32, i8)]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let sign_map: HashMap<u32, i8> = forced_signs.iter().copied().collect();
        let out_indices = self.all_output_indices();

        let ni = self.num_inputs as usize;
        let mut jac = Vec::with_capacity(out_indices.len());

        for &out_idx in out_indices {
            let adjoints = self.reverse_with_forced_signs(out_idx, &sign_map);
            jac.push(adjoints[..ni].to_vec());
        }

        jac
    }

    /// Compute the Clarke generalized Jacobian via limiting Jacobian enumeration.
    ///
    /// 1. Runs `forward_nonsmooth` to detect all kink operations and their branches.
    /// 2. Identifies "active" kinks (|switching_value| < `tol`).
    /// 3. Enumerates all 2^k sign combinations for the k active kinks.
    /// 4. For each combination, computes a limiting Jacobian via forced reverse sweeps.
    ///
    /// Returns the nonsmooth info and a vector of limiting Jacobians.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ClarkeError::TooManyKinks`] if the number of active kinks exceeds
    /// the limit (default 20, overridden by `max_active_kinks`).
    pub fn clarke_jacobian(
        &mut self,
        inputs: &[F],
        tol: F,
        max_active_kinks: Option<usize>,
    ) -> Result<(crate::nonsmooth::NonsmoothInfo<F>, Vec<Vec<Vec<F>>>), crate::nonsmooth::ClarkeError>
    {
        #![allow(clippy::type_complexity)]
        let info = self.forward_nonsmooth(inputs);
        let active: Vec<&crate::nonsmooth::KinkEntry<F>> = info
            .active_kinks(tol)
            .into_iter()
            .filter(|k| opcode::has_nontrivial_subdifferential(k.opcode))
            .collect();
        let k = active.len();
        let limit = max_active_kinks.unwrap_or(20);

        if k > limit {
            return Err(crate::nonsmooth::ClarkeError::TooManyKinks { count: k, limit });
        }

        let active_indices: Vec<u32> = active.iter().map(|e| e.tape_index).collect();

        // Build sign_map from all (non-active) kinks using their natural branches,
        // then override active kinks per combination.
        let base_signs: HashMap<u32, i8> = info
            .kinks
            .iter()
            .map(|e| (e.tape_index, e.branch))
            .collect();

        let out_indices = self.all_output_indices();
        let ni = self.num_inputs as usize;

        let num_combos = 1usize << k;
        let mut jacobians = Vec::with_capacity(num_combos);

        for combo in 0..num_combos {
            let mut sign_map = base_signs.clone();
            for (bit, &idx) in active_indices.iter().enumerate() {
                let sign: i8 = if (combo >> bit) & 1 == 0 { 1 } else { -1 };
                sign_map.insert(idx, sign);
            }

            let mut jac = Vec::with_capacity(out_indices.len());
            for &out_idx in out_indices {
                let adjoints = self.reverse_with_forced_signs(out_idx, &sign_map);
                jac.push(adjoints[..ni].to_vec());
            }
            jacobians.push(jac);
        }

        Ok((info, jacobians))
    }

    /// Re-evaluate the tape at new inputs (forward sweep).
    ///
    /// Overwrites `values` in-place — no allocation.
    pub fn forward(&mut self, inputs: &[F]) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        // Overwrite input values.
        for (i, &v) in inputs.iter().enumerate() {
            self.values[i] = v;
        }

        // Re-evaluate all non-Input, non-Const ops.
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = self
                        .custom_second_args
                        .get(&(i as u32))
                        .map(|&bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    self.values[i] = self.custom_ops[cb_idx as usize].eval(a, b);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        self.values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    self.values[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Forward sweep with nonsmooth branch tracking.
    ///
    /// Calls [`forward`](Self::forward) to evaluate the tape, then scans for
    /// nonsmooth operations and records which branch was taken at each one.
    ///
    /// Tracked operations:
    /// - `Abs`, `Min`, `Max` — kinks with nontrivial subdifferentials
    /// - `Signum`, `Floor`, `Ceil`, `Round`, `Trunc` — step-function
    ///   discontinuities (zero derivative on both sides, tracked for proximity
    ///   detection only)
    ///
    /// Returns [`crate::NonsmoothInfo`] containing all kink entries in tape order.
    pub fn forward_nonsmooth(&mut self, inputs: &[F]) -> crate::nonsmooth::NonsmoothInfo<F> {
        self.forward(inputs);

        let mut kinks = Vec::new();
        for i in 0..self.opcodes.len() {
            let op = self.opcodes[i];
            if !opcode::is_nonsmooth(op) {
                continue;
            }

            let [a_idx, b_idx] = self.arg_indices[i];
            let a = self.values[a_idx as usize];

            match op {
                OpCode::Abs => {
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a,
                        branch: if a >= F::zero() { 1 } else { -1 },
                    });
                }
                OpCode::Max => {
                    let b = self.values[b_idx as usize];
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - b,
                        branch: if a >= b { 1 } else { -1 },
                    });
                }
                OpCode::Min => {
                    let b = self.values[b_idx as usize];
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - b,
                        branch: if a <= b { 1 } else { -1 },
                    });
                }
                OpCode::Signum => {
                    // Kink at x = 0 (same as Abs).
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a,
                        branch: if a >= F::zero() { 1 } else { -1 },
                    });
                }
                OpCode::Floor | OpCode::Ceil | OpCode::Round | OpCode::Trunc => {
                    // Kink at integer values. switching_value = distance to
                    // nearest integer: zero exactly at kink points, works
                    // symmetrically for both approach directions.
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - a.round(),
                        branch: if a - a.floor() < F::from(0.5).unwrap() {
                            1
                        } else {
                            -1
                        },
                    });
                }
                _ => unreachable!(),
            }
        }

        crate::nonsmooth::NonsmoothInfo { kinks }
    }

    /// Core reverse sweep loop shared by all scalar reverse sweep variants.
    ///
    /// Expects `adjoints` to be pre-seeded by the caller (length = `num_variables`).
    /// Reads primal values from `values` (either `self.values` or an external buffer).
    /// When `forced_signs` is `Some`, uses forced partials at matching tape indices.
    fn reverse_sweep_core(
        &self,
        adjoints: &mut [F],
        values: &[F],
        forced_signs: Option<&HashMap<u32, i8>>,
    ) {
        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoints[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    adjoints[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt.map(|bi| values[bi as usize]).unwrap_or(F::zero());
                    let r = values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoints[bi as usize] = adjoints[bi as usize] + db * adj;
                    }
                }
                op => {
                    adjoints[i] = F::zero();
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    let r = values[i];

                    let (da, db) = match forced_signs.and_then(|fs| fs.get(&(i as u32))) {
                        Some(&sign) => opcode::forced_reverse_partials(op, a, b, r, sign),
                        None => opcode::reverse_partials(op, a, b, r),
                    };

                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        adjoints[b_idx as usize] = adjoints[b_idx as usize] + db * adj;
                    }
                }
            }
        }
    }

    /// Reverse sweep: compute adjoints seeded at the output.
    ///
    /// Returns the full adjoint vector (length = `num_variables`).
    pub fn reverse(&self, seed_index: u32) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();
        self.reverse_sweep_core(&mut adjoints, &self.values, None);
        adjoints
    }

    /// Reverse sweep with forced branch choices at specified tape indices.
    fn reverse_with_forced_signs(
        &self,
        seed_index: u32,
        forced_signs: &HashMap<u32, i8>,
    ) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();
        self.reverse_sweep_core(&mut adjoints, &self.values, Some(forced_signs));
        adjoints
    }

    /// Forward evaluation into an external buffer.
    ///
    /// Reads opcodes, constants, and argument indices from `self`, but writes
    /// computed values into `values_buf` instead of `self.values`. This allows
    /// parallel evaluation of the same tape at different inputs without cloning.
    pub fn forward_into(&self, inputs: &[F], values_buf: &mut Vec<F>) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        let n = self.num_variables as usize;
        values_buf.clear();
        values_buf.resize(n, F::zero());

        // Copy constant values from the tape, then overwrite inputs.
        values_buf.copy_from_slice(&self.values[..n]);
        for (i, &v) in inputs.iter().enumerate() {
            values_buf[i] = v;
        }

        // Re-evaluate all non-Input, non-Const ops.
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = values_buf[a_idx as usize];
                    let b = self
                        .custom_second_args
                        .get(&(i as u32))
                        .map(|&bi| values_buf[bi as usize])
                        .unwrap_or(F::zero());
                    values_buf[i] = self.custom_ops[cb_idx as usize].eval(a, b);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = values_buf[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        values_buf[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    values_buf[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep reading from an external values buffer.
    ///
    /// Like [`reverse`](Self::reverse) but reads primal values from `values`
    /// instead of `self.values`. Pair with [`forward_into`](Self::forward_into)
    /// for parallel evaluation.
    pub fn reverse_from(&self, values: &[F], seed_index: u32) -> Vec<F> {
        let n = self.num_variables as usize;
        assert_eq!(values.len(), n, "values buffer has wrong length");
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();
        self.reverse_sweep_core(&mut adjoints, values, None);
        adjoints
    }

    /// Forward + reverse: compute the gradient at new inputs.
    ///
    /// Returns only the input adjoints (indices `0..num_inputs`).
    pub fn gradient(&mut self, inputs: &[F]) -> Vec<F> {
        self.forward(inputs);
        let adjoints = self.reverse(self.output_index);
        adjoints[..self.num_inputs as usize].to_vec()
    }

    /// Like [`gradient`](Self::gradient) but reuses a caller-provided buffer
    /// for the adjoint vector, avoiding allocation on repeated calls.
    pub fn gradient_with_buf(&mut self, inputs: &[F], adjoint_buf: &mut Vec<F>) -> Vec<F> {
        self.forward(inputs);

        let n = self.num_variables as usize;
        adjoint_buf.clear();
        adjoint_buf.resize(n, F::zero());
        adjoint_buf[self.output_index as usize] = F::one();

        self.reverse_sweep_core(adjoint_buf, &self.values, None);
        adjoint_buf[..self.num_inputs as usize].to_vec()
    }

    // ── Forward-over-reverse (second-order) ──

    /// Forward sweep with tangent-carrying numbers. Reads opcodes and constants
    /// from `self`, writing results into `buf`. Does not mutate the tape.
    ///
    /// Generic over `T: NumFloat` so it works with both `Dual<F>` and
    /// `DualVec<F, N>`.
    pub fn forward_tangent<T: NumFloat>(&self, inputs: &[T], buf: &mut Vec<T>) {
        self.forward_tangent_inner(inputs, buf, |i, a_t, b_t| {
            // First-order chain rule: evaluate on primals, convert partials to T.
            let [a_idx, cb_idx] = self.arg_indices[i];
            let a_primal = self.values[a_idx as usize];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let b_primal = b_idx_opt
                .map(|bi| self.values[bi as usize])
                .unwrap_or(F::zero());
            let result = self.custom_ops[cb_idx as usize].eval(a_primal, b_primal);
            let (da, db) = self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, result);
            let result_t = T::from(result).unwrap();
            let da_t = T::from(da).unwrap();
            let db_t = T::from(db).unwrap();
            let a_re_t = T::from(a_primal).unwrap();
            let b_re_t = T::from(b_primal).unwrap();
            result_t + da_t * (a_t - a_re_t) + db_t * (b_t - b_re_t)
        });
    }

    /// Forward sweep specialized for `Dual<F>`, calling [`CustomOp::eval_dual`]
    /// so that custom ops propagate tangent information for second-order derivatives.
    fn forward_tangent_dual(&self, inputs: &[Dual<F>], buf: &mut Vec<Dual<F>>) {
        self.forward_tangent_inner(inputs, buf, |i, a_t, b_t| {
            let [_a_idx, cb_idx] = self.arg_indices[i];
            self.custom_ops[cb_idx as usize].eval_dual(a_t, b_t)
        });
    }

    /// Common forward-tangent loop. The `handle_custom` closure receives
    /// `(tape_index, a_value, b_value)` for custom op slots and returns
    /// the result to store.
    fn forward_tangent_inner<T: NumFloat>(
        &self,
        inputs: &[T],
        buf: &mut Vec<T>,
        handle_custom: impl Fn(usize, T, T) -> T,
    ) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());

        let mut input_idx = 0usize;
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input => {
                    buf[i] = inputs[input_idx];
                    input_idx += 1;
                }
                OpCode::Const => {
                    buf[i] = T::from(self.values[i]).unwrap();
                }
                OpCode::Custom => {
                    let [a_idx, _cb_idx] = self.arg_indices[i];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let a_t = buf[a_idx as usize];
                    let b_t = b_idx_opt.map(|bi| buf[bi as usize]).unwrap_or(T::zero());
                    buf[i] = handle_custom(i, a_t, b_t);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = buf[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    buf[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep with tangent-carrying adjoints. Uses values from
    /// [`forward_tangent`](Self::forward_tangent). Uses [`IsAllZero`] to
    /// safely skip zero adjoints without dropping tangent contributions.
    fn reverse_tangent<T: NumFloat + IsAllZero>(&self, tangent_vals: &[T], buf: &mut Vec<T>) {
        self.reverse_tangent_inner(tangent_vals, buf, |i| {
            // First-order: convert primal-float partials to T.
            let [a_idx, cb_idx] = self.arg_indices[i];
            let a_primal = self.values[a_idx as usize];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let b_primal = b_idx_opt
                .map(|bi| self.values[bi as usize])
                .unwrap_or(F::zero());
            let r_primal = self.values[i];
            let (da, db) = self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, r_primal);
            (T::from(da).unwrap(), T::from(db).unwrap())
        });
    }

    /// Reverse sweep specialized for `Dual<F>`, calling [`CustomOp::partials_dual`]
    /// so that custom op partials carry tangent information for second-order derivatives.
    fn reverse_tangent_dual(&self, tangent_vals: &[Dual<F>], buf: &mut Vec<Dual<F>>) {
        self.reverse_tangent_inner(tangent_vals, buf, |i| {
            let [a_idx, cb_idx] = self.arg_indices[i];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let a_dual = tangent_vals[a_idx as usize];
            let b_dual = b_idx_opt
                .map(|bi| tangent_vals[bi as usize])
                .unwrap_or(Dual::constant(F::zero()));
            let r_dual = tangent_vals[i];
            self.custom_ops[cb_idx as usize].partials_dual(a_dual, b_dual, r_dual)
        });
    }

    /// Common reverse-tangent loop. The `custom_partials` closure receives
    /// `tape_index` for custom op slots and returns `(da, db)` as T-valued partials.
    fn reverse_tangent_inner<T: NumFloat + IsAllZero>(
        &self,
        tangent_vals: &[T],
        buf: &mut Vec<T>,
        custom_partials: impl Fn(usize) -> (T, T),
    ) {
        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());
        buf[self.output_index as usize] = T::one();

        for i in (0..self.opcodes.len()).rev() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, _cb_idx] = self.arg_indices[i];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let (da_t, db_t) = custom_partials(i);
                    buf[a_idx as usize] = buf[a_idx as usize] + da_t * adj;
                    if let Some(bi) = b_idx_opt {
                        buf[bi as usize] = buf[bi as usize] + db_t * adj;
                    }
                }
                op => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = tangent_vals[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        tangent_vals[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    let r = tangent_vals[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    buf[a_idx as usize] = buf[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize] = buf[b_idx as usize] + db * adj;
                    }
                }
            }
        }
    }

    /// Hessian-vector product via forward-over-reverse.
    ///
    /// Returns `(gradient, H·v)` where both are `Vec<F>` of length
    /// [`num_inputs`](Self::num_inputs). The tape is not mutated.
    pub fn hvp(&self, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
        let mut dual_vals = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.hvp_with_buf(x, v, &mut dual_vals, &mut adjoint_buf)
    }

    /// Like [`hvp`](Self::hvp) but reuses caller-provided buffers to avoid
    /// allocation on repeated calls (e.g. inside [`hessian`](Self::hessian)).
    pub fn hvp_with_buf(
        &self,
        x: &[F],
        v: &[F],
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) -> (Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v.len(), n, "wrong number of directions");

        let dual_inputs: Vec<Dual<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| Dual::new(xi, vi))
            .collect();

        self.forward_tangent_dual(&dual_inputs, dual_vals_buf);
        self.reverse_tangent_dual(dual_vals_buf, adjoint_buf);

        let gradient: Vec<F> = (0..n).map(|i| adjoint_buf[i].re).collect();
        let hvp: Vec<F> = (0..n).map(|i| adjoint_buf[i].eps).collect();
        (gradient, hvp)
    }

    /// Like [`hvp_with_buf`](Self::hvp_with_buf) but also reuses a caller-provided
    /// input buffer, eliminating all allocations on repeated calls.
    fn hvp_with_all_bufs(
        &self,
        x: &[F],
        v: &[F],
        dual_input_buf: &mut Vec<Dual<F>>,
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) {
        // Reuse the input buffer instead of allocating
        dual_input_buf.clear();
        dual_input_buf.extend(x.iter().zip(v.iter()).map(|(&xi, &vi)| Dual::new(xi, vi)));

        self.forward_tangent_dual(dual_input_buf, dual_vals_buf);
        self.reverse_tangent_dual(dual_vals_buf, adjoint_buf);
    }

    /// Full Hessian matrix via `n` Hessian-vector products.
    ///
    /// Returns `(value, gradient, hessian)` where `hessian[i][j] = ∂²f/∂x_i∂x_j`.
    /// The tape is not mutated.
    pub fn hessian(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        for j in 0..n {
            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf
                .extend((0..n).map(|i| Dual::new(x[i], if i == j { F::one() } else { F::zero() })));

            self.forward_tangent_dual(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent_dual(&dual_vals_buf, &mut adjoint_buf);

            if j == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for (row, adj) in hessian.iter_mut().zip(adjoint_buf.iter()) {
                row[j] = adj.eps;
            }
        }

        (value, gradient, hessian)
    }

    /// Full Hessian matrix via batched forward-over-reverse.
    ///
    /// Processes `ceil(n/N)` batches instead of `n` individual HVPs,
    /// computing N Hessian columns simultaneously.
    pub fn hessian_vec<const N: usize>(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let num_batches = n.div_ceil(N);
        for batch in 0..num_batches {
            let base = batch * N;

            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let col = base + lane;
                    if col < n && i == col {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if batch == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for lane in 0..N {
                let col = base + lane;
                if col >= n {
                    break;
                }
                for i in 0..n {
                    hessian[i][col] = adjoint_buf[i].eps[lane];
                }
            }
        }

        (value, gradient, hessian)
    }

    /// Detect the structural sparsity pattern of the Hessian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets.
    /// At nonlinear operations, marks cross-pairs as potential Hessian interactions.
    pub fn detect_sparsity(&self) -> crate::sparse::SparsityPattern {
        crate::sparse::detect_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            self.num_inputs as usize,
            self.num_variables as usize,
        )
    }

    /// Detect the structural sparsity pattern of the Jacobian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets (first-order).
    /// For each output, determines which inputs it depends on.
    pub fn detect_jacobian_sparsity(&self) -> crate::sparse::JacobianSparsityPattern {
        let out_indices = self.all_output_indices();
        crate::sparse::detect_jacobian_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            self.num_inputs as usize,
            self.num_variables as usize,
            out_indices,
        )
    }

    /// Compute a sparse Hessian using structural sparsity detection and graph coloring.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)` where
    /// `hessian_values[k]` corresponds to `(pattern.rows[k], pattern.cols[k])`.
    ///
    /// For problems with sparse Hessians, this requires only `chromatic_number`
    /// HVP calls instead of `n`, which can be dramatically fewer for banded
    /// or sparse interaction structures.
    pub fn sparse_hessian(&self, x: &[F]) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut v = vec![F::zero(); n];

        for color in 0..num_colors {
            // Form direction vector: v[i] = 1 if colors[i] == color, else 0
            for i in 0..n {
                v[i] = if colors[i] == color {
                    F::one()
                } else {
                    F::zero()
                };
            }

            self.hvp_with_all_bufs(
                x,
                &v,
                &mut dual_input_buf,
                &mut dual_vals_buf,
                &mut adjoint_buf,
            );

            if color == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            // Extract Hessian entries for this color.
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    hessian_values[k] = adjoint_buf[row as usize].eps;
                }
            }
        }

        (value, gradient, pattern, hessian_values)
    }

    /// Batched sparse Hessian: packs N colors per sweep using DualVec.
    ///
    /// Reduces the number of forward+reverse sweeps from `num_colors` to
    /// `ceil(num_colors / N)`. Each sweep processes N colors simultaneously.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)`.
    pub fn sparse_hessian_vec<const N: usize>(
        &self,
        x: &[F],
    ) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            // Build DualVec inputs: lane k has v[i]=1 if colors[i] == base_color+k
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let target_color = base_color + lane as u32;
                    if target_color < num_colors && colors[i] == target_color {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if batch == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            // Extract Hessian entries: for entry (row, col) with colors[col] == base_color+lane,
            // read adjoint_buf[row].eps[lane]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                let col_color = colors[col as usize];
                if col_color >= base_color && col_color < base_color + N as u32 {
                    let lane = (col_color - base_color) as usize;
                    hessian_values[k] = adjoint_buf[row as usize].eps[lane];
                }
            }
        }

        (value, gradient, pattern, hessian_values)
    }

    // ── Batch evaluation ──

    /// Evaluate the gradient at multiple input points.
    ///
    /// Returns one gradient vector per input point.
    pub fn gradient_batch(&mut self, inputs: &[&[F]]) -> Vec<Vec<F>> {
        inputs.iter().map(|x| self.gradient(x)).collect()
    }

    // ── Higher-order derivatives ──

    /// Third-order directional derivative: `∑_{jk} (∂³f/∂x_i∂x_j∂x_k) v1_j v2_k`.
    ///
    /// Given directions `v1` and `v2`, computes:
    /// - `gradient`: `∇f(x)`
    /// - `hvp`: `H(x) · v1` (Hessian-vector product)
    /// - `third`: `(∂/∂v2)(H · v1)` (third-order tensor contracted with v1 and v2)
    ///
    /// Uses `Dual<Dual<F>>` (nested dual numbers): inner tangent for `v1`,
    /// outer tangent for `v2`.
    pub fn third_order_hvvp(&self, x: &[F], v1: &[F], v2: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v1.len(), n, "wrong v1 length");
        assert_eq!(v2.len(), n, "wrong v2 length");

        // Build Dual<Dual<F>> inputs:
        //   inner.re = x[i], inner.eps = v1[i]  (for HVP direction)
        //   outer.re = inner, outer.eps = v2[i]  (for third-order direction)
        //
        // outer = Dual { re: Dual(x[i], v1[i]), eps: Dual(v2[i], 0) }
        let dd_inputs: Vec<Dual<Dual<F>>> = (0..n)
            .map(|i| Dual {
                re: Dual::new(x[i], v1[i]),
                eps: Dual::new(v2[i], F::zero()),
            })
            .collect();

        let mut dd_vals: Vec<Dual<Dual<F>>> = Vec::new();
        let mut dd_adj: Vec<Dual<Dual<F>>> = Vec::new();

        self.forward_tangent(&dd_inputs, &mut dd_vals);
        self.reverse_tangent(&dd_vals, &mut dd_adj);

        let gradient: Vec<F> = (0..n).map(|i| dd_adj[i].re.re).collect();
        let hvp: Vec<F> = (0..n).map(|i| dd_adj[i].re.eps).collect();
        let third: Vec<F> = (0..n).map(|i| dd_adj[i].eps.eps).collect();

        (gradient, hvp, third)
    }

    /// Forward-reverse Taylor pass for gradient + higher-order directional adjoints.
    ///
    /// Builds Taylor inputs `x_i(t) = x_i + v_i * t` (with zero higher coefficients),
    /// runs `forward_tangent`, then `reverse_tangent` to get Taylor-valued adjoints.
    ///
    /// Returns `(output, adjoints)` where:
    /// - `output` is the Taylor expansion of `f` along direction `v`
    /// - `adjoints[i].coeff(0)` = `∂f/∂x_i` (gradient)
    /// - `adjoints[i].coeff(1)` = `Σ_j (∂²f/∂x_i∂x_j) v_j` (HVP)
    /// - `adjoints[i].derivative(k)` = k-th order directional adjoint
    ///
    /// For K=2, the HVP component is equivalent to [`hvp`](Self::hvp).
    /// For K≥3, yields additional higher-order information in the same pass.
    ///
    /// Like [`hvp`](Self::hvp), takes `&self` and does not call `forward(x)`
    /// before the Taylor pass. Custom ops will use primal values from recording time.
    #[cfg(feature = "taylor")]
    pub fn taylor_grad<const K: usize>(
        &self,
        x: &[F],
        v: &[F],
    ) -> (Taylor<F, K>, Vec<Taylor<F, K>>) {
        let mut fwd_buf = Vec::new();
        let mut adj_buf = Vec::new();
        self.taylor_grad_with_buf(x, v, &mut fwd_buf, &mut adj_buf)
    }

    /// Like [`taylor_grad`](Self::taylor_grad) but reuses caller-provided buffers
    /// to avoid allocation on repeated calls.
    #[cfg(feature = "taylor")]
    pub fn taylor_grad_with_buf<const K: usize>(
        &self,
        x: &[F],
        v: &[F],
        fwd_buf: &mut Vec<Taylor<F, K>>,
        adj_buf: &mut Vec<Taylor<F, K>>,
    ) -> (Taylor<F, K>, Vec<Taylor<F, K>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v.len(), n, "wrong number of directions");

        // Build Taylor inputs: x_i(t) = x_i + v_i * t
        let taylor_inputs: Vec<Taylor<F, K>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| {
                let mut coeffs = [F::zero(); K];
                coeffs[0] = xi;
                if K > 1 {
                    coeffs[1] = vi;
                }
                Taylor::new(coeffs)
            })
            .collect();

        self.forward_tangent(&taylor_inputs, fwd_buf);
        let output = fwd_buf[self.output_index as usize];
        self.reverse_tangent(fwd_buf, adj_buf);

        (output, adj_buf[..n].to_vec())
    }

    // ── ODE Taylor integration ──

    /// Compute the Taylor expansion of the ODE solution `y(t)` to order K.
    ///
    /// Given a tape representing the right-hand side `f: R^n → R^n` of the ODE
    /// `y' = f(y)`, and an initial condition `y(0) = y0`, computes the Taylor
    /// coefficients `y_0, y_1, ..., y_{K-1}` such that
    /// `y(t) ≈ y_0 + y_1·t + y_2·t² + ... + y_{K-1}·t^{K-1}`.
    ///
    /// The tape must have `num_outputs == num_inputs` (autonomous ODE: f maps R^n → R^n).
    ///
    /// Returns one `Taylor<F, K>` per state variable. Use [`Taylor::eval_at`] to
    /// evaluate at a step size `h`, or inspect coefficients for error estimation.
    #[cfg(feature = "taylor")]
    pub fn ode_taylor_step<const K: usize>(&self, y0: &[F]) -> Vec<Taylor<F, K>> {
        let mut buf = Vec::new();
        self.ode_taylor_step_with_buf(y0, &mut buf)
    }

    /// Like [`ode_taylor_step`](Self::ode_taylor_step) but reuses a caller-provided
    /// buffer to avoid allocation on repeated calls.
    #[cfg(feature = "taylor")]
    pub fn ode_taylor_step_with_buf<const K: usize>(
        &self,
        y0: &[F],
        buf: &mut Vec<Taylor<F, K>>,
    ) -> Vec<Taylor<F, K>> {
        let n = self.num_inputs as usize;
        assert_eq!(y0.len(), n, "y0 length must match num_inputs");
        assert_eq!(
            self.num_outputs(),
            n,
            "ODE tape must have num_outputs == num_inputs (f: R^n -> R^n)"
        );

        let out_indices = self.all_output_indices();

        let mut y_coeffs = vec![[F::zero(); K]; n];
        for i in 0..n {
            y_coeffs[i][0] = y0[i];
        }

        for k in 0..K - 1 {
            let inputs: Vec<Taylor<F, K>> = (0..n).map(|i| Taylor::new(y_coeffs[i])).collect();

            self.forward_tangent(&inputs, buf);

            let divisor = F::from(k + 1).unwrap();
            for i in 0..n {
                y_coeffs[i][k + 1] = buf[out_indices[i] as usize].coeff(k) / divisor;
            }
        }

        (0..n).map(|i| Taylor::new(y_coeffs[i])).collect()
    }

    // ── Tape optimizations ──

    /// Eliminate dead (unreachable) entries from the tape.
    ///
    /// Walks backward from all outputs, marks reachable entries, then compacts
    /// the tape in-place with an index remap. Inputs are never removed.
    /// Core DCE: reachability walk from `seeds`, compact tape, return index remap.
    ///
    /// Shared by [`dead_code_elimination`](Self::dead_code_elimination) and
    /// [`dead_code_elimination_for_outputs`](Self::dead_code_elimination_for_outputs).
    /// Callers handle output index updates after compaction.
    fn dce_compact(&mut self, seeds: &[u32]) -> Vec<u32> {
        let n = self.opcodes.len();
        let mut reachable = vec![false; n];

        // Mark all inputs as reachable.
        for flag in reachable.iter_mut().take(self.num_inputs as usize) {
            *flag = true;
        }

        let mut stack: Vec<u32> = seeds.to_vec();

        while let Some(idx) = stack.pop() {
            let i = idx as usize;
            if reachable[i] {
                continue;
            }
            reachable[i] = true;
            let [a, b] = self.arg_indices[i];
            if a != UNUSED {
                stack.push(a);
            }
            if b != UNUSED && self.opcodes[i] != OpCode::Powi {
                stack.push(b);
            }
        }

        // Build remap: old index -> new index.
        let mut remap = vec![0u32; n];
        let mut new_idx = 0u32;
        for i in 0..n {
            if reachable[i] {
                remap[i] = new_idx;
                new_idx += 1;
            }
        }
        let new_len = new_idx as usize;

        // Compact in-place.
        let mut write = 0;
        for (read, &is_reachable) in reachable.iter().enumerate().take(n) {
            if is_reachable {
                self.opcodes[write] = self.opcodes[read];
                self.values[write] = self.values[read];
                let [a, b] = self.arg_indices[read];
                let ra = if a != UNUSED {
                    remap[a as usize]
                } else {
                    UNUSED
                };
                let rb = if b != UNUSED && self.opcodes[read] != OpCode::Powi {
                    remap[b as usize]
                } else {
                    b
                };
                self.arg_indices[write] = [ra, rb];
                write += 1;
            }
        }

        self.opcodes.truncate(new_len);
        self.arg_indices.truncate(new_len);
        self.values.truncate(new_len);
        self.num_variables = new_len as u32;

        remap
    }

    pub fn dead_code_elimination(&mut self) {
        let mut seeds = vec![self.output_index];
        seeds.extend_from_slice(&self.output_indices);
        let remap = self.dce_compact(&seeds);
        self.output_index = remap[self.output_index as usize];
        for oi in &mut self.output_indices {
            *oi = remap[*oi as usize];
        }
    }

    /// Eliminate dead code, keeping only the specified outputs alive.
    ///
    /// Like [`dead_code_elimination`](Self::dead_code_elimination) but seeds
    /// reachability only from `active_outputs`. After compaction,
    /// `output_indices` contains only the active outputs (remapped), and
    /// `output_index` is set to the first active output.
    ///
    /// **Note**: Like `dead_code_elimination`, this does not remap
    /// `custom_second_args` (pre-existing limitation).
    ///
    /// # Panics
    /// Panics if `active_outputs` is empty.
    pub fn dead_code_elimination_for_outputs(&mut self, active_outputs: &[u32]) {
        assert!(
            !active_outputs.is_empty(),
            "active_outputs must not be empty"
        );
        let remap = self.dce_compact(active_outputs);
        self.output_indices = active_outputs
            .iter()
            .map(|&oi| remap[oi as usize])
            .collect();
        self.output_index = self.output_indices[0];
    }

    /// Common subexpression elimination.
    ///
    /// Deduplicates identical `(OpCode, arg0, arg1)` triples, normalising
    /// argument order for commutative ops. Finishes with a DCE pass to
    /// remove the now-dead duplicates.
    pub fn cse(&mut self) {
        use std::collections::HashMap;

        let n = self.opcodes.len();
        // Maps canonical (op, arg0, arg1) -> first index that computed it.
        let mut seen: HashMap<(OpCode, u32, u32), u32> = HashMap::new();
        // remap[i] = canonical index for entry i (identity by default).
        let mut remap: Vec<u32> = (0..n as u32).collect();

        let is_commutative = |op: OpCode| -> bool {
            matches!(
                op,
                OpCode::Add | OpCode::Mul | OpCode::Max | OpCode::Min | OpCode::Hypot
            )
        };

        for i in 0..n {
            let op = self.opcodes[i];
            match op {
                OpCode::Input | OpCode::Const => continue,
                _ => {}
            }

            let [mut a, mut b] = self.arg_indices[i];
            // Apply remap to args (except Powi exponent in b).
            a = remap[a as usize];
            if b != UNUSED && op != OpCode::Powi {
                b = remap[b as usize];
            }
            // Update arg_indices with remapped values.
            self.arg_indices[i] = [a, b];

            // Build the canonical key.
            let key = if b == UNUSED {
                // Unary: hash (op, arg0) only; use UNUSED as placeholder.
                (op, a, UNUSED)
            } else if is_commutative(op) {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                (op, lo, hi)
            } else {
                (op, a, b)
            };

            if let Some(&canonical) = seen.get(&key) {
                remap[i] = canonical;
            } else {
                seen.insert(key, i as u32);
            }
        }

        // Apply remap to all arg_indices (for entries that reference CSE'd nodes).
        for i in 0..n {
            let op = self.opcodes[i];
            if matches!(op, OpCode::Input | OpCode::Const) {
                continue;
            }
            let [a, b] = self.arg_indices[i];
            let ra = remap[a as usize];
            let rb = if b != UNUSED && op != OpCode::Powi {
                remap[b as usize]
            } else {
                b
            };
            self.arg_indices[i] = [ra, rb];
        }

        // Update output indices.
        self.output_index = remap[self.output_index as usize];
        for oi in &mut self.output_indices {
            *oi = remap[*oi as usize];
        }

        // DCE removes the now-unreachable duplicate entries.
        self.dead_code_elimination();
    }

    // ── Sparse Jacobian ──

    /// Compute a sparse Jacobian using structural sparsity detection and graph coloring.
    ///
    /// Auto-selects forward-mode (column compression) or reverse-mode (row compression)
    /// based on which requires fewer sweeps.
    ///
    /// Returns `(output_values, pattern, jacobian_values)`.
    pub fn sparse_jacobian(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (col_colors, num_col_colors) = crate::sparse::column_coloring(&pattern);
        let (row_colors, num_row_colors) = crate::sparse::row_coloring(&pattern);

        if num_col_colors <= num_row_colors {
            let jac_values =
                self.sparse_jacobian_forward_impl(x, &pattern, &col_colors, num_col_colors);
            let outputs = self.output_values();
            (outputs, pattern, jac_values)
        } else {
            let jac_values =
                self.sparse_jacobian_reverse_impl(x, &pattern, &row_colors, num_row_colors);
            let outputs = self.output_values();
            (outputs, pattern, jac_values)
        }
    }

    /// Sparse Jacobian via forward-mode (column compression).
    pub fn sparse_jacobian_forward(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);
        let jac_values = self.sparse_jacobian_forward_impl(x, &pattern, &colors, num_colors);
        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Jacobian via reverse-mode (row compression).
    pub fn sparse_jacobian_reverse(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::row_coloring(&pattern);
        let jac_values = self.sparse_jacobian_reverse_impl(x, &pattern, &colors, num_colors);
        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Jacobian with a precomputed sparsity pattern and coloring.
    ///
    /// Skips re-detection of sparsity on repeated calls. Use `column_coloring` colors
    /// for forward mode or `row_coloring` colors for reverse mode. The `forward` flag
    /// selects the mode.
    pub fn sparse_jacobian_with_pattern(
        &mut self,
        x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
        forward_mode: bool,
    ) -> (Vec<F>, Vec<F>) {
        self.forward(x);
        let jac_values = if forward_mode {
            self.sparse_jacobian_forward_impl(x, pattern, colors, num_colors)
        } else {
            self.sparse_jacobian_reverse_impl(x, pattern, colors, num_colors)
        };
        let outputs = self.output_values();
        (outputs, jac_values)
    }

    /// Forward-mode sparse Jacobian implementation (column compression).
    ///
    /// Each color group seeds a forward pass with tangent 1.0 in the columns
    /// sharing that color. The resulting tangent at each output gives J[row][col].
    fn sparse_jacobian_forward_impl(
        &self,
        x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<Dual<F>> = Vec::new();

        for color in 0..num_colors {
            // Build Dual inputs: tangent = 1 for inputs with this color
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                Dual::new(
                    x[i],
                    if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    },
                )
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            // Extract Jacobian entries: for entry (row, col) with colors[col] == color,
            // the tangent at output_indices[row] gives J[row][col]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    jac_values[k] = dual_vals_buf[out_indices[row as usize] as usize].eps;
                }
            }
        }

        jac_values
    }

    /// Reverse-mode sparse Jacobian implementation (row compression).
    ///
    /// Each color group seeds a reverse pass with adjoint 1.0 at the outputs
    /// sharing that color.
    fn sparse_jacobian_reverse_impl(
        &self,
        _x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let m = self.num_outputs();
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

        for color in 0..num_colors {
            // Build seeds: weight = 1 for outputs with this color
            let seeds: Vec<F> = (0..m)
                .map(|i| {
                    if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();

            let adjoints = self.reverse_seeded_full(&seeds, out_indices);

            // Extract Jacobian entries: for entry (row, col) with colors[row] == color,
            // adjoint[col] gives J[row][col]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[row as usize] == color {
                    jac_values[k] = adjoints[col as usize];
                }
            }
        }

        jac_values
    }

    /// Reverse sweep with weighted seeds, returning full adjoint vector.
    fn reverse_seeded_full(&self, seeds: &[F], out_indices: &[u32]) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];

        for (&out_idx, &weight) in out_indices.iter().zip(seeds.iter()) {
            if weight == F::zero() {
                continue;
            }
            adjoints[out_idx as usize] = adjoints[out_idx as usize] + weight;
        }

        self.reverse_sweep_core(&mut adjoints, &self.values, None);
        adjoints
    }

    /// Batched sparse Jacobian: packs N colors per forward sweep using DualVec.
    ///
    /// Reduces the number of forward sweeps from `num_colors` to
    /// `ceil(num_colors / N)`.
    pub fn sparse_jacobian_vec<const N: usize>(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);

        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let target_color = base_color + lane as u32;
                    if target_color < num_colors && colors[i] == target_color {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                let col_color = colors[col as usize];
                if col_color >= base_color && col_color < base_color + N as u32 {
                    let lane = (col_color - base_color) as usize;
                    jac_values[k] = dual_vals_buf[out_indices[row as usize] as usize].eps[lane];
                }
            }
        }

        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Dense Jacobian via forward mode (one forward-tangent pass per input).
    ///
    /// More efficient than reverse mode when `num_inputs < num_outputs`.
    pub fn jacobian_forward(&self, x: &[F]) -> Vec<Vec<F>> {
        let n = self.num_inputs as usize;

        let out_indices = self.all_output_indices();
        let m = out_indices.len();

        let mut jac = vec![vec![F::zero(); n]; m];
        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<Dual<F>> = Vec::new();

        // Indexing by `col` is clearer than enumerate here: col seeds the tangent direction
        #[allow(clippy::needless_range_loop)]
        for col in 0..n {
            dual_input_buf.clear();
            dual_input_buf.extend(
                (0..n).map(|i| Dual::new(x[i], if i == col { F::one() } else { F::zero() })),
            );

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (row_idx, &out_idx) in out_indices.iter().enumerate() {
                jac[row_idx][col] = dual_vals_buf[out_idx as usize].eps;
            }
        }

        jac
    }

    /// Dense Jacobian via cross-country (vertex) elimination.
    ///
    /// Builds a linearized DAG from the tape, then eliminates intermediate
    /// vertices in Markowitz order. For functions where `m ≈ n` and the
    /// graph has moderate connectivity, this can require fewer operations
    /// than either pure forward mode (`n` passes) or reverse mode (`m` passes).
    pub fn jacobian_cross_country(&mut self, inputs: &[F]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let out_indices = self.all_output_indices();

        let mut graph = crate::cross_country::LinearizedGraph::from_tape(
            &self.opcodes,
            &self.arg_indices,
            &self.values,
            self.num_inputs as usize,
            out_indices,
            &self.custom_ops,
            &self.custom_second_args,
        );

        graph.eliminate_all();
        graph.extract_jacobian()
    }

    /// Sparse Hessian with a precomputed sparsity pattern and coloring.
    ///
    /// Skips re-detection on repeated calls (e.g. in solver loops).
    pub fn sparse_hessian_with_pattern(
        &self,
        x: &[F],
        pattern: &crate::sparse::SparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> (F, Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut v = vec![F::zero(); n];

        for color in 0..num_colors {
            for i in 0..n {
                v[i] = if colors[i] == color {
                    F::one()
                } else {
                    F::zero()
                };
            }

            self.hvp_with_all_bufs(
                x,
                &v,
                &mut dual_input_buf,
                &mut dual_vals_buf,
                &mut adjoint_buf,
            );

            if color == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    hessian_values[k] = adjoint_buf[row as usize].eps;
                }
            }
        }

        (value, gradient, hessian_values)
    }

    /// Run all tape optimizations: CSE followed by DCE.
    ///
    /// In debug builds, validates internal consistency after optimization.
    pub fn optimize(&mut self) {
        self.cse();
        self.dead_code_elimination();

        // Validate internal consistency in debug builds.
        #[cfg(debug_assertions)]
        {
            let n = self.opcodes.len();
            // All arg_indices must point to valid entries.
            for i in 0..n {
                let [a, b] = self.arg_indices[i];
                match self.opcodes[i] {
                    OpCode::Input | OpCode::Const => {
                        assert_eq!(a, UNUSED, "Input/Const should have UNUSED args");
                        assert_eq!(b, UNUSED, "Input/Const should have UNUSED args");
                    }
                    OpCode::Powi => {
                        assert!(
                            (a as usize) < n,
                            "Powi arg0 {} out of bounds (tape len {})",
                            a,
                            n
                        );
                    }
                    _ => {
                        assert!(
                            (a as usize) < i,
                            "arg0 {} not before op {} (tape len {})",
                            a,
                            i,
                            n
                        );
                        if b != UNUSED {
                            assert!(
                                (b as usize) < i,
                                "arg1 {} not before op {} (tape len {})",
                                b,
                                i,
                                n
                            );
                        }
                    }
                }
            }
            // output_index must be valid.
            assert!(
                (self.output_index as usize) < n,
                "output_index {} out of bounds (tape len {})",
                self.output_index,
                n
            );
            for &oi in &self.output_indices {
                assert!(
                    (oi as usize) < n,
                    "output_indices entry {} out of bounds (tape len {})",
                    oi,
                    n
                );
            }
            // num_inputs must be preserved.
            let input_count = self
                .opcodes
                .iter()
                .filter(|&&op| op == OpCode::Input)
                .count();
            assert_eq!(
                input_count, self.num_inputs as usize,
                "num_inputs mismatch after optimization"
            );
        }
    }
}

impl<F: Float> Default for BytecodeTape<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════
//  Parallel methods (rayon)
// ══════════════════════════════════════════════

#[cfg(feature = "parallel")]
impl<F: Float> BytecodeTape<F> {
    /// Parallel gradient: forward + reverse using external buffers.
    ///
    /// Takes `&self` instead of `&mut self`, enabling shared access across threads.
    pub fn gradient_par(&self, inputs: &[F]) -> Vec<F> {
        let mut values_buf = Vec::new();
        self.forward_into(inputs, &mut values_buf);
        let adjoints = self.reverse_from(&values_buf, self.output_index);
        adjoints[..self.num_inputs as usize].to_vec()
    }

    /// Parallel Jacobian: one reverse sweep per output, parallelized.
    ///
    /// Returns `J[i][j] = ∂f_i/∂x_j`.
    pub fn jacobian_par(&self, inputs: &[F]) -> Vec<Vec<F>> {
        use rayon::prelude::*;

        let mut values_buf = Vec::new();
        self.forward_into(inputs, &mut values_buf);

        let out_indices = self.all_output_indices();

        let ni = self.num_inputs as usize;
        out_indices
            .par_iter()
            .map(|&out_idx| {
                let adjoints = self.reverse_from(&values_buf, out_idx);
                adjoints[..ni].to_vec()
            })
            .collect()
    }

    /// Parallel Hessian: one HVP per column, parallelized over columns.
    ///
    /// Returns `(value, gradient, hessian)`.
    pub fn hessian_par(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        // Compute gradient and value from column 0 (serial).
        let dual_input_buf: Vec<Dual<F>> = (0..n)
            .map(|i| Dual::new(x[i], if i == 0 { F::one() } else { F::zero() }))
            .collect();
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.forward_tangent_dual(&dual_input_buf, &mut dual_vals_buf);
        self.reverse_tangent_dual(&dual_vals_buf, &mut adjoint_buf);

        let value = dual_vals_buf[self.output_index as usize].re;
        let gradient: Vec<F> = (0..n).map(|i| adjoint_buf[i].re).collect();
        let col0: Vec<F> = (0..n).map(|i| adjoint_buf[i].eps).collect();

        // Parallelize remaining columns.
        let other_cols: Vec<Vec<F>> = (1..n)
            .into_par_iter()
            .map(|j| {
                let inputs: Vec<Dual<F>> = (0..n)
                    .map(|i| Dual::new(x[i], if i == j { F::one() } else { F::zero() }))
                    .collect();
                let mut dv = Vec::new();
                let mut ab = Vec::new();
                self.forward_tangent_dual(&inputs, &mut dv);
                self.reverse_tangent_dual(&dv, &mut ab);
                (0..n).map(|i| ab[i].eps).collect()
            })
            .collect();

        let mut hessian = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            hessian[i][0] = col0[i];
        }
        for (j_minus_1, col) in other_cols.iter().enumerate() {
            let j = j_minus_1 + 1;
            for i in 0..n {
                hessian[i][j] = col[i];
            }
        }

        (value, gradient, hessian)
    }

    /// Parallel sparse Hessian: parallelized over colors.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)`.
    pub fn sparse_hessian_par(
        &self,
        x: &[F],
    ) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        // Compute value/gradient from color 0 (serial).
        let mut v0 = vec![F::zero(); n];
        for i in 0..n {
            v0[i] = if colors[i] == 0 { F::one() } else { F::zero() };
        }
        let di: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], v0[i])).collect();
        let mut dv = Vec::new();
        let mut ab = Vec::new();
        self.forward_tangent_dual(&di, &mut dv);
        self.reverse_tangent_dual(&dv, &mut ab);
        let value = dv[self.output_index as usize].re;
        let gradient: Vec<F> = (0..n).map(|i| ab[i].re).collect();

        // Collect all color results in parallel.
        let color_results: Vec<Vec<Dual<F>>> = (0..num_colors)
            .into_par_iter()
            .map(|color| {
                let mut v = vec![F::zero(); n];
                for i in 0..n {
                    v[i] = if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    };
                }
                let inputs: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], v[i])).collect();
                let mut dv_local = Vec::new();
                let mut ab_local = Vec::new();
                self.forward_tangent_dual(&inputs, &mut dv_local);
                self.reverse_tangent_dual(&dv_local, &mut ab_local);
                ab_local
            })
            .collect();

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let color = colors[col as usize] as usize;
            hessian_values[k] = color_results[color][row as usize].eps;
        }

        (value, gradient, pattern, hessian_values)
    }

    /// Parallel sparse Jacobian: parallelized over colors.
    ///
    /// Auto-selects forward (column compression) or reverse (row compression)
    /// based on `num_outputs` vs `num_inputs`.
    pub fn sparse_jacobian_par(
        &self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut values_buf = Vec::new();
        self.forward_into(x, &mut values_buf);

        let out_indices = self.all_output_indices();
        let m = out_indices.len();
        let outputs: Vec<F> = out_indices
            .iter()
            .map(|&oi| values_buf[oi as usize])
            .collect();

        let jac_pattern = self.detect_jacobian_sparsity();
        let ni = self.num_inputs as usize;

        if m <= n {
            // Row compression (reverse mode)
            let (row_colors, num_colors) = crate::sparse::row_coloring(&jac_pattern);

            let color_results: Vec<Vec<F>> = (0..num_colors)
                .into_par_iter()
                .map(|color| {
                    let n_vars = self.num_variables as usize;
                    let mut adjoints = vec![F::zero(); n_vars];
                    for (i, &oi) in out_indices.iter().enumerate() {
                        if row_colors[i] == color {
                            adjoints[oi as usize] = F::one();
                        }
                    }

                    self.reverse_sweep_core(&mut adjoints, &values_buf, None);
                    adjoints[..ni].to_vec()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in jac_pattern
                .rows
                .iter()
                .zip(jac_pattern.cols.iter())
                .enumerate()
            {
                let color = row_colors[row as usize] as usize;
                jac_values[k] = color_results[color][col as usize];
            }

            (outputs, jac_pattern, jac_values)
        } else {
            // Column compression (forward mode) — parallelize forward tangent sweeps
            let (col_colors, num_colors) = crate::sparse::column_coloring(&jac_pattern);

            let color_results: Vec<Vec<F>> = (0..num_colors)
                .into_par_iter()
                .map(|color| {
                    let dir: Vec<F> = (0..n)
                        .map(|i| {
                            if col_colors[i] == color {
                                F::one()
                            } else {
                                F::zero()
                            }
                        })
                        .collect();
                    // Forward tangent with F (not Dual) direction
                    // We need to do forward_into first, then forward tangent
                    // But forward_tangent uses self.values, so we need a workaround.
                    // Use Dual<F> with tangent = direction.
                    let inputs: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], dir[i])).collect();
                    let mut dv = Vec::new();
                    self.forward_tangent(&inputs, &mut dv);
                    out_indices.iter().map(|&oi| dv[oi as usize].eps).collect()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in jac_pattern
                .rows
                .iter()
                .zip(jac_pattern.cols.iter())
                .enumerate()
            {
                let color = col_colors[col as usize] as usize;
                jac_values[k] = color_results[color][row as usize];
            }

            (outputs, jac_pattern, jac_values)
        }
    }

    /// Evaluate the gradient at multiple input points in parallel.
    ///
    /// Uses `forward_into` + `reverse_from` with per-thread buffers.
    pub fn gradient_batch_par(&self, inputs: &[&[F]]) -> Vec<Vec<F>> {
        use rayon::prelude::*;

        let ni = self.num_inputs as usize;
        let out_idx = self.output_index;

        inputs
            .par_iter()
            .map(|x| {
                let mut values_buf = Vec::new();
                self.forward_into(x, &mut values_buf);
                let adjoints = self.reverse_from(&values_buf, out_idx);
                adjoints[..ni].to_vec()
            })
            .collect()
    }

    /// Compute Hessian at multiple input points in parallel.
    ///
    /// Returns `(value, gradient, hessian)` for each input point.
    pub fn hessian_batch_par(&self, inputs: &[&[F]]) -> Vec<(F, Vec<F>, Vec<Vec<F>>)> {
        use rayon::prelude::*;

        inputs.par_iter().map(|x| self.hessian_par(x)).collect()
    }
}

// ══════════════════════════════════════════════
//  Thread-local active bytecode tape
// ══════════════════════════════════════════════
//  Serde support
// ══════════════════════════════════════════════

#[cfg(feature = "serde")]
mod serde_support {
    use super::*;
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<F: Float + Serialize> Serialize for BytecodeTape<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if !self.custom_ops.is_empty() {
                return Err(serde::ser::Error::custom(
                    "cannot serialize a BytecodeTape containing custom ops; \
                     custom ops must be re-registered after deserialization",
                ));
            }
            let mut s = serializer.serialize_struct("BytecodeTape", 8)?;
            s.serialize_field("opcodes", &self.opcodes)?;
            s.serialize_field("arg_indices", &self.arg_indices)?;
            s.serialize_field("values", &self.values)?;
            s.serialize_field("num_inputs", &self.num_inputs)?;
            s.serialize_field("num_variables", &self.num_variables)?;
            s.serialize_field("output_index", &self.output_index)?;
            s.serialize_field("output_indices", &self.output_indices)?;
            s.serialize_field("custom_second_args", &self.custom_second_args)?;
            s.end()
        }
    }

    impl<'de, F: Float + Deserialize<'de>> Deserialize<'de> for BytecodeTape<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            struct TapeData<F> {
                opcodes: Vec<OpCode>,
                arg_indices: Vec<[u32; 2]>,
                values: Vec<F>,
                num_inputs: u32,
                num_variables: u32,
                output_index: u32,
                #[serde(default)]
                output_indices: Vec<u32>,
                #[serde(default)]
                custom_second_args: HashMap<u32, u32>,
            }

            let data = TapeData::<F>::deserialize(deserializer)?;
            Ok(BytecodeTape {
                opcodes: data.opcodes,
                arg_indices: data.arg_indices,
                values: data.values,
                num_inputs: data.num_inputs,
                num_variables: data.num_variables,
                output_index: data.output_index,
                output_indices: data.output_indices,
                custom_ops: Vec::new(),
                custom_second_args: data.custom_second_args,
            })
        }
    }
}

// ══════════════════════════════════════════════

thread_local! {
    static BTAPE_F32: Cell<*mut BytecodeTape<f32>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_F64: Cell<*mut BytecodeTape<f64>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_DUAL_F32: Cell<*mut BytecodeTape<Dual<f32>>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_DUAL_F64: Cell<*mut BytecodeTape<Dual<f64>>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Trait to select the correct thread-local for a given float type.
///
/// Implemented for `f32`, `f64`, `Dual<f32>`, and `Dual<f64>`, enabling
/// `BReverse<F>` to be used with these base types.
pub trait BtapeThreadLocal: Float {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>>;
}

impl BtapeThreadLocal for f32 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F32
    }
}

impl BtapeThreadLocal for f64 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F64
    }
}

impl BtapeThreadLocal for Dual<f32> {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_DUAL_F32
    }
}

impl BtapeThreadLocal for Dual<f64> {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_DUAL_F64
    }
}

/// Access the active bytecode tape for the current thread.
/// Panics if no tape is active.
#[inline]
pub fn with_active_btape<F: BtapeThreadLocal, R>(f: impl FnOnce(&mut BytecodeTape<F>) -> R) -> R {
    F::btape_cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active bytecode tape. Use echidna::record() to record a function."
        );
        // SAFETY: BtapeGuard guarantees validity for the duration of the
        // recording scope, single-threaded via thread-local.
        let tape = unsafe { &mut *ptr };
        f(tape)
    })
}

/// RAII guard that sets a bytecode tape as the thread-local active tape.
pub struct BtapeGuard<F: BtapeThreadLocal> {
    prev: *mut BytecodeTape<F>,
}

impl<F: BtapeThreadLocal> BtapeGuard<F> {
    /// Activate `tape` as the thread-local bytecode tape.
    pub fn new(tape: &mut BytecodeTape<F>) -> Self {
        let prev = F::btape_cell().with(|cell| {
            let prev = cell.get();
            cell.set(tape as *mut BytecodeTape<F>);
            prev
        });
        BtapeGuard { prev }
    }
}

impl<F: BtapeThreadLocal> Drop for BtapeGuard<F> {
    fn drop(&mut self) {
        F::btape_cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

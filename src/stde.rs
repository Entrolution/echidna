//! Stochastic Taylor Derivative Estimators (STDE).
//!
//! Estimate differential operators (Laplacian, Hessian diagonal, directional
//! derivatives) by pushing random direction vectors through Taylor-mode AD.
//!
//! # How it works
//!
//! For f: R^n -> R at point x, define g(t) = f(x + t*v) where v is a
//! direction vector. The Taylor coefficients of g at t=0 are:
//!
//! - c0 = f(x)
//! - c1 = nabla f(x) . v   (directional first derivative)
//! - c2 = v^T H_f(x) v / 2 (half the directional second derivative)
//!
//! By choosing v appropriately (Rademacher, Gaussian, coordinate basis),
//! we can estimate operators like the Laplacian in O(S*K*L) time instead
//! of O(n^2*L) for the full Hessian.
//!
//! # Variance properties
//!
//! The Hutchinson estimator `(1/S) sum_s v_s^T H v_s` is unbiased when
//! E\[vv^T\] = I. Its variance depends on the distribution of v:
//!
//! - **Rademacher** (entries ±1): Var = `2 sum_{i≠j} H_ij^2`. The diagonal
//!   contributes zero variance since `v_i^2 = 1` always.
//! - **Gaussian** (v ~ N(0,I)): Var = `2 ||H||_F^2`. Higher variance than
//!   Rademacher because `v_i^2 ~ chi-squared(1)` introduces diagonal noise.
//!
//! The [`laplacian_with_control`] function reduces Gaussian variance to match
//! Rademacher by subtracting the exact diagonal contribution (a control
//! variate). This requires the diagonal from [`hessian_diagonal`] (n extra
//! evaluations). For Rademacher directions, the control variate has no effect
//! since the diagonal variance is already zero.
//!
//! **Antithetic sampling** (pairing +v with -v) does **not** reduce variance
//! for trace estimation. The Hutchinson sample `v^T H v` is quadratic (even)
//! in v, so `(-v)^T H (-v) = v^T H v` — antithetic pairs are identical.
//!
//! # Design
//!
//! - **No `rand` dependency**: all functions accept user-provided direction
//!   vectors. The library stays pure; users bring their own RNG.
//! - **`Taylor<F, 3>`** for second-order operators: stack-allocated, Copy,
//!   monomorphized. The order K=3 is statically known.
//! - **`TaylorDyn`** variants for runtime-determined order.
//! - **Panics on misuse**: dimension mismatches panic, following existing
//!   API conventions (`record`, `grad`, `hvp`).

use crate::bytecode_tape::BytecodeTape;
use crate::taylor::Taylor;
use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn, TaylorDynGuard};
use crate::Float;

// ══════════════════════════════════════════════
//  Result types
// ══════════════════════════════════════════════

/// Result of a stochastic estimation with sample statistics.
///
/// Contains the function value, the estimated operator value, and
/// sample statistics (variance, standard error) that quantify
/// estimator quality.
#[derive(Clone, Debug)]
pub struct EstimatorResult<F> {
    /// Function value f(x).
    pub value: F,
    /// Estimated operator value (e.g. Laplacian).
    pub estimate: F,
    /// Sample variance of the per-direction estimates.
    /// Zero when `num_samples == 1` (undefined, clamped to zero).
    pub sample_variance: F,
    /// Standard error of the mean: `sqrt(sample_variance / num_samples)`.
    pub standard_error: F,
    /// Number of direction samples used.
    pub num_samples: usize,
}

// ══════════════════════════════════════════════
//  Low-level: single-direction jet propagation
// ══════════════════════════════════════════════

/// Propagate direction `v` through tape using second-order Taylor mode.
///
/// Constructs `Taylor<F, 3>` inputs where `input[i] = [x[i], v[i], 0]`,
/// runs `forward_tangent`, and extracts the output coefficients.
///
/// Returns `(f(x), nabla_f . v, v^T H v / 2)`.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`.
pub fn taylor_jet_2nd<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    v: &[F],
) -> (F, F, F) {
    let mut buf = Vec::new();
    taylor_jet_2nd_with_buf(tape, x, v, &mut buf)
}

/// Like [`taylor_jet_2nd`] but reuses a caller-provided buffer to avoid
/// reallocation across multiple calls.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`.
pub fn taylor_jet_2nd_with_buf<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    v: &[F],
    buf: &mut Vec<Taylor<F, 3>>,
) -> (F, F, F) {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(v.len(), n, "v.len() must match tape.num_inputs()");

    let inputs: Vec<Taylor<F, 3>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Taylor::new([xi, vi, F::zero()]))
        .collect();

    tape.forward_tangent(&inputs, buf);

    let out = buf[tape.output_index()];
    (out.coeffs[0], out.coeffs[1], out.coeffs[2])
}

// ══════════════════════════════════════════════
//  Mid-level: batch direction evaluation
// ══════════════════════════════════════════════

/// Evaluate multiple directions through the tape.
///
/// Returns `(value, first_order, second_order)` where:
/// - `value` = f(x)
/// - `first_order[s]` = nabla_f . v_s  (directional first derivative)
/// - `second_order[s]` = v_s^T H v_s / 2  (half directional second derivative)
///
/// # Panics
///
/// Panics if any direction's length does not match `tape.num_inputs()`.
pub fn directional_derivatives<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> (F, Vec<F>, Vec<F>) {
    let mut buf = Vec::new();
    let mut first_order = Vec::with_capacity(directions.len());
    let mut second_order = Vec::with_capacity(directions.len());
    let mut value = F::zero();

    for v in directions {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        first_order.push(c1);
        second_order.push(c2);
    }

    (value, first_order, second_order)
}

// ══════════════════════════════════════════════
//  High-level: operator estimators
// ══════════════════════════════════════════════

/// Estimate the Laplacian (trace of Hessian) via Hutchinson's trace estimator.
///
/// Directions must satisfy E[vv^T] = I (e.g. Rademacher vectors with entries
/// +/-1, or standard Gaussian vectors). The estimator is:
///
///   Laplacian ~ (1/S) * sum_s 2*c2_s
///
/// where c2_s is the second Taylor coefficient for direction s.
///
/// Returns `(value, laplacian_estimate)`.
///
/// Note: coordinate basis vectors do **not** satisfy E[vv^T] = I and will
/// give tr(H)/n instead of tr(H). Use [`hessian_diagonal`] and sum for exact
/// computation via coordinate directions.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> (F, F) {
    assert!(!directions.is_empty(), "directions must not be empty");

    let (value, _, second_order) = directional_derivatives(tape, x, directions);

    let two = F::from(2.0).unwrap();
    let s = F::from(directions.len()).unwrap();
    let sum: F = second_order.iter().fold(F::zero(), |acc, &c2| acc + two * c2);
    let laplacian = sum / s;

    (value, laplacian)
}

/// Estimate the Laplacian with sample statistics via Hutchinson's trace estimator.
///
/// Same estimator as [`laplacian`], but additionally computes sample variance
/// and standard error using Welford's online algorithm (numerically stable,
/// single pass).
///
/// Each direction produces a sample `2 * c2_s`. The returned statistics
/// describe the distribution of these samples.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian_with_stats<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");

    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut value = F::zero();

    // Welford's online algorithm
    let mut mean = F::zero();
    let mut m2 = F::zero();

    for (k, v) in directions.iter().enumerate() {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        let sample = two * c2;

        let k1 = F::from(k + 1).unwrap();
        let delta = sample - mean;
        mean = mean + delta / k1;
        let delta2 = sample - mean;
        m2 = m2 + delta * delta2;
    }

    let s = directions.len();
    let sf = F::from(s).unwrap();
    let (sample_variance, standard_error) = if s > 1 {
        let var = m2 / (sf - F::one());
        (var, (var / sf).sqrt())
    } else {
        (F::zero(), F::zero())
    };

    EstimatorResult {
        value,
        estimate: mean,
        sample_variance,
        standard_error,
        num_samples: s,
    }
}

/// Estimate the Laplacian with a diagonal control variate.
///
/// Uses the exact Hessian diagonal (from [`hessian_diagonal`]) as a control
/// variate to reduce estimator variance. Each raw Hutchinson sample
/// `raw_s = v^T H v = 2 * c2_s` is adjusted:
///
/// ```text
/// adjusted_s = raw_s - sum_j(D_jj * v_j^2) + tr(D)
/// ```
///
/// where D is the diagonal of H. The adjustment subtracts the noisy diagonal
/// contribution and adds back its exact expectation `tr(D)`.
///
/// **Effect by distribution**:
/// - **Gaussian**: reduces variance from `2||H||_F^2` to `2 sum_{i≠j} H_ij^2`
///   (matching Rademacher performance).
/// - **Rademacher**: no effect, since `v_j^2 = 1` always, so the adjustment
///   is `sum_j D_jj * 1 - tr(D) = 0`.
///
/// Returns an [`EstimatorResult`] with statistics computed over the adjusted
/// samples.
///
/// # Panics
///
/// Panics if `directions` is empty, if any direction's length does not match
/// `tape.num_inputs()`, or if `control_diagonal.len() != tape.num_inputs()`.
pub fn laplacian_with_control<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
    control_diagonal: &[F],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    assert_eq!(
        control_diagonal.len(),
        n,
        "control_diagonal.len() must match tape.num_inputs()"
    );

    let two = F::from(2.0).unwrap();
    let trace_control: F = control_diagonal.iter().copied().fold(F::zero(), |a, b| a + b);

    let mut buf = Vec::new();
    let mut value = F::zero();

    // Welford's online algorithm
    let mut mean = F::zero();
    let mut m2 = F::zero();

    for (k, v) in directions.iter().enumerate() {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;

        let raw = two * c2;

        // Control variate: subtract v^T D v, add back E[v^T D v] = tr(D)
        let cv: F = control_diagonal
            .iter()
            .zip(v.iter())
            .fold(F::zero(), |acc, (&d, &vi)| acc + d * vi * vi);
        let adjusted = raw - cv + trace_control;

        let k1 = F::from(k + 1).unwrap();
        let delta = adjusted - mean;
        mean = mean + delta / k1;
        let delta2 = adjusted - mean;
        m2 = m2 + delta * delta2;
    }

    let s = directions.len();
    let sf = F::from(s).unwrap();
    let (sample_variance, standard_error) = if s > 1 {
        let var = m2 / (sf - F::one());
        (var, (var / sf).sqrt())
    } else {
        (F::zero(), F::zero())
    };

    EstimatorResult {
        value,
        estimate: mean,
        sample_variance,
        standard_error,
        num_samples: s,
    }
}

/// Exact Hessian diagonal via n coordinate-direction evaluations.
///
/// For each coordinate j, pushes basis vector e_j through the tape and
/// reads `2 * c2`, which equals `d^2 f / dx_j^2`.
///
/// Returns `(value, diag)` where `diag[j] = d^2 f / dx_j^2`.
pub fn hessian_diagonal<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
) -> (F, Vec<F>) {
    let mut buf = Vec::new();
    hessian_diagonal_with_buf(tape, x, &mut buf)
}

/// Like [`hessian_diagonal`] but reuses a caller-provided buffer.
pub fn hessian_diagonal_with_buf<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    buf: &mut Vec<Taylor<F, 3>>,
) -> (F, Vec<F>) {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let two = F::from(2.0).unwrap();
    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    // Build basis vector once, mutate the hot coordinate
    let mut e = vec![F::zero(); n];
    for j in 0..n {
        e[j] = F::one();
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &e, buf);
        value = c0;
        diag.push(two * c2);
        e[j] = F::zero();
    }

    (value, diag)
}

// ══════════════════════════════════════════════
//  TaylorDyn variants (runtime order)
// ══════════════════════════════════════════════

/// Propagate direction `v` through tape using `TaylorDyn` with the given order.
///
/// Creates a `TaylorDynGuard` internally, builds `TaylorDyn` inputs from
/// `(x, v)` with coefficients `[x_i, v_i, 0, ..., 0]`, runs `forward_tangent`,
/// and returns the full coefficient vector of the output.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`,
/// or if `order < 2`.
pub fn taylor_jet_dyn<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    v: &[F],
    order: usize,
) -> Vec<F> {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(v.len(), n, "v.len() must match tape.num_inputs()");
    assert!(order >= 2, "order must be >= 2");

    let _guard = TaylorDynGuard::<F>::new(order);

    let inputs: Vec<TaylorDyn<F>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| {
            let mut coeffs = vec![F::zero(); order];
            coeffs[0] = xi;
            coeffs[1] = vi;
            TaylorDyn::from_coeffs(&coeffs)
        })
        .collect();

    let mut buf = Vec::new();
    tape.forward_tangent(&inputs, &mut buf);

    buf[tape.output_index()].coeffs()
}

/// Estimate the Laplacian via `TaylorDyn` (runtime-determined order).
///
/// Uses order 3 (coefficients c0, c1, c2) internally. Manages its own
/// arena guard.
///
/// Returns `(value, laplacian_estimate)`.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian_dyn<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> (F, F) {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let _guard = TaylorDynGuard::<F>::new(3);

    let two = F::from(2.0).unwrap();
    let s = F::from(directions.len()).unwrap();
    let mut sum = F::zero();
    let mut value = F::zero();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for v in directions {
        assert_eq!(v.len(), n, "direction length must match tape.num_inputs()");

        let inputs: Vec<TaylorDyn<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| TaylorDyn::from_coeffs(&[xi, vi, F::zero()]))
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out = buf[tape.output_index()];
        let coeffs = out.coeffs();
        value = coeffs[0];
        let c2 = coeffs[2];
        sum = sum + two * c2;
    }

    (value, sum / s)
}

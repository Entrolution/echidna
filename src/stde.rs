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
//!
//! # Const-Generic Higher-Order Diagonal
//!
//! [`diagonal_kth_order_const`] is a stack-allocated variant of
//! [`diagonal_kth_order`] for compile-time-known derivative order. It uses
//! `Taylor<F, ORDER>` directly (no `TaylorDyn` arena), which is faster when
//! the order is statically known. `ORDER = k + 1` where k is the derivative
//! order. For f32, practical limit is `ORDER ≤ 14` (k ≤ 13) since `k! > 2^23`
//! causes precision loss.
//!
//! # Higher-Order Estimation
//!
//! [`diagonal_kth_order`] generalises [`hessian_diagonal`] from k=2 to
//! arbitrary k, computing exact `[∂^k u/∂x_j^k]` for all coordinates via
//! `TaylorDyn` jets of order k+1. The stochastic variant
//! [`diagonal_kth_order_stochastic`] subsamples coordinates for an unbiased
//! estimate of `Σ_j ∂^k u/∂x_j^k`.
//!
//! # Dense STDE for Positive-Definite Operators
//!
//! [`dense_stde_2nd`] estimates `tr(C · H_u(x)) = Σ_{ij} C_{ij} ∂²u/∂x_i∂x_j`
//! for a positive-definite coefficient matrix C. The caller provides a Cholesky
//! factor L (such that `C = L L^T`) and standard Gaussian vectors z. Internally,
//! `v = L · z` is computed, then pushed through a second-order Taylor jet. When
//! `L = I`, this reduces to the Hutchinson Laplacian estimator.
//!
//! # Parabolic PDE σ-Transform
//!
//! [`parabolic_diffusion`] computes `½ tr(σσ^T · Hess u)` for parabolic PDEs
//! (Fokker-Planck, Black-Scholes, HJB) by pushing columns of σ through
//! second-order Taylor jets. This avoids forming the off-diagonal Hessian
//! entries that a naïve `tr(A·H)` computation would require.
//!
//! # Sparse STDE (requires `stde` + `diffop`)
//!
//! `stde_sparse` estimates arbitrary differential operators `Lu(x)` by
//! sampling sparse k-jets from a `SparseSamplingDistribution` built via
//! `DiffOp::sparse_distribution`. Each sample is a single forward
//! pushforward; the per-sample estimator is `sign(C_α) · Z · D^α u` where
//! Z = Σ|C_α| is the normalization constant. This implements the core
//! contribution of Shi et al. (NeurIPS 2024).

use crate::bytecode_tape::BytecodeTape;
use crate::dual::Dual;
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
//  Welford online accumulator
// ══════════════════════════════════════════════

/// Welford's online algorithm for incremental mean and variance.
struct WelfordAccumulator<F> {
    mean: F,
    m2: F,
    count: usize,
}

impl<F: Float> WelfordAccumulator<F> {
    fn new() -> Self {
        Self {
            mean: F::zero(),
            m2: F::zero(),
            count: 0,
        }
    }

    fn update(&mut self, sample: F) {
        self.count += 1;
        let k1 = F::from(self.count).unwrap();
        let delta = sample - self.mean;
        self.mean = self.mean + delta / k1;
        let delta2 = sample - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    fn finalize(&self) -> (F, F, F) {
        let nf = F::from(self.count).unwrap();
        if self.count > 1 {
            let var = self.m2 / (nf - F::one());
            (self.mean, var, (var / nf).sqrt())
        } else {
            (self.mean, F::zero(), F::zero())
        }
    }
}

// ══════════════════════════════════════════════
//  Estimator trait + built-in estimators
// ══════════════════════════════════════════════

/// Trait for stochastic estimators that combine Taylor jet coefficients into
/// a per-direction sample.
///
/// Given the Taylor coefficients `(c0, c1, c2)` from propagating a random
/// direction through a tape, an `Estimator` produces a scalar sample whose
/// expectation (over random directions with `E[vv^T] = I`) equals the
/// desired quantity.
pub trait Estimator<F: Float> {
    /// Compute one sample from the Taylor jet coefficients.
    ///
    /// - `c0` = f(x)
    /// - `c1` = ∇f(x)·v  (directional first derivative)
    /// - `c2` = v^T H v / 2  (half directional second derivative)
    fn sample(&self, c0: F, c1: F, c2: F) -> F;
}

/// Hutchinson trace estimator: estimates tr(H) = Laplacian.
///
/// Each sample is `2 * c2 = v^T H v`. Since `E[v^T H v] = tr(H)` when
/// `E[vv^T] = I`, the mean of these samples converges to the Laplacian.
pub struct Laplacian;

impl<F: Float> Estimator<F> for Laplacian {
    #[inline]
    fn sample(&self, _c0: F, _c1: F, c2: F) -> F {
        F::from(2.0).unwrap() * c2
    }
}

/// Estimates `||∇f||²` (squared gradient norm).
///
/// Each sample is `c1² = (∇f·v)²`. Since `E[(∇f·v)²] = ∇f^T E[vv^T] ∇f = ||∇f||²`
/// when `E[vv^T] = I`, the mean converges to the squared gradient norm.
///
/// Useful for score matching loss functions where `||∇ log p||²` appears.
pub struct GradientSquaredNorm;

impl<F: Float> Estimator<F> for GradientSquaredNorm {
    #[inline]
    fn sample(&self, _c0: F, c1: F, _c2: F) -> F {
        c1 * c1
    }
}

// ══════════════════════════════════════════════
//  Generic estimation pipeline
// ══════════════════════════════════════════════

/// Result of a divergence estimation.
///
/// Separate from [`EstimatorResult`] because the function output is a vector
/// (`values: Vec<F>`) rather than a scalar (`value: F`).
#[derive(Clone, Debug)]
pub struct DivergenceResult<F> {
    /// Function output vector f(x).
    pub values: Vec<F>,
    /// Estimated divergence (trace of the Jacobian).
    pub estimate: F,
    /// Sample variance of per-direction estimates.
    pub sample_variance: F,
    /// Standard error of the mean.
    pub standard_error: F,
    /// Number of direction samples used.
    pub num_samples: usize,
}

/// Estimate a quantity using the given [`Estimator`] and Welford's online algorithm.
///
/// Evaluates the tape at `x` for each direction, computes the estimator's sample
/// from the Taylor jet, and aggregates with running mean and variance.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn estimate<F: Float>(
    estimator: &impl Estimator<F>,
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for v in directions.iter() {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        acc.update(estimator.sample(c0, c1, c2));
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: directions.len(),
    }
}

/// Estimate a quantity using importance-weighted samples (West's 1979 algorithm).
///
/// Each direction `directions[s]` has an associated weight `weights[s]`.
/// The weighted mean is `Σ(w_s * sample_s) / Σ(w_s)` and the variance uses
/// the reliability-weight Bessel correction: `M2 / (W - W2/W)` where
/// `W = Σw_s` and `W2 = Σw_s²`.
///
/// # Panics
///
/// Panics if `directions` is empty, `weights.len() != directions.len()`,
/// or any direction's length does not match `tape.num_inputs()`.
pub fn estimate_weighted<F: Float>(
    estimator: &impl Estimator<F>,
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
    weights: &[F],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    assert_eq!(
        weights.len(),
        directions.len(),
        "weights.len() must match directions.len()"
    );

    let mut buf = Vec::new();
    let mut value = F::zero();

    // West's (1979) weighted online algorithm
    let mut w_sum = F::zero();
    let mut w_sum2 = F::zero();
    let mut mean = F::zero();
    let mut m2 = F::zero();

    for (k, v) in directions.iter().enumerate() {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        let s = estimator.sample(c0, c1, c2);
        let w = weights[k];

        w_sum = w_sum + w;
        w_sum2 = w_sum2 + w * w;
        let delta = s - mean;
        mean = mean + (w / w_sum) * delta;
        let delta2 = s - mean;
        m2 = m2 + w * delta * delta2;
    }

    let n = directions.len();
    let denom = w_sum - w_sum2 / w_sum;
    let (sample_variance, standard_error) = if n > 1 && denom > F::zero() {
        let var = m2 / denom;
        let nf = F::from(n).unwrap();
        (var, (var / nf).sqrt())
    } else {
        (F::zero(), F::zero())
    };

    EstimatorResult {
        value,
        estimate: mean,
        sample_variance,
        standard_error,
        num_samples: n,
    }
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
pub fn taylor_jet_2nd<F: Float>(tape: &BytecodeTape<F>, x: &[F], v: &[F]) -> (F, F, F) {
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
pub fn laplacian<F: Float>(tape: &BytecodeTape<F>, x: &[F], directions: &[&[F]]) -> (F, F) {
    assert!(!directions.is_empty(), "directions must not be empty");

    let (value, _, second_order) = directional_derivatives(tape, x, directions);

    let two = F::from(2.0).unwrap();
    let s = F::from(directions.len()).unwrap();
    let sum: F = second_order
        .iter()
        .fold(F::zero(), |acc, &c2| acc + two * c2);
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
    estimate(&Laplacian, tape, x, directions)
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
    let trace_control: F = control_diagonal
        .iter()
        .copied()
        .fold(F::zero(), |a, b| a + b);

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for v in directions.iter() {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;

        let raw = two * c2;

        // Control variate: subtract v^T D v, add back E[v^T D v] = tr(D)
        let cv: F = control_diagonal
            .iter()
            .zip(v.iter())
            .fold(F::zero(), |acc, (&d, &vi)| acc + d * vi * vi);
        acc.update(raw - cv + trace_control);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: directions.len(),
    }
}

/// Exact Hessian diagonal via n coordinate-direction evaluations.
///
/// For each coordinate j, pushes basis vector e_j through the tape and
/// reads `2 * c2`, which equals `d^2 f / dx_j^2`.
///
/// Returns `(value, diag)` where `diag[j] = d^2 f / dx_j^2`.
pub fn hessian_diagonal<F: Float>(tape: &BytecodeTape<F>, x: &[F]) -> (F, Vec<F>) {
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
//  Higher-order diagonal estimation
// ══════════════════════════════════════════════

/// Exact k-th order diagonal: `[∂^k u/∂x_j^k for j in 0..n]`.
///
/// Pushes n basis vectors through order-(k+1) `TaylorDyn` jets. For each
/// coordinate j, the input jet has `coeffs_j = [x_j, 1, 0, ..., 0]` and all
/// other inputs `coeffs_i = [x_i, 0, ..., 0]`. The output coefficient at
/// index k stores `∂^k u/∂x_j^k / k!`, so the derivative is `k! * coeffs[k]`.
///
/// Requires `F: Float + TaylorArenaLocal` because the jet order is
/// runtime-determined (unlike [`hessian_diagonal`] which uses const-generic
/// `Taylor<F, 3>`).
///
/// # Panics
///
/// Panics if `k < 2`, `k > 20` (factorial overflow guard for f64), or if
/// `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
) -> (F, Vec<F>) {
    let mut buf = Vec::new();
    diagonal_kth_order_with_buf(tape, x, k, &mut buf)
}

/// Like [`diagonal_kth_order`] but reuses a caller-provided buffer.
pub fn diagonal_kth_order_with_buf<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
    buf: &mut Vec<TaylorDyn<F>>,
) -> (F, Vec<F>) {
    assert!(k >= 2, "k must be >= 2 (use gradient for k=1)");
    assert!(k <= 20, "k must be <= 20 (k! overflows f64 for k > 20)");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = k + 1; // number of Taylor coefficients
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    for j in 0..n {
        // Build TaylorDyn inputs: coeffs_j = [x_j, 1, 0, ..., 0], others = [x_i, 0, ..., 0]
        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];
        diag.push(k_factorial * out_coeffs[k]);
    }

    (value, diag)
}

/// Exact k-th order diagonal using const-generic `Taylor<F, ORDER>`.
///
/// `ORDER = k + 1` where `k = ORDER - 1` is the derivative order. Uses
/// stack-allocated Taylor coefficients — faster than [`diagonal_kth_order`]
/// when k is known at compile time.
///
/// # Precision Note
///
/// For f32, `k! > 2^23` when k ≥ 13 (ORDER ≥ 14), causing precision loss.
/// The existing [`diagonal_kth_order`] has a runtime guard `k ≤ 20` for f64;
/// the const-generic version has no such guard but users should be aware
/// that for f32, ORDER ≤ 14 is the practical limit.
///
/// # Panics
///
/// Compile-time error if `ORDER < 3` (i.e., k < 2).
/// Runtime panic if `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order_const<F: Float, const ORDER: usize>(
    tape: &BytecodeTape<F>,
    x: &[F],
) -> (F, Vec<F>) {
    let mut buf: Vec<Taylor<F, ORDER>> = Vec::new();
    diagonal_kth_order_const_with_buf(tape, x, &mut buf)
}

/// Like [`diagonal_kth_order_const`] but reuses a caller-provided buffer.
pub fn diagonal_kth_order_const_with_buf<F: Float, const ORDER: usize>(
    tape: &BytecodeTape<F>,
    x: &[F],
    buf: &mut Vec<Taylor<F, ORDER>>,
) -> (F, Vec<F>) {
    const { assert!(ORDER >= 3, "ORDER must be >= 3 (k=ORDER-1 >= 2)") }

    let k = ORDER - 1;
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    for j in 0..n {
        let inputs: Vec<Taylor<F, ORDER>> = (0..n)
            .map(|i| {
                let mut coeffs = [F::zero(); ORDER];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                Taylor::new(coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, buf);

        let out = buf[tape.output_index()];
        value = out.coeffs[0];
        diag.push(k_factorial * out.coeffs[k]);
    }

    (value, diag)
}

/// Stochastic k-th order diagonal estimate: `Σ_j ∂^k u/∂x_j^k`.
///
/// Evaluates only the coordinate indices in `sampled_indices`, then scales
/// by `n / |J|` to produce an unbiased estimate of the full sum.
///
/// # Panics
///
/// Panics if `sampled_indices` is empty, `k < 2`, `k > 20`, or if
/// `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order_stochastic<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    assert!(k >= 2, "k must be >= 2 (use gradient for k=1)");
    assert!(k <= 20, "k must be <= 20 (k! overflows f64 for k > 20)");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = k + 1;
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let nf = F::from(n).unwrap();

    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for &j in sampled_indices {
        assert!(j < n, "sampled index {} out of bounds (n={})", j, n);

        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];
        // Per-sample: k! * coeffs[k] (the diagonal entry for coordinate j)
        acc.update(k_factorial * out_coeffs[k]);
    }

    let (mean, sample_variance, standard_error) = acc.finalize();

    // Unbiased estimator for Σ_j d_j: n * mean(sampled d_j's)
    EstimatorResult {
        value,
        estimate: mean * nf,
        sample_variance,
        standard_error,
        num_samples: sampled_indices.len(),
    }
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

// ══════════════════════════════════════════════
//  Hutch++ trace estimator
// ══════════════════════════════════════════════

/// Modified Gram-Schmidt orthonormalisation, in-place.
///
/// Orthonormalises the columns of the matrix represented as `columns: &mut Vec<Vec<F>>`.
/// Drops near-zero columns (norm < epsilon). Returns the rank (number of retained columns).
fn modified_gram_schmidt<F: Float>(columns: &mut Vec<Vec<F>>, epsilon: F) -> usize {
    let mut rank = 0;
    let mut i = 0;
    while i < columns.len() {
        // Orthogonalise against all previously accepted columns
        for j in 0..rank {
            // Split to satisfy the borrow checker: j < rank <= i
            let (left, right) = columns.split_at_mut(i);
            let qj = &left[j];
            let ci = &mut right[0];
            let dot: F = qj
                .iter()
                .zip(ci.iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
            for (c, &q) in ci.iter_mut().zip(qj.iter()) {
                *c = *c - dot * q;
            }
        }

        // Compute norm
        let norm_sq: F = columns[i].iter().fold(F::zero(), |acc, &v| acc + v * v);
        let norm = norm_sq.sqrt();

        if norm < epsilon {
            // Drop this column (near-zero after projection)
            columns.swap_remove(i);
            // Don't increment i — swapped element needs processing
        } else {
            // Normalise
            let inv_norm = F::one() / norm;
            for v in columns[i].iter_mut() {
                *v = *v * inv_norm;
            }
            // Move to rank position
            if i != rank {
                columns.swap(i, rank);
            }
            rank += 1;
            i += 1;
        }
    }
    columns.truncate(rank);
    rank
}

/// Hutch++ trace estimator (Meyer et al. 2021) for the Laplacian.
///
/// Achieves O(1/S²) convergence for matrices with decaying eigenvalues by
/// splitting the work into:
///
/// 1. **Sketch phase**: k HVPs (via `tape.hvp_with_buf`) produce columns of H·S,
///    which are orthonormalised via Modified Gram-Schmidt to give a basis Q.
/// 2. **Exact subspace trace**: For each q_i in Q, `taylor_jet_2nd` gives
///    q_i^T H q_i. Sum = tr(Q^T H Q) — this part has zero variance.
/// 3. **Residual Hutchinson**: For each stochastic direction g_s, project out Q
///    (g' = g - Q(Q^T g)) and estimate the residual trace via `taylor_jet_2nd`.
/// 4. **Total** = exact_trace + residual_mean.
///
/// The variance and standard error in the result refer to the residual only.
///
/// # Arguments
///
/// - `sketch_directions`: k directions for the sketch phase. Typically Rademacher
///   or Gaussian. More directions capture more of the spectrum exactly.
/// - `stochastic_directions`: S directions for residual estimation. Rademacher recommended.
///
/// # Cost
///
/// k HVPs (≈2k forward passes) + k Taylor jets (exact subspace) + S Taylor jets
/// (residual) + O(k²·n) for Gram-Schmidt.
///
/// # Panics
///
/// Panics if `sketch_directions` or `stochastic_directions` is empty, or if any
/// direction's length does not match `tape.num_inputs()`.
pub fn laplacian_hutchpp<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sketch_directions: &[&[F]],
    stochastic_directions: &[&[F]],
) -> EstimatorResult<F> {
    assert!(
        !sketch_directions.is_empty(),
        "sketch_directions must not be empty"
    );
    assert!(
        !stochastic_directions.is_empty(),
        "stochastic_directions must not be empty"
    );

    let n = tape.num_inputs();
    let two = F::from(2.0).unwrap();
    let eps = F::from(1e-12).unwrap();

    // ── Step 1: Sketch — k HVPs to get columns of H·S ──
    let mut dual_vals_buf = Vec::new();
    let mut adjoint_buf = Vec::new();
    let mut hs_columns: Vec<Vec<F>> = Vec::with_capacity(sketch_directions.len());

    for s in sketch_directions {
        assert_eq!(
            s.len(),
            n,
            "sketch direction length must match tape.num_inputs()"
        );
        let (_grad, hvp) = tape.hvp_with_buf(x, s, &mut dual_vals_buf, &mut adjoint_buf);
        hs_columns.push(hvp);
    }

    // ── Step 2: QR via Modified Gram-Schmidt ──
    let rank = modified_gram_schmidt(&mut hs_columns, eps);
    let q = &hs_columns; // q[0..rank] are orthonormal basis vectors

    // ── Step 3: Exact subspace trace ──
    let mut taylor_buf = Vec::new();
    let mut value = F::zero();
    let mut exact_trace = F::zero();

    for qi in q.iter().take(rank) {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, qi, &mut taylor_buf);
        value = c0;
        exact_trace = exact_trace + two * c2; // q_i^T H q_i
    }

    // ── Step 4: Residual Hutchinson ──
    // For each stochastic direction g, project out Q: g' = g - Q(Q^T g)
    let mut acc = WelfordAccumulator::new();
    let mut projected = vec![F::zero(); n];

    for g in stochastic_directions.iter() {
        assert_eq!(
            g.len(),
            n,
            "stochastic direction length must match tape.num_inputs()"
        );

        // g' = g - Q(Q^T g)
        projected.copy_from_slice(g);
        for qi in q.iter().take(rank) {
            let dot: F = qi
                .iter()
                .zip(g.iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
            for (p, &qv) in projected.iter_mut().zip(qi.iter()) {
                *p = *p - dot * qv;
            }
        }

        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &projected, &mut taylor_buf);
        value = c0;
        acc.update(two * c2);
    }

    let (residual_mean, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate: exact_trace + residual_mean,
        sample_variance,
        standard_error,
        num_samples: stochastic_directions.len(),
    }
}

// ══════════════════════════════════════════════
//  Divergence estimator
// ══════════════════════════════════════════════

/// Estimate the divergence (trace of Jacobian) of a vector field f: R^n → R^n.
///
/// Uses Hutchinson's trace estimator with first-order forward-mode AD (`Dual<F>`).
/// The tape must be recorded via [`record_multi`](crate::record_multi) with
/// `num_inputs == num_outputs`.
///
/// For each random direction v with `E[vv^T] = I`:
/// 1. Seed `Dual` inputs: `Dual(x_i, v_i)`
/// 2. Forward pass: `tape.forward_tangent(&dual_inputs, &mut buf)`
/// 3. Sample = Σ_i v_i * buf\[out_indices\[i\]\].eps = v^T J v
///
/// The mean of these samples converges to tr(J) = div(f).
///
/// # Panics
///
/// Panics if `directions` is empty, if `tape.num_outputs() != tape.num_inputs()`,
/// or if any direction's length does not match `tape.num_inputs()`.
pub fn divergence<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> DivergenceResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    let m = tape.num_outputs();
    assert_eq!(
        m, n,
        "divergence requires num_outputs ({}) == num_inputs ({})",
        m, n
    );
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let out_indices = tape.all_output_indices();
    let mut buf: Vec<Dual<F>> = Vec::new();
    let mut values = vec![F::zero(); m];
    let mut acc = WelfordAccumulator::new();

    for (k, v) in directions.iter().enumerate() {
        assert_eq!(v.len(), n, "direction length must match tape.num_inputs()");

        // Build Dual inputs
        let dual_inputs: Vec<Dual<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| Dual::new(xi, vi))
            .collect();

        tape.forward_tangent(&dual_inputs, &mut buf);

        // Extract function values (from first direction only is fine — all give same primal)
        if k == 0 {
            for (i, &oi) in out_indices.iter().enumerate() {
                values[i] = buf[oi as usize].re;
            }
        }

        // sample = v^T J v = Σ_i v_i * (J v)_i
        let sample: F = v
            .iter()
            .zip(out_indices.iter())
            .fold(F::zero(), |acc, (&vi, &oi)| acc + vi * buf[oi as usize].eps);

        acc.update(sample);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    DivergenceResult {
        values,
        estimate,
        sample_variance,
        standard_error,
        num_samples: directions.len(),
    }
}

// ══════════════════════════════════════════════
//  Parabolic PDE σ-transform
// ══════════════════════════════════════════════

/// Estimate the diffusion term `½ tr(σσ^T · Hess u)` for parabolic PDEs.
///
/// Uses the σ-transform: `½ tr(σσ^T H) = ½ Σ_i (σ·e_i)^T H (σ·e_i)`
/// where `σ·e_i` are the columns of σ (paper Eq 19, Appendix E).
///
/// Each column pushforward gives `(σ·e_i)^T H (σ·e_i)` via the second-order
/// Taylor coefficient: `2 * c2`. The diffusion term is `½ * Σ 2*c2 = Σ c2`.
///
/// # Arguments
///
/// - `sigma_columns`: the columns of σ as slices, each of length `tape.num_inputs()`.
///
/// Returns `(value, diffusion_term)`.
///
/// # Panics
///
/// Panics if `sigma_columns` is empty or any column's length does not match
/// `tape.num_inputs()`.
pub fn parabolic_diffusion<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sigma_columns: &[&[F]],
) -> (F, F) {
    assert!(!sigma_columns.is_empty(), "sigma_columns must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let half = F::from(0.5).unwrap();
    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut sum = F::zero();

    for col in sigma_columns {
        assert_eq!(
            col.len(),
            n,
            "sigma column length must match tape.num_inputs()"
        );
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, col, &mut buf);
        value = c0;
        sum = sum + two * c2; // (σ·e_i)^T H (σ·e_i)
    }

    (value, half * sum)
}

/// Stochastic version of [`parabolic_diffusion`]: subsample column indices.
///
/// Estimate = `(d / |J|) · ½ · Σ_{i∈J} (σ·e_i)^T H (σ·e_i)`.
///
/// # Panics
///
/// Panics if `sampled_indices` is empty or any index is out of bounds.
pub fn parabolic_diffusion_stochastic<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sigma_columns: &[&[F]],
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    let n = tape.num_inputs();
    let d = sigma_columns.len();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let two = F::from(2.0).unwrap();
    let half = F::from(0.5).unwrap();
    let df = F::from(d).unwrap();

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for &i in sampled_indices {
        assert!(i < d, "sampled index {} out of bounds (d={})", i, d);
        let col = sigma_columns[i];
        assert_eq!(
            col.len(),
            n,
            "sigma column length must match tape.num_inputs()"
        );
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, col, &mut buf);
        value = c0;
        acc.update(two * c2); // (σ·e_i)^T H (σ·e_i)
    }

    let (mean, sample_variance, standard_error) = acc.finalize();

    // Unbiased estimator for ½ Σ_i (σ·e_i)^T H (σ·e_i): d * mean * ½
    EstimatorResult {
        value,
        estimate: mean * df * half,
        sample_variance,
        standard_error,
        num_samples: sampled_indices.len(),
    }
}

// ══════════════════════════════════════════════
//  Dense STDE for positive-definite operators
// ══════════════════════════════════════════════

/// Dense STDE for a positive-definite 2nd-order operator.
///
/// Given a Cholesky factor `L` (lower-triangular) such that `C = L L^T`,
/// and standard Gaussian vectors `z_s ~ N(0, I)`, this computes
/// `v_s = L · z_s` internally and estimates
/// `tr(C · H_u(x)) = Σ_{ij} C_{ij} ∂²u/∂x_i∂x_j`.
///
/// The key insight: if `E[v v^T] = C`, then `E[v^T H v] = tr(C · H)`.
/// With `v = L · z` where `z ~ N(0, I)`, we get `E[vv^T] = L L^T = C`.
///
/// The caller provides `z_vectors` (standard Gaussian samples) and
/// `cholesky_rows` (the rows of L — only lower-triangular entries matter).
///
/// **Indefinite C deferred**: For indefinite operators, manually split into
/// `C⁺ - C⁻`, compute Cholesky factors for each, and call twice.
///
/// **No `rand` dependency**: callers provide z_vectors.
///
/// # Panics
///
/// Panics if `z_vectors` is empty, `cholesky_rows.len()` does not match
/// `tape.num_inputs()`, or any vector has the wrong length.
pub fn dense_stde_2nd<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    cholesky_rows: &[&[F]],
    z_vectors: &[&[F]],
) -> EstimatorResult<F> {
    assert!(!z_vectors.is_empty(), "z_vectors must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(
        cholesky_rows.len(),
        n,
        "cholesky_rows.len() must match tape.num_inputs()"
    );

    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut v = vec![F::zero(); n];
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for z in z_vectors.iter() {
        assert_eq!(z.len(), n, "z_vector length must match tape.num_inputs()");

        // Compute v = L · z (lower-triangular mat-vec)
        for i in 0..n {
            let row = cholesky_rows[i];
            let mut sum = F::zero();
            // Only use lower-triangular entries: j <= i
            for j in 0..=i {
                sum = sum + row[j] * z[j];
            }
            v[i] = sum;
        }

        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &v, &mut buf);
        value = c0;
        // 2 * c2 = v^T H v, and E[v^T H v] = tr(C · H) when E[vv^T] = C
        acc.update(two * c2);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: z_vectors.len(),
    }
}

// ══════════════════════════════════════════════
//  Sparse STDE (requires stde + diffop)
// ══════════════════════════════════════════════

#[cfg(feature = "diffop")]
use crate::diffop::SparseSamplingDistribution;

/// Sparse STDE: estimate `Lu(x)` by sampling sparse k-jets.
///
/// Each sample index corresponds to an entry in `dist`. For each sample,
/// one forward pushforward of a sparse k-jet is performed.
///
/// The per-sample estimator is: `sign(C_α) · Z · prefactor · coeffs[k]`
/// where `Z = dist.normalization()` (paper Eq 13).
///
/// Paper reference: Eq 13, Section 4.3.
///
/// # Panics
///
/// Panics if `sampled_indices` is empty or any index is out of bounds.
#[cfg(feature = "diffop")]
pub fn stde_sparse<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    dist: &SparseSamplingDistribution<F>,
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = dist.jet_order() + 1;
    let z = dist.normalization();
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for &idx in sampled_indices {
        let entry = dist.entry(idx);

        // Build TaylorDyn inputs: coeffs[0] = x[i] for all, then set
        // coeffs[slot] = 1/slot! for active variables per entry.input_coeffs
        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                for &(var, slot, inv_fact) in entry.input_coeffs() {
                    if var == i && slot < order {
                        coeffs[slot] = inv_fact;
                    }
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];

        // Extract: raw = coeffs[output_coeff_index] * extraction_prefactor
        let raw = out_coeffs[entry.output_coeff_index()] * entry.extraction_prefactor();
        // Sample = sign * Z * raw
        let sample = entry.sign() * z * raw;
        acc.update(sample);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: sampled_indices.len(),
    }
}

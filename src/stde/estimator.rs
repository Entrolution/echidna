use crate::Float;

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

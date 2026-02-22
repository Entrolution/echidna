use num_traits::Float;

/// Parameters controlling convergence checks.
#[derive(Debug, Clone)]
pub struct ConvergenceParams<F> {
    /// Maximum number of iterations (default: 100).
    pub max_iter: usize,
    /// Gradient norm tolerance: stop when `||g|| < grad_tol` (default: 1e-8).
    pub grad_tol: F,
    /// Step size tolerance: stop when `||x_{k+1} - x_k|| < step_tol` (default: 1e-12).
    pub step_tol: F,
    /// Function change tolerance: stop when `|f_{k+1} - f_k| < func_tol` (default: 0, disabled).
    pub func_tol: F,
}

impl Default for ConvergenceParams<f64> {
    fn default() -> Self {
        ConvergenceParams {
            max_iter: 100,
            grad_tol: 1e-8,
            step_tol: 1e-12,
            func_tol: 0.0,
        }
    }
}

impl Default for ConvergenceParams<f32> {
    fn default() -> Self {
        ConvergenceParams {
            max_iter: 100,
            grad_tol: 1e-5,
            step_tol: 1e-7,
            func_tol: 0.0,
        }
    }
}

/// Compute the L2 norm of a vector.
pub fn norm<F: Float>(v: &[F]) -> F {
    let mut s = F::zero();
    for &x in v {
        s = s + x * x;
    }
    s.sqrt()
}

/// Compute the dot product of two vectors.
pub fn dot<F: Float>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    let mut s = F::zero();
    for i in 0..a.len() {
        s = s + a[i] * b[i];
    }
    s
}

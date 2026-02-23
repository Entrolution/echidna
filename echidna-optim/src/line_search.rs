use num_traits::Float;

use crate::convergence::dot;
use crate::objective::Objective;

/// Parameters for the backtracking Armijo line search.
#[derive(Debug, Clone)]
pub struct ArmijoParams<F> {
    /// Sufficient decrease parameter (default: 1e-4).
    pub c: F,
    /// Backtracking factor (default: 0.5).
    pub rho: F,
    /// Initial step size (default: 1.0).
    pub alpha_init: F,
    /// Minimum step size before declaring failure (default: 1e-16).
    pub alpha_min: F,
}

impl Default for ArmijoParams<f64> {
    fn default() -> Self {
        ArmijoParams {
            c: 1e-4,
            rho: 0.5,
            alpha_init: 1.0,
            alpha_min: 1e-16,
        }
    }
}

impl Default for ArmijoParams<f32> {
    fn default() -> Self {
        ArmijoParams {
            c: 1e-4,
            rho: 0.5,
            alpha_init: 1.0,
            alpha_min: 1e-8,
        }
    }
}

/// Result of a successful line search.
#[derive(Debug)]
pub struct LineSearchResult<F> {
    /// The accepted step size.
    pub alpha: F,
    /// Objective value at `x + alpha * d`.
    pub value: F,
    /// Gradient at `x + alpha * d`.
    pub gradient: Vec<F>,
    /// Number of function evaluations used.
    pub evals: usize,
}

/// Backtracking line search satisfying the Armijo (sufficient decrease) condition.
///
/// Searches for `alpha` such that `f(x + alpha*d) <= f(x) + c * alpha * g^T d`.
///
/// Returns `None` if `alpha` falls below `alpha_min` (line search failure).
pub fn backtracking_armijo<F: Float, O: Objective<F>>(
    obj: &mut O,
    x: &[F],
    d: &[F],
    f_x: F,
    grad_x: &[F],
    params: &ArmijoParams<F>,
) -> Option<LineSearchResult<F>> {
    let n = x.len();
    let dg = dot(grad_x, d);

    // Not a descent direction — caller should handle this
    if dg >= F::zero() {
        return None;
    }

    let mut alpha = params.alpha_init;
    let mut x_new = vec![F::zero(); n];
    let mut evals = 0;

    loop {
        if alpha < params.alpha_min {
            return None;
        }

        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }

        let (f_new, g_new) = obj.eval_grad(&x_new);
        evals += 1;

        // Armijo condition: f(x + alpha*d) <= f(x) + c * alpha * g^T d
        if f_new <= f_x + params.c * alpha * dg {
            return Some(LineSearchResult {
                alpha,
                value: f_new,
                gradient: g_new,
                evals,
            });
        }

        alpha = alpha * params.rho;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple quadratic objective for testing: f(x) = 0.5 * (x0^2 + x1^2)
    struct Quadratic;

    impl Objective<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }

        fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
            let f = 0.5 * (x[0] * x[0] + x[1] * x[1]);
            let g = vec![x[0], x[1]];
            (f, g)
        }
    }

    #[test]
    fn armijo_quadratic_descent() {
        let mut obj = Quadratic;
        let x = vec![2.0, 3.0];
        let (f_x, grad) = obj.eval_grad(&x);
        // Steepest descent direction
        let d: Vec<f64> = grad.iter().map(|&g| -g).collect();

        let result =
            backtracking_armijo(&mut obj, &x, &d, f_x, &grad, &ArmijoParams::default()).unwrap();

        assert!(result.alpha > 0.0);
        assert!(result.value < f_x, "line search should decrease objective");
    }

    #[test]
    fn armijo_full_step_on_quadratic() {
        let mut obj = Quadratic;
        let x = vec![2.0, 3.0];
        let (f_x, grad) = obj.eval_grad(&x);
        let d: Vec<f64> = grad.iter().map(|&g| -g).collect();

        let result =
            backtracking_armijo(&mut obj, &x, &d, f_x, &grad, &ArmijoParams::default()).unwrap();

        // For a quadratic, steepest descent with alpha=1 satisfies Armijo with c=1e-4
        assert!(
            (result.alpha - 1.0).abs() < 1e-12,
            "full step should be accepted on quadratic, got alpha={}",
            result.alpha
        );
    }

    #[test]
    fn armijo_non_descent_returns_none() {
        let mut obj = Quadratic;
        let x = vec![2.0, 3.0];
        let (f_x, grad) = obj.eval_grad(&x);
        // Ascent direction (same as gradient)
        let d = grad.clone();

        let result = backtracking_armijo(&mut obj, &x, &d, f_x, &grad, &ArmijoParams::default());
        assert!(result.is_none());
    }
}

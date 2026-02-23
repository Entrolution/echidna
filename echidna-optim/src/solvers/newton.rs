use num_traits::Float;

use crate::convergence::{norm, ConvergenceParams};
use crate::linalg::lu_solve;
use crate::line_search::{backtracking_armijo, ArmijoParams};
use crate::objective::Objective;
use crate::result::{OptimResult, TerminationReason};

/// Configuration for the Newton solver.
#[derive(Debug, Clone)]
pub struct NewtonConfig<F> {
    /// Convergence parameters.
    pub convergence: ConvergenceParams<F>,
    /// Line search parameters.
    pub line_search: ArmijoParams<F>,
}

impl Default for NewtonConfig<f64> {
    fn default() -> Self {
        NewtonConfig {
            convergence: ConvergenceParams::default(),
            line_search: ArmijoParams::default(),
        }
    }
}

impl Default for NewtonConfig<f32> {
    fn default() -> Self {
        NewtonConfig {
            convergence: ConvergenceParams::default(),
            line_search: ArmijoParams::default(),
        }
    }
}

/// Newton's method with LU-based Hessian solve and Armijo line search.
///
/// Minimizes `obj` starting from `x0`. At each iteration, solves `H * delta = -g`
/// via LU factorization, then performs a backtracking line search along `delta`.
///
/// Requires `obj` to implement `eval_hessian`.
pub fn newton<F: Float, O: Objective<F>>(
    obj: &mut O,
    x0: &[F],
    config: &NewtonConfig<F>,
) -> OptimResult<F> {
    let n = x0.len();

    if config.convergence.max_iter == 0 {
        return OptimResult {
            x: x0.to_vec(),
            value: F::zero(),
            gradient: vec![F::zero(); n],
            gradient_norm: F::zero(),
            iterations: 0,
            func_evals: 0,
            termination: TerminationReason::NumericalError,
        };
    }

    let mut x = x0.to_vec();
    let (mut f_val, mut grad, mut hess) = obj.eval_hessian(&x);
    let mut func_evals = 1usize;
    let mut grad_norm = norm(&grad);

    if grad_norm < config.convergence.grad_tol {
        return OptimResult {
            x,
            value: f_val,
            gradient: grad,
            gradient_norm: grad_norm,
            iterations: 0,
            func_evals,
            termination: TerminationReason::GradientNorm,
        };
    }

    for iter in 0..config.convergence.max_iter {
        // Solve H * delta = -g
        let neg_grad: Vec<F> = grad.iter().map(|&g| F::zero() - g).collect();
        let delta = match lu_solve(&hess, &neg_grad) {
            Some(d) => d,
            None => {
                return OptimResult {
                    x,
                    value: f_val,
                    gradient: grad,
                    gradient_norm: grad_norm,
                    iterations: iter,
                    func_evals,
                    termination: TerminationReason::NumericalError,
                };
            }
        };

        // Line search along Newton direction
        let ls = match backtracking_armijo(obj, &x, &delta, f_val, &grad, &config.line_search) {
            Some(ls) => ls,
            None => {
                return OptimResult {
                    x,
                    value: f_val,
                    gradient: grad,
                    gradient_norm: grad_norm,
                    iterations: iter,
                    func_evals,
                    termination: TerminationReason::LineSearchFailed,
                };
            }
        };
        func_evals += ls.evals;

        // Update x
        let mut step_norm_sq = F::zero();
        for i in 0..n {
            let step = ls.alpha * delta[i];
            step_norm_sq = step_norm_sq + step * step;
            x[i] = x[i] + step;
        }

        let f_prev = f_val;

        // Re-evaluate with Hessian at new point
        let result = obj.eval_hessian(&x);
        func_evals += 1;
        f_val = result.0;
        grad = result.1;
        hess = result.2;
        grad_norm = norm(&grad);

        // Convergence checks
        if grad_norm < config.convergence.grad_tol {
            return OptimResult {
                x,
                value: f_val,
                gradient: grad,
                gradient_norm: grad_norm,
                iterations: iter + 1,
                func_evals,
                termination: TerminationReason::GradientNorm,
            };
        }

        if step_norm_sq.sqrt() < config.convergence.step_tol {
            return OptimResult {
                x,
                value: f_val,
                gradient: grad,
                gradient_norm: grad_norm,
                iterations: iter + 1,
                func_evals,
                termination: TerminationReason::StepSize,
            };
        }

        if config.convergence.func_tol > F::zero()
            && (f_prev - f_val).abs() < config.convergence.func_tol
        {
            return OptimResult {
                x,
                value: f_val,
                gradient: grad,
                gradient_norm: grad_norm,
                iterations: iter + 1,
                func_evals,
                termination: TerminationReason::FunctionChange,
            };
        }
    }

    OptimResult {
        x,
        value: f_val,
        gradient: grad,
        gradient_norm: grad_norm,
        iterations: config.convergence.max_iter,
        func_evals,
        termination: TerminationReason::MaxIterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rosenbrock;

    impl Objective<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }

        fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            let f = a * a + 100.0 * b * b;
            let g0 = -2.0 * a - 400.0 * x[0] * b;
            let g1 = 200.0 * b;
            (f, vec![g0, g1])
        }

        fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            let f = a * a + 100.0 * b * b;
            let g0 = -2.0 * a - 400.0 * x[0] * b;
            let g1 = 200.0 * b;

            let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
            let h01 = -400.0 * x[0];
            let h11 = 200.0;

            (f, vec![g0, g1], vec![vec![h00, h01], vec![h01, h11]])
        }
    }

    #[test]
    fn newton_rosenbrock() {
        let mut obj = Rosenbrock;
        let config = NewtonConfig::default();
        let result = newton(&mut obj, &[0.0, 0.0], &config);

        assert_eq!(result.termination, TerminationReason::GradientNorm);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-6,
            "x[0] = {}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-6,
            "x[1] = {}, expected 1.0",
            result.x[1]
        );
        assert!(result.gradient_norm < 1e-8);
    }

    #[test]
    fn newton_singular_hessian() {
        struct SingularAtOrigin;

        impl Objective<f64> for SingularAtOrigin {
            fn dim(&self) -> usize {
                2
            }

            fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
                let f = x[0] * x[0] + x[1] * x[1];
                (f, vec![2.0 * x[0], 2.0 * x[1]])
            }

            fn eval_hessian(&mut self, _x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
                // Return a singular Hessian
                (1.0, vec![1.0, 1.0], vec![vec![1.0, 1.0], vec![1.0, 1.0]])
            }
        }

        let mut obj = SingularAtOrigin;
        let config = NewtonConfig::default();
        let result = newton(&mut obj, &[2.0, 3.0], &config);

        assert_eq!(result.termination, TerminationReason::NumericalError);
    }
}

use num_traits::Float;

use crate::convergence::{check_convergence, dot, norm, ConvergenceParams};
use crate::linalg::lu_solve;
use crate::line_search::{backtracking_armijo_with_evals, ArmijoParams};
use crate::objective::Objective;
use crate::result::{NewtonDiagnostics, OptimResult, SolverDiagnostics, TerminationReason};

/// Configuration for the Newton solver.
#[derive(Debug, Clone)]
pub struct NewtonConfig<F> {
    /// Convergence parameters.
    pub convergence: ConvergenceParams<F>,
    /// Line search parameters.
    pub line_search: ArmijoParams<F>,
}

// One generic impl, same shape as LbfgsConfig's (TrustRegionConfig stays
// per-precision: a generic Default there would need F::from(100.0).unwrap()).
impl<F: Float> Default for NewtonConfig<F>
where
    ConvergenceParams<F>: Default,
    ArmijoParams<F>: Default,
{
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
    let mut diag = NewtonDiagnostics::default();

    if config.convergence.max_iter == 0 {
        return OptimResult::assemble(
            x0.to_vec(),
            F::nan(),
            vec![F::nan(); n],
            F::nan(),
            0,
            0,
            TerminationReason::NumericalError,
            SolverDiagnostics::Newton(diag),
        );
    }

    let mut x = x0.to_vec();
    let (mut f_val, mut grad, mut hess) = obj.eval_hessian(&x);
    let mut func_evals = 1usize;
    let mut grad_norm = norm(&grad);

    // NaN/Inf detection
    if !grad_norm.is_finite() || !f_val.is_finite() {
        return OptimResult::assemble(
            x,
            f_val,
            grad,
            grad_norm,
            0,
            func_evals,
            TerminationReason::NumericalError,
            SolverDiagnostics::Newton(diag),
        );
    }

    if grad_norm < config.convergence.grad_tol {
        return OptimResult::assemble(
            x,
            f_val,
            grad,
            grad_norm,
            0,
            func_evals,
            TerminationReason::GradientNorm,
            SolverDiagnostics::Newton(diag),
        );
    }

    for iter in 0..config.convergence.max_iter {
        // Solve H * delta = -g
        let neg_grad: Vec<F> = grad.iter().map(|&g| -g).collect();
        let raw_delta = lu_solve(&hess, &neg_grad);

        // Check whether `delta` is a descent direction (gᵀ·delta < 0). An
        // indefinite Hessian (common near saddle points on non-convex
        // problems) can produce a direction that points uphill. In that
        // case, or when the solve fails outright, fall back to steepest
        // descent: `delta = -grad`. This keeps Newton usable on non-convex
        // problems instead of returning `NumericalError` or
        // `LineSearchFailed` at the first saddle.
        let delta = match raw_delta {
            Some(d) if dot(&grad, &d) < F::zero() => d,
            _ => {
                diag.fallback_steps += 1;
                neg_grad
            }
        };

        // Line search along Newton (or fallback steepest-descent) direction
        let ls = match backtracking_armijo_with_evals(
            obj,
            &x,
            &delta,
            f_val,
            &grad,
            &config.line_search,
            &mut func_evals,
        ) {
            Some(ls) => ls,
            None => {
                return OptimResult::assemble(
                    x,
                    f_val,
                    grad,
                    grad_norm,
                    iter,
                    func_evals,
                    TerminationReason::LineSearchFailed,
                    SolverDiagnostics::Newton(diag),
                );
            }
        };
        // `func_evals` already includes this search's evaluations via the
        // accumulator (which also survives the failure path above).
        diag.line_search_backtracks += ls.evals.saturating_sub(1);

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

        // NaN/Inf detection
        if !grad_norm.is_finite() || !f_val.is_finite() {
            return OptimResult::assemble(
                x,
                f_val,
                grad,
                grad_norm,
                iter + 1,
                func_evals,
                TerminationReason::NumericalError,
                SolverDiagnostics::Newton(diag),
            );
        }

        // Convergence checks (gradient, step, relative function change).
        if let Some(reason) = check_convergence(
            grad_norm,
            step_norm_sq.sqrt(),
            f_prev,
            f_val,
            &config.convergence,
        ) {
            return OptimResult::assemble(
                x,
                f_val,
                grad,
                grad_norm,
                iter + 1,
                func_evals,
                reason,
                SolverDiagnostics::Newton(diag),
            );
        }
    }

    OptimResult::assemble(
        x,
        f_val,
        grad,
        grad_norm,
        config.convergence.max_iter,
        func_evals,
        TerminationReason::MaxIterations,
        SolverDiagnostics::Newton(diag),
    )
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

        // Indefinite-fallback path: when the LU solve fails, Newton now
        // falls back to steepest descent. This test objective is
        // inconsistent (eval_grad and eval_hessian describe different
        // functions), so the Armijo search eventually fails. Either
        // termination is acceptable; the important contract is that the
        // solver doesn't silently report success on a pathological input.
        assert!(
            matches!(
                result.termination,
                TerminationReason::NumericalError | TerminationReason::LineSearchFailed
            ),
            "expected NumericalError or LineSearchFailed, got {:?}",
            result.termination
        );
    }
}

use num_traits::Float;

use crate::convergence::{dot, norm, ConvergenceParams};
use crate::line_search::{backtracking_armijo, ArmijoParams};
use crate::objective::Objective;
use crate::result::{OptimResult, TerminationReason};

/// Configuration for the L-BFGS solver.
#[derive(Debug, Clone)]
pub struct LbfgsConfig<F> {
    /// Number of recent (s, y) pairs to store (default: 10).
    pub memory: usize,
    /// Convergence parameters.
    pub convergence: ConvergenceParams<F>,
    /// Line search parameters.
    pub line_search: ArmijoParams<F>,
}

impl Default for LbfgsConfig<f64> {
    fn default() -> Self {
        LbfgsConfig {
            memory: 10,
            convergence: ConvergenceParams::default(),
            line_search: ArmijoParams::default(),
        }
    }
}

impl Default for LbfgsConfig<f32> {
    fn default() -> Self {
        LbfgsConfig {
            memory: 10,
            convergence: ConvergenceParams::default(),
            line_search: ArmijoParams::default(),
        }
    }
}

/// L-BFGS optimization.
///
/// Minimizes `obj` starting from `x0` using the limited-memory BFGS method
/// with two-loop recursion and backtracking Armijo line search.
pub fn lbfgs<F: Float, O: Objective<F>>(
    obj: &mut O,
    x0: &[F],
    config: &LbfgsConfig<F>,
) -> OptimResult<F> {
    let n = x0.len();

    // Config validation
    if config.memory == 0 || config.convergence.max_iter == 0 {
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
    let (mut f_val, mut grad) = obj.eval_grad(&x);
    let mut func_evals = 1usize;
    let mut grad_norm = norm(&grad);

    // Check initial convergence
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

    // L-BFGS history buffers: store most recent `m` pairs
    let m = config.memory;
    let mut s_hist: Vec<Vec<F>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<F>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<F> = Vec::with_capacity(m);

    for iter in 0..config.convergence.max_iter {
        // Two-loop recursion to compute H_k * g_k
        let d = two_loop_recursion(&grad, &s_hist, &y_hist, &rho_hist);

        // Line search
        let ls = match backtracking_armijo(obj, &x, &d, f_val, &grad, &config.line_search) {
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

        // Compute s = x_new - x, y = g_new - g
        let mut s = vec![F::zero(); n];
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            let x_new_i = x[i] + ls.alpha * d[i];
            s[i] = x_new_i - x[i];
            y[i] = ls.gradient[i] - grad[i];
            x[i] = x_new_i;
        }

        let f_prev = f_val;
        f_val = ls.value;
        grad = ls.gradient;
        grad_norm = norm(&grad);

        // Update history
        let sy = dot(&s, &y);
        if sy > F::zero() {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(F::one() / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

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

        let step_norm = norm_step(ls.alpha, &d);
        if step_norm < config.convergence.step_tol {
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

/// L-BFGS two-loop recursion: compute d = -H_k * g_k.
fn two_loop_recursion<F: Float>(
    grad: &[F],
    s_hist: &[Vec<F>],
    y_hist: &[Vec<F>],
    rho_hist: &[F],
) -> Vec<F> {
    let k = s_hist.len();
    let n = grad.len();

    // q = g
    let mut q: Vec<F> = grad.to_vec();

    // First loop: newest to oldest
    let mut alpha = vec![F::zero(); k];
    for i in (0..k).rev() {
        alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
        for j in 0..n {
            q[j] = q[j] - alpha[i] * y_hist[i][j];
        }
    }

    // Initial Hessian approximation: H_0 = gamma * I
    // gamma = s^T y / y^T y (from the most recent pair)
    let mut r = q;
    if k > 0 {
        let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
        let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
        if yy > F::zero() {
            let gamma = sy / yy;
            for v in r.iter_mut() {
                *v = *v * gamma;
            }
        }
    }

    // Second loop: oldest to newest
    for i in 0..k {
        let beta = rho_hist[i] * dot(&y_hist[i], &r);
        for j in 0..n {
            r[j] = r[j] + (alpha[i] - beta) * s_hist[i][j];
        }
    }

    // Negate: d = -H * g
    for v in r.iter_mut() {
        *v = F::zero() - *v;
    }

    r
}

fn norm_step<F: Float>(alpha: F, d: &[F]) -> F {
    let mut s = F::zero();
    for &di in d {
        let step = alpha * di;
        s = s + step * step;
    }
    s.sqrt()
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
    }

    #[test]
    fn lbfgs_rosenbrock() {
        let mut obj = Rosenbrock;
        let config = LbfgsConfig::default();
        let result = lbfgs(&mut obj, &[0.0, 0.0], &config);

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
    fn lbfgs_already_converged() {
        let mut obj = Rosenbrock;
        let config = LbfgsConfig::default();
        let result = lbfgs(&mut obj, &[1.0, 1.0], &config);

        assert_eq!(result.termination, TerminationReason::GradientNorm);
        assert_eq!(result.iterations, 0);
    }
}

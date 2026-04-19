use num_traits::Float;

use crate::convergence::{dot, norm, ConvergenceParams};
use crate::line_search::{backtracking_armijo, ArmijoParams};
use crate::objective::Objective;
use crate::result::{LbfgsDiagnostics, OptimResult, SolverDiagnostics, TerminationReason};

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
    let mut diag = LbfgsDiagnostics::default();

    // Config validation
    if config.memory == 0 || config.convergence.max_iter == 0 {
        return OptimResult {
            x: x0.to_vec(),
            value: F::nan(),
            gradient: vec![F::nan(); n],
            gradient_norm: F::nan(),
            iterations: 0,
            func_evals: 0,
            termination: TerminationReason::NumericalError,
            diagnostics: SolverDiagnostics::Lbfgs(diag),
        };
    }

    let mut x = x0.to_vec();
    let (mut f_val, mut grad) = obj.eval_grad(&x);
    let mut func_evals = 1usize;
    let mut grad_norm = norm(&grad);

    // NaN/Inf detection
    if !grad_norm.is_finite() || !f_val.is_finite() {
        return OptimResult {
            x,
            value: f_val,
            gradient: grad,
            gradient_norm: grad_norm,
            iterations: 0,
            func_evals,
            termination: TerminationReason::NumericalError,
            diagnostics: SolverDiagnostics::Lbfgs(diag),
        };
    }

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
            diagnostics: SolverDiagnostics::Lbfgs(diag),
        };
    }

    // L-BFGS history buffers: store most recent `m` pairs
    let m = config.memory;
    let mut s_hist: Vec<Vec<F>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<F>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<F> = Vec::with_capacity(m);

    for iter in 0..config.convergence.max_iter {
        // Two-loop recursion. Returns `(direction, gamma_clamp_hit)` so
        // we can count clamps without threading a `&mut usize`.
        let (d, gamma_clamped) = two_loop_recursion(&grad, &s_hist, &y_hist, &rho_hist);
        if gamma_clamped {
            diag.gamma_clamp_hits += 1;
        }

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
                    diagnostics: SolverDiagnostics::Lbfgs(diag),
                };
            }
        };
        func_evals += ls.evals;
        // `ls.evals` counts every trial point including the first (alpha = 1).
        // A backtrack is any trial beyond the first.
        diag.line_search_backtracks += ls.evals.saturating_sub(1);

        // Compute s = x_new - x, y = g_new - g
        let mut s = vec![F::zero(); n];
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            // Compute s = alpha * d directly instead of (x + alpha*d) - x
            // to avoid cancellation when ||x|| >> alpha*||d||
            s[i] = ls.alpha * d[i];
            y[i] = ls.gradient[i] - grad[i];
            x[i] = x[i] + s[i];
        }

        let f_prev = f_val;
        f_val = ls.value;
        grad = ls.gradient;
        grad_norm = norm(&grad);

        // Update history. Filter out pairs with near-zero curvature. The
        // original guard `sy > eps * yy` has units `[s]·[y]` on the LHS and
        // `[y]²` on the RHS — dimensionally inconsistent, so it behaves
        // differently for "tall" vs "short" `y` vectors. Use a Cauchy-Schwarz
        // normalized filter `sy > eps * sqrt(ss * yy)` which is dimensionally
        // consistent (`cos θ > eps`) and reliably selects pairs with
        // non-trivial curvature regardless of vector magnitudes.
        let sy = dot(&s, &y);
        let ss = dot(&s, &s);
        let yy = dot(&y, &y);
        let cs_scale = (ss * yy).sqrt();
        if sy > F::epsilon() * cs_scale {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
                diag.pairs_evicted_by_memory += 1;
            }
            rho_hist.push(F::one() / sy);
            s_hist.push(s);
            y_hist.push(y);
            diag.pairs_accepted += 1;
        } else {
            diag.pairs_curvature_rejected += 1;
        }

        // NaN/Inf detection
        if !grad_norm.is_finite() || !f_val.is_finite() {
            return OptimResult {
                x,
                value: f_val,
                gradient: grad,
                gradient_norm: grad_norm,
                iterations: iter + 1,
                func_evals,
                termination: TerminationReason::NumericalError,
                diagnostics: SolverDiagnostics::Lbfgs(diag),
            };
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
                diagnostics: SolverDiagnostics::Lbfgs(diag),
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
                diagnostics: SolverDiagnostics::Lbfgs(diag),
            };
        }

        // Relative func_tol: absolute `|f_prev - f_val| < tol` is scale-
        // blind — a tolerance of 1e-8 means ULP-precision on large-
        // magnitude objectives (|f| ≈ 1e8) and impossibly tight on tiny
        // ones. Scale by `(1 + |f|)` so the criterion tracks the problem.
        if config.convergence.func_tol > F::zero()
            && (f_prev - f_val).abs() < config.convergence.func_tol * (F::one() + f_val.abs())
        {
            return OptimResult {
                x,
                value: f_val,
                gradient: grad,
                gradient_norm: grad_norm,
                iterations: iter + 1,
                func_evals,
                termination: TerminationReason::FunctionChange,
                diagnostics: SolverDiagnostics::Lbfgs(diag),
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
        diagnostics: SolverDiagnostics::Lbfgs(diag),
    }
}

/// L-BFGS two-loop recursion: compute `d = -H_k * g_k`.
///
/// Returns `(direction, gamma_clamp_hit)`. The boolean is `true` when
/// the initial gamma had to be clamped to `[1e-3, 1e3]` or replaced
/// with `1.0` because `sy/yy` was non-finite — surfaced into
/// `LbfgsDiagnostics::gamma_clamp_hits` by the caller.
fn two_loop_recursion<F: Float>(
    grad: &[F],
    s_hist: &[Vec<F>],
    y_hist: &[Vec<F>],
    rho_hist: &[F],
) -> (Vec<F>, bool) {
    let k = s_hist.len();
    let n = grad.len();
    let mut gamma_clamp_hit = false;

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
        if yy > F::epsilon() {
            let raw_gamma = sy / yy;
            // Clamp `gamma` to [1e-3, 1e3]. A curvature pair that just
            // barely passed the acceptance filter can have `sy/yy ≈ eps`
            // — the two-loop recursion then scales the direction by that
            // tiny factor, Armijo backtracks to alpha_min, and the
            // search reports LineSearchFailed. Bounded gamma keeps the
            // direction magnitude in a line-search-friendly range.
            let lo = F::from(1e-3).unwrap();
            let hi = F::from(1e3).unwrap();
            // Detect the clamp via comparison rather than `clamped != raw`
            // so we don't rely on float-equality. Clamp boundary values
            // (`raw_gamma == 1e-3` or `1e3` exactly) are not flagged —
            // they pass through unchanged and don't represent the
            // pathology the counter tracks.
            let gamma = if raw_gamma.is_finite() {
                if raw_gamma < lo || raw_gamma > hi {
                    gamma_clamp_hit = true;
                }
                raw_gamma.max(lo).min(hi)
            } else {
                gamma_clamp_hit = true;
                F::one()
            };
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

    (r, gamma_clamp_hit)
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

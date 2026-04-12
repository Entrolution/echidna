use num_traits::Float;

use crate::convergence::{dot, norm, ConvergenceParams};
use crate::objective::Objective;
use crate::result::{OptimResult, TerminationReason};

/// Configuration for the trust-region solver.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig<F> {
    /// Initial trust-region radius (default: 1.0).
    pub initial_radius: F,
    /// Maximum trust-region radius (default: 100.0).
    pub max_radius: F,
    /// Minimum trust-region radius (default: `F::epsilon()`).
    /// Solver returns `NumericalError` if radius shrinks below this.
    pub min_radius: F,
    /// Acceptance threshold for the ratio of actual to predicted reduction (default: 0.1).
    pub eta: F,
    /// Maximum CG iterations per trust-region subproblem (default: 2 * dim).
    /// If 0, defaults to 2 * dim.
    pub max_cg_iter: usize,
    /// Convergence parameters.
    pub convergence: ConvergenceParams<F>,
}

impl Default for TrustRegionConfig<f64> {
    fn default() -> Self {
        TrustRegionConfig {
            initial_radius: 1.0,
            max_radius: 100.0,
            min_radius: f64::EPSILON,
            eta: 0.1,
            max_cg_iter: 0,
            convergence: ConvergenceParams::default(),
        }
    }
}

impl Default for TrustRegionConfig<f32> {
    fn default() -> Self {
        TrustRegionConfig {
            initial_radius: 1.0,
            max_radius: 100.0,
            min_radius: f32::EPSILON,
            eta: 0.1,
            max_cg_iter: 0,
            convergence: ConvergenceParams::default(),
        }
    }
}

/// Trust-region optimization using Steihaug-Toint CG.
///
/// Minimizes `obj` starting from `x0`. Uses Hessian-vector products
/// (via `obj.hvp()`) to solve the trust-region subproblem approximately
/// with truncated conjugate gradients (Steihaug-Toint).
pub fn trust_region<F: Float, O: Objective<F>>(
    obj: &mut O,
    x0: &[F],
    config: &TrustRegionConfig<F>,
) -> OptimResult<F> {
    let n = x0.len();

    if config.convergence.max_iter == 0
        || config.initial_radius <= F::zero()
        || config.max_radius <= F::zero()
    {
        return OptimResult {
            x: x0.to_vec(),
            value: F::nan(),
            gradient: vec![F::nan(); n],
            gradient_norm: F::nan(),
            iterations: 0,
            func_evals: 0,
            termination: TerminationReason::NumericalError,
        };
    }

    let max_cg = if config.max_cg_iter == 0 {
        2 * n
    } else {
        config.max_cg_iter
    };

    let mut x = x0.to_vec();
    let (mut f_val, mut grad) = obj.eval_grad(&x);
    let mut func_evals = 1usize;
    let mut grad_norm = norm(&grad);
    let mut radius = config.initial_radius;

    if !grad_norm.is_finite() || !f_val.is_finite() {
        return OptimResult {
            x,
            value: f_val,
            gradient: grad,
            gradient_norm: grad_norm,
            iterations: 0,
            func_evals,
            termination: TerminationReason::NumericalError,
        };
    }

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

    let two = F::one() + F::one();
    let quarter = F::one() / (two * two);
    let three_quarter = F::one() - quarter;

    for iter in 0..config.convergence.max_iter {
        // Solve the trust-region subproblem with Steihaug-Toint CG
        let step = steihaug_cg(obj, &x, &grad, radius, max_cg, &mut func_evals);

        // Predicted reduction: -g^T s - 0.5 * s^T H s
        // Note: this recomputes H*s outside of CG. For stateful objectives (e.g.,
        // stochastic HVP estimators), this may diverge from the H*s used inside
        // steihaug_cg. A future optimization could return gs/shs from CG directly.
        let (_, hvp_result) = obj.hvp(&x, &step);
        func_evals += 1;
        let gs = dot(&grad, &step);
        let shs = dot(&step, &hvp_result);
        let predicted = F::zero() - gs - shs / two;

        // Actual reduction
        let mut x_new = vec![F::zero(); n];
        for i in 0..n {
            x_new[i] = x[i] + step[i];
        }
        let (f_new, g_new) = obj.eval_grad(&x_new);
        func_evals += 1;
        let actual = f_val - f_new;

        let step_norm = norm(&step);

        // Guard: reject step unconditionally when predicted reduction is non-positive.
        // The quadratic model predicts the step makes things worse — the subproblem is unreliable.
        if predicted <= F::zero() {
            radius = (quarter * radius).max(config.min_radius);
            continue;
        }

        // Ratio of actual to predicted reduction
        let rho = if predicted.abs() < F::epsilon() {
            if actual >= F::zero() {
                F::one()
            } else {
                F::zero()
            }
        } else {
            actual / predicted
        };

        // Update trust-region radius
        if rho < quarter {
            radius = (quarter * radius).max(config.min_radius);
        } else if rho > three_quarter && (step_norm - radius).abs() < F::epsilon() * radius {
            // Step was on the boundary and rho is good — expand
            radius = (two * radius).min(config.max_radius);
        }
        // Otherwise keep radius unchanged

        // Accept or reject step
        if rho > config.eta {
            let f_prev = f_val;
            x = x_new;
            f_val = f_new;
            grad = g_new;
            grad_norm = norm(&grad);

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
                };
            }

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
        // If rejected, loop again with smaller radius
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

/// Steihaug-Toint truncated CG for the trust-region subproblem.
///
/// Approximately minimizes `m(s) = g^T s + 0.5 s^T H s` subject to `||s|| <= radius`.
/// Returns the step `s`.
fn steihaug_cg<F: Float, O: Objective<F>>(
    obj: &mut O,
    x: &[F],
    grad: &[F],
    radius: F,
    max_iter: usize,
    func_evals: &mut usize,
) -> Vec<F> {
    let n = grad.len();
    let mut s = vec![F::zero(); n];
    let mut r: Vec<F> = grad.to_vec();
    let mut d: Vec<F> = r.iter().map(|&ri| F::zero() - ri).collect();
    let mut r_dot_r = dot(&r, &r);
    let cg_tol = F::epsilon().sqrt() * r_dot_r.sqrt(); // relative to initial residual (= ||grad||)

    if r_dot_r.sqrt() < cg_tol {
        return s;
    }

    for _ in 0..max_iter {
        // H * d via hvp
        let (_, hd) = obj.hvp(x, &d);
        *func_evals += 1;

        let d_hd = dot(&d, &hd);

        // Negative curvature: go to the boundary
        if d_hd <= F::zero() {
            let tau = boundary_tau(&s, &d, radius);
            for i in 0..n {
                s[i] = s[i] + tau * d[i];
            }
            return s;
        }

        let alpha = r_dot_r / d_hd;

        // Check if step would leave the trust region
        let mut s_next = vec![F::zero(); n];
        for i in 0..n {
            s_next[i] = s[i] + alpha * d[i];
        }
        if norm(&s_next) >= radius {
            let tau = boundary_tau(&s, &d, radius);
            for i in 0..n {
                s[i] = s[i] + tau * d[i];
            }
            return s;
        }

        s = s_next;

        // Update residual
        for i in 0..n {
            r[i] = r[i] + alpha * hd[i];
        }
        let r_dot_r_new = dot(&r, &r);

        if r_dot_r_new.sqrt() < cg_tol {
            return s;
        }

        let beta = r_dot_r_new / r_dot_r;
        r_dot_r = r_dot_r_new;

        for i in 0..n {
            d[i] = F::zero() - r[i] + beta * d[i];
        }
    }

    s
}

/// Find `tau > 0` such that `||s + tau * d|| = radius`.
///
/// Solves `||s + tau * d||^2 = radius^2` for the positive root.
fn boundary_tau<F: Float>(s: &[F], d: &[F], radius: F) -> F {
    let dd = dot(d, d);
    if dd < F::epsilon() {
        return F::zero();
    }
    let sd = dot(s, d);
    let ss = dot(s, s);
    let two = F::one() + F::one();

    // tau^2 * dd + 2*tau*sd + ss = radius^2
    // Quadratic: a*tau^2 + b*tau + c = 0
    let a = dd;
    let b = two * sd;
    let c = ss - radius * radius;

    let disc = b * b - (two + two) * a * c;
    if disc < F::zero() {
        return F::zero();
    }

    // Use numerically stable quadratic formula (Vieta's) to avoid
    // catastrophic cancellation when |b| ≈ sqrt(disc).
    let sqrt_disc = disc.sqrt();
    // Compute the root with larger magnitude first (no cancellation)
    let neg_b = F::zero() - b;
    let r_large = if neg_b >= F::zero() {
        (neg_b + sqrt_disc) / (two * a)
    } else {
        (neg_b - sqrt_disc) / (two * a)
    };
    // Second root via Vieta's formula: tau1 * tau2 = c / a
    let r_small = if r_large.abs() < F::epsilon() {
        F::zero()
    } else {
        c / (a * r_large)
    };

    let (tau1, tau2) = if r_large < r_small {
        (r_large, r_small)
    } else {
        (r_small, r_large)
    };

    // Return smallest positive root
    if tau1 > F::zero() {
        tau1
    } else if tau2 > F::zero() {
        tau2
    } else {
        F::zero()
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

        fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
            // H = [[2 - 400*(x1 - 3*x0^2), -400*x0],
            //       [-400*x0,                  200  ]]
            let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
            let h01 = -400.0 * x[0];
            let h11 = 200.0;

            let hv0 = h00 * v[0] + h01 * v[1];
            let hv1 = h01 * v[0] + h11 * v[1];

            let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let g1 = 200.0 * (x[1] - x[0] * x[0]);

            (vec![g0, g1], vec![hv0, hv1])
        }
    }

    #[test]
    fn trust_region_rosenbrock() {
        let mut obj = Rosenbrock;
        let config = TrustRegionConfig {
            convergence: ConvergenceParams {
                max_iter: 200,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = trust_region(&mut obj, &[0.0, 0.0], &config);

        assert_eq!(
            result.termination,
            TerminationReason::GradientNorm,
            "terminated with {:?} after {} iterations",
            result.termination,
            result.iterations
        );
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
    }

    struct Rosenbrock4D;

    impl Objective<f64> for Rosenbrock4D {
        fn dim(&self) -> usize {
            4
        }

        fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
            let mut f = 0.0;
            let mut g = vec![0.0; 4];
            for i in 0..3 {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                f += a * a + 100.0 * b * b;
                g[i] += -2.0 * a - 400.0 * x[i] * b;
                g[i + 1] += 200.0 * b;
            }
            (f, g)
        }

        fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let n = 4;
            let mut hv = vec![0.0; n];
            let mut g = vec![0.0; n];

            for i in 0..3 {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];

                g[i] += -2.0 * a - 400.0 * x[i] * b;
                g[i + 1] += 200.0 * b;

                let h_ii = 2.0 - 400.0 * (x[i + 1] - 3.0 * x[i] * x[i]);
                let h_ij = -400.0 * x[i];
                let h_jj = 200.0;

                hv[i] += h_ii * v[i] + h_ij * v[i + 1];
                hv[i + 1] += h_ij * v[i] + h_jj * v[i + 1];
            }

            (g, hv)
        }
    }

    #[test]
    fn trust_region_rosenbrock_4d() {
        let mut obj = Rosenbrock4D;
        let config = TrustRegionConfig {
            convergence: ConvergenceParams {
                max_iter: 500,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = trust_region(&mut obj, &[0.0, 0.0, 0.0, 0.0], &config);

        assert_eq!(
            result.termination,
            TerminationReason::GradientNorm,
            "terminated with {:?} after {} iterations, grad_norm={}",
            result.termination,
            result.iterations,
            result.gradient_norm
        );
        for i in 0..4 {
            assert!(
                (result.x[i] - 1.0).abs() < 1e-5,
                "x[{}] = {}, expected 1.0",
                i,
                result.x[i]
            );
        }
    }

    #[test]
    fn boundary_tau_nearly_parallel() {
        // When s is near the trust-region boundary and d is nearly parallel,
        // c ≈ 0 so disc ≈ b², making (-b + sqrt(disc)) suffer cancellation.
        // Vieta's formula avoids this.
        let s = [1.0 - 1e-14, 0.0]; // very close to boundary
        let d = [1.0, 1e-10]; // nearly parallel to s
        let radius = 1.0;
        let tau = boundary_tau(&s, &d, radius);
        // tau should satisfy ||s + tau*d|| ≈ radius
        let norm_sq = (s[0] + tau * d[0]).powi(2) + (s[1] + tau * d[1]).powi(2);
        assert!(
            (norm_sq.sqrt() - radius).abs() < 1e-10,
            "||s + tau*d|| = {}, expected {}, tau = {}",
            norm_sq.sqrt(),
            radius,
            tau
        );
        assert!(tau > 0.0, "tau should be positive, got {}", tau);
    }
}

//! Phase 6 regression tests: curvature filters, descent fallback,
//! piggyback finiteness, and trust-region NaN detection. (The scale-aware
//! singularity and compensated-summation cases live as unit tests beside
//! `linalg::lu_solve` and `convergence::norm`.)

use echidna_optim::objective::Objective;
use echidna_optim::result::TerminationReason;
use echidna_optim::{lbfgs, newton, trust_region, LbfgsConfig, NewtonConfig, TrustRegionConfig};

struct IllScaled;
impl Objective<f64> for IllScaled {
    fn dim(&self) -> usize {
        2
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        // f(x,y) = x^2 + 1e12 * y^2 — highly ill-scaled but convex.
        let f = x[0] * x[0] + 1e12 * x[1] * x[1];
        let g = vec![2.0 * x[0], 2.0e12 * x[1]];
        (f, g)
    }
}

#[test]
fn m32_m33_lbfgs_converges_on_ill_scaled_quadratic() {
    let mut obj = IllScaled;
    let mut cfg = LbfgsConfig::<f64>::default();
    cfg.convergence.max_iter = 200;
    cfg.convergence.grad_tol = 1e-6;
    let result = lbfgs(&mut obj, &[1.0, 1.0e-3], &cfg);
    assert!(
        matches!(
            result.termination,
            TerminationReason::GradientNorm
                | TerminationReason::StepSize
                | TerminationReason::FunctionChange
        ),
        "unexpected termination: {:?}",
        result.termination
    );
    assert!(result.gradient_norm < 1e-3);
}

// M34: Newton descent-direction fallback actually makes progress when the
// Newton step is uphill. Builds an objective where eval_hessian returns an
// indefinite Hessian causing `-H⁻¹·g` to point uphill (dot with grad > 0).
// Without the fallback, Armijo fails at iteration 0 → LineSearchFailed.
// With the fallback, Newton substitutes `-grad` (descent) and f_val drops.
struct UphillNewton;
impl Objective<f64> for UphillNewton {
    fn dim(&self) -> usize {
        2
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        // f(x, y) = x^2/2 + y — linear in y, so decreasing y monotonically
        // decreases f. Newton steepest-descent fallback uses -grad = [-x, -1]
        // which is always a descent direction for this f.
        let f = 0.5 * x[0] * x[0] + x[1];
        let g = vec![x[0], 1.0];
        (f, g)
    }
    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        // H = diag(1, -1) has eigenvalues {1, -1}: indefinite, non-singular.
        // At x=[0,0], grad=[0,1], d = -H⁻¹·g = -diag(1,-1)·[0,1] = [0, 1].
        // dot(grad, d) = 0*0 + 1*1 = 1 > 0 — NOT descent → fallback triggers.
        (f, g, vec![vec![1.0, 0.0], vec![0.0, -1.0]])
    }
}

#[test]
fn m34_newton_descent_fallback_makes_progress() {
    let mut obj = UphillNewton;
    let mut cfg = NewtonConfig::<f64>::default();
    cfg.convergence.max_iter = 20;
    let f_initial = 0.5 * 0.0_f64 * 0.0 + 0.0; // f(0,0) = 0.0
    let result = newton(&mut obj, &[0.0, 0.0], &cfg);
    // With fallback, y decreases, so f drops below the initial value. Without
    // fallback, Newton would hit LineSearchFailed before taking any step.
    assert_ne!(
        result.termination,
        TerminationReason::LineSearchFailed,
        "fallback must prevent LineSearchFailed at iter 0 (uphill Newton)"
    );
    assert!(
        result.value < f_initial,
        "fallback should have decreased f below {}, got {}",
        f_initial,
        result.value
    );
}

// M47: trust_region should return NumericalError when hvp produces NaN mid-
// iteration even if initial eval_grad is finite. This exercises the new
// M47 check at predicted/actual/rho time, rather than the pre-existing
// initial-gradient finite guard.
struct FiniteButNanHvp;
impl Objective<f64> for FiniteButNanHvp {
    fn dim(&self) -> usize {
        1
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        // f(x) = x^2/2 — finite everywhere.
        (0.5 * x[0] * x[0], vec![x[0]])
    }
    fn hvp(&mut self, x: &[f64], _v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Gradient is finite, but HVP injects NaN into the Hessian action.
        (vec![x[0]], vec![f64::NAN])
    }
}

#[test]
fn m47_trust_region_detects_nan_hvp_mid_iteration() {
    let mut obj = FiniteButNanHvp;
    let mut cfg = TrustRegionConfig::<f64>::default();
    cfg.convergence.max_iter = 50;
    let result = trust_region(&mut obj, &[1.0], &cfg);
    assert_eq!(
        result.termination,
        TerminationReason::NumericalError,
        "NaN HVP must yield NumericalError, not {:?}",
        result.termination
    );
}

// M47: also cover the case where eval_grad returns NaN — the pre-existing
// initial-gradient guard handles this, but we keep a regression for it.
struct NanFunc;
impl Objective<f64> for NanFunc {
    fn dim(&self) -> usize {
        1
    }
    fn eval_grad(&mut self, _x: &[f64]) -> (f64, Vec<f64>) {
        (f64::NAN, vec![f64::NAN])
    }
    fn eval_hessian(&mut self, _x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        (f64::NAN, vec![f64::NAN], vec![vec![1.0]])
    }
}

#[test]
fn m47_trust_region_detects_nan_grad_at_start() {
    let mut obj = NanFunc;
    let cfg = TrustRegionConfig::<f64>::default();
    let result = trust_region(&mut obj, &[1.0], &cfg);
    assert_eq!(result.termination, TerminationReason::NumericalError);
}

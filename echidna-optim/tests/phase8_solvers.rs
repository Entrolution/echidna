//! Phase 8 Commit 3 regressions — trust-region / solver polish.
//!
//! Covers L28 (min_radius floor triggers NumericalError) and L33 (relative
//! func_tol across trust_region, lbfgs, and newton). L29/L30/L31/L32 are
//! internal numerical stability improvements exercised by the existing
//! trust-region convergence tests; they don't need new discriminating
//! tests because the old code merely stalled or lost precision rather
//! than returning wrong answers.

use echidna_optim::objective::Objective;
use echidna_optim::result::TerminationReason;
use echidna_optim::{lbfgs, newton, trust_region, LbfgsConfig, NewtonConfig, TrustRegionConfig};

// ─────────────────────────────────────────────────────────────────────
// L28: trust-region radius collapse → NumericalError
// ─────────────────────────────────────────────────────────────────────

// `predicted <= 0` branch collapse. An objective whose HVP always reports
// huge negative curvature drives the CG subproblem to the trust-region
// boundary with `predicted <= 0`, triggering the shrink path.
struct PredictedNegative;
impl Objective<f64> for PredictedNegative {
    fn dim(&self) -> usize {
        1
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        (x[0] * x[0] + 1.0, vec![2.0 * x[0]])
    }
    fn hvp(&mut self, x: &[f64], _v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (vec![2.0 * x[0]], vec![-1e20])
    }
}

#[test]
fn l28_trust_region_collapse_via_predicted_negative() {
    let mut obj = PredictedNegative;
    let mut cfg = TrustRegionConfig::<f64>::default();
    cfg.convergence.max_iter = 1000;
    cfg.min_radius = 1.0;
    cfg.initial_radius = 2.0;
    let result = trust_region(&mut obj, &[1.0], &cfg);
    assert_eq!(
        result.termination,
        TerminationReason::NumericalError,
        "`predicted<=0` shrink must return NumericalError, got {:?}",
        result.termination
    );
}

// (The `rho < quarter` shrink-to-collapse path shares the same
// `NumericalError` return logic as the `predicted <= 0` path above;
// constructing a discriminating test for it is awkward because
// objectives that keep `rho` below quarter for enough iterations also
// tend to converge on gradient or step-size first. The primary L28 fix
// is verified here and further exercised by the existing trust-region
// Rosenbrock tests which never spuriously terminate on radius collapse.)

// ─────────────────────────────────────────────────────────────────────
// L33: func_tol scales with |f|
// ─────────────────────────────────────────────────────────────────────

// An objective where FunctionChange is the natural termination reason
// but absolute vs relative tol give very different trigger points.
// `f(x) = 1e10 + (x - 1)^10` has tiny gradient magnitude near x = 1,
// so the solver keeps making ever-smaller steps. The step-to-step |Δf|
// drops below `1e-6 * 1e10 = 1e4` (relative tol trips) much earlier
// than it drops below `1e-6` (absolute tol, which wouldn't trip until
// near-ULP precision on f).
struct FlatValleyLargeOffset;
impl Objective<f64> for FlatValleyLargeOffset {
    fn dim(&self) -> usize {
        1
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let d = x[0] - 1.0;
        let f = 1e10 + d.powi(10);
        let g = 10.0 * d.powi(9);
        (f, vec![g])
    }
    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let d = x[0] - 1.0;
        let h = 90.0 * d.powi(8);
        (f, g, vec![vec![h]])
    }
    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (_, g) = self.eval_grad(x);
        let d = x[0] - 1.0;
        let h = 90.0 * d.powi(8);
        (g, vec![h * v[0]])
    }
}

#[test]
fn l33_lbfgs_func_tol_is_relative_for_large_magnitude() {
    let mut obj = FlatValleyLargeOffset;
    let mut cfg = LbfgsConfig::<f64>::default();
    cfg.convergence.max_iter = 500;
    cfg.convergence.grad_tol = 1e-40;
    cfg.convergence.step_tol = 0.0;
    cfg.convergence.func_tol = 1e-6;
    let result = lbfgs(&mut obj, &[2.0], &cfg);
    // Relative tol fires once |Δf| < 1e-6 · 1e10 ≈ 1e4 in a few iterations.
    // Absolute tol would need |Δf| < 1e-6, which for this objective
    // requires near-ULP precision on `f`, far past `max_iter`.
    assert_eq!(
        result.termination,
        TerminationReason::FunctionChange,
        "relative func_tol must fire on large-magnitude objective, got {:?}",
        result.termination
    );
}

#[test]
fn l33_newton_func_tol_is_relative_for_large_magnitude() {
    let mut obj = FlatValleyLargeOffset;
    let mut cfg = NewtonConfig::<f64>::default();
    cfg.convergence.max_iter = 500;
    cfg.convergence.grad_tol = 1e-40;
    cfg.convergence.step_tol = 0.0;
    cfg.convergence.func_tol = 1e-6;
    let result = newton(&mut obj, &[2.0], &cfg);
    assert_eq!(
        result.termination,
        TerminationReason::FunctionChange,
        "relative func_tol must fire on large-magnitude objective, got {:?}",
        result.termination
    );
}

#[test]
fn l33_trust_region_func_tol_is_relative_for_large_magnitude() {
    let mut obj = FlatValleyLargeOffset;
    let mut cfg = TrustRegionConfig::<f64>::default();
    cfg.convergence.max_iter = 500;
    cfg.convergence.grad_tol = 1e-40;
    cfg.convergence.step_tol = 0.0;
    cfg.convergence.func_tol = 1e-6;
    let result = trust_region(&mut obj, &[2.0], &cfg);
    assert_eq!(
        result.termination,
        TerminationReason::FunctionChange,
        "relative func_tol must fire on large-magnitude objective, got {:?}",
        result.termination
    );
}

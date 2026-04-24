//! WS 4 — Solver diagnostics regressions.
//!
//! Pins the per-solver counters added to `OptimResult.diagnostics`. Each
//! test uses a deterministic short-iteration adversarial objective and
//! asserts EXACT counts where possible — threshold-style assertions
//! cannot catch double-counting bugs.

use echidna_optim::objective::Objective;
use echidna_optim::result::{
    LbfgsDiagnostics, NewtonDiagnostics, SolverDiagnostics, TrustRegionDiagnostics,
};
use echidna_optim::{
    lbfgs, newton, trust_region, LbfgsConfig, NewtonConfig, TerminationReason, TrustRegionConfig,
};

fn lbfgs_diag(d: &SolverDiagnostics) -> &LbfgsDiagnostics {
    match d {
        SolverDiagnostics::Lbfgs(l) => l,
        other => panic!("expected Lbfgs diagnostics, got {:?}", other),
    }
}

fn newton_diag(d: &SolverDiagnostics) -> &NewtonDiagnostics {
    match d {
        SolverDiagnostics::Newton(n) => n,
        other => panic!("expected Newton diagnostics, got {:?}", other),
    }
}

fn tr_diag(d: &SolverDiagnostics) -> &TrustRegionDiagnostics {
    match d {
        SolverDiagnostics::TrustRegion(t) => t,
        other => panic!("expected TrustRegion diagnostics, got {:?}", other),
    }
}

// ─────────────────────────────────────────────────────────────────────
// L-BFGS — Cauchy-Schwarz curvature filter pins curvature_rejected
// vs accepted asymmetrically.
// ─────────────────────────────────────────────────────────────────────

// `f(x) = 0` everywhere. The gradient is zero, so y_k = g_{k+1} - g_k = 0
// for every step — sy = 0 always, and the filter rejects every pair.
struct ZeroFunc;
impl Objective<f64> for ZeroFunc {
    fn dim(&self) -> usize {
        2
    }
    fn eval_grad(&mut self, _x: &[f64]) -> (f64, Vec<f64>) {
        (0.0, vec![0.0, 0.0])
    }
}

#[test]
fn lbfgs_zero_objective_terminates_on_initial_grad_norm_with_no_pairs() {
    // The initial grad_norm = 0 < grad_tol, so the solver returns at iter 0
    // before any pair is even constructed. Both counters must be zero.
    let mut obj = ZeroFunc;
    let cfg = LbfgsConfig::<f64>::default();
    let result = lbfgs(&mut obj, &[1.0, 1.0], &cfg);
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    let d = lbfgs_diag(&result.diagnostics);
    assert_eq!(d.pairs_accepted, 0);
    assert_eq!(d.pairs_curvature_rejected, 0);
    assert_eq!(d.gamma_clamp_hits, 0);
    assert_eq!(d.line_search_backtracks, 0);
    assert_eq!(d.pairs_evicted_by_memory, 0);
}

// L-BFGS on Rosenbrock should accept some curvature pairs and probably
// reject zero (well-conditioned region) — pin pairs_accepted > 0.
struct Rosenbrock;
impl Objective<f64> for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        (f, vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b])
    }
}

#[test]
fn lbfgs_rosenbrock_accepts_curvature_pairs() {
    let mut obj = Rosenbrock;
    let cfg = LbfgsConfig::<f64>::default();
    let result = lbfgs(&mut obj, &[0.0, 0.0], &cfg);
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    let d = lbfgs_diag(&result.diagnostics);
    // Rosenbrock has well-defined curvature — most pairs should pass the
    // CS filter. Pin a lower bound; the specific number drifts with
    // tolerances and we just want non-trivial accumulation.
    assert!(
        d.pairs_accepted >= 5,
        "expected several accepted pairs, got {:?}",
        d
    );
}

// Memory eviction: with `memory = 3` and a problem that converges in
// 6+ iterations accepting every pair, eviction count = (accepted - 3).
#[test]
fn lbfgs_memory_eviction_count_matches_accepted_minus_memory() {
    let mut obj = Rosenbrock;
    let cfg = LbfgsConfig::<f64> {
        memory: 3,
        convergence: echidna_optim::ConvergenceParams {
            max_iter: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[0.0, 0.0], &cfg);
    let d = lbfgs_diag(&result.diagnostics);
    // Eviction = max(0, accepted - memory). The memory-eviction branch
    // only runs when the buffer is full and a NEW pair is being pushed,
    // so the relation `evicted = max(0, accepted - memory)` holds
    // exactly.
    let expected_evicted = d.pairs_accepted.saturating_sub(cfg.memory);
    assert_eq!(
        d.pairs_evicted_by_memory, expected_evicted,
        "evicted ({}) != accepted ({}) - memory ({})",
        d.pairs_evicted_by_memory, d.pairs_accepted, cfg.memory
    );
}

// ─────────────────────────────────────────────────────────────────────
// L-BFGS — gamma clamp counter
// ─────────────────────────────────────────────────────────────────────

// Highly ill-scaled quadratic: f(x, y) = x² + 1e12·y². The L-BFGS gamma
// estimate sy/yy will land far outside [1e-3, 1e3] so the clamp fires
// on every iteration that has at least one pair in history.
struct IllScaled;
impl Objective<f64> for IllScaled {
    fn dim(&self) -> usize {
        2
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = x[0] * x[0] + 1e12 * x[1] * x[1];
        (f, vec![2.0 * x[0], 2.0e12 * x[1]])
    }
}

#[test]
fn lbfgs_gamma_clamp_hits_on_ill_scaled_quadratic() {
    let mut obj = IllScaled;
    let mut cfg = LbfgsConfig::<f64>::default();
    cfg.convergence.max_iter = 200;
    cfg.convergence.grad_tol = 1e-6;
    let result = lbfgs(&mut obj, &[1.0, 1.0e-3], &cfg);
    let d = lbfgs_diag(&result.diagnostics);
    assert!(
        d.gamma_clamp_hits >= 1,
        "expected gamma clamp on ill-scaled problem, got {:?}",
        d
    );
}

// ─────────────────────────────────────────────────────────────────────
// Newton — fallback_steps
// ─────────────────────────────────────────────────────────────────────

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
        // Singular Hessian — LU returns None every iter, fallback fires.
        (1.0, vec![1.0, 1.0], vec![vec![1.0, 1.0], vec![1.0, 1.0]])
    }
}

#[test]
fn newton_fallback_increments_on_singular_hessian() {
    let mut obj = SingularAtOrigin;
    let cfg = NewtonConfig::<f64>::default();
    let result = newton(&mut obj, &[2.0, 3.0], &cfg);
    let d = newton_diag(&result.diagnostics);
    // The pre-existing Phase 6 test pins that this terminates with
    // NumericalError or LineSearchFailed — meaning at least one
    // iteration was attempted and the fallback fired at least once.
    assert!(
        d.fallback_steps >= 1,
        "expected fallback on singular Hessian, got fallback_steps = {}",
        d.fallback_steps
    );
}

// Newton on a well-behaved problem must NOT use the fallback.
#[test]
fn newton_well_conditioned_does_not_fall_back() {
    struct Quadratic;
    impl Objective<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
            let f = x[0] * x[0] + x[1] * x[1];
            (f, vec![2.0 * x[0], 2.0 * x[1]])
        }
        fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
            let (f, g) = self.eval_grad(x);
            (f, g, vec![vec![2.0, 0.0], vec![0.0, 2.0]])
        }
    }
    let mut obj = Quadratic;
    let cfg = NewtonConfig::<f64>::default();
    let result = newton(&mut obj, &[3.0, -2.0], &cfg);
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    let d = newton_diag(&result.diagnostics);
    assert_eq!(
        d.fallback_steps, 0,
        "well-conditioned problem must never trigger Newton fallback"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Trust-region — radius shrink branches and CG iter count
// ─────────────────────────────────────────────────────────────────────

// HVP returns a huge negative value, so `predicted = -gs - 0.5·shs ≤ 0`
// every iteration → the `predicted <= 0` branch (radius_shrinks_bad_model)
// fires. The `rho < quarter` branch should NOT fire because we never
// reach it.
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
fn trust_region_predicted_negative_only_increments_bad_model_counter() {
    let mut obj = PredictedNegative;
    let mut cfg = TrustRegionConfig::<f64>::default();
    cfg.convergence.max_iter = 10;
    cfg.min_radius = 1.0;
    cfg.initial_radius = 2.0;
    let result = trust_region(&mut obj, &[1.0], &cfg);
    let d = tr_diag(&result.diagnostics);
    assert!(
        d.radius_shrinks_bad_model >= 1,
        "expected at least one bad-model shrink, got {:?}",
        d
    );
    // The two branches are mutually exclusive on each iteration; an
    // accidental double-count would put a value in the low-rho counter
    // even though the `continue` after bad-model skips the rho check.
    assert_eq!(
        d.radius_shrinks_low_rho, 0,
        "low-rho counter must stay at zero when only the bad-model \
         branch fires; got {}",
        d.radius_shrinks_low_rho
    );
}

#[test]
fn trust_region_rosenbrock_accumulates_cg_iters() {
    struct RosenbrockHvp;
    impl Objective<f64> for RosenbrockHvp {
        fn dim(&self) -> usize {
            2
        }
        fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            (
                a * a + 100.0 * b * b,
                vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
            )
        }
        fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
            let h01 = -400.0 * x[0];
            let h11 = 200.0;
            let (_, g) = self.eval_grad(x);
            (g, vec![h00 * v[0] + h01 * v[1], h01 * v[0] + h11 * v[1]])
        }
    }
    let mut obj = RosenbrockHvp;
    let mut cfg = TrustRegionConfig::<f64>::default();
    cfg.convergence.max_iter = 200;
    let result = trust_region(&mut obj, &[0.0, 0.0], &cfg);
    let d = tr_diag(&result.diagnostics);
    assert!(
        d.cg_inner_iters >= result.iterations,
        "cg_inner_iters ({}) should be at least the outer iter count ({})",
        d.cg_inner_iters,
        result.iterations
    );
}

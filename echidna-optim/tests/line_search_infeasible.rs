//! `backtracking_armijo` must reject trial points where the objective returns
//! a non-finite value. Prior to the fix, `f_new = -Inf` satisfied the Armijo
//! comparison vacuously (`-Inf <= anything` is true), causing the solver to
//! step past the feasible region toward `-Inf` and remain there.

use echidna_optim::line_search::{backtracking_armijo, ArmijoParams};
use echidna_optim::objective::Objective;

/// A 1-D objective that is finite on `[0, 1]` and `-Inf` outside. Simulates a
/// log-barrier or log-likelihood that blows up at the domain boundary.
struct BoundedDomain;

impl Objective<f64> for BoundedDomain {
    fn dim(&self) -> usize {
        1
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let xi = x[0];
        if (0.0..=1.0).contains(&xi) {
            // Simple quadratic: f(x) = (x - 0.5)², gradient 2(x-0.5)
            let f = (xi - 0.5) * (xi - 0.5);
            let g = 2.0 * (xi - 0.5);
            (f, vec![g])
        } else {
            // Off-domain: the user's model blows up.
            (f64::NEG_INFINITY, vec![f64::NAN])
        }
    }
}

#[test]
fn armijo_does_not_walk_off_to_neg_infinity() {
    let mut obj = BoundedDomain;

    // Start at x = 0.9, head toward x = 10 with d = 10 (far outside domain).
    // Pre-fix: Armijo accepts α=1 since f_new = -Inf <= anything, solver
    // reports success at x = 10.9 with f = -Inf. Post-fix: trial points with
    // f_new = -Inf are rejected, α backtracks until α < alpha_min, returning None.
    let x = vec![0.9_f64];
    let d = vec![10.0_f64];
    let f_x = (0.9_f64 - 0.5_f64).powi(2);
    let grad = vec![2.0 * (0.9 - 0.5)];

    let params = ArmijoParams {
        c: 1e-4,
        rho: 0.5,
        alpha_init: 1.0,
        alpha_min: 1e-16,
    };

    let result = backtracking_armijo(&mut obj, &x, &d, f_x, &grad, &params);
    // No finite α keeps us in [0, 1], so the search must fail cleanly.
    // The critical invariant: if we DID get a result, its value must be finite.
    match result {
        None => {
            // Correct: search failed rather than returning an -Inf step.
        }
        Some(r) => {
            assert!(
                r.value.is_finite(),
                "line search accepted a non-finite objective value: {}",
                r.value
            );
            assert!(
                r.gradient.iter().all(|g| g.is_finite()),
                "line search accepted a non-finite gradient"
            );
        }
    }
}

#[test]
fn armijo_still_accepts_feasible_step() {
    // Same objective, but head toward a point inside the domain.
    let mut obj = BoundedDomain;
    let x = vec![0.2_f64];
    let d = vec![0.3_f64]; // target = 0.5, still in [0, 1]
    let f_x = (0.2_f64 - 0.5_f64).powi(2);
    let grad = vec![2.0 * (0.2 - 0.5)];

    let result = backtracking_armijo(&mut obj, &x, &d, f_x, &grad, &ArmijoParams::default());
    let r = result.expect("feasible step should succeed");
    assert!(r.value.is_finite());
    assert!(r.value < f_x, "step should reduce objective");
}

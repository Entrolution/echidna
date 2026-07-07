//! A failed line search has still spent its objective evaluations, and the
//! solver's reported `func_evals` must include them. The discriminating
//! failure path is backtracking all the way to `alpha_min`: the other two
//! failure modes (bad params, non-descent direction) return before any
//! evaluation and cannot expose an undercount.

use echidna_optim::line_search::{backtracking_armijo_with_evals, ArmijoParams};
use echidna_optim::objective::Objective;
use echidna_optim::{lbfgs, newton, LbfgsConfig, NewtonConfig, TerminationReason};

/// Objective whose gradient claims unit descent but whose value JUMPS away
/// from the origin: every trial point sees `f_new = 2 > f_x = 1`, so the
/// Armijo condition can never hold and every search backtracks to
/// `alpha_min`, failing after a full complement of evaluations. (A merely
/// flat objective is not enough: for tiny `α` the Armijo right-hand side
/// `f_x + c·α·gᵀd` rounds to exactly `f_x` and `f_new <= f_x` succeeds
/// vacuously.)
struct RisesOffOrigin {
    calls: usize,
}

impl RisesOffOrigin {
    fn value_at(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            2.0
        }
    }
}

impl Objective<f64> for RisesOffOrigin {
    fn dim(&self) -> usize {
        1
    }
    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        self.calls += 1;
        (Self::value_at(x[0]), vec![-1.0])
    }
    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        self.calls += 1;
        (Self::value_at(x[0]), vec![-1.0], vec![vec![1.0]])
    }
}

#[test]
fn failed_search_reports_its_evaluations() {
    let mut obj = RisesOffOrigin { calls: 0 };
    let mut func_evals = 0usize;
    let result = backtracking_armijo_with_evals(
        &mut obj,
        &[0.0],
        &[1.0],
        1.0,
        &[-1.0],
        &ArmijoParams::default(),
        &mut func_evals,
    );
    assert!(result.is_none(), "Armijo can never be satisfied here");
    assert!(
        func_evals > 10,
        "exhausting alpha_min must take many evaluations, got {func_evals}"
    );
    assert_eq!(
        func_evals, obj.calls,
        "accumulator must match the objective's true call count"
    );
}

#[test]
fn lbfgs_func_evals_includes_failed_line_search() {
    let mut obj = RisesOffOrigin { calls: 0 };
    let result = lbfgs(&mut obj, &[0.0], &LbfgsConfig::default());
    assert_eq!(result.termination, TerminationReason::LineSearchFailed);
    assert_eq!(
        result.func_evals, obj.calls,
        "reported func_evals ({}) must equal the objective's true call count ({})",
        result.func_evals, obj.calls
    );
    assert!(
        result.func_evals > 10,
        "the failed search's evals must be included"
    );
}

#[test]
fn newton_func_evals_includes_failed_line_search() {
    let mut obj = RisesOffOrigin { calls: 0 };
    let result = newton(&mut obj, &[0.0], &NewtonConfig::default());
    assert_eq!(result.termination, TerminationReason::LineSearchFailed);
    assert_eq!(
        result.func_evals, obj.calls,
        "reported func_evals ({}) must equal the objective's true call count ({})",
        result.func_evals, obj.calls
    );
}

//! Semantic tests tying `src/bytecode_tape/optimize.rs` to the TLA+ specs under
//! `specs/tape_optimizer/`.
//!
//! The structural invariants (`InputPrefixInvariant`, `DAGOrderInvariant`,
//! `ValidRefsInvariant`, `OutputValidInvariant`, `InputsPreserved`,
//! `CSERemapMonotone`, `CSERemapIdempotent`, `DCEInputsReachable`,
//! `PostOptValid`) are enforced by `debug_assertions` in `optimize()`. The
//! `post_optimize_structural_asserts_hold_across_corpus` test is gated on
//! `cfg(debug_assertions)` so the structural coverage claim is honest:
//! the test is only present, and only passes meaningfully, in debug builds
//! (which is the default for `cargo test`). Under `cargo test --release`
//! the test is compiled out rather than silently providing false assurance.
//!
//! The semantic property (`IdempotencyProperty`: `optimize(optimize(t)) =
//! optimize(t)`) is tested explicitly here by observing tape length and
//! gradient values across two successive optimise passes — if the second pass
//! changed anything structurally or semantically, one of the two would drift.

#![cfg(feature = "bytecode")]

use echidna::{record, BReverse};
use num_traits::Float;

fn assert_close(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "length mismatch in {}", ctx);
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < tol,
            "{}: mismatch at [{}]: {} vs {}",
            ctx,
            i,
            a[i],
            b[i]
        );
    }
}

// A diverse corpus of functions that exercise different optimizer paths:
// CSE opportunities, constants, commutative normalization, non-commutative
// ops, unary ops, and deep nesting.
type RecFn = fn(&[BReverse<f64>]) -> BReverse<f64>;

fn f_cse_heavy(x: &[BReverse<f64>]) -> BReverse<f64> {
    // Repeated subexpressions: (x0 + x1), (x0 * x1).
    let a = x[0] + x[1];
    let b = x[0] + x[1];
    let c = x[0] * x[1];
    let d = x[0] * x[1];
    a * b + c * d
}

fn f_commutative_norm(x: &[BReverse<f64>]) -> BReverse<f64> {
    // (x0 + x1) and (x1 + x0) should canonicalize to one node.
    let a = x[0] + x[1];
    let b = x[1] + x[0];
    a * b
}

fn f_noncommutative(x: &[BReverse<f64>]) -> BReverse<f64> {
    // (x0 - x1) and (x1 - x0) must NOT be merged.
    let a = x[0] - x[1];
    let b = x[1] - x[0];
    a * b
}

fn f_unary_chain(x: &[BReverse<f64>]) -> BReverse<f64> {
    // Unary ops stacked, with shared subexpression at the base. Exercises
    // unary opcode paths (sin, exp) through the CSE/DCE pipeline.
    let base = x[0] * x[0] + x[1] * x[1];
    let a = base.sin() + x[0];
    let b = base.exp() + x[1];
    a * b
}

fn f_mixed_ops(x: &[BReverse<f64>]) -> BReverse<f64> {
    // Multiplication, addition, subtraction — no scalar literals.
    let a = x[0] * x[0];
    let b = x[1] * x[1];
    let c = x[0] * x[1];
    let d = a + b - c;
    d * d
}

fn f_deep_nesting(x: &[BReverse<f64>]) -> BReverse<f64> {
    // Chain of binary ops that reuse intermediates.
    let a = x[0] + x[1];
    let b = a * x[0];
    let c = b + a;
    let d = c * b;
    d + a
}

const CORPUS: &[(RecFn, &[f64], &str)] = &[
    (f_cse_heavy, &[1.5, -0.4], "cse_heavy"),
    (f_commutative_norm, &[0.3, 0.7], "commutative_norm"),
    (f_noncommutative, &[0.6, -1.1], "noncommutative"),
    (f_unary_chain, &[0.2, 0.8], "unary_chain"),
    (f_mixed_ops, &[1.2, 1.0], "mixed_ops"),
    (f_deep_nesting, &[-0.5, 0.4], "deep_nesting"),
];

// ----------------------------------------------------------------------------
// IdempotencyProperty — `optimize(optimize(t)) = optimize(t)`.
//
// Observed as: after the first optimise, (num_ops, gradient) must be stable
// under a second optimise.
// ----------------------------------------------------------------------------
#[test]
fn optimize_is_idempotent() {
    for &(f, x0, name) in CORPUS {
        let (mut tape, _) = record(f, x0);

        tape.optimize();
        let ops_after_first = tape.num_ops();
        let grad_after_first = tape.gradient(x0);

        tape.optimize();
        let ops_after_second = tape.num_ops();
        let grad_after_second = tape.gradient(x0);

        assert_eq!(
            ops_after_first, ops_after_second,
            "{}: optimize is not idempotent — ops changed {} -> {}",
            name, ops_after_first, ops_after_second
        );
        assert_close(
            &grad_after_first,
            &grad_after_second,
            1e-12,
            &format!("{}: gradient changed after second optimize", name),
        );
    }
}

// ----------------------------------------------------------------------------
// PostOptValid + ValidRefsInvariant + DAGOrderInvariant + OutputValidInvariant
// + InputsPreserved + CSERemapMonotone + CSERemapIdempotent + DCEInputsReachable
//
// All upheld by `debug_assertions` in `optimize()`. This test exercises
// `optimize` on the full corpus so the assertions fire under each shape.
// Gated on `cfg(debug_assertions)` because those assertions are stripped
// under `cargo test --release`; running this test in release mode would
// give a false-pass signal, so we compile it out instead.
// ----------------------------------------------------------------------------
#[cfg(debug_assertions)]
#[test]
fn post_optimize_structural_asserts_hold_across_corpus() {
    for &(f, x0, name) in CORPUS {
        let (mut tape, _) = record(f, x0);
        let num_inputs_before = tape.num_inputs();
        tape.optimize();
        // InputsPreserved (also checked by the debug assertion).
        assert_eq!(
            tape.num_inputs(),
            num_inputs_before,
            "{}: num_inputs changed across optimize ({} -> {})",
            name,
            num_inputs_before,
            tape.num_inputs()
        );
    }
}

// ----------------------------------------------------------------------------
// Optimise preserves semantic correctness: gradient after optimise matches
// gradient from a freshly recorded, unoptimised tape.
//
// Not a TLA+ invariant per se, but a necessary companion — the TLA+ specs
// verify structure, Rust tests verify that structure maps to correct values.
// ----------------------------------------------------------------------------
#[test]
fn optimize_preserves_gradient() {
    for &(f, x0, name) in CORPUS {
        let (mut raw, _) = record(f, x0);
        let grad_raw = raw.gradient(x0);

        let (mut opt, _) = record(f, x0);
        opt.optimize();
        let grad_opt = opt.gradient(x0);

        assert_close(
            &grad_raw,
            &grad_opt,
            1e-12,
            &format!("{}: gradient changed by optimize", name),
        );
    }
}

// ----------------------------------------------------------------------------
// CSE must actually reduce redundant subexpressions. This guards against a
// regression where the optimiser silently becomes a no-op (which would still
// satisfy idempotency and correctness).
// ----------------------------------------------------------------------------
#[test]
fn optimize_actually_reduces_cse_heavy_tape() {
    let (mut tape, _) = record(f_cse_heavy, &[1.5, -0.4]);
    let ops_before = tape.num_ops();
    tape.optimize();
    let ops_after = tape.num_ops();
    assert!(
        ops_after < ops_before,
        "cse_heavy: optimize did not reduce tape ({} -> {})",
        ops_before,
        ops_after
    );
}

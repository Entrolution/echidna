//! Phase 5 structural-assertion regression tests.
//!
//! Each fix here turns a silent-wrong behavior into a loud panic or
//! `Result::Err`. The tests verify the assertion fires on the problem
//! input shape, and (for the non-assert items) that the happy path
//! still works.

#![cfg(feature = "bytecode")]

use echidna::{BReverse, BytecodeTape};
use std::sync::Arc;

// A constant-derivative custom op used by the custom-op rejection tests below
// (`taylor_grad`, `ode_taylor_step`, `third_order_hvvp`). `jacobian_forward`
// itself now supports custom ops exactly via `CustomOp::eval_dual` — its
// positive coverage lives in `tests/custom_primitives.rs`.
struct Scale;
impl echidna::CustomOp<f64> for Scale {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        2.0 * a
    }
    fn partials(&self, _a: f64, _b: f64, _r: f64) -> (f64, f64) {
        (2.0, 0.0)
    }
}

// ── hessian / hvp reject multi-output tapes ─────────────────────────

fn rosenbrock_multi(x: &[BReverse<f64>]) -> Vec<BReverse<f64>> {
    let r0 = x[0] * x[0];
    let r1 = x[1] * x[1];
    vec![r0, r1]
}

#[test]
#[should_panic(expected = "scalar-output")]
fn hessian_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.hessian(&[1.0_f64, 2.0]);
}

#[test]
#[should_panic(expected = "scalar-output")]
fn hvp_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.hvp(&[1.0_f64, 2.0], &[1.0, 0.0]);
}

#[test]
#[should_panic(expected = "scalar-output")]
fn hessian_vec_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.hessian_vec::<2>(&[1.0_f64, 2.0]);
}

// Sanity: scalar-output tapes still work.
#[test]
fn hessian_on_scalar_output_still_works() {
    let (tape, _) = echidna::record(
        |x: &[BReverse<f64>]| x[0] * x[0] + x[1] * x[1],
        &[1.0_f64, 2.0],
    );
    let (_val, _grad, h) = tape.hessian(&[1.0_f64, 2.0]);
    assert_eq!(h.len(), 2);
    // H of x² + y² is diag(2, 2).
    assert!((h[0][0] - 2.0).abs() < 1e-12);
    assert!((h[1][1] - 2.0).abs() < 1e-12);
}

// ── M15: taylor_grad / ode_taylor_step reject custom ops ────────────

#[cfg(feature = "taylor")]
#[test]
#[should_panic(expected = "custom ops")]
fn taylor_grad_rejects_custom_ops() {
    let x = [1.0_f64];
    let mut tape = BytecodeTape::with_capacity(10);
    let handle = tape.register_custom(Arc::new(Scale));
    let idx = tape.new_input(x[0]);
    let input = BReverse::from_tape(x[0], idx);
    let output = {
        let _guard = echidna::bytecode_tape::BtapeGuard::new(&mut tape);
        input.custom_unary(handle, 2.0 * x[0])
    };
    tape.set_output(output.index());
    let _ = tape.taylor_grad::<3>(&x, &[1.0]);
}

#[cfg(feature = "taylor")]
#[test]
#[should_panic(expected = "custom ops")]
fn ode_taylor_step_rejects_custom_ops() {
    let x = [1.0_f64];
    let mut tape = BytecodeTape::with_capacity(10);
    let handle = tape.register_custom(Arc::new(Scale));
    let idx = tape.new_input(x[0]);
    let input = BReverse::from_tape(x[0], idx);
    let output = {
        let _guard = echidna::bytecode_tape::BtapeGuard::new(&mut tape);
        input.custom_unary(handle, 2.0 * x[0])
    };
    tape.set_output(output.index());
    tape.set_outputs(&[output.index()]);
    let _ = tape.ode_taylor_step::<3>(&x);
}

// ── M25: stde_gpu rejects multi-output tapes ─────────────────────────
//
// These tests would require a GPU backend to exercise; the check fires at
// runtime in `laplacian_gpu` when passed a multi-output tape. Covered
// structurally by the source-level check (tape.num_outputs != 1 → Err).

// ── third_order_hvvp: reject custom ops and multi-output tapes ───────
//
// Like `hessian_vec`, the nested-dual sweep linearizes custom ops, so 2nd/3rd
// order through them is only first-order-accurate; and it seeds the single
// `output_index`, so a multi-output tape silently gets one output's tensor.

#[test]
#[should_panic(expected = "custom ops")]
fn third_order_hvvp_rejects_custom_ops() {
    let x = [1.0_f64];
    let mut tape = BytecodeTape::with_capacity(10);
    let handle = tape.register_custom(Arc::new(Scale));
    let idx = tape.new_input(x[0]);
    let input = BReverse::from_tape(x[0], idx);
    let output = {
        let _guard = echidna::bytecode_tape::BtapeGuard::new(&mut tape);
        input.custom_unary(handle, 2.0 * x[0])
    };
    tape.set_output(output.index());
    let _ = tape.third_order_hvvp(&x, &[1.0], &[1.0]);
}

#[test]
#[should_panic(expected = "scalar-output")]
fn third_order_hvvp_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.third_order_hvvp(&[1.0_f64, 2.0], &[1.0, 0.0], &[0.0, 1.0]);
}

// ── sparse-Hessian family rejects multi-output tapes ─────────────────
//
// Mirrors the dense `hessian`/`hvp`/`hessian_vec` scalar-output guard.

#[test]
#[should_panic(expected = "scalar-output")]
fn sparse_hessian_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.sparse_hessian(&[1.0_f64, 2.0]);
}

#[cfg(feature = "parallel")]
#[test]
#[should_panic(expected = "scalar-output")]
fn sparse_hessian_par_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.sparse_hessian_par(&[1.0_f64, 2.0]);
}

#[cfg(feature = "parallel")]
#[test]
#[should_panic(expected = "scalar-output")]
fn hessian_par_rejects_multi_output_tape() {
    let (tape, _) = echidna::record_multi(rosenbrock_multi, &[1.0_f64, 2.0]);
    let _ = tape.hessian_par(&[1.0_f64, 2.0]);
}

// Sanity: scalar-output sparse Hessian still works.
#[test]
fn sparse_hessian_on_scalar_output_still_works() {
    let (tape, _) = echidna::record(
        |x: &[BReverse<f64>]| x[0] * x[0] + x[1] * x[1],
        &[1.0_f64, 2.0],
    );
    let (_val, _grad, _pattern, h) = tape.sparse_hessian(&[1.0_f64, 2.0]);
    // H of x² + y² is diag(2, 2); the two diagonal entries are present.
    assert!(h.iter().any(|&v| (v - 2.0).abs() < 1e-12));
}

// ── Thread-local active-pointer guards ──
//
// Both tape families share one reentrance/null-check mechanism; these pin
// its two panic paths per family (nothing else in the suite attempts a
// reentrant access or a guardless one).

#[test]
#[should_panic(expected = "No active tape")]
fn with_active_tape_without_guard_panics() {
    echidna::tape::with_active_tape::<f64, _>(|_| ());
}

#[test]
#[should_panic(expected = "No active bytecode tape")]
fn with_active_btape_without_guard_panics() {
    echidna::bytecode_tape::with_active_btape::<f64, _>(|_| ());
}

#[test]
#[should_panic(expected = "reentrant with_active_tape")]
fn reentrant_with_active_tape_panics() {
    use echidna::tape::{with_active_tape, Tape, TapeGuard};
    let mut tape: Tape<f64> = Tape::new();
    let _guard = TapeGuard::new(&mut tape);
    with_active_tape::<f64, _>(|_| {
        with_active_tape::<f64, _>(|_| ());
    });
}

#[test]
#[should_panic(expected = "reentrant with_active_btape")]
fn reentrant_with_active_btape_panics() {
    use echidna::bytecode_tape::{with_active_btape, BtapeGuard};
    let mut tape: BytecodeTape<f64> = BytecodeTape::new();
    let _guard = BtapeGuard::new(&mut tape);
    with_active_btape::<f64, _>(|_| {
        with_active_btape::<f64, _>(|_| ());
    });
}

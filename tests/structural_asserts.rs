//! Phase 5 structural-assertion regression tests.
//!
//! Each fix here turns a silent-wrong behavior into a loud panic or
//! `Result::Err`. The tests verify the assertion fires on the problem
//! input shape, and (for the non-assert items) that the happy path
//! still works.

#![cfg(feature = "bytecode")]

use echidna::{BReverse, BytecodeTape};
use std::sync::Arc;

// ── M14: jacobian_forward rejects custom ops ────────────────────────

struct Scale;
impl echidna::CustomOp<f64> for Scale {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        2.0 * a
    }
    fn partials(&self, _a: f64, _b: f64, _r: f64) -> (f64, f64) {
        (2.0, 0.0)
    }
}

#[test]
#[should_panic(expected = "custom ops")]
fn jacobian_forward_rejects_custom_ops() {
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
    let _ = tape.jacobian_forward(&x);
}

// ── M16: hessian / hvp reject multi-output tapes ────────────────────

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

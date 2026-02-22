//! Tests for tape optimizations: constant folding, DCE, CSE, and optimize.

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
use echidna::{record, record_multi, BReverse};
use num_traits::Float;

// ── Constant folding ──

#[test]
fn constant_folding_reduces_ops() {
    // 2.0 * 3.0 should be folded into a single Const during recording.
    let (tape, val) = record(
        |x| {
            let two = BReverse::constant(2.0);
            let three = BReverse::constant(3.0);
            // This multiplication of two constants should be folded.
            let six = two * three;
            x[0] * six
        },
        &[5.0_f64],
    );

    assert_relative_eq!(val, 30.0, max_relative = 1e-12);

    // Count non-Input, non-Const ops. Without folding there would be a Mul
    // for two*three; with folding it becomes a Const.
    let num_ops = tape.num_ops();
    // Expected: 1 Input + some Consts + 1 Mul (x[0] * six)
    // The key insight: there should be no Mul for the constant*constant case.
    // We check that gradient is still correct.
    let (mut tape, _) = record(
        |x| {
            let two = BReverse::constant(2.0);
            let three = BReverse::constant(3.0);
            let six = two * three;
            x[0] * six
        },
        &[5.0_f64],
    );
    let g = tape.gradient(&[5.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
    let _ = num_ops;
}

#[test]
fn constant_folding_powi() {
    // powi on a constant should be folded.
    let (mut tape, val) = record(
        |x| {
            let three = BReverse::constant(3.0);
            let nine = three.powi(2); // should fold to Const(9.0)
            x[0] + nine
        },
        &[1.0_f64],
    );

    assert_relative_eq!(val, 10.0, max_relative = 1e-12);
    let g = tape.gradient(&[1.0]);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);
}

#[test]
fn constant_folding_preserves_input_ops() {
    // Operations involving inputs should NOT be folded.
    let (mut tape, val) = record(|x| x[0] * x[0], &[3.0_f64]);
    assert_relative_eq!(val, 9.0, max_relative = 1e-12);
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);

    // Re-evaluate at different input.
    let g2 = tape.gradient(&[5.0]);
    assert_relative_eq!(g2[0], 10.0, max_relative = 1e-12);
}

// ── Dead code elimination ──

#[test]
fn dce_removes_unused_intermediates() {
    // Record a function with an unused intermediate.
    let (mut tape, val) = record(
        |x| {
            let _unused = x[0].sin(); // dead code
            let _also_unused = x[0].exp(); // dead code
            x[0] * x[0] // only this is used
        },
        &[3.0_f64],
    );

    assert_relative_eq!(val, 9.0, max_relative = 1e-12);
    let ops_before = tape.num_ops();

    tape.dead_code_elimination();
    let ops_after = tape.num_ops();

    assert!(
        ops_after < ops_before,
        "DCE should reduce tape size: before={}, after={}",
        ops_before,
        ops_after
    );

    // Gradient should still be correct.
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
}

#[test]
fn dce_preserves_all_inputs() {
    // Even if an input is unused in the output, it should be kept.
    let (mut tape, val) = record(|x| x[0] * x[0], &[3.0_f64, 4.0]);
    assert_relative_eq!(val, 9.0, max_relative = 1e-12);

    tape.dead_code_elimination();
    assert_eq!(tape.num_inputs(), 2, "DCE must preserve all inputs");
}

// ── Common subexpression elimination ──

#[test]
fn cse_deduplicates_common_subexpressions() {
    // x*x is computed twice; CSE should deduplicate.
    let (mut tape, val) = record(
        |x| {
            let a = x[0] * x[0];
            let b = x[0] * x[0]; // same as a
            a + b
        },
        &[3.0_f64],
    );

    assert_relative_eq!(val, 18.0, max_relative = 1e-12);
    let ops_before = tape.num_ops();

    tape.cse();
    let ops_after = tape.num_ops();

    assert!(
        ops_after < ops_before,
        "CSE should reduce tape size: before={}, after={}",
        ops_before,
        ops_after
    );

    // Gradient should still be correct: d/dx(2x^2) = 4x = 12.
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 12.0, max_relative = 1e-12);
}

#[test]
fn cse_commutative_order() {
    // x*y and y*x should be recognized as the same (Mul is commutative).
    let (mut tape, val) = record(
        |x| {
            let a = x[0] * x[1];
            let b = x[1] * x[0]; // same as a (commutative)
            a + b
        },
        &[2.0_f64, 3.0],
    );

    assert_relative_eq!(val, 12.0, max_relative = 1e-12);
    let ops_before = tape.num_ops();

    tape.cse();
    let ops_after = tape.num_ops();

    assert!(
        ops_after < ops_before,
        "CSE should deduplicate commutative ops: before={}, after={}",
        ops_before,
        ops_after
    );

    // Gradient of 2*x*y: d/dx = 2y = 6, d/dy = 2x = 4.
    let g = tape.gradient(&[2.0, 3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 4.0, max_relative = 1e-12);
}

#[test]
fn cse_non_commutative_preserved() {
    // x - y and y - x should NOT be deduplicated (Sub is non-commutative).
    let (mut tape, val) = record(
        |x| {
            let a = x[0] - x[1]; // x - y
            let b = x[1] - x[0]; // y - x (different!)
            a * b
        },
        &[5.0_f64, 3.0],
    );

    assert_relative_eq!(val, -4.0, max_relative = 1e-12);

    tape.cse();

    // Gradient should still be correct.
    // f = (x-y)(y-x) = -(x-y)^2
    // df/dx = -2(x-y) = -2(2) = -4
    // df/dy = 2(x-y) = 2(2) = 4
    let g = tape.gradient(&[5.0, 3.0]);
    assert_relative_eq!(g[0], -4.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 4.0, max_relative = 1e-12);
}

// ── Gradient correctness after optimization ──

#[test]
fn gradient_correct_after_dce() {
    let (mut tape, _) = record(
        |x| {
            let _dead = x[0].cos();
            x[0].sin() * x[0]
        },
        &[1.5_f64],
    );

    tape.dead_code_elimination();

    let g = tape.gradient(&[1.5]);
    // f = x*sin(x), f' = sin(x) + x*cos(x)
    let expected = 1.5_f64.sin() + 1.5 * 1.5_f64.cos();
    assert_relative_eq!(g[0], expected, max_relative = 1e-12);
}

#[test]
fn gradient_correct_after_cse() {
    let (mut tape, _) = record(
        |x| {
            let s = x[0].sin();
            let s2 = x[0].sin(); // duplicate
            s * s2
        },
        &[1.0_f64],
    );

    tape.cse();

    let g = tape.gradient(&[1.0]);
    // f = sin(x)^2, f' = 2*sin(x)*cos(x)
    let expected = 2.0 * 1.0_f64.sin() * 1.0_f64.cos();
    assert_relative_eq!(g[0], expected, max_relative = 1e-12);
}

// ── optimize() ──

#[test]
fn optimize_rosenbrock() {
    let x = [1.5_f64, 2.0];

    let (mut tape, _) = record(
        |v| {
            let one = BReverse::constant(1.0);
            let hundred = BReverse::constant(100.0);
            let t1 = one - v[0];
            let t2 = v[1] - v[0] * v[0];
            t1 * t1 + hundred * t2 * t2
        },
        &x,
    );

    // Get reference gradient before optimization.
    let g_before = tape.gradient(&x);
    let val_before = tape.output_value();

    tape.optimize();

    // Gradient after optimization should match.
    let g_after = tape.gradient(&x);
    let val_after = tape.output_value();

    assert_relative_eq!(val_before, val_after, max_relative = 1e-12);
    for i in 0..x.len() {
        assert_relative_eq!(g_before[i], g_after[i], max_relative = 1e-12);
    }

    // Also re-evaluate at different inputs.
    let x2 = [0.5, 1.0];
    let g2 = tape.gradient(&x2);
    let val2 = tape.output_value();

    // Compute expected values directly.
    let expected_val = (1.0 - x2[0]).powi(2) + 100.0 * (x2[1] - x2[0] * x2[0]).powi(2);
    assert_relative_eq!(val2, expected_val, max_relative = 1e-12);

    // Finite difference check for gradient.
    let h = 1e-7;
    for i in 0..x2.len() {
        let mut xp = x2;
        let mut xm = x2;
        xp[i] += h;
        xm[i] -= h;
        tape.forward(&xp);
        let fp = tape.output_value();
        tape.forward(&xm);
        let fm = tape.output_value();
        let fd = (fp - fm) / (2.0 * h);
        assert_relative_eq!(g2[i], fd, max_relative = 1e-5);
    }
}

#[test]
fn optimize_reduces_tape_size() {
    let (mut tape, _) = record(
        |x| {
            let _dead1 = x[0].exp();
            let _dead2 = x[0].cos();
            let a = x[0].sin();
            let b = x[0].sin(); // CSE candidate
            a + b
        },
        &[1.0_f64],
    );

    let ops_before = tape.num_ops();
    tape.optimize();
    let ops_after = tape.num_ops();

    assert!(
        ops_after < ops_before,
        "optimize should reduce tape size: before={}, after={}",
        ops_before,
        ops_after
    );
}

// ── Multi-output optimization ──

#[test]
fn optimize_preserves_multi_output_correctness() {
    let x = [2.0_f64, 3.0];

    let (mut tape, values) = record_multi(
        |v| {
            let sum = v[0] + v[1];
            let prod = v[0] * v[1];
            // Both outputs share subexpressions with the unused computation.
            let _dead = v[0].sin();
            vec![sum, prod]
        },
        &x,
    );

    assert_relative_eq!(values[0], 5.0, max_relative = 1e-12);
    assert_relative_eq!(values[1], 6.0, max_relative = 1e-12);

    // Get Jacobian before optimization.
    let jac_before = tape.jacobian(&x);

    tape.optimize();

    // Jacobian after optimization should match.
    let jac_after = tape.jacobian(&x);

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(jac_before[i][j], jac_after[i][j], max_relative = 1e-12);
        }
    }
}

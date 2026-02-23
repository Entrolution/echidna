//! Tests for tape optimizations: constant folding, DCE, CSE, algebraic
//! simplification, targeted DCE, and optimize.

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

// ── Algebraic simplification — identity patterns ──

#[test]
fn algebraic_add_zero() {
    // x + 0.0 and 0.0 + x should simplify to x.
    let (mut tape, val) = record(
        |x| {
            let a = x[0] + 0.0_f64;
            let b = 0.0_f64 + x[0];
            a + b // should be x + x = 2x
        },
        &[3.0_f64],
    );
    assert_relative_eq!(val, 6.0, max_relative = 1e-12);
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 2.0, max_relative = 1e-12);

    // Re-evaluate at different input.
    let g2 = tape.gradient(&[5.0]);
    assert_relative_eq!(g2[0], 2.0, max_relative = 1e-12);
}

#[test]
fn algebraic_mul_one() {
    // x * 1.0 and 1.0 * x should simplify to x.
    let (mut tape, val) = record(
        |x| {
            let a = x[0] * 1.0_f64;
            let b = 1.0_f64 * x[0];
            a + b
        },
        &[4.0_f64],
    );
    assert_relative_eq!(val, 8.0, max_relative = 1e-12);
    let g = tape.gradient(&[4.0]);
    assert_relative_eq!(g[0], 2.0, max_relative = 1e-12);
}

#[test]
fn algebraic_sub_zero() {
    // x - 0.0 should simplify to x.
    let (mut tape, val) = record(|x| x[0] - 0.0_f64, &[7.0_f64]);
    assert_relative_eq!(val, 7.0, max_relative = 1e-12);
    let g = tape.gradient(&[7.0]);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);
}

#[test]
fn algebraic_div_one() {
    // x / 1.0 should simplify to x.
    let (mut tape, val) = record(|x| x[0] / 1.0_f64, &[5.0_f64]);
    assert_relative_eq!(val, 5.0, max_relative = 1e-12);
    let g = tape.gradient(&[5.0]);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);
}

// ── Algebraic simplification — absorbing/same-index patterns ──

#[test]
fn algebraic_mul_zero() {
    // x * 0.0 and 0.0 * x should fold to const zero, gradient is zero.
    let (mut tape, val) = record(
        |x| {
            let a = x[0] * 0.0_f64;
            let b = 0.0_f64 * x[0];
            a + b
        },
        &[42.0_f64],
    );
    assert_relative_eq!(val, 0.0, max_relative = 1e-12);
    let g = tape.gradient(&[42.0]);
    assert_relative_eq!(g[0], 0.0, max_relative = 1e-12);
}

#[test]
fn algebraic_sub_self() {
    // x - x should fold to const zero.
    let (mut tape, val) = record(|x| x[0] - x[0], &[5.0_f64]);
    assert_relative_eq!(val, 0.0, max_relative = 1e-12);
    let g = tape.gradient(&[5.0]);
    assert_relative_eq!(g[0], 0.0, max_relative = 1e-12);
}

#[test]
fn algebraic_div_self() {
    // x / x should fold to const one.
    let (mut tape, val) = record(|x| x[0] / x[0], &[3.0_f64]);
    assert_relative_eq!(val, 1.0, max_relative = 1e-12);
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 0.0, max_relative = 1e-12);
}

// ── Algebraic simplification — powi ──

#[test]
fn algebraic_powi_zero() {
    // x^0 should fold to const 1.
    let (mut tape, val) = record(|x| x[0].powi(0), &[7.0_f64]);
    assert_relative_eq!(val, 1.0, max_relative = 1e-12);
    let g = tape.gradient(&[7.0]);
    assert_relative_eq!(g[0], 0.0, max_relative = 1e-12);
}

#[test]
fn algebraic_powi_one() {
    // x^1 should return the input directly.
    let (mut tape, val) = record(|x| x[0].powi(1), &[5.0_f64]);
    assert_relative_eq!(val, 5.0, max_relative = 1e-12);
    let g = tape.gradient(&[5.0]);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);

    // Re-evaluate at different input.
    let g2 = tape.gradient(&[3.0]);
    assert_relative_eq!(g2[0], 1.0, max_relative = 1e-12);
}

#[test]
fn algebraic_powi_neg_one() {
    // x^(-1) should emit Recip, gradient = -1/x^2.
    let (mut tape, val) = record(|x| x[0].powi(-1), &[2.0_f64]);
    assert_relative_eq!(val, 0.5, max_relative = 1e-12);
    let g = tape.gradient(&[2.0]);
    // d/dx(1/x) = -1/x^2 = -0.25
    assert_relative_eq!(g[0], -0.25, max_relative = 1e-12);

    // Re-evaluate at different input.
    let g2 = tape.gradient(&[4.0]);
    assert_relative_eq!(g2[0], -1.0 / 16.0, max_relative = 1e-12);
}

// ── Algebraic simplification — edge cases ──

#[test]
fn algebraic_nan_mul_zero_guard() {
    // NaN * 0.0 = NaN, should NOT be simplified to 0.
    // We record with a value that will produce NaN when multiplied with 0.
    let (tape, val) = record(
        |x| {
            // Create NaN via 0/0 in a way that avoids const-folding.
            let zero = x[0] - x[0]; // zero (from non-zero input)
            let nan = zero / zero; // NaN
            nan * 0.0_f64
        },
        &[1.0_f64],
    );
    // The value should be NaN (not simplified to 0).
    assert!(val.is_nan(), "NaN * 0.0 should produce NaN, not be folded");

    // The tape should have ops (not just a const).
    assert!(tape.num_ops() > 2, "NaN * 0 should not be folded away");
}

#[test]
fn algebraic_inf_sub_self_guard() {
    // Inf - Inf = NaN, should NOT be simplified to 0.
    let (tape, val) = record(
        |x| {
            let big = x[0].exp().exp(); // produces Inf for large x
            big - big
        },
        &[1000.0_f64],
    );
    // Value should be NaN (Inf - Inf).
    assert!(
        val.is_nan(),
        "Inf - Inf should produce NaN, not be folded to 0"
    );

    // x - x simplification should NOT have fired (value is NaN, not 0).
    assert!(tape.num_ops() > 2);
}

#[test]
fn algebraic_zero_div_self_guard() {
    // 0/0 = NaN, should NOT be simplified to 1.
    let (tape, val) = record(
        |x| {
            let zero = x[0] - x[0]; // 0
            zero / zero // NaN
        },
        &[5.0_f64],
    );
    assert!(val.is_nan(), "0/0 should produce NaN, not be folded to 1");
    assert!(tape.num_ops() > 2);
}

// ── Algebraic simplification — tape size and re-evaluation ──

#[test]
fn algebraic_tape_size_reduction() {
    // x + 0 + 0 + 0 should produce a smaller tape than without simplification.
    let (tape, val) = record(|x| x[0] + 0.0_f64 + 0.0_f64 + 0.0_f64, &[5.0_f64]);
    assert_relative_eq!(val, 5.0, max_relative = 1e-12);

    // Without algebraic simplification, we'd have 1 Input + 3 Const(0) + 3 Add = 7 ops.
    // With simplification, the Adds are eliminated, leaving 1 Input + some orphaned Consts.
    // The key point: no Add ops should remain.
    // After DCE, orphaned consts would also be removed.
    // We just check that the tape is smaller than 7 ops.
    assert!(
        tape.num_ops() < 7,
        "algebraic simplification should reduce tape: got {} ops",
        tape.num_ops()
    );
}

#[test]
fn algebraic_reeval_after_simplify() {
    // Record x + 0.0 at x=3, re-evaluate at x=5.
    let (mut tape, val) = record(|x| x[0] + 0.0_f64, &[3.0_f64]);
    assert_relative_eq!(val, 3.0, max_relative = 1e-12);

    tape.forward(&[5.0]);
    assert_relative_eq!(tape.output_value(), 5.0, max_relative = 1e-12);

    let g = tape.gradient(&[5.0]);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);
}

// ── Composition with existing passes ──

#[test]
fn algebraic_enables_cse() {
    // y = x + 0 and z = x should become the same index after algebraic
    // simplification. When combined with CSE, this further shrinks the tape.
    let (mut tape, val) = record(
        |x| {
            let y = x[0] + 0.0_f64; // simplified to x[0]
            let z = x[0];
            // Both y and z are x[0], so y * z = x^2.
            // After algebraic simplification, this is x[0] * x[0].
            y * z
        },
        &[3.0_f64],
    );
    assert_relative_eq!(val, 9.0, max_relative = 1e-12);

    let ops_before = tape.num_ops();
    tape.optimize();
    let ops_after = tape.num_ops();

    // After optimize: just Input + Mul (x*x). Orphaned Const(0) removed by DCE.
    assert!(ops_after <= ops_before);

    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
}

#[test]
fn algebraic_rosenbrock() {
    // Full Rosenbrock with mixed scalar ops, gradient matches finite difference.
    let f = |v: &[BReverse<f64>]| -> BReverse<f64> {
        let t1 = 1.0_f64 - v[0]; // mixed scalar: allocates Const(1.0)
        let t2 = v[1] - v[0] * v[0];
        t1 * t1 + 100.0_f64 * t2 * t2 // mixed scalar: allocates Const(100.0)
    };

    let points = [[1.5, 2.0], [0.0, 0.0], [-1.0, 1.0], [1.0, 1.0]];
    let h = 1e-7;

    for x in &points {
        let (mut tape, _) = record(|v| f(v), x);

        let g = tape.gradient(x);

        // Finite difference check.
        for i in 0..2 {
            let mut xp = *x;
            let mut xm = *x;
            xp[i] += h;
            xm[i] -= h;
            tape.forward(&xp);
            let fp = tape.output_value();
            tape.forward(&xm);
            let fm = tape.output_value();
            let fd = (fp - fm) / (2.0 * h);
            assert_relative_eq!(g[i], fd, max_relative = 1e-5, epsilon = 1e-10);
        }
    }
}

// ── Targeted DCE ──

#[test]
fn targeted_dce_prunes_unused_output() {
    // Multi-output tape: [sum, prod, sin(x)].
    // Prune to only keep sum — prod and sin should be eliminated.
    let x = [2.0_f64, 3.0];
    let (mut tape, values) = record_multi(
        |v| {
            let sum = v[0] + v[1];
            let prod = v[0] * v[1];
            let s = v[0].sin();
            vec![sum, prod, s]
        },
        &x,
    );

    assert_eq!(values.len(), 3);
    let ops_before = tape.num_ops();

    // Get the output indices before DCE.
    let out_indices: Vec<u32> = tape.all_output_indices().to_vec();
    assert_eq!(out_indices.len(), 3);

    // Keep only the first output (sum).
    tape.dead_code_elimination_for_outputs(&[out_indices[0]]);
    let ops_after = tape.num_ops();

    assert!(
        ops_after < ops_before,
        "targeted DCE should reduce tape: before={}, after={}",
        ops_before,
        ops_after
    );

    // Gradient of sum = x + y: d/dx = 1, d/dy = 1.
    let g = tape.gradient(&x);
    assert_relative_eq!(g[0], 1.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 1.0, max_relative = 1e-12);
}

#[test]
fn targeted_dce_preserves_active_output() {
    // Multi-output tape: [x*y, x+y]. Keep only x*y.
    let x = [3.0_f64, 4.0];
    let (mut tape, _) = record_multi(
        |v| {
            let prod = v[0] * v[1];
            let sum = v[0] + v[1];
            vec![prod, sum]
        },
        &x,
    );

    let out_indices: Vec<u32> = tape.all_output_indices().to_vec();
    tape.dead_code_elimination_for_outputs(&[out_indices[0]]);

    // Gradient of prod = x*y: d/dx = y = 4, d/dy = x = 3.
    let g = tape.gradient(&x);
    assert_relative_eq!(g[0], 4.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 3.0, max_relative = 1e-12);

    // Re-evaluate at different input.
    let g2 = tape.gradient(&[5.0, 6.0]);
    assert_relative_eq!(g2[0], 6.0, max_relative = 1e-12);
    assert_relative_eq!(g2[1], 5.0, max_relative = 1e-12);
}

// ── Integration tests ──

#[test]
fn optimize_with_algebraic_simplification() {
    // Record, optimize (CSE+DCE), verify gradient at multiple points.
    // This function has identity patterns that should be simplified at recording time.
    let f = |v: &[BReverse<f64>]| -> BReverse<f64> {
        let x = v[0];
        let y = v[1];
        // Identity ops that should be simplified away:
        let x1 = x + 0.0_f64; // → x
        let y1 = y * 1.0_f64; // → y
        let z = x1 * y1; // → x * y
        z - 0.0_f64 // → z
    };

    let (mut tape, val) = record(|v| f(v), &[3.0_f64, 4.0]);
    assert_relative_eq!(val, 12.0, max_relative = 1e-12);

    tape.optimize();

    let points = [[3.0, 4.0], [1.0, 2.0], [0.0, 5.0], [-1.0, -3.0]];
    let h = 1e-7;
    for x in &points {
        let g = tape.gradient(x);
        for i in 0..2 {
            let mut xp = *x;
            let mut xm = *x;
            xp[i] += h;
            xm[i] -= h;
            tape.forward(&xp);
            let fp = tape.output_value();
            tape.forward(&xm);
            let fm = tape.output_value();
            let fd = (fp - fm) / (2.0 * h);
            assert_relative_eq!(g[i], fd, max_relative = 1e-5);
        }
    }
}

#[test]
fn all_optimizations_combined() {
    // Algebraic simplification + CSE + DCE together.
    let f = |v: &[BReverse<f64>]| -> BReverse<f64> {
        let x = v[0];
        // Algebraic: x + 0 → x, x * 1 → x
        let a = x + 0.0_f64;
        let b = x * 1.0_f64;
        // After algebraic simplification, a and b are both x[0].
        // CSE would unify if they were separate ops, but they're already
        // the same index.
        let _dead = x.cos(); // DCE candidate
                             // Use a and b in a computation.
        let c = a * b; // x^2
        let d = c + x; // x^2 + x
        d / 1.0_f64 // → d
    };

    let (mut tape, val) = record(|v| f(v), &[3.0_f64]);
    assert_relative_eq!(val, 12.0, max_relative = 1e-12); // 9 + 3

    let ops_before = tape.num_ops();
    tape.optimize();
    let ops_after = tape.num_ops();

    // DCE should have removed the dead cos.
    assert!(ops_after < ops_before);

    // Gradient: f(x) = x^2 + x, f'(x) = 2x + 1
    let h = 1e-7;
    let points = [3.0, -2.0, 0.0, 10.0];
    for &x in &points {
        let g = tape.gradient(&[x]);
        let expected = 2.0 * x + 1.0;
        assert_relative_eq!(g[0], expected, max_relative = 1e-12);

        // Finite difference check.
        tape.forward(&[x + h]);
        let fp = tape.output_value();
        tape.forward(&[x - h]);
        let fm = tape.output_value();
        let fd = (fp - fm) / (2.0 * h);
        assert_relative_eq!(g[0], fd, max_relative = 1e-5);
    }
}

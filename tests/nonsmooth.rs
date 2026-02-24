//! Tests for nonsmooth extensions: branch tracking, kink detection, Clarke subdifferential.

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
use echidna::opcode::OpCode;
use echidna::record;
use num_traits::Float;

// ══════════════════════════════════════════════
//  R6a: Forward nonsmooth — kink detection
// ══════════════════════════════════════════════

#[test]
fn forward_nonsmooth_detects_abs_kink() {
    // f(x) = |x| at x = 0
    let (mut tape, _) = record(|x| x[0].abs(), &[0.0]);
    let info = tape.forward_nonsmooth(&[0.0]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, echidna::opcode::OpCode::Abs);
    assert_relative_eq!(info.kinks[0].switching_value, 0.0);
    assert_eq!(info.kinks[0].branch, 1); // x >= 0 → +1
}

#[test]
fn forward_nonsmooth_detects_abs_negative() {
    // f(x) = |x| at x = -3
    let (mut tape, _) = record(|x| x[0].abs(), &[-3.0]);
    let info = tape.forward_nonsmooth(&[-3.0]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].branch, -1); // x < 0 → -1
    assert_relative_eq!(info.kinks[0].switching_value, -3.0);
}

#[test]
fn forward_nonsmooth_detects_max_kink() {
    // f(x, y) = max(x, y) at x = y = 1
    let (mut tape, _) = record(|x| x[0].max(x[1]), &[1.0, 1.0]);
    let info = tape.forward_nonsmooth(&[1.0, 1.0]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, echidna::opcode::OpCode::Max);
    assert_relative_eq!(info.kinks[0].switching_value, 0.0); // a - b = 0
    assert_eq!(info.kinks[0].branch, 1); // a >= b → +1
}

#[test]
fn forward_nonsmooth_detects_min_kink() {
    // f(x, y) = min(x, y) at x = y = 2
    let (mut tape, _) = record(|x| x[0].min(x[1]), &[2.0, 2.0]);
    let info = tape.forward_nonsmooth(&[2.0, 2.0]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, echidna::opcode::OpCode::Min);
    assert_relative_eq!(info.kinks[0].switching_value, 0.0);
    assert_eq!(info.kinks[0].branch, 1); // a <= b → +1
}

#[test]
fn forward_nonsmooth_smooth_function() {
    // f(x) = x^2 + sin(x) — no nonsmooth ops at all
    let (mut tape, _) = record(|x| x[0] * x[0] + x[0].sin(), &[1.0]);
    let info = tape.forward_nonsmooth(&[1.0]);
    assert!(info.kinks.is_empty());
    assert!(info.is_smooth(1e-10));
}

#[test]
fn forward_nonsmooth_multiple_kinks() {
    // f(x, y) = |x| + max(x, y) + min(x, y)
    let (mut tape, _) = record(
        |x| x[0].abs() + x[0].max(x[1]) + x[0].min(x[1]),
        &[0.0, 0.0],
    );
    let info = tape.forward_nonsmooth(&[0.0, 0.0]);
    assert_eq!(info.kinks.len(), 3); // abs, max, min
}

#[test]
fn nonsmooth_signature_consistency() {
    // Same input → same signature
    let (mut tape, _) = record(|x| x[0].abs() + x[0].max(x[1]), &[1.0, 2.0]);
    let info1 = tape.forward_nonsmooth(&[1.0, 2.0]);
    let info2 = tape.forward_nonsmooth(&[1.0, 2.0]);
    assert_eq!(info1.signature(), info2.signature());
}

#[test]
fn active_kinks_tolerance() {
    // f(x) = |x| at x near 0
    let (mut tape, _) = record(|x| x[0].abs(), &[1e-6]);
    let info = tape.forward_nonsmooth(&[1e-6]);

    // With tight tolerance, no active kinks
    assert_eq!(info.active_kinks(1e-8).len(), 0);
    assert!(info.is_smooth(1e-8));

    // With loose tolerance, kink is active
    assert_eq!(info.active_kinks(1e-4).len(), 1);
    assert!(!info.is_smooth(1e-4));
}

// ══════════════════════════════════════════════
//  R6b: Jacobian limiting + Clarke subdifferential
// ══════════════════════════════════════════════

#[test]
fn jacobian_limiting_abs_positive() {
    // f(x) = |x|, forced sign +1 → derivative = +1
    let (mut tape, _) = record(|x| x[0].abs(), &[0.0]);

    // Need to find the tape index of the abs op
    let info = tape.forward_nonsmooth(&[0.0]);
    let abs_idx = info.kinks[0].tape_index;

    let jac = tape.jacobian_limiting(&[0.0], &[(abs_idx, 1)]);
    assert_relative_eq!(jac[0][0], 1.0, max_relative = 1e-12);
}

#[test]
fn jacobian_limiting_abs_negative() {
    // f(x) = |x|, forced sign -1 → derivative = -1
    let (mut tape, _) = record(|x| x[0].abs(), &[0.0]);

    let info = tape.forward_nonsmooth(&[0.0]);
    let abs_idx = info.kinks[0].tape_index;

    let jac = tape.jacobian_limiting(&[0.0], &[(abs_idx, -1)]);
    assert_relative_eq!(jac[0][0], -1.0, max_relative = 1e-12);
}

#[test]
fn jacobian_limiting_max_branches() {
    // f(x, y) = max(x, y)
    let (mut tape, _) = record(|x| x[0].max(x[1]), &[1.0, 1.0]);

    let info = tape.forward_nonsmooth(&[1.0, 1.0]);
    let max_idx = info.kinks[0].tape_index;

    // Force first branch (a wins): ∂max/∂x = 1, ∂max/∂y = 0
    let jac_a = tape.jacobian_limiting(&[1.0, 1.0], &[(max_idx, 1)]);
    assert_relative_eq!(jac_a[0][0], 1.0, max_relative = 1e-12);
    assert_relative_eq!(jac_a[0][1], 0.0, max_relative = 1e-12);

    // Force second branch (b wins): ∂max/∂x = 0, ∂max/∂y = 1
    let jac_b = tape.jacobian_limiting(&[1.0, 1.0], &[(max_idx, -1)]);
    assert_relative_eq!(jac_b[0][0], 0.0, max_relative = 1e-12);
    assert_relative_eq!(jac_b[0][1], 1.0, max_relative = 1e-12);
}

#[test]
fn jacobian_limiting_matches_standard_smooth() {
    // At a smooth point, jacobian_limiting with no forced signs should match jacobian
    let (mut tape, _) = record(|x| x[0] * x[0] + x[1], &[3.0, 4.0]);

    let jac_std = tape.jacobian(&[3.0, 4.0]);
    let jac_lim = tape.jacobian_limiting(&[3.0, 4.0], &[]);

    assert_relative_eq!(jac_std[0][0], jac_lim[0][0], max_relative = 1e-12);
    assert_relative_eq!(jac_std[0][1], jac_lim[0][1], max_relative = 1e-12);
}

#[test]
fn clarke_single_kink() {
    // f(x) = |x| at x = 0 → Clarke = {+1, -1}
    let (mut tape, _) = record(|x| x[0].abs(), &[0.0]);

    let (info, jacobians) = tape.clarke_jacobian(&[0.0], 1e-8, None).unwrap();
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(jacobians.len(), 2); // 2^1 combinations

    // One should be +1, the other -1
    let mut derivs: Vec<f64> = jacobians.iter().map(|j| j[0][0]).collect();
    derivs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_relative_eq!(derivs[0], -1.0, max_relative = 1e-12);
    assert_relative_eq!(derivs[1], 1.0, max_relative = 1e-12);
}

#[test]
fn clarke_two_kinks() {
    // f(x, y) = |x| + max(x, y) at x = 0, y = 0 → 2 active kinks → 4 Jacobians
    let (mut tape, _) = record(|x| x[0].abs() + x[0].max(x[1]), &[0.0, 0.0]);

    let (info, jacobians) = tape.clarke_jacobian(&[0.0, 0.0], 1e-8, None).unwrap();
    assert_eq!(info.active_kinks(1e-8).len(), 2);
    assert_eq!(jacobians.len(), 4); // 2^2 = 4 combinations
}

#[test]
fn clarke_smooth_single_jacobian() {
    // Smooth function at evaluation point → no active kinks → 1 Jacobian
    let (mut tape, _) = record(|x| x[0].abs(), &[5.0]);

    let (info, jacobians) = tape.clarke_jacobian(&[5.0], 1e-8, None).unwrap();
    assert!(info.is_smooth(1e-8));
    assert_eq!(jacobians.len(), 1); // 2^0 = 1

    // Should be the standard derivative: +1 (since x > 0)
    assert_relative_eq!(jacobians[0][0][0], 1.0, max_relative = 1e-12);
}

#[test]
fn clarke_too_many_kinks_error() {
    // Build a function with many abs kinks — set a low limit
    // f(x) = |x| + |x| + |x| (tape will have 3 abs ops)
    let (mut tape, _) = record(|x| x[0].abs() + x[0].abs() + x[0].abs(), &[0.0]);

    // With max_active_kinks = 2, should fail because we have 3 active kinks
    let result = tape.clarke_jacobian(&[0.0], 1e-8, Some(2));
    assert!(result.is_err());
    match result.unwrap_err() {
        echidna::ClarkeError::TooManyKinks { count, limit } => {
            assert_eq!(count, 3);
            assert_eq!(limit, 2);
        }
    }
}

// ══════════════════════════════════════════════
//  Extended nonsmooth ops (5.5): Signum, Floor, Ceil, Round, Trunc
// ══════════════════════════════════════════════

#[test]
fn forward_nonsmooth_detects_signum_kink() {
    // f(x) = signum(x) at x ≈ 0 → kink detected, switching_value ≈ 0
    let (mut tape, _) = record(|x| x[0].signum(), &[1e-12]);
    let info = tape.forward_nonsmooth(&[1e-12]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, OpCode::Signum);
    assert_relative_eq!(info.kinks[0].switching_value, 1e-12, max_relative = 1e-6);
    assert_eq!(info.kinks[0].branch, 1); // x >= 0 → +1
}

#[test]
fn forward_nonsmooth_detects_signum_away() {
    // f(x) = signum(x) at x = 5 → kink detected but not active at small tol
    let (mut tape, _) = record(|x| x[0].signum(), &[5.0]);
    let info = tape.forward_nonsmooth(&[5.0]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, OpCode::Signum);
    assert_relative_eq!(info.kinks[0].switching_value, 5.0);
    // Not active at tol = 0.1
    assert_eq!(info.active_kinks(0.1).len(), 0);
}

#[test]
fn forward_nonsmooth_detects_floor_kink() {
    // f(x) = floor(x) at x ≈ 2.0 → kink detected, switching_value ≈ 0
    let (mut tape, _) = record(|x| x[0].floor(), &[2.0001]);
    let info = tape.forward_nonsmooth(&[2.0001]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, OpCode::Floor);
    // switching_value = 2.0001 - round(2.0001) = 2.0001 - 2.0 = 0.0001
    assert_relative_eq!(info.kinks[0].switching_value, 0.0001, max_relative = 1e-6);
    // Active at tol = 0.01
    assert_eq!(info.active_kinks(0.01).len(), 1);
}

#[test]
fn forward_nonsmooth_detects_floor_away() {
    // f(x) = floor(x) at x = 2.7 → kink detected but not active
    let (mut tape, _) = record(|x| x[0].floor(), &[2.7]);
    let info = tape.forward_nonsmooth(&[2.7]);
    assert_eq!(info.kinks.len(), 1);
    // switching_value = 2.7 - round(2.7) = 2.7 - 3.0 = -0.3
    assert_relative_eq!(info.kinks[0].switching_value, -0.3, max_relative = 1e-6);
    // Not active at tol = 0.1
    assert_eq!(info.active_kinks(0.1).len(), 0);
}

#[test]
fn forward_nonsmooth_detects_ceil_kink() {
    // f(x) = ceil(x) at x ≈ 3.0 → kink detected
    let (mut tape, _) = record(|x| x[0].ceil(), &[2.9999]);
    let info = tape.forward_nonsmooth(&[2.9999]);
    assert_eq!(info.kinks.len(), 1);
    assert_eq!(info.kinks[0].opcode, OpCode::Ceil);
    // switching_value = 2.9999 - round(2.9999) = 2.9999 - 3.0 = -0.0001
    assert_relative_eq!(info.kinks[0].switching_value, -0.0001, max_relative = 1e-3);
    assert_eq!(info.active_kinks(0.01).len(), 1);
}

#[test]
fn forward_nonsmooth_detects_round_trunc() {
    // round(x) and trunc(x) near integers → kinks detected
    let (mut tape, _) = record(|x| x[0].round() + x[0].trunc(), &[4.001]);
    let info = tape.forward_nonsmooth(&[4.001]);
    assert_eq!(info.kinks.len(), 2);
    assert_eq!(info.kinks[0].opcode, OpCode::Round);
    assert_eq!(info.kinks[1].opcode, OpCode::Trunc);
    // Both near integer → active
    assert_eq!(info.active_kinks(0.01).len(), 2);
}

#[test]
fn clarke_filters_trivial_kinks() {
    // f(x) = |x| + signum(x) near x = 0
    // Both abs and signum kinks are active, but Clarke should only enumerate
    // the abs kink (2 Jacobians), not the signum kink (which would give 4).
    let (mut tape, _) = record(|x| x[0].abs() + x[0].signum(), &[0.0]);

    let (info, jacobians) = tape.clarke_jacobian(&[0.0], 1e-8, None).unwrap();
    // info.kinks has both abs and signum entries
    assert_eq!(info.kinks.len(), 2);
    // But active kinks with nontrivial subdifferential → only abs
    assert_eq!(jacobians.len(), 2); // 2^1 = 2 (only abs), not 2^2 = 4
}

#[test]
fn forced_partials_step_functions_zero() {
    // forced_reverse_partials for step functions always returns (0, 0)
    use echidna::opcode::forced_reverse_partials;
    let ops = [
        OpCode::Signum,
        OpCode::Floor,
        OpCode::Ceil,
        OpCode::Round,
        OpCode::Trunc,
    ];
    for op in ops {
        let (da, db) = forced_reverse_partials(op, 1.5_f64, 0.0, 1.0, 1);
        assert_eq!(da, 0.0, "expected zero da for {:?} with sign +1", op);
        assert_eq!(db, 0.0, "expected zero db for {:?} with sign +1", op);

        let (da, db) = forced_reverse_partials(op, 1.5, 0.0, 1.0, -1);
        assert_eq!(da, 0.0, "expected zero da for {:?} with sign -1", op);
        assert_eq!(db, 0.0, "expected zero db for {:?} with sign -1", op);
    }
}

#[test]
fn signum_records_to_tape() {
    // Signed::signum() on BReverse now records OpCode::Signum (not a constant).
    // We verify by checking the tape contains a Signum opcode.
    use num_traits::Signed;

    let (tape, val) = record(
        |x| {
            // Call signum through the Signed trait (takes &self)
            Signed::signum(&x[0])
        },
        &[3.0],
    );
    assert_eq!(val, 1.0);
    assert!(
        tape.opcodes_slice().contains(&OpCode::Signum),
        "expected OpCode::Signum in tape, got: {:?}",
        tape.opcodes_slice()
    );
}

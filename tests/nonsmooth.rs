//! Tests for nonsmooth extensions: branch tracking, kink detection, Clarke subdifferential.

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
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

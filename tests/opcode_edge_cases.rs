#![cfg(feature = "bytecode")]

use echidna::opcode::{
    eval_forward, forced_reverse_partials, powi_exp_encode, reverse_partials, OpCode,
};

const TOL: f64 = 1e-10;

// ══════════════════════════════════════════════
//  eval_forward: NaN propagation
// ══════════════════════════════════════════════

#[test]
fn nan_propagates_through_add() {
    assert!(eval_forward(OpCode::Add, f64::NAN, 1.0).is_nan());
    assert!(eval_forward(OpCode::Add, 1.0_f64, f64::NAN).is_nan());
}

#[test]
fn nan_propagates_through_mul() {
    assert!(eval_forward(OpCode::Mul, f64::NAN, 1.0).is_nan());
    assert!(eval_forward(OpCode::Mul, 1.0_f64, f64::NAN).is_nan());
}

#[test]
fn nan_propagates_through_unary() {
    assert!(eval_forward(OpCode::Neg, f64::NAN, 0.0).is_nan());
    assert!(eval_forward(OpCode::Exp, f64::NAN, 0.0).is_nan());
    assert!(eval_forward(OpCode::Sin, f64::NAN, 0.0).is_nan());
    assert!(eval_forward(OpCode::Sqrt, f64::NAN, 0.0).is_nan());
    assert!(eval_forward(OpCode::Abs, f64::NAN, 0.0).is_nan());
}

#[test]
fn nan_propagates_through_div() {
    assert!(eval_forward(OpCode::Div, f64::NAN, 1.0).is_nan());
    assert!(eval_forward(OpCode::Div, 1.0_f64, f64::NAN).is_nan());
}

// ══════════════════════════════════════════════
//  eval_forward: Infinity
// ══════════════════════════════════════════════

#[test]
fn exp_of_large_is_inf() {
    assert!(eval_forward(OpCode::Exp, 1000.0_f64, 0.0).is_infinite());
}

#[test]
fn ln_of_zero_is_neg_inf() {
    let r = eval_forward(OpCode::Ln, 0.0_f64, 0.0);
    assert!(r.is_infinite() && r < 0.0);
}

#[test]
fn div_by_zero_is_inf() {
    let r = eval_forward(OpCode::Div, 1.0_f64, 0.0);
    assert!(r.is_infinite());
}

#[test]
fn recip_of_zero_is_inf() {
    let r = eval_forward(OpCode::Recip, 0.0_f64, 0.0);
    assert!(r.is_infinite());
}

#[test]
fn sqrt_of_negative_is_nan() {
    assert!(eval_forward(OpCode::Sqrt, -1.0_f64, 0.0).is_nan());
}

#[test]
fn ln_of_negative_is_nan() {
    assert!(eval_forward(OpCode::Ln, -1.0_f64, 0.0).is_nan());
}

// ══════════════════════════════════════════════
//  eval_forward: boundary values
// ══════════════════════════════════════════════

#[test]
fn max_min_basic() {
    assert_eq!(eval_forward(OpCode::Max, 3.0_f64, 5.0), 5.0);
    assert_eq!(eval_forward(OpCode::Max, 5.0_f64, 3.0), 5.0);
    assert_eq!(eval_forward(OpCode::Min, 3.0_f64, 5.0), 3.0);
    assert_eq!(eval_forward(OpCode::Min, 5.0_f64, 3.0), 3.0);
}

#[test]
fn max_min_equal() {
    assert_eq!(eval_forward(OpCode::Max, 4.0_f64, 4.0), 4.0);
    assert_eq!(eval_forward(OpCode::Min, 4.0_f64, 4.0), 4.0);
}

#[test]
fn abs_of_zero() {
    assert_eq!(eval_forward(OpCode::Abs, 0.0_f64, 0.0), 0.0);
    assert_eq!(eval_forward(OpCode::Abs, -0.0_f64, 0.0), 0.0);
}

#[test]
fn abs_positive_negative() {
    assert_eq!(eval_forward(OpCode::Abs, 3.5_f64, 0.0), 3.5);
    assert_eq!(eval_forward(OpCode::Abs, -3.5_f64, 0.0), 3.5);
}

// ══════════════════════════════════════════════
//  eval_forward: Powi with encoded exponent
// ══════════════════════════════════════════════

#[test]
fn powi_zero_exponent() {
    let b = powi_exp_encode(0) as f64;
    assert!((eval_forward(OpCode::Powi, 5.0_f64, b) - 1.0).abs() < TOL);
}

#[test]
fn powi_positive_exponent() {
    let b = powi_exp_encode(3) as f64;
    assert!((eval_forward(OpCode::Powi, 2.0_f64, b) - 8.0).abs() < TOL);
}

#[test]
fn powi_negative_exponent() {
    let b = powi_exp_encode(-2) as f64;
    assert!((eval_forward(OpCode::Powi, 2.0_f64, b) - 0.25).abs() < TOL);
}

// ══════════════════════════════════════════════
//  eval_forward: rounding ops
// ══════════════════════════════════════════════

#[test]
fn floor_ceil_round_trunc_fract() {
    assert_eq!(eval_forward(OpCode::Floor, 2.7_f64, 0.0), 2.0);
    assert_eq!(eval_forward(OpCode::Ceil, 2.3_f64, 0.0), 3.0);
    assert_eq!(eval_forward(OpCode::Round, 2.5_f64, 0.0), 3.0);
    assert_eq!(eval_forward(OpCode::Trunc, -2.7_f64, 0.0), -2.0);
    assert!((eval_forward(OpCode::Fract, 2.7_f64, 0.0) - 0.7).abs() < TOL);
}

// ══════════════════════════════════════════════
//  reverse_partials: basic correctness
// ══════════════════════════════════════════════

#[test]
fn reverse_add() {
    let (da, db) = reverse_partials(OpCode::Add, 2.0_f64, 3.0, 5.0);
    assert!((da - 1.0).abs() < TOL);
    assert!((db - 1.0).abs() < TOL);
}

#[test]
fn reverse_sub() {
    let (da, db) = reverse_partials(OpCode::Sub, 5.0_f64, 3.0, 2.0);
    assert!((da - 1.0).abs() < TOL);
    assert!((db - (-1.0)).abs() < TOL);
}

#[test]
fn reverse_mul() {
    let (da, db) = reverse_partials(OpCode::Mul, 3.0_f64, 4.0, 12.0);
    assert!((da - 4.0).abs() < TOL); // ∂(a*b)/∂a = b
    assert!((db - 3.0).abs() < TOL); // ∂(a*b)/∂b = a
}

#[test]
fn reverse_div() {
    let (da, db) = reverse_partials(OpCode::Div, 6.0_f64, 3.0, 2.0);
    assert!((da - 1.0 / 3.0).abs() < TOL);
    assert!((db - (-6.0 / 9.0)).abs() < TOL);
}

#[test]
fn reverse_exp() {
    let a = 1.5_f64;
    let r = a.exp();
    let (da, _) = reverse_partials(OpCode::Exp, a, 0.0, r);
    assert!((da - r).abs() < TOL);
}

#[test]
fn reverse_ln() {
    let a = 2.0_f64;
    let r = a.ln();
    let (da, _) = reverse_partials(OpCode::Ln, a, 0.0, r);
    assert!((da - 0.5).abs() < TOL);
}

#[test]
fn reverse_sin() {
    let a = 0.7_f64;
    let r = a.sin();
    let (da, _) = reverse_partials(OpCode::Sin, a, 0.0, r);
    assert!((da - a.cos()).abs() < TOL);
}

#[test]
fn reverse_cos() {
    let a = 0.7_f64;
    let r = a.cos();
    let (da, _) = reverse_partials(OpCode::Cos, a, 0.0, r);
    assert!((da - (-a.sin())).abs() < TOL);
}

#[test]
fn reverse_sqrt() {
    let a = 4.0_f64;
    let r = a.sqrt();
    let (da, _) = reverse_partials(OpCode::Sqrt, a, 0.0, r);
    assert!((da - 0.25).abs() < TOL);
}

#[test]
fn reverse_neg() {
    let (da, db) = reverse_partials(OpCode::Neg, 5.0_f64, 0.0, -5.0);
    assert!((da - (-1.0)).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn reverse_recip() {
    let a = 2.0_f64;
    let (da, _) = reverse_partials(OpCode::Recip, a, 0.0, 0.5);
    assert!((da - (-0.25)).abs() < TOL);
}

#[test]
fn reverse_powi() {
    let a = 2.0_f64;
    let b = powi_exp_encode(3) as f64;
    let r = 8.0;
    let (da, _) = reverse_partials(OpCode::Powi, a, b, r);
    assert!((da - 12.0).abs() < TOL);
}

// ══════════════════════════════════════════════
//  reverse_partials: edge cases
// ══════════════════════════════════════════════

#[test]
fn reverse_hypot() {
    let (da, db) = reverse_partials(OpCode::Hypot, 3.0_f64, 4.0, 5.0);
    assert!((da - 0.6).abs() < TOL);
    assert!((db - 0.8).abs() < TOL);
}

#[test]
fn reverse_atan2() {
    let a = 1.0_f64;
    let b = 1.0_f64;
    let r = a.atan2(b);
    let (da, db) = reverse_partials(OpCode::Atan2, a, b, r);
    assert!((da - 0.5).abs() < TOL);
    assert!((db - (-0.5)).abs() < TOL);
}

#[test]
fn reverse_max_first_wins() {
    let (da, db) = reverse_partials(OpCode::Max, 5.0_f64, 3.0, 5.0);
    assert!((da - 1.0).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn reverse_max_second_wins() {
    let (da, db) = reverse_partials(OpCode::Max, 3.0_f64, 5.0, 5.0);
    assert!(da.abs() < TOL);
    assert!((db - 1.0).abs() < TOL);
}

#[test]
fn reverse_min_first_wins() {
    let (da, db) = reverse_partials(OpCode::Min, 3.0_f64, 5.0, 3.0);
    assert!((da - 1.0).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn reverse_abs_positive() {
    let (da, _) = reverse_partials(OpCode::Abs, 3.0_f64, 0.0, 3.0);
    assert!((da - 1.0).abs() < TOL);
}

#[test]
fn reverse_abs_negative() {
    let (da, _) = reverse_partials(OpCode::Abs, -3.0_f64, 0.0, 3.0);
    assert!((da - (-1.0)).abs() < TOL);
}

#[test]
fn reverse_rounding_ops_have_zero_derivatives() {
    for op in [
        OpCode::Signum,
        OpCode::Floor,
        OpCode::Ceil,
        OpCode::Round,
        OpCode::Trunc,
    ] {
        let (da, db) = reverse_partials(op, 2.7_f64, 0.0, 0.0);
        assert!(da.abs() < TOL, "{:?} should have zero derivative", op);
        assert!(db.abs() < TOL, "{:?} should have zero derivative", op);
    }
}

#[test]
fn reverse_fract_has_unit_derivative() {
    let (da, _) = reverse_partials(OpCode::Fract, 2.7_f64, 0.0, 0.7);
    assert!((da - 1.0).abs() < TOL);
}

// ══════════════════════════════════════════════
//  reverse_partials: NaN in derivatives
// ══════════════════════════════════════════════

#[test]
fn reverse_div_by_zero_gives_inf() {
    let (da, _) = reverse_partials(OpCode::Div, 1.0_f64, 0.0, f64::INFINITY);
    assert!(da.is_infinite());
}

#[test]
fn reverse_sqrt_at_zero_gives_inf() {
    let (da, _) = reverse_partials(OpCode::Sqrt, 0.0_f64, 0.0, 0.0);
    assert!(da.is_infinite());
}

#[test]
fn reverse_ln_at_zero_gives_inf() {
    let (da, _) = reverse_partials(OpCode::Ln, 0.0_f64, 0.0, f64::NEG_INFINITY);
    assert!(da.is_infinite());
}

// ══════════════════════════════════════════════
//  forced_reverse_partials
// ══════════════════════════════════════════════

#[test]
fn forced_abs_positive_sign() {
    let (da, db) = forced_reverse_partials(OpCode::Abs, -3.0_f64, 0.0, 3.0, 1);
    assert!((da - 1.0).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn forced_abs_negative_sign() {
    let (da, db) = forced_reverse_partials(OpCode::Abs, 3.0_f64, 0.0, 3.0, -1);
    assert!((da - (-1.0)).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn forced_max_first_branch() {
    let (da, db) = forced_reverse_partials(OpCode::Max, 1.0_f64, 5.0, 5.0, 1);
    assert!((da - 1.0).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn forced_max_second_branch() {
    let (da, db) = forced_reverse_partials(OpCode::Max, 5.0_f64, 1.0, 5.0, -1);
    assert!(da.abs() < TOL);
    assert!((db - 1.0).abs() < TOL);
}

#[test]
fn forced_min_first_branch() {
    let (da, db) = forced_reverse_partials(OpCode::Min, 5.0_f64, 1.0, 1.0, 1);
    assert!((da - 1.0).abs() < TOL);
    assert!(db.abs() < TOL);
}

#[test]
fn forced_min_second_branch() {
    let (da, db) = forced_reverse_partials(OpCode::Min, 1.0_f64, 5.0, 1.0, -1);
    assert!(da.abs() < TOL);
    assert!((db - 1.0).abs() < TOL);
}

#[test]
fn forced_non_nonsmooth_delegates() {
    let a = 3.0_f64;
    let b = 4.0;
    let r = a * b;
    let (da1, db1) = reverse_partials(OpCode::Mul, a, b, r);
    let (da2, db2) = forced_reverse_partials(OpCode::Mul, a, b, r, 1);
    assert!((da1 - da2).abs() < TOL);
    assert!((db1 - db2).abs() < TOL);
}

// ══════════════════════════════════════════════
//  powi_exp_encode round-trip
// ══════════════════════════════════════════════

#[test]
fn powi_encode_positive() {
    assert_eq!(powi_exp_encode(5), 5);
}

#[test]
fn powi_encode_zero() {
    assert_eq!(powi_exp_encode(0), 0);
}

#[test]
fn powi_encode_negative() {
    assert_eq!(powi_exp_encode(-1), u32::MAX);
}

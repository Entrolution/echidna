//! Tests for Laurent<F, K> — singularity analysis via Laurent series arithmetic.

#![cfg(feature = "laurent")]

use approx::assert_relative_eq;
use echidna::Laurent;
use num_traits::Float;

type L4 = Laurent<f64, 4>;

// ══════════════════════════════════════════════
//  Construction and basic properties
// ══════════════════════════════════════════════

#[test]
fn constant_and_variable() {
    let c: L4 = Laurent::constant(3.0);
    assert_relative_eq!(c.value(), 3.0);
    assert_eq!(c.pole_order(), 0);
    assert!(!c.has_pole());

    let v: L4 = Laurent::variable(2.0);
    assert_relative_eq!(v.value(), 2.0);
    assert_eq!(v.pole_order(), 0);
    assert_relative_eq!(v.coeff(0), 2.0);
    assert_relative_eq!(v.coeff(1), 1.0);
}

// ══════════════════════════════════════════════
//  Regular arithmetic (should match Taylor)
// ══════════════════════════════════════════════

#[test]
fn add_sub_regular() {
    let a: L4 = Laurent::variable(1.0); // 1 + t
    let b: L4 = Laurent::constant(2.0); // 2
    let sum = a + b;
    assert_relative_eq!(sum.value(), 3.0);
    assert_relative_eq!(sum.coeff(1), 1.0); // t coefficient preserved

    let diff = a - b;
    assert_relative_eq!(diff.value(), -1.0);
    assert_relative_eq!(diff.coeff(1), 1.0);
}

#[test]
fn mul_regular() {
    let a: L4 = Laurent::variable(2.0); // 2 + t
    let b: L4 = Laurent::variable(3.0); // 3 + t
    let prod = a * b;
    // (2 + t)(3 + t) = 6 + 5t + t²
    assert_relative_eq!(prod.value(), 6.0);
    assert_relative_eq!(prod.coeff(1), 5.0);
    assert_relative_eq!(prod.coeff(2), 1.0);
}

// ══════════════════════════════════════════════
//  Pole creation and arithmetic
// ══════════════════════════════════════════════

#[test]
fn div_creates_pole() {
    // 1 / t → pole_order = -1
    let one: L4 = Laurent::constant(1.0);
    let t: L4 = Laurent::variable(0.0); // 0 + t (zero at origin, pole_order becomes 1 after normalize)
    let result = one / t;
    assert_eq!(result.pole_order(), -1);
    assert_relative_eq!(result.leading_coefficient(), 1.0);
    assert!(result.has_pole());
    assert!(result.value().is_infinite());
}

#[test]
fn pole_cancellation() {
    // t * (1/t) = 1
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let product = t * inv_t;
    assert_eq!(product.pole_order(), 0);
    assert_relative_eq!(product.value(), 1.0, max_relative = 1e-12);
}

#[test]
fn pole_arithmetic() {
    // (1/t²) + (1/t) — different pole orders must align
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;    // pole_order = -1
    let inv_t2 = inv_t * inv_t;                 // pole_order = -2

    let sum = inv_t2 + inv_t;
    // 1/t² + 1/t → pole_order = -2, coeffs[0] = 1 (t^{-2}), coeffs[1] = 1 (t^{-1})
    assert_eq!(sum.pole_order(), -2);
    assert_relative_eq!(sum.coeff(-2), 1.0);
    assert_relative_eq!(sum.coeff(-1), 1.0);
}

#[test]
fn residue_extraction() {
    // Build a Laurent series with a specific residue at t^{-1}
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(3.0) / t; // 3/t → residue = 3
    assert_relative_eq!(inv_t.residue(), 3.0);

    // 1/t² has residue 0 (no t^{-1} term)
    let inv_t2 = Laurent::constant(1.0) / (t * t);
    assert_relative_eq!(inv_t2.residue(), 0.0);
}

// ══════════════════════════════════════════════
//  Transcendentals with poles
// ══════════════════════════════════════════════

#[test]
fn sqrt_even_pole() {
    // sqrt(1/t²) = 1/t (when pole_order is -2, even → ok)
    let t: L4 = Laurent::variable(0.0);
    let inv_t2 = Laurent::constant(1.0) / (t * t);
    let result = inv_t2.sqrt();
    assert_eq!(result.pole_order(), -1);
    assert_relative_eq!(result.leading_coefficient(), 1.0, max_relative = 1e-10);
}

#[test]
fn exp_of_pole_is_nan() {
    // exp(1/t) → essential singularity → NaN
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let result = inv_t.exp();
    assert!(result.value().is_nan());
}

#[test]
fn ln_of_zero_is_nan() {
    // ln(t) where t→0 → logarithmic singularity → NaN
    let t: L4 = Laurent::variable(0.0);
    let result = t.ln();
    assert!(result.value().is_nan());
}

#[test]
fn powi_with_poles() {
    // (1/t)^3 → pole_order = -3
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let result = inv_t.powi(3);
    assert_eq!(result.pole_order(), -3);
    assert_relative_eq!(result.leading_coefficient(), 1.0, max_relative = 1e-10);
}

// ══════════════════════════════════════════════
//  Regular matches Taylor
// ══════════════════════════════════════════════

#[test]
fn regular_matches_taylor() {
    use echidna::Taylor;

    // For pole_order=0, Laurent arithmetic should match Taylor exactly
    type T4 = Taylor<f64, 4>;

    let lt: L4 = Laurent::variable(2.0);
    let tt: T4 = Taylor::variable(2.0);

    // exp
    let le = lt.exp();
    let te = tt.exp();
    for k in 0..4 {
        assert_relative_eq!(le.coeff(k as i32), te.coeff(k), max_relative = 1e-12);
    }

    // sin
    let ls = lt.sin();
    let ts = tt.sin();
    for k in 0..4 {
        assert_relative_eq!(ls.coeff(k as i32), ts.coeff(k), max_relative = 1e-12);
    }
}

// ══════════════════════════════════════════════
//  Tape integration
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
#[test]
fn tape_forward_tangent() {
    use echidna::record;

    // f(x) = x² + sin(x)
    let (tape, _) = record(|x| x[0] * x[0] + x[0].sin(), &[1.0]);

    let x: L4 = Laurent::variable(1.0);
    let mut buf = Vec::new();
    tape.forward_tangent(&[x], &mut buf);
    let result = buf[tape.output_index()];

    // value should be 1 + sin(1)
    assert_relative_eq!(result.value(), 1.0 + 1.0_f64.sin(), max_relative = 1e-12);
    assert_eq!(result.pole_order(), 0);
}

// ══════════════════════════════════════════════
//  Singularity detection
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
#[test]
fn singularity_detection_reciprocal() {
    use echidna::record;

    // f(x) = 1/x — has a pole at x = 0
    let (tape, _) = record(|x| x[0].recip(), &[1.0]);

    // Evaluate at x = t (variable with value 0) to detect the singularity
    let x: L4 = Laurent::variable(0.0);
    let mut buf = Vec::new();
    tape.forward_tangent(&[x], &mut buf);
    let result = buf[tape.output_index()];

    // Should detect a pole: pole_order < 0
    assert!(result.has_pole());
    assert_eq!(result.pole_order(), -1);
}

// ══════════════════════════════════════════════
//  Normalization
// ══════════════════════════════════════════════

#[test]
fn normalization_strips_zeros() {
    // Create a Laurent with leading zeros: [0, 0, 1, 0] at pole_order = -2
    // After normalization: [1, 0] at pole_order = 0
    let l: L4 = Laurent::new([0.0, 0.0, 1.0, 0.0], -2);
    assert_eq!(l.pole_order(), 0);
    assert_relative_eq!(l.leading_coefficient(), 1.0);
    assert_relative_eq!(l.value(), 1.0);
}

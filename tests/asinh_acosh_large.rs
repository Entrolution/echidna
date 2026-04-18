//! Regression tests for `asinh` / `acosh` derivatives at large arguments.
//!
//! The pre-fix derivative `1/sqrt(x²±1)` overflows for `|x| > sqrt(f64::MAX)`,
//! silently zeroing the gradient. The `|x|>1e8` switch to `|1/x|/sqrt(1±1/x²)`
//! keeps the result representable for any `x` in the function's domain.

use echidna::{Dual, DualVec};
use num_traits::Float;

/// `asinh(x)` at `x = 1e200`: derivative ≈ `1/|x| = 1e-200`, representable.
#[test]
fn dual_asinh_large_x() {
    let x = Dual::new(1e200_f64, 1.0);
    let y = x.asinh();
    assert!(y.re.is_finite());
    assert!(y.eps.is_finite());
    assert!(y.eps > 0.0, "asinh is monotone increasing; derivative > 0");
    let expected = 1e-200_f64;
    let rel_err = (y.eps - expected).abs() / expected;
    assert!(rel_err < 1e-10, "eps = {}, expected ≈ {}", y.eps, expected);
}

/// `asinh(x)` at `x = -1e200`: asinh is odd, asinh' is even. derivative ≈ 1e-200.
#[test]
fn dual_asinh_large_negative_x() {
    let x = Dual::new(-1e200_f64, 1.0);
    let y = x.asinh();
    assert!(y.eps.is_finite());
    let expected = 1e-200_f64;
    let rel_err = (y.eps - expected).abs() / expected;
    assert!(rel_err < 1e-10, "eps = {}, expected ≈ {}", y.eps, expected);
}

/// `asinh(x)` at small x (below threshold) — behavior must be unchanged.
#[test]
fn dual_asinh_small_x_unchanged() {
    let x = Dual::new(2.0_f64, 1.0);
    let y = x.asinh();
    // d/dx asinh(2) = 1/sqrt(5)
    let expected = 1.0_f64 / 5.0_f64.sqrt();
    assert!((y.eps - expected).abs() < 1e-12);
}

/// `acosh(x)` at `x = 1e200`: derivative ≈ 1e-200.
#[test]
fn dual_acosh_large_x() {
    let x = Dual::new(1e200_f64, 1.0);
    let y = x.acosh();
    assert!(y.eps.is_finite());
    let expected = 1e-200_f64;
    let rel_err = (y.eps - expected).abs() / expected;
    assert!(rel_err < 1e-10, "eps = {}, expected ≈ {}", y.eps, expected);
}

/// `acosh(x)` at small x (in-domain) — behavior unchanged.
#[test]
fn dual_acosh_small_x_unchanged() {
    let x = Dual::new(2.0_f64, 1.0);
    let y = x.acosh();
    // d/dx acosh(2) = 1/sqrt(3)
    let expected = 1.0_f64 / 3.0_f64.sqrt();
    assert!((y.eps - expected).abs() < 1e-12);
}

/// `acosh(1)` boundary: derivative diverges to +∞ (vertical tangent).
/// Both formulas must produce the same +∞ result.
#[test]
fn dual_acosh_at_one_diverges() {
    let x = Dual::new(1.0_f64, 1.0);
    let y = x.acosh();
    assert_eq!(y.re, 0.0);
    assert!(y.eps.is_infinite() && y.eps > 0.0);
}

/// DualVec variant.
#[test]
fn dual_vec_asinh_large() {
    let x: DualVec<f64, 2> = DualVec {
        re: 1e200,
        eps: [1.0, 0.0],
    };
    let y = x.asinh();
    assert!(y.eps[0].is_finite());
    let expected = 1e-200_f64;
    assert!(((y.eps[0] - expected) / expected).abs() < 1e-10);
    assert_eq!(y.eps[1], 0.0);
}

/// Reverse-mode variant.
#[test]
fn reverse_asinh_large_x() {
    let g = echidna::grad(|x: &[echidna::Reverse<f64>]| x[0].asinh(), &[1e200_f64]);
    assert!(g[0].is_finite());
    let expected = 1e-200_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
}

#[test]
fn reverse_acosh_large_x() {
    let g = echidna::grad(|x: &[echidna::Reverse<f64>]| x[0].acosh(), &[1e200_f64]);
    assert!(g[0].is_finite());
    let expected = 1e-200_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
}

/// BReverse (bytecode tape) exercises `OpCode::Asinh` reverse partial.
#[cfg(feature = "bytecode")]
#[test]
fn breverse_asinh_large_x() {
    use echidna::BReverse;
    let (mut tape, _) = echidna::record(|x: &[BReverse<f64>]| x[0].asinh(), &[1e200_f64]);
    let g = tape.gradient(&[1e200_f64]);
    assert!(g[0].is_finite());
    let expected = 1e-200_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
}

#[cfg(feature = "bytecode")]
#[test]
fn breverse_acosh_large_x() {
    use echidna::BReverse;
    let (mut tape, _) = echidna::record(|x: &[BReverse<f64>]| x[0].acosh(), &[1e200_f64]);
    let g = tape.gradient(&[1e200_f64]);
    assert!(g[0].is_finite());
    let expected = 1e-200_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
}

//! Regression tests for `atan2` partials at extreme magnitudes.
//!
//! The pre-fix implementation computed the partials via `h*h`, which
//! overflowed for `|h| > sqrt(f64::MAX) ≈ 1.3e154` and underflowed for
//! `|h| < sqrt(f64::MIN_POSITIVE)`. The factored `(y/h)/h` form is
//! representable for any `h > 0`.

use echidna::{Dual, DualVec};
use num_traits::Float;

/// `atan2(1e200, 1e200)`: `h = sqrt(2)·1e200`, naive `h*h` overflows to +∞.
/// True partial: `b/(a²+b²) = 5e-201`.
#[test]
fn dual_atan2_large_magnitudes_finite_tangent() {
    let y = Dual::new(1e200_f64, 1.0);
    let x = Dual::constant(1e200_f64);
    let result = y.atan2(x);
    assert!(result.re.is_finite());
    assert!(result.eps.is_finite(), "tangent overflowed: {}", result.eps);
    assert!(result.eps != 0.0, "tangent underflowed to zero");
    // Expected: x/(x²+y²) = 1e200/(2·1e400) = 5e-201
    let expected = 5e-201_f64;
    let rel_err = (result.eps - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-10,
        "eps = {}, expected ≈ {}",
        result.eps,
        expected
    );
}

/// `atan2(1e-200, 1e-200)`: `h = sqrt(2)·1e-200`, naive `h*h` underflows to 0,
/// the old zero-guard fired and returned `(0, 0)`. True partial: `5e199`.
#[test]
fn dual_atan2_tiny_magnitudes_finite_tangent() {
    let y = Dual::new(1e-200_f64, 1.0);
    let x = Dual::constant(1e-200_f64);
    let result = y.atan2(x);
    assert!(result.re.is_finite());
    assert!(result.eps.is_finite());
    assert!(result.eps != 0.0, "tangent should be huge, not zero");
    // Expected: 1e-200/(2·1e-400) = 5e199
    let expected = 5e199_f64;
    let rel_err = (result.eps - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-10,
        "eps = {}, expected ≈ {}",
        result.eps,
        expected
    );
}

/// Mixed: very large y, small x. `atan2 → π/2`, `d/dy ≈ 0`, `d/dx ≈ -1/y`.
#[test]
fn dual_atan2_mixed_magnitudes() {
    let y = Dual::new(1e200_f64, 1.0);
    let x = Dual::constant(1.0_f64);
    let result = y.atan2(x);
    // d/dy atan2(y,x) = x/(x²+y²) ≈ 1/y² for y>>x = 1e-400 (underflows in f64 — 0 is fine)
    // Actual tangent = x/(x²+y²) * 1.0 — underflow to 0 is mathematically acceptable here
    assert!(result.re.is_finite());
    assert!(result.eps.is_finite());
}

/// Sanity: ordinary inputs match the pre-fix code's output.
#[test]
fn dual_atan2_ordinary_inputs_unchanged() {
    let y = Dual::new(3.0_f64, 1.0);
    let x = Dual::constant(4.0_f64);
    let result = y.atan2(x);
    // atan2(3, 4) = atan(0.75)
    let expected_primal = 3.0_f64.atan2(4.0);
    let expected_eps = 4.0_f64 / (9.0 + 16.0); // x/(x²+y²) = 4/25 = 0.16
    assert!((result.re - expected_primal).abs() < 1e-12);
    assert!((result.eps - expected_eps).abs() < 1e-12);
}

/// Origin: mathematical singularity. Convention: return (atan2(0,0) = 0, eps = 0).
#[test]
fn dual_atan2_origin_returns_zero_eps() {
    let y = Dual::new(0.0_f64, 1.0);
    let x = Dual::constant(0.0_f64);
    let result = y.atan2(x);
    assert_eq!(result.eps, 0.0, "at origin, eps should be 0 by convention");
}

/// NaN input: NaN primal, any eps is acceptable but must not panic.
#[test]
fn dual_atan2_nan_primal() {
    let y = Dual::new(f64::NAN, 1.0);
    let x = Dual::constant(1.0_f64);
    let result = y.atan2(x);
    assert!(result.re.is_nan() || !result.re.is_finite());
}

/// DualVec variant of the large-magnitude case.
#[test]
fn dual_vec_atan2_large_magnitudes() {
    let y: DualVec<f64, 2> = DualVec {
        re: 1e200,
        eps: [1.0, 0.0],
    };
    let x: DualVec<f64, 2> = DualVec {
        re: 1e200,
        eps: [0.0, 1.0],
    };
    let r = y.atan2(x);
    assert!(r.eps[0].is_finite());
    assert!(r.eps[1].is_finite());
    assert!(r.eps[0] != 0.0);
    assert!(r.eps[1] != 0.0);
    // d/dy = x/(x²+y²) = 1e200 / 2e400 = 5e-201
    // d/dx = -y/(x²+y²) = -5e-201
    let expected = 5e-201_f64;
    assert!(((r.eps[0] - expected) / expected).abs() < 1e-10);
    assert!(((-r.eps[1] - expected) / expected).abs() < 1e-10);
}

/// Reverse-mode variant. The `h*h` bug was also present in `Reverse::atan2`.
#[test]
fn reverse_atan2_large_magnitudes() {
    let g = echidna::grad(
        |x: &[echidna::Reverse<f64>]| x[0].atan2(x[1]),
        &[1e200_f64, 1e200_f64],
    );
    assert!(g[0].is_finite());
    assert!(g[1].is_finite());
    assert!(g[0] != 0.0, "dx gradient underflowed to zero");
    assert!(g[1] != 0.0, "dy gradient underflowed to zero");
    // d/dy atan2(y,x) = x/(x²+y²), d/dx = -y/(x²+y²) — both magnitude 5e-201.
    let expected = 5e-201_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
    assert!(((-g[1] - expected) / expected).abs() < 1e-10);
}

/// BReverse (bytecode tape) routes atan2 through `OpCode::Atan2` reverse partial,
/// which had the same bug. This test exercises the opcode path.
#[cfg(feature = "bytecode")]
#[test]
fn breverse_atan2_large_magnitudes() {
    use echidna::BReverse;
    let (mut tape, _) = echidna::record(
        |x: &[BReverse<f64>]| x[0].atan2(x[1]),
        &[1e200_f64, 1e200_f64],
    );
    let g = tape.gradient(&[1e200_f64, 1e200_f64]);
    assert!(g[0].is_finite());
    assert!(g[1].is_finite());
    assert!(g[0] != 0.0);
    assert!(g[1] != 0.0);
    let expected = 5e-201_f64;
    assert!(((g[0] - expected) / expected).abs() < 1e-10);
    assert!(((-g[1] - expected) / expected).abs() < 1e-10);
}

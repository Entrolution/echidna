//! Regression tests for `powf` with a negative base and a constant integer
//! exponent.
//!
//! `f(x) = x^n` for integer `n` is well-defined and has a well-defined
//! derivative `n·x^(n-1)` at negative `x`. The pre-fix implementation of
//! `powf` always went through `exp(b · ln(a))`, which produces a NaN
//! tangent (because `ln(a)` is NaN for `a < 0`) even when `b` is an
//! integer constant whose contribution to `eps` is algebraically zero.

use echidna::{Dual, DualVec};
use num_traits::Float;

// ── Dual ────────────────────────────────────────────────────────────

#[test]
fn dual_powf_negative_base_cubed() {
    let x = Dual::new(-2.0_f64, 1.0);
    let y = x.powf(Dual::constant(3.0));
    // (-2)^3 = -8; d/dx x^3 |_{x=-2} = 3x² = 12
    assert!((y.re - (-8.0)).abs() < 1e-12, "re = {}", y.re);
    assert!((y.eps - 12.0).abs() < 1e-12, "eps = {}", y.eps);
}

#[test]
fn dual_powf_negative_base_squared() {
    let x = Dual::new(-3.0_f64, 1.0);
    let y = x.powf(Dual::constant(2.0));
    // (-3)^2 = 9; d/dx x² |_{x=-3} = -6
    assert!((y.re - 9.0).abs() < 1e-12);
    assert!((y.eps - (-6.0)).abs() < 1e-12);
}

#[test]
fn dual_powf_negative_base_negative_exponent() {
    let x = Dual::new(-2.0_f64, 1.0);
    let y = x.powf(Dual::constant(-3.0));
    // (-2)^(-3) = -1/8; d/dx x^(-3) = -3x^(-4) = -3/16 at x=-2
    assert!((y.re - (-0.125)).abs() < 1e-12);
    assert!((y.eps - (-3.0 / 16.0)).abs() < 1e-12);
}

#[test]
fn dual_powf_fractional_exponent_of_negative_stays_nan() {
    // Non-integer exponent of a negative base is genuinely complex —
    // the dispatch must NOT rescue this to a finite value.
    let x = Dual::new(-2.0_f64, 1.0);
    let y = x.powf(Dual::constant(2.5));
    assert!(y.re.is_nan(), "re should be NaN, got {}", y.re);
}

#[test]
fn dual_powf_positive_base_unchanged() {
    // Non-dispatch path: positive base with integer constant exponent also
    // goes through `powi` now, but must produce the same result as before.
    let x = Dual::new(2.0_f64, 1.0);
    let y = x.powf(Dual::constant(3.0));
    assert!((y.re - 8.0).abs() < 1e-12);
    assert!((y.eps - 12.0).abs() < 1e-12);
}

#[test]
fn dual_powf_live_exponent_still_uses_exp_ln() {
    // When `n.eps != 0`, dispatch must NOT fire — the ln(x) term is real
    // and carries real tangent information for positive `x`.
    let x = Dual::new(2.0_f64, 0.0);
    let n = Dual::new(3.0_f64, 1.0); // live tangent in exponent
    let y = x.powf(n);
    // 2^3 = 8; d/dn 2^n |_{n=3} = 2^3 · ln(2) = 8·ln(2)
    assert!((y.re - 8.0).abs() < 1e-12);
    let expected = 8.0 * 2.0_f64.ln();
    assert!((y.eps - expected).abs() < 1e-12);
}

// ── DualVec ─────────────────────────────────────────────────────────

#[test]
fn dual_vec_powf_negative_base() {
    let x: DualVec<f64, 2> = DualVec {
        re: -2.0,
        eps: [1.0, 0.0],
    };
    let y = x.powf(DualVec {
        re: 3.0,
        eps: [0.0, 0.0],
    });
    assert!((y.re - (-8.0)).abs() < 1e-12);
    assert!((y.eps[0] - 12.0).abs() < 1e-12);
    assert_eq!(y.eps[1], 0.0);
}

// ── Taylor ──────────────────────────────────────────────────────────

#[cfg(feature = "taylor")]
#[test]
fn taylor_powf_negative_base() {
    use echidna::Taylor;
    // Taylor series [-2, 1, 0, 0] — first-order perturbation of x=-2.
    let x: Taylor<f64, 4> = Taylor::new([-2.0, 1.0, 0.0, 0.0]);
    // Exponent is a constant Taylor [3, 0, 0, 0].
    let n: Taylor<f64, 4> = Taylor::constant(3.0);
    let y = x.powf(n);
    // (x + t)^3 at x = -2: (-2+t)^3 = -8 + 12t - 6t² + t³
    assert!((y.coeffs[0] - (-8.0)).abs() < 1e-12);
    assert!((y.coeffs[1] - 12.0).abs() < 1e-12);
    assert!((y.coeffs[2] - (-6.0)).abs() < 1e-12);
    assert!((y.coeffs[3] - 1.0).abs() < 1e-12);
}

// ── Laurent ─────────────────────────────────────────────────────────

#[cfg(feature = "laurent")]
#[test]
fn laurent_powf_negative_base() {
    use echidna::Laurent;
    let x: Laurent<f64, 4> = Laurent::new([-2.0, 1.0, 0.0, 0.0], 0);
    let n: Laurent<f64, 4> = Laurent::constant(3.0);
    let y = x.powf(n);
    // Same expansion as Taylor case: -8 + 12t - 6t² + t³
    assert!((y.coeff(0) - (-8.0)).abs() < 1e-12);
    assert!((y.coeff(1) - 12.0).abs() < 1e-12);
    assert!((y.coeff(2) - (-6.0)).abs() < 1e-12);
    assert!((y.coeff(3) - 1.0).abs() < 1e-12);
}

// ── Reverse ─────────────────────────────────────────────────────────

#[test]
fn reverse_powf_negative_base() {
    let g = echidna::grad(
        |x: &[echidna::Reverse<f64>]| x[0].powf(echidna::Reverse::constant(3.0)),
        &[-2.0_f64],
    );
    // d/dx x^3 |_{x=-2} = 3x² = 12
    assert!((g[0] - 12.0).abs() < 1e-12, "grad = {}", g[0]);
}

// ── BReverse (bytecode tape opcode safety net) ───────────────────────

#[cfg(feature = "bytecode")]
#[test]
fn breverse_powf_negative_base_constant_exponent() {
    use echidna::BReverse;
    let (mut tape, _) = echidna::record(
        |x: &[BReverse<f64>]| x[0].powf(BReverse::constant(3.0)),
        &[-2.0_f64],
    );
    let g = tape.gradient(&[-2.0_f64]);
    // Safety net: OpCode::Powf reverse partial sets db = 0 for a <= 0, so
    // the NaN that would have propagated through `r * a.ln()` is replaced
    // by 0. Since `n` is CONSTANT, its slot is dropped during push_binary
    // regardless — but the replay path on a mutated tape still uses the
    // opcode partial, and the safety net keeps it finite.
    assert!(g[0].is_finite());
    assert!((g[0] - 12.0).abs() < 1e-12, "grad = {}", g[0]);
}

// ── powi i32::MIN edge (constant-integer dispatch round-trip check) ─

#[test]
fn dual_powf_i32_min_exponent() {
    // i32::MIN as f64 round-trips losslessly, so dispatch should fire.
    // (Result is astronomically small but must not panic / NaN.)
    let x = Dual::new(2.0_f64, 1.0);
    let y = x.powf(Dual::constant(i32::MIN as f64));
    // 2^(-2^31) underflows to 0; derivative `n · 2^(n-1)` also underflows to 0.
    assert!(y.re == 0.0 || y.re.is_finite());
    assert!(y.eps.is_finite(), "eps = {}", y.eps);
}

#[test]
fn dual_powf_out_of_i32_range_falls_through() {
    // 2^32 is outside i32 range; dispatch must NOT fire (to_i32 returns None).
    // This is only a test that the code doesn't panic — for positive base the
    // exp/ln path is well-defined anyway.
    let x = Dual::new(2.0_f64, 1.0);
    let y = x.powf(Dual::constant((i32::MAX as f64) + 2.0));
    assert!(y.re.is_finite() || y.re.is_infinite()); // 2^2e9 overflows to +inf, that's fine
}

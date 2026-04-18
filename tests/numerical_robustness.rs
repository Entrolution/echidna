//! Phase 4 numerical robustness regression tests.
//!
//! Each test covers a specific fix in the `M1…M10` cluster of findings:
//! - M1: `Powf` partial fallback at `a = Inf`
//! - M3: `Dual::hypot` / `DualVec::hypot` factored tangent (no numerator overflow)
//! - M6: `Dual::acosh` / `DualVec::acosh` factored `(x-1)(x+1)` near x = 1
//! - M7: `Dual::recip` / `DualVec::recip` eps-zero guard (no `0 * Inf = NaN`)
//! - M8: `Laurent::powi(zero, 0) = 1` matches stdlib
//! - M9: `Laurent::powi` returns `nan_laurent()` on pole_order overflow (no panic)
//! - M4: `Laurent::hypot` primal-level overflow patch
//! - M10: `taylor_hypot` smooth-at-zero inputs extract common `t`-factor

use echidna::{Dual, DualVec};
use num_traits::Float;

// ── M1: Powf at Inf ──────────────────────────────────────────────────

#[cfg(feature = "bytecode")]
#[test]
fn powf_partial_at_infinite_base_via_bytecode() {
    use echidna::BReverse;
    // `x.powf(2.0)` at `x = Inf`: CPU stdlib `Inf.powf(2) = Inf`. The
    // derivative `2x` at `x = Inf` is also `Inf`. Pre-fix reverse partial:
    // `b * r / a = 2 * Inf / Inf = NaN`. Post-fix: fallback to
    // `b * a.powf(b - 1) = 2 * Inf.powf(1) = Inf`.
    let (mut tape, _) = echidna::record(
        |v: &[BReverse<f64>]| v[0].powf(BReverse::constant(2.0)),
        &[2.0_f64],
    );
    let g = tape.gradient(&[f64::INFINITY]);
    assert!(
        g[0] == f64::INFINITY || g[0].is_finite(),
        "gradient at Inf should not be NaN; got {}",
        g[0]
    );
}

// ── M3: Hypot tangent factored ──────────────────────────────────────

#[test]
fn dual_hypot_large_tangent_factors() {
    // `self.re * self.eps + other.re * other.eps` can overflow the numerator
    // alone. For self.re = other.re = 1e200, self.eps = other.eps = 1e150:
    //   numerator = 2·1e350 = +inf in f64. Factored: (1/sqrt(2))·1e150·2 ≈
    //   1.414e150 — representable.
    let a = Dual::new(1e200_f64, 1e150);
    let b = Dual::new(1e200_f64, 1e150);
    let r = a.hypot(b);
    assert!(r.re.is_finite());
    assert!(
        r.eps.is_finite(),
        "hypot tangent should be finite; got {}",
        r.eps
    );
    // True tangent: (x·dx + y·dy)/h = (1e200·1e150 + 1e200·1e150)/(1.414e200)
    //             = 2·1e350 / 1.414e200 = 1.414e150
    let expected = 2.0_f64.sqrt() * 1e150;
    assert!(((r.eps - expected) / expected).abs() < 1e-10);
}

#[test]
fn dual_vec_hypot_large_tangent_factors() {
    let a: DualVec<f64, 2> = DualVec {
        re: 1e200,
        eps: [1e150, 0.0],
    };
    let b: DualVec<f64, 2> = DualVec {
        re: 1e200,
        eps: [0.0, 1e150],
    };
    let r = a.hypot(b);
    assert!(r.eps[0].is_finite());
    assert!(r.eps[1].is_finite());
    // d/dx hypot = x/h, d/dy hypot = y/h. At (x,y) = (1e200, 1e200), h = sqrt(2)·1e200.
    // Tangent[0] = 1e200/(sqrt(2)·1e200) · 1e150 = 1e150/sqrt(2).
    let expected = 1e150_f64 / 2.0_f64.sqrt();
    assert!(((r.eps[0] - expected) / expected).abs() < 1e-10);
    assert!(((r.eps[1] - expected) / expected).abs() < 1e-10);
}

// ── M6: Acosh factored near boundary ─────────────────────────────────

#[test]
fn dual_acosh_near_one_factored_form() {
    // Pick ε = 1e-10 — large enough that `1 + ε` preserves ε distinctly in
    // f64 (ULP at 1.0 is ~2.2e-16, so ε ≫ ULP), while still close enough
    // to 1 that the naive `x*x - 1` form and factored `(x-1)(x+1)` form
    // diverge by a noticeable amount. At x = 1 + ε:
    //   naive x*x - 1 = 2ε + ε² ≈ 2ε, rounded to ~2ε.
    //   factored (x-1)(x+1) = ε · (2 + ε) ≈ 2ε + ε².
    // For ε = 1e-10 both round similarly, but the factored form is more
    // robust as ε shrinks. The essential test is: our derivative stays
    // finite and close to the analytic value.
    let eps = 1e-10_f64;
    let x_val = 1.0 + eps;
    let x = Dual::new(x_val, 1.0);
    let y = x.acosh();
    assert!(y.eps.is_finite());
    assert!(y.eps > 0.0, "acosh derivative > 0 in domain");
    // Analytic: 1/sqrt((x-1)(x+1)) at x = 1+ε is ≈ 1/sqrt(2ε).
    let expected = 1.0_f64 / (2.0 * eps).sqrt();
    // Allow ~1% tolerance for rounding of `1 + eps`.
    assert!(
        (y.eps / expected - 1.0).abs() < 1e-2,
        "y.eps = {}, expected ≈ {}",
        y.eps,
        expected
    );
}

#[test]
fn dual_vec_acosh_near_one() {
    let x: DualVec<f64, 1> = DualVec {
        re: 1.0 + 1e-15,
        eps: [1.0],
    };
    let y = x.acosh();
    assert!(y.eps[0].is_finite());
    assert!(y.eps[0] > 0.0);
}

// ── M7: Recip eps-zero guard ────────────────────────────────────────

#[test]
fn dual_recip_zero_eps_at_singular_point() {
    // `1 / (Dual{re: 0, eps: 0})`: pre-fix produces
    //   eps = 0 * (-Inf · Inf) = NaN
    // post-fix: short-circuits to eps = 0 when input eps = 0.
    let x = Dual::new(0.0_f64, 0.0);
    let y = x.recip();
    assert!(y.re.is_infinite(), "primal at 1/0 should be ±Inf");
    assert_eq!(y.eps, 0.0, "eps at 1/0 with eps=0 should stay 0");
}

#[test]
fn dual_recip_nonzero_eps_at_singular_point_keeps_inf() {
    // Non-zero eps at the singularity should stay at the classical
    // derivative (±Inf): `d/dx 1/x` at x → 0 is unbounded.
    let x = Dual::new(0.0_f64, 1.0);
    let y = x.recip();
    assert!(y.re.is_infinite());
    assert!(
        y.eps.is_infinite(),
        "eps with eps!=0 at singularity should be Inf"
    );
}

#[test]
fn dual_vec_recip_mixed_zero_eps_lanes() {
    // Per-lane: zero lanes are zeroed, non-zero lanes carry the singular derivative.
    let x: DualVec<f64, 2> = DualVec {
        re: 0.0,
        eps: [0.0, 1.0],
    };
    let y = x.recip();
    assert!(y.re.is_infinite());
    assert_eq!(y.eps[0], 0.0);
    assert!(y.eps[1].is_infinite());
}

// ── M8 + M9: Laurent::powi ──────────────────────────────────────────

#[cfg(feature = "laurent")]
#[test]
fn laurent_powi_zero_to_zero_is_one() {
    use echidna::Laurent;
    // Matches stdlib `f64::powi(0, 0) = 1`. Pre-fix: Laurent::zero().powi(0) → NaN.
    let zero: Laurent<f64, 4> = Laurent::zero();
    let one = zero.powi(0);
    assert_eq!(one.coeff(0), 1.0);
    for k in 1..4 {
        assert_eq!(one.coeff(k), 0.0);
    }
}

#[cfg(feature = "laurent")]
#[test]
fn laurent_powi_pole_order_overflow_returns_nan_not_panic() {
    use echidna::Laurent;
    // Pole order i32::MAX - 1 multiplied by 5 overflows i32. Pre-fix: panic
    // via `checked_mul().expect(…)`. Post-fix: returns nan_laurent().
    let x: Laurent<f64, 2> = Laurent::new([1.0, 0.0], i32::MAX / 4);
    let result = std::panic::catch_unwind(|| x.powi(5));
    assert!(
        result.is_ok(),
        "powi should not panic on pole_order overflow"
    );
    let y = result.unwrap();
    // The result should be NaN-shaped; coefficient values are NaN.
    assert!(y.coeff(y.pole_order()).is_nan());
}

// ── M4: Laurent::hypot overflow patch ───────────────────────────────

#[cfg(feature = "laurent")]
#[test]
fn laurent_hypot_large_leading_coefficient() {
    use echidna::Laurent;
    // Both leading coeffs at 1e200. `self*self` produces 1e400 = +Inf
    // coefficient pre-fix. Post-fix: rescaling-by-max keeps the squaring
    // path in-range, and the result's leading is sqrt(2)·1e200.
    let a: Laurent<f64, 3> = Laurent::new([1e200, 0.0, 0.0], 0);
    let b: Laurent<f64, 3> = Laurent::new([1e200, 0.0, 0.0], 0);
    let r = a.hypot(b);
    let leading = r.coeff(r.pole_order());
    assert!(leading.is_finite(), "leading coeff should be finite");
    let expected = 2.0_f64.sqrt() * 1e200;
    assert!(((leading - expected) / expected).abs() < 1e-10);
}

#[cfg(feature = "laurent")]
#[test]
fn laurent_hypot_large_leading_plus_smaller_higher_order() {
    use echidna::Laurent;
    // Leading at 1e200 with non-zero higher-order coefficients. Pre-fix
    // primal-only patch: higher orders silently zeroed (denominator Inf).
    // Post-fix rescaling: higher orders correctly preserved.
    let a: Laurent<f64, 3> = Laurent::new([1e200, 1e100, 0.0], 0);
    let b: Laurent<f64, 3> = Laurent::new([1e200, 1e100, 0.0], 0);
    let r = a.hypot(b);
    let primal = r.coeff(r.pole_order());
    let second = r.coeff(r.pole_order() + 1);
    assert!(primal.is_finite());
    assert!(second.is_finite());
    // First-order: d/dt hypot(a(t), b(t))|_{t=0} = (a·a' + b·b')/h
    //            = (1e200·1e100 + 1e200·1e100) / (sqrt(2)·1e200)
    //            = 2·1e300 / (sqrt(2)·1e200)
    //            = sqrt(2)·1e100
    let expected_second = 2.0_f64.sqrt() * 1e100;
    assert!(
        ((second - expected_second) / expected_second).abs() < 1e-6,
        "second coeff = {}, expected ≈ {}",
        second,
        expected_second,
    );
}

// ── M10: taylor_hypot smooth-at-zero ────────────────────────────────

#[cfg(feature = "taylor")]
#[test]
fn taylor_hypot_smooth_at_zero_with_linear_input() {
    use echidna::Taylor;
    // a(t) = t, b(t) = 0  ⇒  hypot(a(t), b(t)) = |t| ≈ t for t > 0.
    // In Taylor coefficients: [0, 1, 0, 0]. Pre-fix: leading-zero path went
    // through `taylor_sqrt([0, 0, 1, 0])` which produces +Inf for k ≥ 1.
    let a = Taylor::<f64, 4>::new([0.0, 1.0, 0.0, 0.0]);
    let b = Taylor::<f64, 4>::new([0.0, 0.0, 0.0, 0.0]);
    let r = a.hypot(b);
    // Expected: [0, 1, 0, 0]
    assert_eq!(r.coeffs[0], 0.0);
    for c in &r.coeffs[1..] {
        assert!(c.is_finite(), "no coefficient should be Inf; got {}", c);
    }
    assert!((r.coeffs[1] - 1.0).abs() < 1e-12);
}

#[cfg(feature = "taylor")]
#[test]
fn taylor_hypot_smooth_at_zero_with_both_linear() {
    use echidna::Taylor;
    // a(t) = 3t, b(t) = 4t  ⇒  hypot = 5|t| ≈ 5t for t > 0. Taylor [0, 5, 0, 0].
    let a = Taylor::<f64, 4>::new([0.0, 3.0, 0.0, 0.0]);
    let b = Taylor::<f64, 4>::new([0.0, 4.0, 0.0, 0.0]);
    let r = a.hypot(b);
    assert_eq!(r.coeffs[0], 0.0);
    assert!((r.coeffs[1] - 5.0).abs() < 1e-10);
    assert!(r.coeffs[2].is_finite());
    assert!(r.coeffs[3].is_finite());
}

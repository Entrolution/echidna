//! Phase 8 Commit 1 regressions — Taylor / Laurent domain guards.
//!
//! Covers L6, L7 (checked pole_order arithmetic), L8 (taylor_sqrt negative
//! a[0]), L9 (Taylor::rem zero divisor).

#![cfg(feature = "laurent")]

use echidna::{Laurent, Taylor};

// A NaN Laurent is built with `pole_order = 0` and `coeffs = [NaN; K]`.
// `coeff(0)..coeff(K-1)` therefore covers the stored slots.
fn laurent_is_nan<const K: usize>(l: &Laurent<f64, K>) -> bool {
    (0..K as i32).all(|k| l.coeff(k).is_nan())
}

fn taylor_is_nan<const K: usize>(t: &Taylor<f64, K>) -> bool {
    (0..K).all(|k| t.coeff(k).is_nan())
}

// L6: Laurent::recip on pole_order = i32::MIN cannot be negated without
// overflow. The fix returns a NaN Laurent instead of silently wrapping.
#[test]
fn l6_laurent_recip_pole_order_i32_min_returns_nan() {
    let l = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], i32::MIN);
    let r = l.recip();
    assert!(
        laurent_is_nan(&r),
        "recip of i32::MIN-pole Laurent must yield NaN coeffs"
    );
}

// L7: Laurent Mul on two pole_orders whose sum overflows i32.
#[test]
fn l7_laurent_mul_pole_order_overflow_returns_nan() {
    let a = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], i32::MAX - 1);
    let b = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], 3);
    let r = a * b;
    assert!(
        laurent_is_nan(&r),
        "Mul with overflowing pole_order must yield NaN"
    );
}

// L7: Laurent Div on two pole_orders whose difference underflows i32.
#[test]
fn l7_laurent_div_pole_order_underflow_returns_nan() {
    let a = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], i32::MIN + 1);
    let b = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], 3);
    let r = a / b;
    assert!(
        laurent_is_nan(&r),
        "Div with underflowing pole_order must yield NaN"
    );
}

// L8: taylor_sqrt for a Taylor series with a[0] < 0 should produce a fully
// NaN output, not a mix of NaN primal and silently-computed higher coeffs.
#[test]
fn l8_taylor_sqrt_negative_a0_returns_nan() {
    let a = Taylor::<f64, 4>::new([-1.0, 2.0, 3.0, 4.0]);
    let r = a.sqrt();
    assert!(
        taylor_is_nan(&r),
        "sqrt of negative-a0 Taylor must yield all-NaN coeffs"
    );
}

// taylor_hypot with both leading primals zero must peel ANY higher-order signal,
// not just order 1. `hypot(t², 0) = t²`; the old guard only checked order 1, so it
// fell through to the square→sqrt path and returned `[0, Inf, …]`. Assert per
// coefficient — Taylor `==` compares only `coeff(0)`.
#[test]
fn taylor_hypot_higher_order_leading_zero() {
    let zero = Taylor::<f64, 4>::new([0.0, 0.0, 0.0, 0.0]);

    // hypot(t², 0) = t²  →  [0, 0, 1, 0]
    let r = Taylor::<f64, 4>::new([0.0, 0.0, 1.0, 0.0]).hypot(zero);
    assert_eq!(r.coeff(0), 0.0);
    assert_eq!(
        r.coeff(1),
        0.0,
        "coeff(1) must be 0 (was Inf), got {}",
        r.coeff(1)
    );
    assert!(
        (r.coeff(2) - 1.0).abs() < 1e-12,
        "coeff(2) = {}",
        r.coeff(2)
    );
    assert_eq!(r.coeff(3), 0.0);

    // hypot(t³, 0) = t³  →  [0, 0, 0, 1]
    let r3 = Taylor::<f64, 4>::new([0.0, 0.0, 0.0, 1.0]).hypot(zero);
    assert!(
        (r3.coeff(3) - 1.0).abs() < 1e-12,
        "coeff(3) = {}",
        r3.coeff(3)
    );
    for k in 0..3 {
        assert_eq!(r3.coeff(k), 0.0, "coeff({k}) = {}", r3.coeff(k));
    }
}

// Laurent(t², pole 2).hypot(0) rebases to a common pole, introducing leading
// zeros with an order-2 signal — reaching the kernel's `scale==0` branch (contra
// the old "unreachable" comment). Must normalize to a clean pole-2 t².
#[test]
fn laurent_hypot_rebased_leading_zero() {
    let r = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], 2).hypot(Laurent::<f64, 4>::zero());
    assert_eq!(r.pole_order(), 2, "pole_order = {}", r.pole_order());
    assert!(
        (r.coeff(2) - 1.0).abs() < 1e-12,
        "coeff(2) = {}",
        r.coeff(2)
    );
    assert_eq!(r.coeff(0), 0.0);
    assert_eq!(r.coeff(1), 0.0);
}

// L9: Taylor::rem with zero divisor must not silently produce Inf/NaN in
// individual coefficient slots — it should return a uniformly NaN result.
#[test]
fn l9_taylor_rem_zero_divisor_returns_nan() {
    let a = Taylor::<f64, 4>::new([3.0, 1.0, 0.0, 0.0]);
    let b = Taylor::<f64, 4>::new([0.0, 1.0, 0.0, 0.0]);
    let r = a % b;
    assert!(
        taylor_is_nan(&r),
        "rem with zero-divisor Taylor must yield all-NaN"
    );
}

// Scalar-variant `Rem`: `Taylor % scalar` and `scalar % Taylor` must apply the
// same zero-divisor guard as the Taylor%Taylor impl. Previously `Taylor % 0`
// left `coeffs[0] % 0 = NaN` beside finite higher coefficients, and
// `scalar % zero-const-Taylor` produced an Inf/NaN mix.
#[test]
fn taylor_rem_scalar_zero_divisor_returns_nan() {
    let a = Taylor::<f64, 4>::new([3.0, 1.0, 2.0, 0.0]);
    let r = a % 0.0_f64;
    assert!(taylor_is_nan(&r), "Taylor % 0 must yield all-NaN");
}

#[test]
fn scalar_rem_taylor_zero_divisor_returns_nan() {
    let b = Taylor::<f64, 4>::new([0.0, 1.0, 0.0, 0.0]);
    let r = 7.0_f64 % b;
    assert!(
        taylor_is_nan(&r),
        "scalar % zero-const Taylor must yield all-NaN"
    );
}

// TaylorDyn scalar Rem variants delegate to the guarded generic impl — anchor
// that the delegation genuinely carries the zero-divisor guard.
#[cfg(feature = "taylor")]
#[test]
fn taylor_dyn_rem_scalar_zero_divisor_returns_nan() {
    use echidna::{TaylorDyn, TaylorDynGuard};

    let _guard = TaylorDynGuard::<f64>::new(4);
    let a = TaylorDyn::<f64>::from_coeffs(&[3.0, 1.0, 2.0, 0.0]);
    let r = a % 0.0_f64;
    assert!(
        r.coeffs().iter().all(|c| c.is_nan()),
        "TaylorDyn % 0 must yield all-NaN, got {:?}",
        r.coeffs()
    );

    let b = TaylorDyn::<f64>::from_coeffs(&[0.0, 1.0, 0.0, 0.0]);
    let r2 = 7.0_f64 % b;
    assert!(
        r2.coeffs().iter().all(|c| c.is_nan()),
        "scalar % zero-const TaylorDyn must yield all-NaN, got {:?}",
        r2.coeffs()
    );
}

// L9 sibling on TaylorDyn: the dynamic-sized Taylor must receive the
// same zero-divisor guard as the static-sized Taylor.
#[cfg(feature = "taylor")]
#[test]
fn l9_taylor_dyn_rem_zero_divisor_returns_nan() {
    use echidna::{TaylorDyn, TaylorDynGuard};

    let _guard = TaylorDynGuard::<f64>::new(4);
    let a = TaylorDyn::<f64>::from_coeffs(&[3.0, 1.0, 0.0, 0.0]);
    let b = TaylorDyn::<f64>::from_coeffs(&[0.0, 1.0, 0.0, 0.0]);
    let r = a % b;
    let coeffs = r.coeffs();
    assert!(
        coeffs.iter().all(|c| c.is_nan()),
        "TaylorDyn::rem with zero-divisor must yield all-NaN, got {:?}",
        coeffs
    );
}

// Series domain-NaN: the Taylor log / atanh / acosh kernels must return an
// all-NaN jet strictly outside the real domain (mirroring the scalar Phase-4
// convention) rather than a NaN primal beside finite higher coefficients.
#[test]
fn taylor_log_atanh_acosh_domain_nan() {
    // ln / log2 / log10: leading coefficient < 0.
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-2.0, 1.0, 1.0, 0.0]).ln()
    ));
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-2.0, 1.0, 1.0, 0.0]).log2()
    ));
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-2.0, 1.0, 1.0, 0.0]).log10()
    ));
    // ln_1p: leading coefficient < -1.
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-2.0, 1.0, 0.0, 0.0]).ln_1p()
    ));
    // atanh: |leading| > 1.
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([1.5, 1.0, 0.0, 0.0]).atanh()
    ));
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-1.5, 1.0, 0.0, 0.0]).atanh()
    ));
    // acosh: leading coefficient < 1 (the a[0] <= -1 gap plus the -1<a<1 range).
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([-2.0, 1.0, 0.0, 0.0]).acosh()
    ));
    assert!(taylor_is_nan(
        &Taylor::<f64, 4>::new([0.5, 1.0, 0.0, 0.0]).acosh()
    ));
}

// In-domain jets are unchanged (regression anchor, prevents over-guarding).
#[test]
fn taylor_ln_in_domain_unchanged() {
    let r = Taylor::<f64, 4>::new([2.0, 1.0, 0.0, 0.0]).ln();
    assert!((r.coeff(0) - 2.0_f64.ln()).abs() < 1e-12);
    // d/dt ln(2+t) = 1/(2+t) = 0.5 at t=0.
    assert!((r.coeff(1) - 0.5).abs() < 1e-12);
    assert!(r.coeff(0).is_finite());
}

// The `a[0]==0` boundary for ln stays the IEEE branch-point singularity (-Inf),
// NOT NaN — the guard is strictly `< 0`, matching the scalar boundary convention.
#[test]
fn taylor_ln_zero_leading_stays_singular_not_nan() {
    let r = Taylor::<f64, 4>::new([0.0, 1.0, 0.0, 0.0]).ln();
    assert!(
        r.coeff(0).is_infinite() && r.coeff(0) < 0.0,
        "ln(0) leading must be -Inf, got {}",
        r.coeff(0)
    );
    assert!(!r.coeff(0).is_nan());
}

// Laurent `atanh` / `ln_1p` have no independent domain guard — they inherit it
// from the kernel fix and now return all-NaN out of domain (Laurent `ln`/`log2`/
// `log10` were already guarded independently).
#[test]
fn laurent_atanh_ln_1p_domain_nan() {
    assert!(laurent_is_nan(
        &Laurent::<f64, 4>::new([1.5, 1.0, 0.0, 0.0], 0).atanh()
    ));
    assert!(laurent_is_nan(
        &Laurent::<f64, 4>::new([-2.0, 1.0, 0.0, 0.0], 0).ln_1p()
    ));
}

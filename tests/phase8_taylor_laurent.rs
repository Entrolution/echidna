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

//! `Laurent::zero()` must behave as the additive identity regardless of the
//! other operand's `pole_order`. The pre-fix Add/Sub impls aligned both sides
//! to `p_out = min(p1, p2) = 0` and then asserted `shift < K`, which panicked
//! on `self + Laurent::zero()` whenever `|self.pole_order| >= K`.

#![cfg(feature = "laurent")]

use echidna::Laurent;

#[test]
fn add_zero_to_large_pole_order() {
    // pole_order = 4, K = 4 → shift1 = 4 = K, which used to trip the assertion
    // `shift1 < K` before the short-circuit. After the fix, the zero-identity
    // check fires first.
    let x: Laurent<f64, 4> = Laurent::new([1.0, 2.0, 3.0, 4.0], 4);
    let y = x + Laurent::<f64, 4>::zero();
    // Result must equal x: coeffs[0] is the leading coefficient at
    // `t^pole_order`, addressable via coeff(pole_order).
    assert_eq!(y.pole_order(), 4);
    assert_eq!(y.coeff(4), 1.0);
    assert_eq!(y.coeff(7), 4.0);
}

#[test]
fn add_zero_left_large_pole_order() {
    let x: Laurent<f64, 4> = Laurent::new([1.0, 2.0, 3.0, 4.0], 4);
    let y = Laurent::<f64, 4>::zero() + x;
    assert_eq!(y.pole_order(), 4);
    assert_eq!(y.coeff(4), 1.0);
    assert_eq!(y.coeff(7), 4.0);
}

#[test]
fn sub_zero_from_large_pole_order() {
    let x: Laurent<f64, 4> = Laurent::new([1.0, 2.0, 3.0, 4.0], 4);
    let y = x - Laurent::<f64, 4>::zero();
    assert_eq!(y.pole_order(), 4);
    assert_eq!(y.coeff(4), 1.0);
}

#[test]
fn sub_from_zero_large_pole_order() {
    let x: Laurent<f64, 4> = Laurent::new([1.0, 2.0, 3.0, 4.0], 4);
    let y = Laurent::<f64, 4>::zero() - x;
    assert_eq!(y.pole_order(), 4);
    assert_eq!(y.coeff(4), -1.0);
    assert_eq!(y.coeff(7), -4.0);
}

#[test]
fn powi_then_add_zero() {
    // Regression for the original motivating case: `Laurent::variable(0.0).powi(K)`
    // produces a pole_order that equals K, after which `+ zero()` used to panic.
    let x: Laurent<f64, 4> = Laurent::variable(0.0);
    let y = x.powi(4);
    let _sum = y + Laurent::<f64, 4>::zero();
    // Test passes if it doesn't panic.
}

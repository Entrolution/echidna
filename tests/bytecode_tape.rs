//! Tests for BReverse + BytecodeTape — mirrors reverse_mode.rs.

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
use echidna::{record, BReverse};
use num_traits::Float;

/// Run a single-variable bytecode reverse-mode differentiation.
fn brev_grad(f: impl FnOnce(BReverse<f64>) -> BReverse<f64>, x_val: f64) -> f64 {
    let (mut tape, _) = record(|x| f(x[0]), &[x_val]);
    let g = tape.gradient(&[x_val]);
    g[0]
}

/// Central finite difference for comparison.
fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-7;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn check_brev_elemental(
    f_brev: impl FnOnce(BReverse<f64>) -> BReverse<f64>,
    f_f64: impl Fn(f64) -> f64,
    x: f64,
    tol: f64,
) {
    let grad = brev_grad(f_brev, x);
    let expected = finite_diff(&f_f64, x);
    assert_relative_eq!(grad, expected, max_relative = tol);
}

// ── Arithmetic ──

#[test]
fn x_squared() {
    let grad = brev_grad(|x| x * x, 3.0);
    assert_relative_eq!(grad, 6.0, max_relative = 1e-12);
}

#[test]
fn x_times_y() {
    let (mut tape, _) = record(|x| x[0] * x[1], &[3.0, 4.0]);
    let g = tape.gradient(&[3.0, 4.0]);
    assert_relative_eq!(g[0], 4.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 3.0, max_relative = 1e-12);
}

#[test]
fn diamond_pattern() {
    let grad = brev_grad(|x| x * x + x * x * x, 2.0);
    assert_relative_eq!(grad, 4.0 + 12.0, max_relative = 1e-12);
}

#[test]
fn fan_out() {
    let grad = brev_grad(|x| x + x + x, 5.0);
    assert_relative_eq!(grad, 3.0, max_relative = 1e-12);
}

#[test]
fn chain_depth() {
    let grad = brev_grad(
        |x| {
            let a = x * x;
            let b = a * a;
            b * b
        },
        2.0,
    );
    assert_relative_eq!(grad, 8.0 * 2.0_f64.powi(7), max_relative = 1e-10);
}

// ── Elementals ──

#[test]
fn sin() {
    check_brev_elemental(|x| x.sin(), |x| x.sin(), 1.0, 1e-5);
}

#[test]
fn cos() {
    check_brev_elemental(|x| x.cos(), |x| x.cos(), 1.0, 1e-5);
}

#[test]
fn tan() {
    check_brev_elemental(|x| x.tan(), |x| x.tan(), 0.5, 1e-5);
}

#[test]
fn exp() {
    check_brev_elemental(|x| x.exp(), |x| x.exp(), 1.0, 1e-5);
}

#[test]
fn ln() {
    check_brev_elemental(|x| x.ln(), |x| x.ln(), 2.0, 1e-5);
}

#[test]
fn sqrt() {
    check_brev_elemental(|x| x.sqrt(), |x| x.sqrt(), 4.0, 1e-5);
}

#[test]
fn recip() {
    check_brev_elemental(|x| x.recip(), |x| x.recip(), 2.5, 1e-5);
}

#[test]
fn powi() {
    check_brev_elemental(|x| x.powi(3), |x| x.powi(3), 2.0, 1e-5);
}

#[test]
fn tanh() {
    check_brev_elemental(|x| x.tanh(), |x| x.tanh(), 1.0, 1e-5);
}

#[test]
fn asin() {
    check_brev_elemental(|x| x.asin(), |x| x.asin(), 0.5, 1e-5);
}

#[test]
fn acos() {
    check_brev_elemental(|x| x.acos(), |x| x.acos(), 0.5, 1e-5);
}

#[test]
fn atan() {
    check_brev_elemental(|x| x.atan(), |x| x.atan(), 1.0, 1e-5);
}

#[test]
fn sinh() {
    check_brev_elemental(|x| x.sinh(), |x| x.sinh(), 1.0, 1e-5);
}

#[test]
fn cosh() {
    check_brev_elemental(|x| x.cosh(), |x| x.cosh(), 1.0, 1e-5);
}

#[test]
fn asinh() {
    check_brev_elemental(|x| x.asinh(), |x| x.asinh(), 1.0, 1e-5);
}

#[test]
fn acosh() {
    check_brev_elemental(|x| x.acosh(), |x| x.acosh(), 2.0, 1e-5);
}

#[test]
fn atanh() {
    check_brev_elemental(|x| x.atanh(), |x| x.atanh(), 0.5, 1e-5);
}

#[test]
fn exp2() {
    check_brev_elemental(|x| x.exp2(), |x| x.exp2(), 1.5, 1e-5);
}

#[test]
fn log2() {
    check_brev_elemental(|x| x.log2(), |x| x.log2(), 2.0, 1e-5);
}

#[test]
fn log10() {
    check_brev_elemental(|x| x.log10(), |x| x.log10(), 2.0, 1e-5);
}

#[test]
fn cbrt() {
    check_brev_elemental(|x| x.cbrt(), |x| x.cbrt(), 8.0, 1e-5);
}

#[test]
fn exp_m1() {
    check_brev_elemental(|x| x.exp_m1(), |x| x.exp_m1(), 0.5, 1e-5);
}

#[test]
fn ln_1p() {
    check_brev_elemental(|x| x.ln_1p(), |x| x.ln_1p(), 0.5, 1e-5);
}

#[test]
fn abs_positive() {
    let grad = brev_grad(|x| x.abs(), 3.0);
    assert_relative_eq!(grad, 1.0, max_relative = 1e-12);
}

#[test]
fn abs_negative() {
    let grad = brev_grad(|x| x.abs(), -3.0);
    assert_relative_eq!(grad, -1.0, max_relative = 1e-12);
}

// ── Compositions ──

#[test]
fn sin_of_exp() {
    let x_val = 0.5;
    let grad = brev_grad(|x| x.exp().sin(), x_val);
    let expected = x_val.exp().cos() * x_val.exp();
    assert_relative_eq!(grad, expected, max_relative = 1e-10);
}

#[test]
fn complex_composition() {
    let x_val = 1.5;
    let grad = brev_grad(|x| x * x.sin() + (x * x).cos(), x_val);
    let expected = x_val.sin() + x_val * x_val.cos() - 2.0 * x_val * (x_val * x_val).sin();
    assert_relative_eq!(grad, expected, max_relative = 1e-10);
}

// ── Constants ──

#[test]
fn constant_addition() {
    let grad = brev_grad(|x| x + BReverse::constant(5.0), 3.0);
    assert_relative_eq!(grad, 1.0, max_relative = 1e-12);
}

#[test]
fn scalar_multiplication() {
    let grad = brev_grad(|x| 3.0 * x, 2.0);
    assert_relative_eq!(grad, 3.0, max_relative = 1e-12);
}

// ── Cross-validation: BReverse vs Adept tape ──

#[test]
fn matches_adept_rosenbrock_2d() {
    let x = [1.5, 2.0];
    let adept = echidna::grad(|v| {
        let one = echidna::Reverse::constant(1.0);
        let hundred = echidna::Reverse::constant(100.0);
        let t1 = one - v[0];
        let t2 = v[1] - v[0] * v[0];
        t1 * t1 + hundred * t2 * t2
    }, &x);

    let (mut tape, _) = record(|v| {
        let one = BReverse::constant(1.0);
        let hundred = BReverse::constant(100.0);
        let t1 = one - v[0];
        let t2 = v[1] - v[0] * v[0];
        t1 * t1 + hundred * t2 * t2
    }, &x);
    let btape = tape.gradient(&x);

    for i in 0..x.len() {
        assert_relative_eq!(adept[i], btape[i], max_relative = 1e-12);
    }
}

#[test]
fn matches_adept_trig_mix() {
    let x = [1.0, 2.0];
    let adept = echidna::grad(|v| v[0].sin() * v[1].exp() + v[0].cos() * v[1].ln(), &x);
    let (mut tape, _) = record(|v| v[0].sin() * v[1].exp() + v[0].cos() * v[1].ln(), &x);
    let btape = tape.gradient(&x);

    for i in 0..x.len() {
        assert_relative_eq!(adept[i], btape[i], max_relative = 1e-12);
    }
}

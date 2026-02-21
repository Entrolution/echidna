use approx::assert_relative_eq;
use echidna::{Dual, Dual64};
use num_traits::Float;

/// Central finite difference: (f(x+h) - f(x-h)) / 2h
fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-7;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Test a dual elemental against finite differences.
fn check_elemental(
    f_dual: impl Fn(Dual64) -> Dual64,
    f_f64: impl Fn(f64) -> f64,
    x: f64,
    tol: f64,
) {
    let d = f_dual(Dual::variable(x));
    let expected_deriv = finite_diff(&f_f64, x);
    assert_relative_eq!(d.re, f_f64(x), max_relative = 1e-12);
    assert_relative_eq!(d.eps, expected_deriv, max_relative = tol);
}

// ── Arithmetic ──

#[test]
fn product_rule() {
    // (3 + ε)(4 + ε) = 12 + 7ε
    let a = Dual::new(3.0, 1.0);
    let b = Dual::new(4.0, 1.0);
    let c = a * b;
    assert_relative_eq!(c.re, 12.0);
    assert_relative_eq!(c.eps, 7.0);
}

#[test]
fn quotient_rule() {
    // d/dx (x / (x+1)) at x=2: 1/(x+1)^2 = 1/9
    let x = Dual::variable(2.0);
    let one = Dual::constant(1.0);
    let y = x / (x + one);
    assert_relative_eq!(y.re, 2.0 / 3.0, max_relative = 1e-12);
    assert_relative_eq!(y.eps, 1.0 / 9.0, max_relative = 1e-12);
}

#[test]
fn mixed_scalar_ops() {
    let x = Dual::<f64>::variable(3.0);
    let y = x * 2.0;
    assert_relative_eq!(y.re, 6.0);
    assert_relative_eq!(y.eps, 2.0);

    let z = 2.0 * x;
    assert_relative_eq!(z.re, 6.0);
    assert_relative_eq!(z.eps, 2.0);

    let w = 1.0 / x;
    assert_relative_eq!(w.re, 1.0 / 3.0, max_relative = 1e-12);
    assert_relative_eq!(w.eps, -1.0 / 9.0, max_relative = 1e-12);
}

// ── Powers ──

#[test]
fn recip() { check_elemental(|x| x.recip(), |x| x.recip(), 2.5, 1e-5); }

#[test]
fn sqrt() { check_elemental(|x| x.sqrt(), |x| x.sqrt(), 4.0, 1e-5); }

#[test]
fn cbrt() { check_elemental(|x| x.cbrt(), |x| x.cbrt(), 8.0, 1e-5); }

#[test]
fn powi() { check_elemental(|x| x.powi(3), |x| x.powi(3), 2.0, 1e-5); }

#[test]
fn powf() {
    let x = Dual::variable(2.0);
    let n = Dual::constant(3.5);
    let y = x.powf(n);
    let expected = finite_diff(|v| v.powf(3.5), 2.0);
    assert_relative_eq!(y.re, 2.0_f64.powf(3.5), max_relative = 1e-12);
    assert_relative_eq!(y.eps, expected, max_relative = 1e-5);
}

// ── Exp/Log ──

#[test]
fn exp() { check_elemental(|x| x.exp(), |x| x.exp(), 1.0, 1e-5); }

#[test]
fn exp2() { check_elemental(|x| x.exp2(), |x| x.exp2(), 1.5, 1e-5); }

#[test]
fn exp_m1() { check_elemental(|x| x.exp_m1(), |x| x.exp_m1(), 0.5, 1e-5); }

#[test]
fn ln() { check_elemental(|x| x.ln(), |x| x.ln(), 2.0, 1e-5); }

#[test]
fn log2() { check_elemental(|x| x.log2(), |x| x.log2(), 2.0, 1e-5); }

#[test]
fn log10() { check_elemental(|x| x.log10(), |x| x.log10(), 2.0, 1e-5); }

#[test]
fn ln_1p() { check_elemental(|x| x.ln_1p(), |x| x.ln_1p(), 0.5, 1e-5); }

// ── Trig ──

#[test]
fn sin() { check_elemental(|x| x.sin(), |x| x.sin(), 1.0, 1e-5); }

#[test]
fn cos() { check_elemental(|x| x.cos(), |x| x.cos(), 1.0, 1e-5); }

#[test]
fn tan() { check_elemental(|x| x.tan(), |x| x.tan(), 0.5, 1e-5); }

#[test]
fn sin_cos() {
    let x = Dual::<f64>::variable(1.0);
    let (s, c) = x.sin_cos();
    assert_relative_eq!(s.re, 1.0_f64.sin(), max_relative = 1e-12);
    assert_relative_eq!(c.re, 1.0_f64.cos(), max_relative = 1e-12);
    assert_relative_eq!(s.eps, 1.0_f64.cos(), max_relative = 1e-12);
    assert_relative_eq!(c.eps, -1.0_f64.sin(), max_relative = 1e-12);
}

#[test]
fn asin() { check_elemental(|x| x.asin(), |x| x.asin(), 0.5, 1e-5); }

#[test]
fn acos() { check_elemental(|x| x.acos(), |x| x.acos(), 0.5, 1e-5); }

#[test]
fn atan() { check_elemental(|x| x.atan(), |x| x.atan(), 1.0, 1e-5); }

#[test]
fn atan2() {
    let y = Dual::<f64>::variable(3.0);
    let x = Dual::constant(4.0);
    let a = y.atan2(x);
    let expected = finite_diff(|v| v.atan2(4.0), 3.0);
    assert_relative_eq!(a.re, 3.0_f64.atan2(4.0), max_relative = 1e-12);
    assert_relative_eq!(a.eps, expected, max_relative = 1e-5);
}

// ── Hyperbolic ──

#[test]
fn sinh() { check_elemental(|x| x.sinh(), |x| x.sinh(), 1.0, 1e-5); }

#[test]
fn cosh() { check_elemental(|x| x.cosh(), |x| x.cosh(), 1.0, 1e-5); }

#[test]
fn tanh() { check_elemental(|x| x.tanh(), |x| x.tanh(), 1.0, 1e-5); }

#[test]
fn asinh() { check_elemental(|x| x.asinh(), |x| x.asinh(), 1.0, 1e-5); }

#[test]
fn acosh() { check_elemental(|x| x.acosh(), |x| x.acosh(), 2.0, 1e-5); }

#[test]
fn atanh() { check_elemental(|x| x.atanh(), |x| x.atanh(), 0.5, 1e-5); }

// ── Misc ──

#[test]
fn abs_positive() {
    let x = Dual::<f64>::variable(3.0);
    let y = x.abs();
    assert_relative_eq!(y.re, 3.0);
    assert_relative_eq!(y.eps, 1.0);
}

#[test]
fn abs_negative() {
    let x = Dual::<f64>::variable(-3.0);
    let y = x.abs();
    assert_relative_eq!(y.re, 3.0);
    assert_relative_eq!(y.eps, -1.0);
}

#[test]
fn hypot() {
    let x = Dual::<f64>::variable(3.0);
    let y = Dual::constant(4.0);
    let h = x.hypot(y);
    assert_relative_eq!(h.re, 5.0, max_relative = 1e-12);
    let expected = finite_diff(|v| v.hypot(4.0), 3.0);
    assert_relative_eq!(h.eps, expected, max_relative = 1e-5);
}

// ── Compositions ──

#[test]
fn sin_of_exp() {
    // d/dx sin(exp(x)) = cos(exp(x)) * exp(x)
    let x_val = 0.5;
    let x = Dual::<f64>::variable(x_val);
    let y = x.exp().sin();
    let expected = x_val.exp().cos() * x_val.exp();
    assert_relative_eq!(y.eps, expected, max_relative = 1e-12);
}

#[test]
fn complex_composition() {
    // f(x) = x * sin(x) + cos(x²)
    // f'(x) = sin(x) + x*cos(x) - 2x*sin(x²)
    let x_val = 1.5;
    let x = Dual::<f64>::variable(x_val);
    let y = x * x.sin() + (x * x).cos();
    let expected = x_val.sin() + x_val * x_val.cos() - 2.0 * x_val * (x_val * x_val).sin();
    assert_relative_eq!(y.eps, expected, max_relative = 1e-12);
}

// ── Edge cases ──

#[test]
fn zero_derivative_funcs() {
    let x = Dual::<f64>::variable(2.7);
    assert_relative_eq!(x.floor().eps, 0.0);
    assert_relative_eq!(x.ceil().eps, 0.0);
    assert_relative_eq!(x.round().eps, 0.0);
    assert_relative_eq!(x.trunc().eps, 0.0);
    assert_relative_eq!(x.signum().eps, 0.0);
}

#[test]
fn fract_preserves_derivative() {
    let x = Dual::<f64>::variable(2.7);
    assert_relative_eq!(x.fract().eps, 1.0);
}

// ── num-traits Float trait ──

#[test]
fn float_trait_methods() {
    let x = Dual64::variable(2.0);

    // Via Float trait
    let y = Float::sin(x);
    assert_relative_eq!(y.re, 2.0_f64.sin(), max_relative = 1e-12);
    assert_relative_eq!(y.eps, 2.0_f64.cos(), max_relative = 1e-12);
}

#[test]
fn from_primitive_zero_derivative() {
    use num_traits::FromPrimitive;
    let x = Dual64::from_f64(3.14).unwrap();
    assert_relative_eq!(x.re, 3.14);
    assert_relative_eq!(x.eps, 0.0);
}

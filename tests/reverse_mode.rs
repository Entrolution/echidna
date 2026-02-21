use approx::assert_relative_eq;
use echidna::Reverse;
use echidna::tape::{Tape, TapeGuard};
use num_traits::Float;

/// Run a single-variable reverse-mode differentiation.
fn reverse_grad(f: impl FnOnce(Reverse<f64>) -> Reverse<f64>, x_val: f64) -> f64 {
    let mut tape = Tape::new();
    let (idx, val) = tape.new_variable(x_val);
    let x = Reverse::from_tape(val, idx);
    let _guard = TapeGuard::new(&mut tape);
    let y = f(x);
    let adjoints = tape.reverse(y.index());
    adjoints[0]
}

/// Central finite difference for comparison.
fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-7;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn check_reverse_elemental(
    f_rev: impl FnOnce(Reverse<f64>) -> Reverse<f64>,
    f_f64: impl Fn(f64) -> f64,
    x: f64,
    tol: f64,
) {
    let grad = reverse_grad(f_rev, x);
    let expected = finite_diff(&f_f64, x);
    assert_relative_eq!(grad, expected, max_relative = tol);
}

// ── Arithmetic ──

#[test]
fn x_squared() {
    let grad = reverse_grad(|x| x * x, 3.0);
    assert_relative_eq!(grad, 6.0, max_relative = 1e-12);
}

#[test]
fn x_times_y() {
    let mut tape = Tape::new();
    let (xi, xv) = tape.new_variable(3.0);
    let (yi, yv) = tape.new_variable(4.0);
    let x = Reverse::from_tape(xv, xi);
    let y = Reverse::from_tape(yv, yi);
    let _guard = TapeGuard::new(&mut tape);
    let z = x * y;
    let adjoints = tape.reverse(z.index());
    assert_relative_eq!(adjoints[0], 4.0, max_relative = 1e-12); // dz/dx = y
    assert_relative_eq!(adjoints[1], 3.0, max_relative = 1e-12); // dz/dy = x
}

#[test]
fn diamond_pattern() {
    // z = f(x) + g(x), both paths use x.
    // f(x) = x², g(x) = x³
    // dz/dx = 2x + 3x²
    let grad = reverse_grad(|x| x * x + x * x * x, 2.0);
    assert_relative_eq!(grad, 4.0 + 12.0, max_relative = 1e-12);
}

#[test]
fn fan_out() {
    // y = x + x + x = 3x
    let grad = reverse_grad(|x| x + x + x, 5.0);
    assert_relative_eq!(grad, 3.0, max_relative = 1e-12);
}

#[test]
fn chain_depth() {
    // y = ((x²)²)² = x^8, dy/dx = 8*x^7
    let grad = reverse_grad(|x| {
        let a = x * x;
        let b = a * a;
        b * b
    }, 2.0);
    assert_relative_eq!(grad, 8.0 * 2.0_f64.powi(7), max_relative = 1e-10);
}

// ── Elementals ──

#[test]
fn sin() { check_reverse_elemental(|x| x.sin(), |x| x.sin(), 1.0, 1e-5); }

#[test]
fn cos() { check_reverse_elemental(|x| x.cos(), |x| x.cos(), 1.0, 1e-5); }

#[test]
fn tan() { check_reverse_elemental(|x| x.tan(), |x| x.tan(), 0.5, 1e-5); }

#[test]
fn exp() { check_reverse_elemental(|x| x.exp(), |x| x.exp(), 1.0, 1e-5); }

#[test]
fn ln() { check_reverse_elemental(|x| x.ln(), |x| x.ln(), 2.0, 1e-5); }

#[test]
fn sqrt() { check_reverse_elemental(|x| x.sqrt(), |x| x.sqrt(), 4.0, 1e-5); }

#[test]
fn recip() { check_reverse_elemental(|x| x.recip(), |x| x.recip(), 2.5, 1e-5); }

#[test]
fn powi() { check_reverse_elemental(|x| x.powi(3), |x| x.powi(3), 2.0, 1e-5); }

#[test]
fn tanh() { check_reverse_elemental(|x| x.tanh(), |x| x.tanh(), 1.0, 1e-5); }

#[test]
fn asin() { check_reverse_elemental(|x| x.asin(), |x| x.asin(), 0.5, 1e-5); }

#[test]
fn acos() { check_reverse_elemental(|x| x.acos(), |x| x.acos(), 0.5, 1e-5); }

#[test]
fn atan() { check_reverse_elemental(|x| x.atan(), |x| x.atan(), 1.0, 1e-5); }

#[test]
fn sinh() { check_reverse_elemental(|x| x.sinh(), |x| x.sinh(), 1.0, 1e-5); }

#[test]
fn cosh() { check_reverse_elemental(|x| x.cosh(), |x| x.cosh(), 1.0, 1e-5); }

#[test]
fn asinh() { check_reverse_elemental(|x| x.asinh(), |x| x.asinh(), 1.0, 1e-5); }

#[test]
fn acosh() { check_reverse_elemental(|x| x.acosh(), |x| x.acosh(), 2.0, 1e-5); }

#[test]
fn atanh() { check_reverse_elemental(|x| x.atanh(), |x| x.atanh(), 0.5, 1e-5); }

#[test]
fn exp2() { check_reverse_elemental(|x| x.exp2(), |x| x.exp2(), 1.5, 1e-5); }

#[test]
fn log2() { check_reverse_elemental(|x| x.log2(), |x| x.log2(), 2.0, 1e-5); }

#[test]
fn log10() { check_reverse_elemental(|x| x.log10(), |x| x.log10(), 2.0, 1e-5); }

#[test]
fn cbrt() { check_reverse_elemental(|x| x.cbrt(), |x| x.cbrt(), 8.0, 1e-5); }

#[test]
fn exp_m1() { check_reverse_elemental(|x| x.exp_m1(), |x| x.exp_m1(), 0.5, 1e-5); }

#[test]
fn ln_1p() { check_reverse_elemental(|x| x.ln_1p(), |x| x.ln_1p(), 0.5, 1e-5); }

#[test]
fn abs_positive() {
    let grad = reverse_grad(|x| x.abs(), 3.0);
    assert_relative_eq!(grad, 1.0, max_relative = 1e-12);
}

#[test]
fn abs_negative() {
    let grad = reverse_grad(|x| x.abs(), -3.0);
    assert_relative_eq!(grad, -1.0, max_relative = 1e-12);
}

// ── Compositions ──

#[test]
fn sin_of_exp() {
    let x_val = 0.5;
    let grad = reverse_grad(|x| x.exp().sin(), x_val);
    let expected = x_val.exp().cos() * x_val.exp();
    assert_relative_eq!(grad, expected, max_relative = 1e-10);
}

#[test]
fn complex_composition() {
    // f(x) = x * sin(x) + cos(x²)
    let x_val = 1.5;
    let grad = reverse_grad(|x| x * x.sin() + (x * x).cos(), x_val);
    let expected = x_val.sin() + x_val * x_val.cos()
        - 2.0 * x_val * (x_val * x_val).sin();
    assert_relative_eq!(grad, expected, max_relative = 1e-10);
}

// ── Constants ──

#[test]
fn constant_addition() {
    // f(x) = x + 5.0
    let grad = reverse_grad(|x| x + Reverse::constant(5.0), 3.0);
    assert_relative_eq!(grad, 1.0, max_relative = 1e-12);
}

#[test]
fn scalar_multiplication() {
    // f(x) = 3.0 * x
    let grad = reverse_grad(|x| 3.0 * x, 2.0);
    assert_relative_eq!(grad, 3.0, max_relative = 1e-12);
}

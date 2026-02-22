#![cfg(feature = "taylor")]

use approx::assert_relative_eq;
use echidna::{Scalar, Taylor, TaylorDyn, TaylorDynGuard};

// ══════════════════════════════════════════════
//  1. Known Taylor series
// ══════════════════════════════════════════════

#[test]
fn exp_taylor_series() {
    // exp(x) around x=0: coeffs = [1, 1, 1/2, 1/6, 1/24]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.exp();
    assert_relative_eq!(result.coeffs[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.5, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 1.0 / 24.0, epsilon = 1e-12);
}

#[test]
fn sin_taylor_series() {
    // sin(x) around x=0: [0, 1, 0, -1/6, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.sin();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], -1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn cos_taylor_series() {
    // cos(x) around x=0: [1, 0, -1/2, 0, 1/24]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.cos();
    assert_relative_eq!(result.coeffs[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], -0.5, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 1.0 / 24.0, epsilon = 1e-12);
}

#[test]
fn ln_1_plus_x_taylor_series() {
    // ln(1+x) around x=0: [0, 1, -1/2, 1/3, -1/4]
    let x = Taylor::<f64, 5>::variable(0.0);
    let one = Taylor::<f64, 5>::constant(1.0);
    let result = (one + x).ln();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], -0.5, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], -0.25, epsilon = 1e-12);
}

#[test]
fn geometric_series() {
    // 1/(1-x) around x=0: [1, 1, 1, 1, 1]
    let x = Taylor::<f64, 5>::variable(0.0);
    let one = Taylor::<f64, 5>::constant(1.0);
    let result = one / (one - x);
    for k in 0..5 {
        assert_relative_eq!(result.coeffs[k], 1.0, epsilon = 1e-12);
    }
}

// ══════════════════════════════════════════════
//  2. Cross-validation with Dual at K=2
// ══════════════════════════════════════════════

#[test]
fn taylor_k2_matches_dual_exp() {
    let x0 = 1.5;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.exp();
    let expected_val = x0.exp();
    let expected_deriv = x0.exp(); // d/dx exp(x) = exp(x)
    assert_relative_eq!(result.coeffs[0], expected_val, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], expected_deriv, epsilon = 1e-12);
}

#[test]
fn taylor_k2_matches_dual_sin() {
    let x0 = 0.7;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.sin();
    assert_relative_eq!(result.coeffs[0], x0.sin(), epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], x0.cos(), epsilon = 1e-12);
}

#[test]
fn taylor_k2_matches_dual_ln() {
    let x0 = 2.0;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.ln();
    assert_relative_eq!(result.coeffs[0], x0.ln(), epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0 / x0, epsilon = 1e-12);
}

#[test]
fn taylor_k2_matches_dual_sqrt() {
    let x0 = 4.0;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.sqrt();
    assert_relative_eq!(result.coeffs[0], x0.sqrt(), epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 0.5 / x0.sqrt(), epsilon = 1e-12);
}

#[test]
fn taylor_k2_matches_dual_atan() {
    let x0 = 0.5;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.atan();
    assert_relative_eq!(result.coeffs[0], x0.atan(), epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0 / (1.0 + x0 * x0), epsilon = 1e-12);
}

#[test]
fn taylor_k2_matches_dual_tanh() {
    let x0 = 0.3;
    let t = Taylor::<f64, 2>::variable(x0);
    let result = t.tanh();
    let c = x0.cosh();
    assert_relative_eq!(result.coeffs[0], x0.tanh(), epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0 / (c * c), epsilon = 1e-12);
}

// ══════════════════════════════════════════════
//  3. Arithmetic
// ══════════════════════════════════════════════

#[test]
fn cauchy_product_known_polynomials() {
    // (1 + x)(1 + x) = 1 + 2x + x²
    // Taylor coeffs: a = [1, 1, 0], b = [1, 1, 0]
    // Result: [1, 2, 1]
    let a = Taylor::<f64, 3>::new([1.0, 1.0, 0.0]);
    let b = Taylor::<f64, 3>::new([1.0, 1.0, 0.0]);
    let c = a * b;
    assert_relative_eq!(c.coeffs[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(c.coeffs[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(c.coeffs[2], 1.0, epsilon = 1e-12);
}

#[test]
fn recursive_division() {
    // (1 + 2x + x²) / (1 + x) = 1 + x
    let a = Taylor::<f64, 3>::new([1.0, 2.0, 1.0]);
    let b = Taylor::<f64, 3>::new([1.0, 1.0, 0.0]);
    let c = a / b;
    assert_relative_eq!(c.coeffs[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(c.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(c.coeffs[2], 0.0, epsilon = 1e-12);
}

// ══════════════════════════════════════════════
//  4. Scalar trait
// ══════════════════════════════════════════════

fn ad_generic_rosenbrock<T: Scalar>(x: T, y: T) -> T {
    let one = T::one();
    let hundred = T::from(100.0).unwrap();
    let t1 = one - x;
    let t2 = y - x * x;
    t1 * t1 + hundred * t2 * t2
}

#[test]
fn taylor_through_scalar_generic() {
    let x = Taylor::<f64, 2>::variable(1.0);
    let y = Taylor::<f64, 2>::constant(1.0);
    let result = ad_generic_rosenbrock(x, y);
    // At (1,1), Rosenbrock = 0
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-10);
    // df/dx at (1,1) = -2(1-x) - 400x(y-x²) = 0
    assert_relative_eq!(result.coeffs[1], 0.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  5. BytecodeTape + Taylor<F, K>
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod bytecode_tests {
    use super::*;

    fn simple_function<T: echidna::Scalar>(x: &[T]) -> T {
        x[0].exp() + x[0].sin()
    }

    #[test]
    fn forward_tangent_with_taylor() {
        let (tape, _) = echidna::record(|x| simple_function(x), &[1.0]);
        let mut buf = Vec::new();
        let x0 = 1.0_f64;
        let inputs = [Taylor::<f64, 3>::variable(x0)];
        tape.forward_tangent(&inputs, &mut buf);
        let output = buf[tape.output_index()];
        assert_relative_eq!(output.coeffs[0], x0.exp() + x0.sin(), epsilon = 1e-10);
        assert_relative_eq!(output.coeffs[1], x0.exp() + x0.cos(), epsilon = 1e-10);
        assert_relative_eq!(
            output.coeffs[2],
            (x0.exp() - x0.sin()) / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn forward_tangent_with_taylor_dyn() {
        let (tape, _) = echidna::record(|x| simple_function(x), &[1.0]);
        let mut buf = Vec::new();
        let x0 = 1.0_f64;
        let _guard = TaylorDynGuard::<f64>::new(3);
        let inputs = [TaylorDyn::variable(x0)];
        tape.forward_tangent(&inputs, &mut buf);
        let output = buf[tape.output_index()];
        let coeffs = output.coeffs();
        assert_relative_eq!(coeffs[0], x0.exp() + x0.sin(), epsilon = 1e-10);
        assert_relative_eq!(coeffs[1], x0.exp() + x0.cos(), epsilon = 1e-10);
        assert_relative_eq!(coeffs[2], (x0.exp() - x0.sin()) / 2.0, epsilon = 1e-10);
    }

    fn multivar_function<T: echidna::Scalar>(x: &[T]) -> T {
        x[0] * x[1] + x[0].sin()
    }

    #[test]
    fn forward_tangent_multivar_taylor() {
        let (tape, _) = echidna::record(|x| multivar_function(x), &[1.0, 2.0]);
        let mut buf = Vec::new();
        let inputs = [
            Taylor::<f64, 3>::variable(1.0),
            Taylor::<f64, 3>::constant(2.0),
        ];
        tape.forward_tangent(&inputs, &mut buf);
        let output = buf[tape.output_index()];
        assert_relative_eq!(output.coeffs[0], 2.0 + 1.0_f64.sin(), epsilon = 1e-10);
        assert_relative_eq!(output.coeffs[1], 2.0 + 1.0_f64.cos(), epsilon = 1e-10);
        assert_relative_eq!(output.coeffs[2], -1.0_f64.sin() / 2.0, epsilon = 1e-10);
    }
}

// ══════════════════════════════════════════════
//  6. TaylorDyn matches Taylor
// ══════════════════════════════════════════════

#[test]
fn taylor_dyn_matches_taylor_exp() {
    let x0 = 1.5;
    // Taylor<f64, 4>
    let ts = Taylor::<f64, 4>::variable(x0).exp();
    // TaylorDyn
    let _guard = TaylorDynGuard::<f64>::new(4);
    let td = TaylorDyn::<f64>::variable(x0).exp();
    let td_coeffs = td.coeffs();
    for k in 0..4 {
        assert_relative_eq!(ts.coeffs[k], td_coeffs[k], epsilon = 1e-12);
    }
}

#[test]
fn taylor_dyn_matches_taylor_sin() {
    let x0 = 0.7;
    let ts = Taylor::<f64, 4>::variable(x0).sin();
    let _guard = TaylorDynGuard::<f64>::new(4);
    let td = TaylorDyn::<f64>::variable(x0).sin();
    let td_coeffs = td.coeffs();
    for k in 0..4 {
        assert_relative_eq!(ts.coeffs[k], td_coeffs[k], epsilon = 1e-12);
    }
}

#[test]
fn taylor_dyn_matches_taylor_arithmetic() {
    let _guard = TaylorDynGuard::<f64>::new(4);
    // Same computation with both types
    let ta = Taylor::<f64, 4>::variable(2.0);
    let tb = Taylor::<f64, 4>::variable(3.0);
    let ts_result = (ta * tb + ta.exp()) / tb.ln();

    let da = TaylorDyn::<f64>::variable(2.0);
    let db = TaylorDyn::<f64>::variable(3.0);
    let td_result = (da * db + da.exp()) / db.ln();
    let td_coeffs = td_result.coeffs();
    for k in 0..4 {
        assert_relative_eq!(ts_result.coeffs[k], td_coeffs[k], epsilon = 1e-10);
    }
}

// ══════════════════════════════════════════════
//  7. Edge cases
// ══════════════════════════════════════════════

#[test]
fn taylor_k1_is_scalar() {
    // K=1: just a primal value, no derivatives
    let x = Taylor::<f64, 1>::variable(3.0);
    let result = x.exp();
    assert_relative_eq!(result.coeffs[0], 3.0_f64.exp(), epsilon = 1e-12);
}

#[test]
fn taylor_constant_propagation() {
    let c = Taylor::<f64, 4>::constant(5.0);
    let result = c.exp();
    assert_relative_eq!(result.coeffs[0], 5.0_f64.exp(), epsilon = 1e-12);
    // All higher coefficients should be zero for a constant input
    for k in 1..4 {
        assert_relative_eq!(result.coeffs[k], 0.0, epsilon = 1e-12);
    }
}

#[test]
fn taylor_discontinuous_floor() {
    let x = Taylor::<f64, 3>::variable(2.7);
    let result = x.floor();
    assert_relative_eq!(result.coeffs[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_derivative_extraction() {
    // exp(x) at x=0: derivative(0) = 1, derivative(1) = 1, derivative(2) = 1, derivative(3) = 1
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.exp();
    assert_relative_eq!(result.derivative(0), 1.0, epsilon = 1e-12); // exp(0) = 1
    assert_relative_eq!(result.derivative(1), 1.0, epsilon = 1e-12); // exp'(0) = 1
    assert_relative_eq!(result.derivative(2), 1.0, epsilon = 1e-12); // exp''(0) = 1
    assert_relative_eq!(result.derivative(3), 1.0, epsilon = 1e-12); // exp'''(0) = 1
    assert_relative_eq!(result.derivative(4), 1.0, epsilon = 1e-12); // exp''''(0) = 1
}

// ══════════════════════════════════════════════
//  8. TaylorDyn arena lifecycle
// ══════════════════════════════════════════════

#[test]
fn taylor_dyn_guard_lifecycle() {
    {
        let _guard = TaylorDynGuard::<f64>::new(3);
        let x = TaylorDyn::<f64>::variable(1.0);
        let result = x.exp();
        assert_relative_eq!(result.value(), 1.0_f64.exp(), epsilon = 1e-12);
    }
    // Guard dropped, arena torn down — creating a new guard works
    {
        let _guard = TaylorDynGuard::<f64>::new(5);
        let x = TaylorDyn::<f64>::variable(2.0);
        let result = x.ln();
        assert_relative_eq!(result.value(), 2.0_f64.ln(), epsilon = 1e-12);
        let coeffs = result.coeffs();
        assert_eq!(coeffs.len(), 5);
    }
}

#[test]
fn taylor_dyn_constant_no_arena_alloc() {
    let _guard = TaylorDynGuard::<f64>::new(3);
    let c = TaylorDyn::<f64>::constant(42.0);
    assert_eq!(c.index(), echidna::taylor_dyn::CONSTANT);
    assert_relative_eq!(c.value(), 42.0, epsilon = 1e-12);
    // Coefficients should be [42, 0, 0]
    let coeffs = c.coeffs();
    assert_relative_eq!(coeffs[0], 42.0, epsilon = 1e-12);
    assert_relative_eq!(coeffs[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(coeffs[2], 0.0, epsilon = 1e-12);
}

// ══════════════════════════════════════════════
//  9. Additional transcendentals
// ══════════════════════════════════════════════

#[test]
fn taylor_tan_series() {
    // tan(x) around x=0: [0, 1, 0, 1/3, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.tan();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_sinh_series() {
    // sinh(x) around x=0: [0, 1, 0, 1/6, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.sinh();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_cosh_series() {
    // cosh(x) around x=0: [1, 0, 1/2, 0, 1/24]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.cosh();
    assert_relative_eq!(result.coeffs[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.5, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 1.0 / 24.0, epsilon = 1e-12);
}

#[test]
fn taylor_atan_series() {
    // atan(x) around x=0: [0, 1, 0, -1/3, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.atan();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], -1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_asin_at_zero() {
    // asin(x) around x=0: [0, 1, 0, 1/6, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.asin();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], 1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_tanh_series() {
    // tanh(x) around x=0: [0, 1, 0, -1/3, 0]
    let x = Taylor::<f64, 5>::variable(0.0);
    let result = x.tanh();
    assert_relative_eq!(result.coeffs[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[3], -1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(result.coeffs[4], 0.0, epsilon = 1e-12);
}

#[test]
fn taylor_dyn_through_scalar_generic() {
    let _guard = TaylorDynGuard::<f64>::new(3);
    let x = TaylorDyn::<f64>::variable(1.0);
    let y = TaylorDyn::<f64>::constant(1.0);
    let result = ad_generic_rosenbrock(x, y);
    assert_relative_eq!(result.value(), 0.0, epsilon = 1e-10);
}

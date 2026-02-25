#![cfg(feature = "diffop")]

use approx::assert_relative_eq;
use echidna::diffop::{JetPlan, MultiIndex};
use echidna::{BReverse, BytecodeTape, Scalar};

fn record_fn(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record(f, x);
    tape
}

// ══════════════════════════════════════════════
//  Test functions (generic over Scalar)
// ══════════════════════════════════════════════

fn f_exp<T: Scalar>(x: &[T]) -> T {
    x[0].exp()
}

fn f_exp_sum<T: Scalar>(x: &[T]) -> T {
    (x[0] + x[1]).exp()
}

fn f_sin_cos<T: Scalar>(x: &[T]) -> T {
    x[0].sin() * x[1].cos()
}

fn f_exp_plus_sq<T: Scalar>(x: &[T]) -> T {
    x[0].exp() + x[1] * x[1]
}

fn f_sin<T: Scalar>(x: &[T]) -> T {
    x[0].sin()
}

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

// ══════════════════════════════════════════════
//  MultiIndex basics
// ══════════════════════════════════════════════

#[test]
fn multi_index_basics() {
    let mi = MultiIndex::new(&[2, 0, 1]);
    assert_eq!(mi.total_order(), 3);
    assert_eq!(mi.num_vars(), 3);
    assert_eq!(mi.orders(), &[2, 0, 1]);
    assert_eq!(mi.active_vars(), vec![(0, 2), (2, 1)]);
}

#[test]
fn multi_index_diagonal() {
    let mi = MultiIndex::diagonal(3, 1, 4);
    assert_eq!(mi.orders(), &[0, 4, 0]);
    assert_eq!(mi.total_order(), 4);
    assert_eq!(mi.active_vars(), vec![(1, 4)]);
}

#[test]
fn multi_index_partial() {
    let mi = MultiIndex::partial(2, 0);
    assert_eq!(mi.orders(), &[1, 0]);
    assert_eq!(mi.total_order(), 1);
}

// ══════════════════════════════════════════════
//  Partition enumeration (indirect)
// ══════════════════════════════════════════════

#[test]
fn partition_enumeration() {
    // d²/dx² with slot 2 => k=4, need 5 coefficients
    let mi = MultiIndex::diagonal(1, 0, 2);
    let plan = JetPlan::<f64>::plan(1, &[mi]);
    assert!(plan.jet_order() >= 5);
}

// ══════════════════════════════════════════════
//  Prefactor computation
// ══════════════════════════════════════════════

#[test]
fn prefactor_computation_ux() {
    // d/dx of f(x,y) = x at (1, 0) => should be 1
    let mi = MultiIndex::partial(2, 0);
    let plan = JetPlan::<f64>::plan(2, &[mi]);
    let tape = record_fn(|x| x[0], &[1.0, 0.0]);
    let result = echidna::diffop::eval_dyn(&plan, &tape, &[1.0, 0.0]);
    assert_relative_eq!(result.derivatives[0], 1.0, epsilon = 1e-10);
}

#[test]
fn prefactor_computation_uxx() {
    // d²/dx² of x² = 2
    let tape = record_fn(|x| x[0] * x[0], &[3.0]);
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[3.0], &[2]);
    assert_relative_eq!(deriv, 2.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  Diagonal derivatives: exp(x)
// ══════════════════════════════════════════════

#[test]
fn diagonal_exp_order_1_to_6() {
    // d^k(exp(x))/dx^k = exp(x) for all k
    let tape = record_fn(f_exp, &[1.0]);
    let expected = 1.0_f64.exp();

    for k in 1..=6u8 {
        let (val, deriv) = echidna::diffop::mixed_partial(&tape, &[1.0], &[k]);
        assert_relative_eq!(val, expected, epsilon = 1e-10);
        assert_relative_eq!(deriv, expected, epsilon = 1e-6, max_relative = 1e-6);
    }
}

// ══════════════════════════════════════════════
//  Diagonal derivatives: polynomial
// ══════════════════════════════════════════════

#[test]
fn diagonal_polynomial() {
    // d^4(x^4)/dx^4 = 24
    let tape = record_fn(
        |x| {
            let x0 = x[0];
            x0 * x0 * x0 * x0
        },
        &[2.0],
    );
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[2.0], &[4]);
    assert_relative_eq!(deriv, 24.0, epsilon = 1e-8);
}

#[test]
fn diagonal_polynomial_third() {
    // d^3(x^3)/dx^3 = 6
    let tape = record_fn(|x| x[0] * x[0] * x[0], &[5.0]);
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[5.0], &[3]);
    assert_relative_eq!(deriv, 6.0, epsilon = 1e-8);
}

#[test]
fn diagonal_polynomial_exceeds_degree() {
    // d^5(x^4)/dx^5 = 0
    let tape = record_fn(
        |x| {
            let x0 = x[0];
            x0 * x0 * x0 * x0
        },
        &[2.0],
    );
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[2.0], &[5]);
    assert_relative_eq!(deriv, 0.0, epsilon = 1e-6);
}

// ══════════════════════════════════════════════
//  Mixed partials: two variables
// ══════════════════════════════════════════════

#[test]
fn mixed_xy_product() {
    // d²(xy)/dxdy = 1
    let tape = record_fn(|x| x[0] * x[1], &[3.0, 4.0]);
    let (val, deriv) = echidna::diffop::mixed_partial(&tape, &[3.0, 4.0], &[1, 1]);
    assert_relative_eq!(val, 12.0, epsilon = 1e-10);
    assert_relative_eq!(deriv, 1.0, epsilon = 1e-10);
}

#[test]
fn mixed_exp_sum() {
    // d²(exp(x+y))/dxdy = exp(x+y)
    let tape = record_fn(f_exp_sum, &[1.0, 2.0]);
    let expected = 3.0_f64.exp();
    let (val, deriv) = echidna::diffop::mixed_partial(&tape, &[1.0, 2.0], &[1, 1]);
    assert_relative_eq!(val, expected, epsilon = 1e-10);
    assert_relative_eq!(deriv, expected, epsilon = 1e-6);
}

#[test]
fn mixed_trig() {
    // d²(sin(x)cos(y))/dxdy = -cos(x)sin(y)
    let tape = record_fn(f_sin_cos, &[1.0, 2.0]);
    let expected = -1.0_f64.cos() * 2.0_f64.sin();
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[1.0, 2.0], &[1, 1]);
    assert_relative_eq!(deriv, expected, epsilon = 1e-6);
}

// ══════════════════════════════════════════════
//  Mixed third-order
// ══════════════════════════════════════════════

#[test]
fn mixed_third_order() {
    // d³(x²y)/dx²dy = 2
    let tape = record_fn(|x| x[0] * x[0] * x[1], &[3.0, 4.0]);
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[3.0, 4.0], &[2, 1]);
    assert_relative_eq!(deriv, 2.0, epsilon = 1e-6);
}

#[test]
fn mixed_third_order_cubic() {
    // d³(xyz)/dxdydz = 1
    let tape = record_fn(|x| x[0] * x[1] * x[2], &[2.0, 3.0, 4.0]);
    let (_, deriv) = echidna::diffop::mixed_partial(&tape, &[2.0, 3.0, 4.0], &[1, 1, 1]);
    assert_relative_eq!(deriv, 1.0, epsilon = 1e-6);
}

// ══════════════════════════════════════════════
//  Hessian cross-validation
// ══════════════════════════════════════════════

#[test]
fn hessian_cross_validation_rosenbrock() {
    let x = [0.5, 1.0, 1.5];
    let tape = record_fn(|v| rosenbrock(v), &x);

    let (val_tape, _grad_tape, hess_tape) = tape.hessian(&x);
    let (val_diffop, grad_diffop, hess_diffop) = echidna::diffop::hessian(&tape, &x);

    assert_relative_eq!(val_tape, val_diffop, epsilon = 1e-10);

    let grad_ref = echidna::grad(|v| rosenbrock(v), &x);
    for i in 0..x.len() {
        assert_relative_eq!(grad_diffop[i], grad_ref[i], epsilon = 1e-6);
    }

    for i in 0..x.len() {
        for j in 0..x.len() {
            assert_relative_eq!(hess_diffop[i][j], hess_tape[i][j], epsilon = 1e-4);
        }
    }
}

#[test]
fn hessian_cross_validation_multiple_points() {
    let points = [
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![-0.5, 2.0],
        vec![3.0, -1.0],
        vec![0.1, 0.9],
    ];

    for x in &points {
        let tape = record_fn(|v| rosenbrock(v), x);
        let (_, _, hess_tape) = tape.hessian(x);
        let (_, _, hess_diffop) = echidna::diffop::hessian(&tape, x);

        for i in 0..x.len() {
            for j in 0..x.len() {
                assert_relative_eq!(hess_diffop[i][j], hess_tape[i][j], epsilon = 1e-4);
            }
        }
    }
}

// ══════════════════════════════════════════════
//  Plan reuse
// ══════════════════════════════════════════════

#[test]
fn plan_reuse() {
    let tape = record_fn(f_exp_plus_sq, &[0.0, 0.0]);

    let indices = vec![
        MultiIndex::diagonal(2, 0, 2), // d²/dx₀² = exp(x₀)
        MultiIndex::diagonal(2, 1, 2), // d²/dx₁² = 2
    ];
    let plan = JetPlan::plan(2, &indices);

    let points = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]];

    for x in &points {
        let result = echidna::diffop::eval_dyn(&plan, &tape, x);
        assert_relative_eq!(result.derivatives[0], x[0].exp(), epsilon = 1e-6);
        assert_relative_eq!(result.derivatives[1], 2.0, epsilon = 1e-6);
    }
}

// ══════════════════════════════════════════════
//  First-order partials
// ══════════════════════════════════════════════

#[test]
fn first_order_matches_gradient() {
    let x = [1.0, 2.0, 3.0];
    let tape = record_fn(|v| rosenbrock(v), &x);
    let grad_ref = echidna::grad(|v| rosenbrock(v), &x);

    for i in 0..x.len() {
        let mi = MultiIndex::partial(3, i);
        let plan = JetPlan::plan(3, &[mi]);
        let result = echidna::diffop::eval_dyn(&plan, &tape, &x);
        assert_relative_eq!(result.derivatives[0], grad_ref[i], epsilon = 1e-6);
    }
}

// ══════════════════════════════════════════════
//  Batch multi-index evaluation
// ══════════════════════════════════════════════

#[test]
fn batch_multi_indices() {
    // f(x,y) = x²y + y³ at (1, 2)
    let tape = record_fn(|x| x[0] * x[0] * x[1] + x[1] * x[1] * x[1], &[1.0, 2.0]);

    let indices = vec![
        MultiIndex::partial(2, 0),     // df/dx = 2xy = 4
        MultiIndex::partial(2, 1),     // df/dy = x² + 3y² = 13
        MultiIndex::diagonal(2, 0, 2), // d²f/dx² = 2y = 4
        MultiIndex::new(&[1, 1]),      // d²f/dxdy = 2x = 2
        MultiIndex::diagonal(2, 1, 2), // d²f/dy² = 6y = 12
    ];

    let plan = JetPlan::plan(2, &indices);
    let result = echidna::diffop::eval_dyn(&plan, &tape, &[1.0, 2.0]);

    assert_relative_eq!(result.value, 10.0, epsilon = 1e-10);
    assert_relative_eq!(result.derivatives[0], 4.0, epsilon = 1e-6);
    assert_relative_eq!(result.derivatives[1], 13.0, epsilon = 1e-6);
    assert_relative_eq!(result.derivatives[2], 4.0, epsilon = 1e-6);
    assert_relative_eq!(result.derivatives[3], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result.derivatives[4], 12.0, epsilon = 1e-6);
}

// ══════════════════════════════════════════════
//  Edge cases
// ══════════════════════════════════════════════

#[test]
fn single_variable_high_order() {
    // sin(x) at x=0: d¹=1, d²=0, d³=-1, d⁴=0
    let tape = record_fn(f_sin, &[0.0]);

    let (_, d1) = echidna::diffop::mixed_partial(&tape, &[0.0], &[1]);
    let (_, d2) = echidna::diffop::mixed_partial(&tape, &[0.0], &[2]);
    let (_, d3) = echidna::diffop::mixed_partial(&tape, &[0.0], &[3]);
    let (_, d4) = echidna::diffop::mixed_partial(&tape, &[0.0], &[4]);

    assert_relative_eq!(d1, 1.0, epsilon = 1e-8);
    assert_relative_eq!(d2, 0.0, epsilon = 1e-6);
    assert_relative_eq!(d3, -1.0, epsilon = 1e-6);
    assert_relative_eq!(d4, 0.0, epsilon = 1e-4);
}

#[test]
#[should_panic(expected = "must provide at least one multi-index")]
fn empty_multi_indices_panics() {
    let _ = JetPlan::<f64>::plan(2, &[]);
}

#[test]
#[should_panic(expected = "multi-index num_vars")]
fn mismatched_num_vars_panics() {
    let mi = MultiIndex::new(&[1, 0, 0]);
    let _ = JetPlan::<f64>::plan(2, &[mi]);
}

//! Tests for tape reuse: record once, evaluate many times.

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
use echidna::{record, BReverse};
use num_traits::Float;

/// Rosenbrock 2D for BReverse.
fn rosenbrock_brev(x: &[BReverse<f64>]) -> BReverse<f64> {
    let one = BReverse::constant(1.0);
    let hundred = BReverse::constant(100.0);
    let t1 = one - x[0];
    let t2 = x[1] - x[0] * x[0];
    t1 * t1 + hundred * t2 * t2
}

/// Rosenbrock 2D, plain f64 → Adept grad.
fn rosenbrock_grad_adept(x: &[f64]) -> Vec<f64> {
    echidna::grad(
        |v| {
            let one = echidna::Reverse::constant(1.0);
            let hundred = echidna::Reverse::constant(100.0);
            let t1 = one - v[0];
            let t2 = v[1] - v[0] * v[0];
            t1 * t1 + hundred * t2 * t2
        },
        x,
    )
}

#[test]
fn tape_reuse_rosenbrock_10_points() {
    let (mut tape, _) = record(rosenbrock_brev, &[1.0, 1.0]);

    let points: Vec<[f64; 2]> = (0..10)
        .map(|i| {
            let t = 0.1 * (i as f64 + 1.0);
            [t, t + 0.5]
        })
        .collect();

    for pt in &points {
        let btape_grad = tape.gradient(pt);
        let adept_grad = rosenbrock_grad_adept(pt);
        for i in 0..2 {
            assert_relative_eq!(btape_grad[i], adept_grad[i], max_relative = 1e-10);
        }
    }
}

/// Trig mix for BReverse.
fn trig_mix_brev(x: &[BReverse<f64>]) -> BReverse<f64> {
    x[0].sin() * x[1].exp() + x[0].cos() * x[1].ln()
}

fn trig_mix_grad_adept(x: &[f64]) -> Vec<f64> {
    echidna::grad(|v| v[0].sin() * v[1].exp() + v[0].cos() * v[1].ln(), x)
}

#[test]
fn tape_reuse_trig_mix_10_points() {
    let (mut tape, _) = record(trig_mix_brev, &[1.0, 2.0]);

    for i in 0..10 {
        let x = [0.5 + 0.2 * i as f64, 1.0 + 0.3 * i as f64];
        let btape_grad = tape.gradient(&x);
        let adept_grad = trig_mix_grad_adept(&x);
        for j in 0..2 {
            assert_relative_eq!(btape_grad[j], adept_grad[j], max_relative = 1e-10);
        }
    }
}

/// Exponential chain.
fn exp_chain_brev(x: &[BReverse<f64>]) -> BReverse<f64> {
    x[0].exp().sin().ln_1p()
}

fn exp_chain_grad_adept(x: &[f64]) -> Vec<f64> {
    echidna::grad(|v| v[0].exp().sin().ln_1p(), x)
}

#[test]
fn tape_reuse_exp_chain() {
    let (mut tape, _) = record(exp_chain_brev, &[0.5]);

    for i in 0..10 {
        let x = [0.1 * (i as f64 + 1.0)];
        let btape_grad = tape.gradient(&x);
        let adept_grad = exp_chain_grad_adept(&x);
        assert_relative_eq!(btape_grad[0], adept_grad[0], max_relative = 1e-10);
    }
}

#[test]
fn gradient_with_buf_matches_gradient() {
    let (mut tape, _) = record(rosenbrock_brev, &[1.0, 1.0]);
    let mut buf = Vec::new();

    for i in 0..10 {
        let x = [0.5 + 0.1 * i as f64, 1.0 + 0.2 * i as f64];
        let g1 = tape.gradient(&x);
        let g2 = tape.gradient_with_buf(&x, &mut buf);
        for j in 0..2 {
            assert_relative_eq!(g1[j], g2[j], max_relative = 1e-15);
        }
    }
}

/// Branch invalidation test: documents the expected limitation.
///
/// When a function has control flow, the tape records one path.
/// Re-evaluating at inputs that take the other path gives wrong results.
#[test]
fn branch_invalidation_documented() {
    // Record with x > 0: tape captures x*x
    let (mut tape, val) = record(
        |x| {
            if x[0] > BReverse::constant(0.0) {
                x[0] * x[0]
            } else {
                -x[0]
            }
        },
        &[2.0],
    );
    assert_relative_eq!(val, 4.0, max_relative = 1e-12);

    // Evaluate at positive x — correct.
    let g = tape.gradient(&[3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);

    // Evaluate at negative x — INCORRECT (tape still uses x*x path).
    // This is the documented limitation. The result is the derivative of x*x
    // evaluated at -1, which is 2*(-1) = -2, not the correct derivative of
    // -x which would be -1.
    let g_wrong = tape.gradient(&[-1.0]);
    // The tape gives d/dx(x^2) at x=-1 = -2, not d/dx(-x) = -1.
    assert_relative_eq!(g_wrong[0], -2.0, max_relative = 1e-12);
}

/// Output value is updated after forward().
#[test]
fn output_value_after_forward() {
    let (mut tape, initial_val) = record(|x| x[0] * x[0], &[3.0]);
    assert_relative_eq!(initial_val, 9.0, max_relative = 1e-12);
    assert_relative_eq!(tape.output_value(), 9.0, max_relative = 1e-12);

    tape.forward(&[5.0]);
    assert_relative_eq!(tape.output_value(), 25.0, max_relative = 1e-12);
}

//! Tests for Adept tape thread-local pooling.
//!
//! Verifies that reusing pooled tapes via `grad()` and `vjp()` produces
//! correct results across sequential calls, varying input sizes, and f32/f64.

use approx::assert_relative_eq;
use num_traits::Float;

#[test]
fn adept_tape_pooling_correctness() {
    // Two sequential grad() calls — second reuses the pooled tape.
    let g1 = echidna::grad(|x| x[0] * x[0] + x[1] * x[1], &[3.0_f64, 4.0]);
    assert_relative_eq!(g1[0], 6.0, max_relative = 1e-10);
    assert_relative_eq!(g1[1], 8.0, max_relative = 1e-10);

    let g2 = echidna::grad(|x| x[0] * x[1] + x[1].sin(), &[2.0_f64, 1.0]);
    assert_relative_eq!(g2[0], 1.0, max_relative = 1e-10);
    assert_relative_eq!(g2[1], 2.0 + 1.0_f64.cos(), max_relative = 1e-10);
}

#[test]
fn adept_tape_pooling_different_sizes() {
    // First call: 2 inputs.
    let g1 = echidna::grad(|x| x[0] * x[1], &[3.0_f64, 5.0]);
    assert_relative_eq!(g1[0], 5.0, max_relative = 1e-10);
    assert_relative_eq!(g1[1], 3.0, max_relative = 1e-10);

    // Second call: 4 inputs — larger tape needed, pooled tape capacity grows.
    let g2 = echidna::grad(|x| x[0] * x[1] + x[2] * x[3], &[1.0_f64, 2.0, 3.0, 4.0]);
    assert_relative_eq!(g2[0], 2.0, max_relative = 1e-10);
    assert_relative_eq!(g2[1], 1.0, max_relative = 1e-10);
    assert_relative_eq!(g2[2], 4.0, max_relative = 1e-10);
    assert_relative_eq!(g2[3], 3.0, max_relative = 1e-10);

    // Third call: back to 2 inputs — pooled tape has excess capacity, still correct.
    let g3 = echidna::grad(|x| x[0].exp() + x[1].ln(), &[1.0_f64, 2.0]);
    assert_relative_eq!(g3[0], 1.0_f64.exp(), max_relative = 1e-10);
    assert_relative_eq!(g3[1], 0.5, max_relative = 1e-10);
}

#[test]
fn adept_tape_pooling_f32() {
    // f32 pool is independent from f64 pool.
    let g1 = echidna::grad(|x| x[0] * x[0] + x[1] * x[1], &[3.0_f32, 4.0]);
    assert_relative_eq!(g1[0], 6.0_f32, max_relative = 1e-5);
    assert_relative_eq!(g1[1], 8.0_f32, max_relative = 1e-5);

    let g2 = echidna::grad(|x| x[0].sin() * x[1], &[1.0_f32, 2.0]);
    assert_relative_eq!(g2[0], 2.0 * 1.0_f32.cos(), max_relative = 1e-5);
    assert_relative_eq!(g2[1], 1.0_f32.sin(), max_relative = 1e-5);
}

#[test]
fn adept_tape_pooling_vjp() {
    // VJP also uses pooling.
    let (vals1, grad1) = echidna::vjp(
        |x| vec![x[0] * x[1], x[0] + x[1]],
        &[3.0_f64, 5.0],
        &[1.0, 1.0],
    );
    assert_relative_eq!(vals1[0], 15.0, max_relative = 1e-10);
    assert_relative_eq!(vals1[1], 8.0, max_relative = 1e-10);
    assert_relative_eq!(grad1[0], 5.0 + 1.0, max_relative = 1e-10); // d(x*y)/dx + d(x+y)/dx
    assert_relative_eq!(grad1[1], 3.0 + 1.0, max_relative = 1e-10); // d(x*y)/dy + d(x+y)/dy

    // Second call reuses pooled tape.
    let (vals2, grad2) = echidna::vjp(|x| vec![x[0] * x[0]], &[4.0_f64], &[1.0]);
    assert_relative_eq!(vals2[0], 16.0, max_relative = 1e-10);
    assert_relative_eq!(grad2[0], 8.0, max_relative = 1e-10);
}

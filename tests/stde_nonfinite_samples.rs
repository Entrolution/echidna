#![cfg(feature = "stde")]
//! Non-finite estimator samples are skipped and counted, not propagated or
//! panicked on: `num_samples` reports the finite (contributing) samples,
//! and an estimator left with zero contributing samples reports NaN rather
//! than a confident 0.

use echidna::stde::{estimate, estimate_weighted, Laplacian};
use echidna::{BReverse, BytecodeTape, Scalar};

fn record_fn(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record(f, x);
    tape
}

/// exp(x0·x1) at (1, 700): the primal e^700 ≈ 1e304 is finite, but a live
/// direction through x0 amplifies the second-order coefficient past f64
/// range (≈ e^700·700²/2 → Inf), so its sample is non-finite while a zero
/// direction's jet stays exactly (e^700, 0, 0) — a finite sample.
fn exp_prod<T: Scalar>(x: &[T]) -> T {
    (x[0] * x[1]).exp()
}

fn quadratic<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

#[test]
fn estimate_skips_nonfinite_samples_and_counts_contributing() {
    let x = [1.0_f64, 700.0];
    let tape = record_fn(exp_prod, &x);
    let live = vec![1.0_f64, 0.0]; // overflowing jet → non-finite sample
    let dead = vec![0.0_f64, 0.0]; // zero jet → sample 0
    let dirs: Vec<&[f64]> = vec![&live, &dead, &dead];

    let r = estimate(&Laplacian, &tape, &x, &dirs);
    assert_eq!(r.num_samples, 2, "one non-finite sample must be skipped");
    assert_eq!(r.estimate, 0.0, "the finite samples are exactly zero");
    assert!(r.standard_error.is_finite());
}

#[test]
fn estimate_all_samples_nonfinite_reports_nan_and_zero_count() {
    let x = [1.0_f64, 700.0];
    let tape = record_fn(exp_prod, &x);
    let live = vec![1.0_f64, 0.0];
    let dirs: Vec<&[f64]> = vec![&live, &live];

    let r = estimate(&Laplacian, &tape, &x, &dirs);
    assert_eq!(r.num_samples, 0);
    assert!(
        r.estimate.is_nan(),
        "no contributing samples must surface NaN, got {}",
        r.estimate
    );
}

#[test]
fn estimate_weighted_skips_nonfinite_samples() {
    let x = [1.0_f64, 700.0];
    let tape = record_fn(exp_prod, &x);
    let live = vec![1.0_f64, 0.0];
    let dead = vec![0.0_f64, 0.0];
    let dirs: Vec<&[f64]> = vec![&live, &dead];
    let weights = vec![1.0_f64, 1.0];

    let r = estimate_weighted(&Laplacian, &tape, &x, &dirs, &weights);
    assert_eq!(r.num_samples, 1, "the non-finite sample must be skipped");
    assert_eq!(r.estimate, 0.0);
}

#[test]
fn estimate_weighted_all_nonfinite_reports_nan() {
    let x = [1.0_f64, 700.0];
    let tape = record_fn(exp_prod, &x);
    let live = vec![1.0_f64, 0.0];
    let dirs: Vec<&[f64]> = vec![&live];
    let weights = vec![1.0_f64];

    let r = estimate_weighted(&Laplacian, &tape, &x, &dirs, &weights);
    assert_eq!(r.num_samples, 0);
    assert!(r.estimate.is_nan());
}

#[test]
fn estimate_weighted_zero_weights_still_count_as_samples() {
    // Zero-weight directions contribute nothing to the statistics but are
    // not "skipped data": with all samples finite, num_samples stays at
    // directions.len().
    let x = [1.0_f64, 2.0];
    let tape = record_fn(quadratic, &x);
    let v1 = vec![1.0_f64, 0.0];
    let v2 = vec![0.0_f64, 1.0];
    let dirs: Vec<&[f64]> = vec![&v1, &v2];
    let weights = vec![1.0_f64, 0.0];

    let r = estimate_weighted(&Laplacian, &tape, &x, &dirs, &weights);
    assert_eq!(r.num_samples, 2);
    assert!(r.estimate.is_finite());
}

#[test]
#[should_panic(expected = "non-negative")]
fn estimate_weighted_negative_weight_panics() {
    let x = [1.0_f64, 2.0];
    let tape = record_fn(quadratic, &x);
    let v1 = vec![1.0_f64, 0.0];
    let dirs: Vec<&[f64]> = vec![&v1];
    let weights = vec![-1.0_f64];
    let _ = estimate_weighted(&Laplacian, &tape, &x, &dirs, &weights);
}

#[test]
#[should_panic(expected = "k must be <= 18")]
fn diagonal_kth_order_const_k19_panics() {
    let x = [1.0_f64];
    let tape = record_fn(|v| v[0] * v[0], &x);
    let _ = echidna::stde::diagonal_kth_order_const::<f64, 20>(&tape, &x);
}

#[test]
fn diagonal_kth_order_constant_tape_recovers_primal() {
    let (tape, _) = echidna::record(|_: &[BReverse<f64>]| BReverse::constant(3.5), &[]);
    let (value, diag) = echidna::stde::diagonal_kth_order(&tape, &[], 3);
    assert_eq!(value, 3.5);
    assert!(diag.is_empty());

    let (value_c, diag_c) = echidna::stde::diagonal_kth_order_const::<f64, 4>(&tape, &[]);
    assert_eq!(value_c, 3.5);
    assert!(diag_c.is_empty());
}

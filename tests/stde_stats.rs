#![cfg(feature = "stde")]

use approx::assert_relative_eq;
use echidna::{BReverse, BytecodeTape, Scalar};

fn record_fn(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record(f, x);
    tape
}

// ══════════════════════════════════════════════
//  Test functions
// ══════════════════════════════════════════════

/// f(x,y) = x^2 + y^2
fn sum_of_squares<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

/// f(x,y,z) = x^2*y + y^3
fn cubic_mix<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] * x[1] + x[1] * x[1] * x[1]
}

/// f(x,y) = x + y (linear, all second derivatives zero)
fn linear_fn<T: Scalar>(x: &[T]) -> T {
    x[0] + x[1]
}

fn exp_plus_sin_2d<T: Scalar>(x: &[T]) -> T {
    x[0].exp() + x[1].sin()
}

// ══════════════════════════════════════════════
//  6. TaylorDyn matches const-generic
// ══════════════════════════════════════════════

#[test]
fn taylor_dyn_matches_const_generic_jet() {
    let x = [1.0, 2.0, 3.0];
    let v: Vec<f64> = vec![0.5, -1.0, 2.0];
    let tape = record_fn(cubic_mix, &x);

    let (c0, c1, c2) = echidna::stde::taylor_jet_2nd(&tape, &x, &v);
    let coeffs = echidna::stde::taylor_jet_dyn(&tape, &x, &v, 3);

    assert_relative_eq!(c0, coeffs[0], epsilon = 1e-12);
    assert_relative_eq!(c1, coeffs[1], epsilon = 1e-12);
    assert_relative_eq!(c2, coeffs[2], epsilon = 1e-12);
}

#[test]
fn laplacian_dyn_matches_const_generic() {
    let x = [1.0, 2.0, 3.0];
    let tape = record_fn(cubic_mix, &x);

    // Use Rademacher vectors for consistent comparison
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2];

    let (val_s, lap_s) = echidna::stde::laplacian(&tape, &x, &dirs);
    let (val_d, lap_d) = echidna::stde::laplacian_dyn(&tape, &x, &dirs);

    assert_relative_eq!(val_s, val_d, epsilon = 1e-12);
    assert_relative_eq!(lap_s, lap_d, epsilon = 1e-12);
}

// ══════════════════════════════════════════════
//  8. Directional derivative verification
// ══════════════════════════════════════════════

#[test]
fn basis_directional_derivatives_equal_gradient_components() {
    let x = [3.0, 4.0];
    let tape = record_fn(sum_of_squares, &x);
    let grad = echidna::grad(sum_of_squares, &x);

    let e0: Vec<f64> = vec![1.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&e0, &e1];

    let (_, first_order, _) = echidna::stde::directional_derivatives(&tape, &x, &dirs);

    assert_relative_eq!(first_order[0], grad[0], epsilon = 1e-10); // 2*3 = 6
    assert_relative_eq!(first_order[1], grad[1], epsilon = 1e-10); // 2*4 = 8
}

// ══════════════════════════════════════════════
//  10. Transcendental function (exp+sin)
// ══════════════════════════════════════════════

#[test]
fn hessian_diagonal_transcendental() {
    // f(x,y) = exp(x) + sin(y)
    // H = [[exp(x), 0], [0, -sin(y)]]
    let x = [1.0_f64, 2.0_f64];
    let tape = record_fn(exp_plus_sin_2d, &x);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &x);
    assert_relative_eq!(diag[0], 1.0_f64.exp(), epsilon = 1e-10);
    assert_relative_eq!(diag[1], -2.0_f64.sin(), epsilon = 1e-10);
}

#[test]
fn laplacian_transcendental_cross_validate() {
    let x = [1.0_f64, 2.0_f64];
    let (_, _, hess) = echidna::hessian(exp_plus_sin_2d, &x);
    let trace: f64 = hess[0][0] + hess[1][1];

    // Use all 4 Rademacher vectors for n=2 (exact)
    let tape = record_fn(exp_plus_sin_2d, &x);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];
    let (_, lap) = echidna::stde::laplacian(&tape, &x, &dirs);

    assert_relative_eq!(trace, lap, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  11. laplacian_with_stats
// ══════════════════════════════════════════════

#[test]
fn stats_matches_laplacian() {
    // laplacian_with_stats should return the same estimate as laplacian
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let (value, lap) = echidna::stde::laplacian(&tape, &[1.0, 2.0], &dirs);
    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0], &dirs);

    assert_relative_eq!(result.value, value, epsilon = 1e-10);
    assert_relative_eq!(result.estimate, lap, epsilon = 1e-10);
    assert_eq!(result.num_samples, 4);
}

#[test]
fn stats_positive_variance() {
    // For a function with off-diagonal Hessian entries, Rademacher samples
    // have nonzero variance: v^T H v differs across directions.
    // H = [[4, 2, 0], [2, 12, 0], [0, 0, 0]]
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let v3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    assert!(
        result.sample_variance > 0.0,
        "expected positive variance for off-diagonal Hessian"
    );
    assert!(result.standard_error > 0.0);
}

#[test]
fn stats_single_sample() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v: Vec<f64> = vec![1.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&v];

    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0], &dirs);
    assert_eq!(result.num_samples, 1);
    assert_relative_eq!(result.sample_variance, 0.0, epsilon = 1e-14);
    assert_relative_eq!(result.standard_error, 0.0, epsilon = 1e-14);
}

#[test]
fn stats_consistency() {
    // Verify standard_error = sqrt(sample_variance / num_samples)
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2];

    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    let expected_se = (result.sample_variance / result.num_samples as f64).sqrt();
    assert_relative_eq!(result.standard_error, expected_se, epsilon = 1e-14);
}

#[test]
fn stats_zero_variance_diagonal_only() {
    // f(x,y) = x^2 + y^2: H = [[2,0],[0,2]], diagonal-only.
    // All Rademacher samples yield v^T H v = 2+2 = 4, so variance = 0.
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0], &dirs);
    assert_relative_eq!(result.estimate, 4.0, epsilon = 1e-10);
    assert_relative_eq!(result.sample_variance, 0.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  12. laplacian_with_control
// ══════════════════════════════════════════════

#[test]
fn control_unbiased() {
    // Control variate should still give correct (unbiased) estimate.
    // Use all 8 Rademacher vectors for n=3 (exact result).
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0, 3.0]);

    let signs: [f64; 2] = [1.0, -1.0];
    let mut vecs = Vec::new();
    for &s0 in &signs {
        for &s1 in &signs {
            for &s2 in &signs {
                vecs.push(vec![s0, s1, s2]);
            }
        }
    }
    let dirs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

    let result = echidna::stde::laplacian_with_control(&tape, &[1.0, 2.0, 3.0], &dirs, &diag);
    assert_relative_eq!(result.estimate, 16.0, epsilon = 1e-10);
}

#[test]
fn control_rademacher_no_effect() {
    // For Rademacher, control variate adjustment is zero (v_j^2 = 1 always).
    // So controlled and uncontrolled estimates should be identical.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0, 3.0]);

    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let v3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let uncontrolled = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    let controlled = echidna::stde::laplacian_with_control(&tape, &[1.0, 2.0, 3.0], &dirs, &diag);

    assert_relative_eq!(controlled.estimate, uncontrolled.estimate, epsilon = 1e-10);
    assert_relative_eq!(
        controlled.sample_variance,
        uncontrolled.sample_variance,
        epsilon = 1e-10
    );
}

#[test]
fn control_gaussian_reduces_variance() {
    // For non-unit-norm entries (simulating Gaussian), control variate
    // should reduce variance. Use directions where v_j^2 != 1.
    // H = [[4, 2, 0], [2, 12, 0], [0, 0, 0]], diag = [4, 12, 0]
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0, 3.0]);

    // Directions with non-unit entries (Gaussian-like)
    let v0: Vec<f64> = vec![0.5, 1.5, 0.8];
    let v1: Vec<f64> = vec![1.2, -0.3, 1.1];
    let v2: Vec<f64> = vec![-0.7, 0.9, -1.4];
    let v3: Vec<f64> = vec![1.8, 0.2, -0.6];
    let v4: Vec<f64> = vec![-0.4, -1.1, 0.3];
    let v5: Vec<f64> = vec![0.9, 1.3, -0.2];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3, &v4, &v5];

    let uncontrolled = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    let controlled = echidna::stde::laplacian_with_control(&tape, &[1.0, 2.0, 3.0], &dirs, &diag);

    // Control variate should reduce sample variance
    assert!(
        controlled.sample_variance < uncontrolled.sample_variance,
        "expected control variate to reduce variance: controlled={} vs uncontrolled={}",
        controlled.sample_variance,
        uncontrolled.sample_variance,
    );
}

#[test]
fn control_zero_diagonal() {
    // With a zero control_diagonal, results match laplacian_with_stats exactly.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let zero_diag = vec![0.0; 3];

    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2];

    let stats = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    let controlled =
        echidna::stde::laplacian_with_control(&tape, &[1.0, 2.0, 3.0], &dirs, &zero_diag);

    assert_relative_eq!(controlled.estimate, stats.estimate, epsilon = 1e-14);
    assert_relative_eq!(
        controlled.sample_variance,
        stats.sample_variance,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        controlled.standard_error,
        stats.standard_error,
        epsilon = 1e-14
    );
}

#[test]
fn cross_validate_stats_with_hessian_trace() {
    // laplacian_with_stats estimate matches hessian() trace for all Rademacher
    let x = [1.0, 2.0];
    let (_, _, hess) = echidna::hessian(sum_of_squares, &x);
    let trace: f64 = hess[0][0] + hess[1][1];

    let tape = record_fn(sum_of_squares, &x);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::laplacian_with_stats(&tape, &x, &dirs);
    assert_relative_eq!(result.estimate, trace, epsilon = 1e-10);
}

#[test]
fn stats_linear_function() {
    // Linear function: all second derivatives zero, estimate should be 0
    let tape = record_fn(linear_fn, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];

    let result = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0], &dirs);
    assert_relative_eq!(result.value, 3.0, epsilon = 1e-10);
    assert_relative_eq!(result.estimate, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.sample_variance, 0.0, epsilon = 1e-10);
}

#[test]
fn estimator_result_fields() {
    // Verify all fields are populated correctly
    let tape = record_fn(sum_of_squares, &[3.0, 4.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];

    let result = echidna::stde::laplacian_with_stats(&tape, &[3.0, 4.0], &dirs);
    assert_relative_eq!(result.value, 25.0, epsilon = 1e-10); // 9 + 16
    assert_relative_eq!(result.estimate, 4.0, epsilon = 1e-10); // tr([[2,0],[0,2]]) = 4
    assert_eq!(result.num_samples, 2);
    // Diagonal Hessian + Rademacher => zero variance
    assert_relative_eq!(result.sample_variance, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.standard_error, 0.0, epsilon = 1e-10);
}

#[test]
#[should_panic(expected = "control_diagonal.len() must match tape.num_inputs()")]
fn control_dimension_mismatch_panics() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let wrong_diag = vec![1.0, 2.0, 3.0]; // n=2 but diag has 3 elements
    let v: Vec<f64> = vec![1.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&v];

    let _ = echidna::stde::laplacian_with_control(&tape, &[1.0, 2.0], &dirs, &wrong_diag);
}

// ══════════════════════════════════════════════
//  16. Refactored stats regression
// ══════════════════════════════════════════════

#[test]
fn refactored_stats_identical() {
    // Verify that the refactored laplacian_with_stats (now delegating to estimate)
    // produces results identical to the original estimate pipeline.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let v3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let stats = echidna::stde::laplacian_with_stats(&tape, &[1.0, 2.0, 3.0], &dirs);
    let generic =
        echidna::stde::estimate(&echidna::stde::Laplacian, &tape, &[1.0, 2.0, 3.0], &dirs);

    assert_relative_eq!(stats.value, generic.value, max_relative = 1e-15);
    assert_relative_eq!(stats.estimate, generic.estimate, max_relative = 1e-15);
    assert_relative_eq!(
        stats.sample_variance,
        generic.sample_variance,
        max_relative = 1e-15
    );
    assert_relative_eq!(
        stats.standard_error,
        generic.standard_error,
        max_relative = 1e-15
    );
    assert_eq!(stats.num_samples, generic.num_samples);
}

// ══════════════════════════════════════════════
//  Welford accumulator edge cases (PR #49 regression tests)
// ══════════════════════════════════════════════

#[test]
fn welford_nearly_identical_samples() {
    let x = [1.0, 2.0];
    let tape = record_fn(sum_of_squares, &x);
    // Directions that produce nearly identical c2 values — exercises the
    // .max(0.0) clamp on Welford variance that prevents NaN from sqrt(negative)
    let eps = f64::EPSILON;
    let v1 = vec![1.0, 0.0];
    let v2 = vec![1.0 + eps, 0.0];
    let v3 = vec![1.0 - eps, 0.0];
    let dirs: Vec<&[f64]> = vec![&v1, &v2, &v3];
    let result = echidna::stde::laplacian_with_stats(&tape, &x, &dirs);
    assert!(
        !result.standard_error.is_nan(),
        "SE should not be NaN for nearly identical samples"
    );
    assert!(result.standard_error >= 0.0, "SE should be non-negative");
}

#[test]
fn estimate_weighted_all_zero_weights() {
    let x = [1.0, 2.0];
    let tape = record_fn(sum_of_squares, &x);
    let v1 = vec![1.0, 0.0];
    let v2 = vec![0.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&v1, &v2];
    let weights = vec![0.0, 0.0];
    let result =
        echidna::stde::estimate_weighted(&echidna::stde::Laplacian, &tape, &x, &dirs, &weights);
    assert!(
        !result.estimate.is_nan(),
        "estimate should not be NaN with zero weights"
    );
    assert!(
        !result.standard_error.is_nan(),
        "SE should not be NaN with zero weights"
    );
}

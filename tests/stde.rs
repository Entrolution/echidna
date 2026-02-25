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
/// Gradient: [2x, 2y]
/// Hessian: [[2, 0], [0, 2]]
/// Laplacian: 4
/// Diagonal: [2, 2]
fn sum_of_squares<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

/// f(x,y) = x*y
/// Gradient: [y, x]
/// Hessian: [[0, 1], [1, 0]]
/// Laplacian: 0
/// Diagonal: [0, 0]
fn product<T: Scalar>(x: &[T]) -> T {
    x[0] * x[1]
}

/// f(x,y,z) = x^2*y + y^3
/// At (1, 2, 3):
/// f = 1*2 + 8 = 10
/// Gradient: [2xy, x^2+3y^2, 0] = [4, 13, 0]
/// Hessian: [[2y, 2x, 0], [2x, 6y, 0], [0, 0, 0]]
///        = [[4, 2, 0], [2, 12, 0], [0, 0, 0]]
/// Laplacian: 4 + 12 + 0 = 16
/// Diagonal: [4, 12, 0]
fn cubic_mix<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] * x[1] + x[1] * x[1] * x[1]
}

/// f(x) = x^3 (1D)
/// f''(x) = 6x, so at x=2: f''=12
fn cube_1d<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] * x[0]
}

/// f(x,y) = x + y (linear, all second derivatives zero)
fn linear_fn<T: Scalar>(x: &[T]) -> T {
    x[0] + x[1]
}

// ══════════════════════════════════════════════
//  1. Known Hessians via Rademacher vectors
// ══════════════════════════════════════════════

#[test]
fn laplacian_sum_of_squares() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    // Rademacher vectors: entries +/-1, E[vv^T] = I
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];

    let (value, lap) = echidna::stde::laplacian(&tape, &[1.0, 2.0], &dirs);
    assert_relative_eq!(value, 5.0, epsilon = 1e-10);
    assert_relative_eq!(lap, 4.0, epsilon = 1e-10);
}

#[test]
fn laplacian_product() {
    let tape = record_fn(product, &[3.0, 4.0]);
    // Rademacher: v^T [[0,1],[1,0]] v = 2*v0*v1
    // For [1,1]: 2. For [1,-1]: -2. Average of 2*c2 = average(2, -2) = 0.
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];

    let (value, lap) = echidna::stde::laplacian(&tape, &[3.0, 4.0], &dirs);
    assert_relative_eq!(value, 12.0, epsilon = 1e-10);
    assert_relative_eq!(lap, 0.0, epsilon = 1e-10);
}

#[test]
fn laplacian_cubic_mix() {
    // H = [[4, 2, 0], [2, 12, 0], [0, 0, 0]], tr(H) = 16
    // Use all 8 Rademacher vectors for exact result
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);

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

    let (value, lap) = echidna::stde::laplacian(&tape, &[1.0, 2.0, 3.0], &dirs);
    assert_relative_eq!(value, 10.0, epsilon = 1e-10);
    assert_relative_eq!(lap, 16.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  2. Hessian diagonal
// ══════════════════════════════════════════════

#[test]
fn hessian_diagonal_sum_of_squares() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let (value, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0]);
    assert_relative_eq!(value, 5.0, epsilon = 1e-10);
    assert_eq!(diag.len(), 2);
    assert_relative_eq!(diag[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(diag[1], 2.0, epsilon = 1e-10);
}

#[test]
fn hessian_diagonal_product() {
    let tape = record_fn(product, &[3.0, 4.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[3.0, 4.0]);
    assert_relative_eq!(diag[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(diag[1], 0.0, epsilon = 1e-10);
}

#[test]
fn hessian_diagonal_cubic_mix() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0, 3.0]);
    assert_eq!(diag.len(), 3);
    assert_relative_eq!(diag[0], 4.0, epsilon = 1e-10); // 2y = 4
    assert_relative_eq!(diag[1], 12.0, epsilon = 1e-10); // 6y = 12
    assert_relative_eq!(diag[2], 0.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  3. Coordinate-basis Laplacian via hessian_diagonal sum
// ══════════════════════════════════════════════

#[test]
fn coordinate_basis_laplacian_via_diagonal_sum() {
    // The exact Laplacian is the sum of the Hessian diagonal.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0, 3.0]);
    let laplacian: f64 = diag.iter().sum();
    assert_relative_eq!(laplacian, 16.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  4. Cross-validation with hessian()
// ══════════════════════════════════════════════

#[test]
fn cross_validate_with_hessian_sum_of_squares() {
    let x = [1.0, 2.0];
    let (val_h, _grad, hess) = echidna::hessian(sum_of_squares, &x);

    let trace: f64 = (0..x.len()).map(|i| hess[i][i]).sum();
    let diag_from_hess: Vec<f64> = (0..x.len()).map(|i| hess[i][i]).collect();

    // Compare Laplacian via Rademacher
    let tape = record_fn(sum_of_squares, &x);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];
    let (val_s, lap) = echidna::stde::laplacian(&tape, &x, &dirs);

    // Compare diagonal
    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &x);

    assert_relative_eq!(val_h, val_s, epsilon = 1e-10);
    assert_relative_eq!(trace, lap, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_from_hess[j], diag[j], epsilon = 1e-10);
    }
}

#[test]
fn cross_validate_with_hessian_cubic_mix() {
    let x = [1.0, 2.0, 3.0];
    let (val_h, _grad, hess) = echidna::hessian(cubic_mix, &x);

    let trace: f64 = (0..x.len()).map(|i| hess[i][i]).sum();
    let diag_from_hess: Vec<f64> = (0..x.len()).map(|i| hess[i][i]).collect();

    // Compare Laplacian via all 8 Rademacher vectors (exact)
    let tape = record_fn(cubic_mix, &x);
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
    let (val_s, lap) = echidna::stde::laplacian(&tape, &x, &dirs);

    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &x);

    assert_relative_eq!(val_h, val_s, epsilon = 1e-10);
    assert_relative_eq!(trace, lap, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_from_hess[j], diag[j], epsilon = 1e-10);
    }
}

// ══════════════════════════════════════════════
//  5. Cross-validation with hvp() / grad()
// ══════════════════════════════════════════════

#[test]
fn directional_derivative_matches_gradient() {
    // c1 for basis vector e_j should equal partial_j f
    let x = [1.0, 2.0, 3.0];
    let tape = record_fn(cubic_mix, &x);

    let grad = echidna::grad(cubic_mix, &x);

    let e0: Vec<f64> = vec![1.0, 0.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0, 0.0];
    let e2: Vec<f64> = vec![0.0, 0.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&e0, &e1, &e2];

    let (_, first_order, _) = echidna::stde::directional_derivatives(&tape, &x, &dirs);

    for j in 0..x.len() {
        assert_relative_eq!(first_order[j], grad[j], epsilon = 1e-10);
    }
}

#[test]
fn directional_derivative_arbitrary_direction() {
    // For arbitrary v, c1 should equal grad . v
    let x = [1.0, 2.0, 3.0];
    let tape = record_fn(cubic_mix, &x);

    let grad = echidna::grad(cubic_mix, &x);
    let v: Vec<f64> = vec![0.5, -1.0, 2.0];
    let expected_c1: f64 = grad.iter().zip(v.iter()).map(|(g, vi)| g * vi).sum();

    let (_, c1, _) = echidna::stde::taylor_jet_2nd(&tape, &x, &v);
    assert_relative_eq!(c1, expected_c1, epsilon = 1e-10);
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
//  7. Edge cases
// ══════════════════════════════════════════════

#[test]
fn n_equals_1() {
    let tape = record_fn(cube_1d, &[2.0]);
    let (value, diag) = echidna::stde::hessian_diagonal(&tape, &[2.0]);
    assert_relative_eq!(value, 8.0, epsilon = 1e-10);
    assert_eq!(diag.len(), 1);
    assert_relative_eq!(diag[0], 12.0, epsilon = 1e-10); // f''(2) = 6*2 = 12
}

#[test]
fn linear_function_all_zeros() {
    // f(x,y) = x + y — all second derivatives zero
    let tape = record_fn(linear_fn, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];

    let (value, lap) = echidna::stde::laplacian(&tape, &[1.0, 2.0], &dirs);
    assert_relative_eq!(value, 3.0, epsilon = 1e-10);
    assert_relative_eq!(lap, 0.0, epsilon = 1e-10);

    let (_, diag) = echidna::stde::hessian_diagonal(&tape, &[1.0, 2.0]);
    assert_relative_eq!(diag[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(diag[1], 0.0, epsilon = 1e-10);
}

#[test]
fn single_direction() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v: Vec<f64> = vec![1.0, 0.0];
    let dirs: Vec<&[f64]> = vec![&v];

    let (_, first_order, second_order) =
        echidna::stde::directional_derivatives(&tape, &[1.0, 2.0], &dirs);
    assert_eq!(first_order.len(), 1);
    assert_eq!(second_order.len(), 1);
    // c1 = grad . e0 = 2*1 = 2
    assert_relative_eq!(first_order[0], 2.0, epsilon = 1e-10);
    // c2 = e0^T H e0 / 2 = 2/2 = 1
    assert_relative_eq!(second_order[0], 1.0, epsilon = 1e-10);
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
//  9. With_buf reuse produces consistent results
// ══════════════════════════════════════════════

#[test]
fn buf_reuse_consistency() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];
    let v = [0.5, -1.0, 2.0];

    let (c0a, c1a, c2a) = echidna::stde::taylor_jet_2nd(&tape, &x, &v);

    let mut buf = Vec::new();
    let (c0b, c1b, c2b) = echidna::stde::taylor_jet_2nd_with_buf(&tape, &x, &v, &mut buf);
    // Reuse same buffer
    let (c0c, c1c, c2c) = echidna::stde::taylor_jet_2nd_with_buf(&tape, &x, &v, &mut buf);

    assert_relative_eq!(c0a, c0b, epsilon = 1e-14);
    assert_relative_eq!(c1a, c1b, epsilon = 1e-14);
    assert_relative_eq!(c2a, c2b, epsilon = 1e-14);

    assert_relative_eq!(c0b, c0c, epsilon = 1e-14);
    assert_relative_eq!(c1b, c1c, epsilon = 1e-14);
    assert_relative_eq!(c2b, c2c, epsilon = 1e-14);
}

// ══════════════════════════════════════════════
//  10. Transcendental function (exp+sin)
// ══════════════════════════════════════════════

fn exp_plus_sin_2d<T: Scalar>(x: &[T]) -> T {
    x[0].exp() + x[1].sin()
}

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
//  13. Estimator trait + generic pipeline
// ══════════════════════════════════════════════

#[test]
fn estimate_laplacian_matches_existing() {
    // estimate(&Laplacian, ...) should produce the same result as laplacian()
    // Use all 4 Rademacher vectors for n=2 (exact).
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let (value, lap) = echidna::stde::laplacian(&tape, &[1.0, 2.0], &dirs);
    let result = echidna::stde::estimate(&echidna::stde::Laplacian, &tape, &[1.0, 2.0], &dirs);

    assert_relative_eq!(result.value, value, epsilon = 1e-10);
    assert_relative_eq!(result.estimate, lap, epsilon = 1e-10);
    assert_eq!(result.num_samples, 4);
}

#[test]
fn estimate_gradient_squared_norm() {
    // f(x,y) = x^2 + y^2 at (3,4): grad = [6, 8], ||grad||^2 = 100
    // Use all 4 Rademacher vectors for exact result.
    let tape = record_fn(sum_of_squares, &[3.0, 4.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::estimate(
        &echidna::stde::GradientSquaredNorm,
        &tape,
        &[3.0, 4.0],
        &dirs,
    );

    assert_relative_eq!(result.value, 25.0, epsilon = 1e-10);
    assert_relative_eq!(result.estimate, 100.0, epsilon = 1e-10);
}

#[test]
fn estimate_weighted_uniform() {
    // Uniform weights should give the same result as unweighted estimate.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let v3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];
    let weights = vec![1.0; 4];

    let unweighted =
        echidna::stde::estimate(&echidna::stde::Laplacian, &tape, &[1.0, 2.0, 3.0], &dirs);
    let weighted = echidna::stde::estimate_weighted(
        &echidna::stde::Laplacian,
        &tape,
        &[1.0, 2.0, 3.0],
        &dirs,
        &weights,
    );

    assert_relative_eq!(weighted.estimate, unweighted.estimate, epsilon = 1e-10);
    assert_eq!(weighted.num_samples, unweighted.num_samples);
}

#[test]
fn estimate_weighted_nonuniform() {
    // Non-uniform weights should produce a valid weighted mean.
    // H = [[2, 0], [0, 2]], all samples = 4, so any weighting gives 4.
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];
    let weights = vec![3.0, 1.0];

    let result = echidna::stde::estimate_weighted(
        &echidna::stde::Laplacian,
        &tape,
        &[1.0, 2.0],
        &dirs,
        &weights,
    );

    // Both samples equal 4, so weighted mean = 4 regardless of weights
    assert_relative_eq!(result.estimate, 4.0, epsilon = 1e-10);
    assert_eq!(result.num_samples, 2);
}

// ══════════════════════════════════════════════
//  14. Hutch++ trace estimator
// ══════════════════════════════════════════════

#[test]
fn hutchpp_diagonal_matrix() {
    // H = diag(2, 2) → sketch captures everything, exact trace = 4, zero residual.
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    // Sketch with 2 orthogonal directions (full rank for n=2)
    let s0: Vec<f64> = vec![1.0, 0.0];
    let s1: Vec<f64> = vec![0.0, 1.0];
    let sketch: Vec<&[f64]> = vec![&s0, &s1];

    // One stochastic direction
    let g0: Vec<f64> = vec![1.0, 1.0];
    let stoch: Vec<&[f64]> = vec![&g0];

    let result = echidna::stde::laplacian_hutchpp(&tape, &x, &sketch, &stoch);
    assert_relative_eq!(result.value, 5.0, epsilon = 1e-10);
    assert_relative_eq!(result.estimate, 4.0, epsilon = 1e-10);
}

#[test]
fn hutchpp_known_eigenvalue_decay() {
    // H = [[4, 2, 0], [2, 12, 0], [0, 0, 0]] from cubic_mix at (1, 2, 3).
    // Eigenvalues: ~12.36, ~3.64, 0. Sketch with 1 direction should capture
    // the dominant eigenvalue, reducing variance vs standard Hutchinson.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    // Sketch with 2 directions
    let s0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let s1: Vec<f64> = vec![1.0, -1.0, 0.0];
    let sketch: Vec<&[f64]> = vec![&s0, &s1];

    // 4 stochastic directions
    let g0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let g1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let g2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let g3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let stoch: Vec<&[f64]> = vec![&g0, &g1, &g2, &g3];

    let hutchpp = echidna::stde::laplacian_hutchpp(&tape, &x, &sketch, &stoch);
    let standard = echidna::stde::laplacian_with_stats(&tape, &x, &stoch);

    // Both should estimate tr(H) = 16, Hutch++ should have lower or equal variance
    assert_relative_eq!(hutchpp.estimate, 16.0, max_relative = 0.5);
    assert!(
        hutchpp.sample_variance <= standard.sample_variance + 1e-10,
        "expected Hutch++ variance ({}) <= standard variance ({})",
        hutchpp.sample_variance,
        standard.sample_variance,
    );
}

#[test]
fn hutchpp_matches_laplacian_unbiased() {
    // With all 8 Rademacher vectors (exact for n=3), both should give exact answer.
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    // Use 2 sketch directions and 8 stochastic
    let s0: Vec<f64> = vec![1.0, 0.0, 0.0];
    let s1: Vec<f64> = vec![0.0, 1.0, 0.0];
    let sketch: Vec<&[f64]> = vec![&s0, &s1];

    let signs: [f64; 2] = [1.0, -1.0];
    let mut vecs = Vec::new();
    for &a in &signs {
        for &b in &signs {
            for &c in &signs {
                vecs.push(vec![a, b, c]);
            }
        }
    }
    let stoch: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

    let hutchpp = echidna::stde::laplacian_hutchpp(&tape, &x, &sketch, &stoch);
    let standard = echidna::stde::laplacian(&tape, &x, &stoch);

    assert_relative_eq!(hutchpp.estimate, 16.0, epsilon = 1e-8);
    assert_relative_eq!(standard.1, 16.0, epsilon = 1e-8);
}

// ══════════════════════════════════════════════
//  15. Divergence estimator
// ══════════════════════════════════════════════

fn record_multi_fn(
    f: impl FnOnce(&[BReverse<f64>]) -> Vec<BReverse<f64>>,
    x: &[f64],
) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record_multi(f, x);
    tape
}

#[test]
fn divergence_identity_field() {
    // f(x) = x → J = I → div = n
    let tape = record_multi_fn(|x| x.to_vec(), &[1.0, 2.0, 3.0]);
    let v0: Vec<f64> = vec![1.0, 1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0, 1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0, -1.0];
    let v3: Vec<f64> = vec![1.0, 1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::divergence(&tape, &[1.0, 2.0, 3.0], &dirs);
    assert_relative_eq!(result.estimate, 3.0, epsilon = 1e-10);
    assert_eq!(result.values.len(), 3);
    assert_relative_eq!(result.values[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.values[1], 2.0, epsilon = 1e-10);
    assert_relative_eq!(result.values[2], 3.0, epsilon = 1e-10);
}

#[test]
fn divergence_linear_field() {
    // f(x,y) = (2x + y, x + 3y) → J = [[2, 1], [1, 3]] → div = 5
    let tape = record_multi_fn(
        |x| {
            let two = x[0] + x[0];
            let three_y = x[1] + x[1] + x[1];
            vec![two + x[1], x[0] + three_y]
        },
        &[1.0, 1.0],
    );

    // All 4 Rademacher vectors for n=2
    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::divergence(&tape, &[1.0, 1.0], &dirs);
    assert_relative_eq!(result.estimate, 5.0, epsilon = 1e-10);
}

#[test]
fn divergence_nonlinear_field() {
    // f(x,y) = (x^2, y^2) → J = [[2x, 0], [0, 2y]] → div = 2x + 2y
    // At (3, 4): div = 14
    let tape = record_multi_fn(|x| vec![x[0] * x[0], x[1] * x[1]], &[3.0, 4.0]);

    let v0: Vec<f64> = vec![1.0, 1.0];
    let v1: Vec<f64> = vec![1.0, -1.0];
    let v2: Vec<f64> = vec![-1.0, 1.0];
    let v3: Vec<f64> = vec![-1.0, -1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1, &v2, &v3];

    let result = echidna::stde::divergence(&tape, &[3.0, 4.0], &dirs);
    assert_relative_eq!(result.estimate, 14.0, epsilon = 1e-10);
    assert_relative_eq!(result.values[0], 9.0, epsilon = 1e-10);
    assert_relative_eq!(result.values[1], 16.0, epsilon = 1e-10);
}

#[test]
#[should_panic(expected = "divergence requires num_outputs (1) == num_inputs (2)")]
fn divergence_dimension_mismatch_panics() {
    // 2 inputs, 1 output → should panic
    let tape = record_multi_fn(|x| vec![x[0]], &[1.0, 2.0]);
    let v: Vec<f64> = vec![1.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&v];

    let _ = echidna::stde::divergence(&tape, &[1.0, 2.0], &dirs);
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
//  17. Higher-order diagonal estimation
// ══════════════════════════════════════════════

/// f(x) = exp(x) — all derivatives equal exp(x).
fn exp_1d<T: Scalar>(x: &[T]) -> T {
    x[0].exp()
}

/// f(x,y) = x^4 + y^4
fn quartic<T: Scalar>(x: &[T]) -> T {
    let x0 = x[0];
    let y0 = x[1];
    x0 * x0 * x0 * x0 + y0 * y0 * y0 * y0
}

#[test]
fn diagonal_kth_order_exp() {
    // ∂^k(exp(x))/∂x^k = exp(x) for k=2,3,4,5
    let tape = record_fn(exp_1d, &[1.0]);
    let expected = 1.0_f64.exp();

    for k in 2..=5 {
        let (val, diag) = echidna::stde::diagonal_kth_order(&tape, &[1.0], k);
        assert_relative_eq!(val, expected, epsilon = 1e-10);
        assert_eq!(diag.len(), 1);
        // Tolerance relaxes with k (higher-order coefficients accumulate error)
        let tol = 10.0_f64.powi(-(12 - k as i32));
        assert_relative_eq!(diag[0], expected, epsilon = tol);
    }
}

#[test]
fn diagonal_kth_order_polynomial() {
    // f(x,y) = x^4 + y^4
    // ∂^4f/∂x^4 = 24, ∂^4f/∂y^4 = 24
    // ∂^5f/∂x^5 = 0, ∂^5f/∂y^5 = 0
    let tape = record_fn(quartic, &[2.0, 3.0]);

    let (_, diag4) = echidna::stde::diagonal_kth_order(&tape, &[2.0, 3.0], 4);
    assert_eq!(diag4.len(), 2);
    assert_relative_eq!(diag4[0], 24.0, epsilon = 1e-6);
    assert_relative_eq!(diag4[1], 24.0, epsilon = 1e-6);

    let (_, diag5) = echidna::stde::diagonal_kth_order(&tape, &[2.0, 3.0], 5);
    assert_eq!(diag5.len(), 2);
    assert_relative_eq!(diag5[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(diag5[1], 0.0, epsilon = 1e-4);
}

#[test]
fn diagonal_kth_order_matches_hessian_diagonal() {
    // k=2 case must match existing hessian_diagonal
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_hd, diag_hd) = echidna::stde::hessian_diagonal(&tape, &x);
    let (val_dk, diag_dk) = echidna::stde::diagonal_kth_order(&tape, &x, 2);

    assert_relative_eq!(val_hd, val_dk, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_hd[j], diag_dk[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_kth_order_stochastic_full_sample() {
    // Full sample (all indices) should give exact sum
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (_, diag) = echidna::stde::diagonal_kth_order(&tape, &x, 4);
    let exact_sum: f64 = diag.iter().sum(); // 24 + 24 = 48

    let all_indices: Vec<usize> = (0..x.len()).collect();
    let result = echidna::stde::diagonal_kth_order_stochastic(&tape, &x, 4, &all_indices);

    assert_relative_eq!(result.estimate, exact_sum, epsilon = 1e-4);
}

#[test]
fn diagonal_kth_order_stochastic_scaling() {
    // Subset estimate should have n/|J| scaling
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    // Sample only index 0: estimate = n/|J| * mean = 2/1 * 24 = 48
    let result = echidna::stde::diagonal_kth_order_stochastic(&tape, &x, 4, &[0]);
    assert_relative_eq!(result.estimate, 48.0, epsilon = 1e-4);
}

#[test]
#[should_panic(expected = "k must be >= 2")]
fn diagonal_kth_order_k_too_small() {
    let tape = record_fn(exp_1d, &[1.0]);
    let _ = echidna::stde::diagonal_kth_order(&tape, &[1.0], 1);
}

#[test]
#[should_panic(expected = "k must be <= 20")]
fn diagonal_kth_order_k_too_large() {
    let tape = record_fn(exp_1d, &[1.0]);
    let _ = echidna::stde::diagonal_kth_order(&tape, &[1.0], 21);
}

// ══════════════════════════════════════════════
//  18. Parabolic diffusion
// ══════════════════════════════════════════════

#[test]
fn parabolic_diffusion_identity_sigma() {
    // σ=I reduces to standard Laplacian: ½ tr(I · H · I) = ½ tr(H)
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let e0: Vec<f64> = vec![1.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0];
    let cols: Vec<&[f64]> = vec![&e0, &e1];

    let (value, diffusion) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    assert_relative_eq!(value, 5.0, epsilon = 1e-10);
    // H = [[2, 0], [0, 2]], tr(H) = 4, ½ tr(H) = 2
    assert_relative_eq!(diffusion, 2.0, epsilon = 1e-10);
}

#[test]
fn parabolic_diffusion_diagonal_sigma() {
    // σ = diag(a₁, a₂): ½ tr(σσ^T H) = ½ Σ a_i² ∂²u/∂x_i²
    // f(x,y) = x²y + y³ at (1,2): H diag = [2y, 6y] = [4, 12]
    // σ = diag(2, 3): ½(4*4 + 9*12) = ½(16 + 108) = 62
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    // σ = diag(2, 3, 0.5) — columns of diagonal matrix
    let c0: Vec<f64> = vec![2.0, 0.0, 0.0];
    let c1: Vec<f64> = vec![0.0, 3.0, 0.0];
    let c2: Vec<f64> = vec![0.0, 0.0, 0.5];
    let cols: Vec<&[f64]> = vec![&c0, &c1, &c2];

    let (_, diffusion) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    // H diag = [4, 12, 0], σ = diag(2,3,0.5)
    // ½(4*4 + 9*12 + 0.25*0) = ½(16 + 108) = 62
    assert_relative_eq!(diffusion, 62.0, epsilon = 1e-10);
}

#[test]
fn parabolic_diffusion_stochastic_unbiased() {
    // Full sample (all indices) matches exact
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let e0: Vec<f64> = vec![1.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0];
    let cols: Vec<&[f64]> = vec![&e0, &e1];

    let (_, exact) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    let result = echidna::stde::parabolic_diffusion_stochastic(&tape, &x, &cols, &[0, 1]);

    assert_relative_eq!(result.estimate, exact, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  19. Const-generic diagonal_kth_order_const
// ══════════════════════════════════════════════

#[test]
fn diagonal_const_matches_dyn_k3() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 2);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_const_matches_dyn_k4() {
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 4>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 3);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-6);
    }
}

#[test]
fn diagonal_const_matches_dyn_k5() {
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 5>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 4);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-4);
    }
}

#[test]
fn diagonal_const_matches_hessian_diagonal() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_hd, diag_hd) = echidna::stde::hessian_diagonal(&tape, &x);
    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);

    assert_relative_eq!(val_hd, val_c, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_hd[j], diag_c[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_const_with_buf_reuse() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_a, diag_a) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);

    let mut buf = Vec::new();
    let (val_b, diag_b) =
        echidna::stde::diagonal_kth_order_const_with_buf::<_, 3>(&tape, &x, &mut buf);
    let (val_c, diag_c) =
        echidna::stde::diagonal_kth_order_const_with_buf::<_, 3>(&tape, &x, &mut buf);

    assert_relative_eq!(val_a, val_b, epsilon = 1e-14);
    assert_relative_eq!(val_b, val_c, epsilon = 1e-14);
    for j in 0..x.len() {
        assert_relative_eq!(diag_a[j], diag_b[j], epsilon = 1e-14);
        assert_relative_eq!(diag_b[j], diag_c[j], epsilon = 1e-14);
    }
}

#[test]
fn diagonal_const_exp_1d() {
    // ∂^k(exp(x))/∂x^k = exp(x) for all k
    let tape = record_fn(exp_1d, &[1.0]);
    let expected = 1.0_f64.exp();

    let (val3, diag3) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &[1.0]);
    assert_relative_eq!(val3, expected, epsilon = 1e-10);
    assert_relative_eq!(diag3[0], expected, epsilon = 1e-10);

    let (_, diag4) = echidna::stde::diagonal_kth_order_const::<_, 4>(&tape, &[1.0]);
    assert_relative_eq!(diag4[0], expected, epsilon = 1e-8);

    let (_, diag5) = echidna::stde::diagonal_kth_order_const::<_, 5>(&tape, &[1.0]);
    assert_relative_eq!(diag5[0], expected, epsilon = 1e-7);
}

// ══════════════════════════════════════════════
//  20. Dense STDE for positive-definite operators
// ══════════════════════════════════════════════

#[test]
fn dense_stde_identity_is_laplacian() {
    // L=I, dense_stde_2nd should equal laplacian (same z_vectors as directions)
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    // Identity Cholesky factor
    let row0: Vec<f64> = vec![1.0, 0.0];
    let row1: Vec<f64> = vec![0.0, 1.0];
    let cholesky: Vec<&[f64]> = vec![&row0, &row1];

    // Use Rademacher vectors as z
    let z0: Vec<f64> = vec![1.0, 1.0];
    let z1: Vec<f64> = vec![1.0, -1.0];
    let z2: Vec<f64> = vec![-1.0, 1.0];
    let z3: Vec<f64> = vec![-1.0, -1.0];
    let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

    let dense_result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);

    // With L=I, v=z, so dense_stde_2nd is the same as laplacian
    let (_, lap) = echidna::stde::laplacian(&tape, &x, &z_vecs);

    assert_relative_eq!(dense_result.estimate, lap, epsilon = 1e-10);
}

#[test]
fn dense_stde_diagonal_scaling() {
    // L=diag(a), C=diag(a²): tr(C·H) = Σ a_j² ∂²u/∂x_j²
    // f(x,y) = x²+y², H=diag(2,2), a=(2,3)
    // tr(C·H) = 4*2 + 9*2 = 26
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let row0: Vec<f64> = vec![2.0, 0.0];
    let row1: Vec<f64> = vec![0.0, 3.0];
    let cholesky: Vec<&[f64]> = vec![&row0, &row1];

    // All 4 Rademacher z-vectors for exact result
    let z0: Vec<f64> = vec![1.0, 1.0];
    let z1: Vec<f64> = vec![1.0, -1.0];
    let z2: Vec<f64> = vec![-1.0, 1.0];
    let z3: Vec<f64> = vec![-1.0, -1.0];
    let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

    let result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);
    assert_relative_eq!(result.estimate, 26.0, epsilon = 1e-10);
}

#[test]
fn dense_stde_off_diagonal() {
    // L with off-diagonal entries, verify against exact tr(C·H) from full Hessian
    // f(x,y,z) = x²y + y³ at (1,2,3)
    // H = [[4, 2, 0], [2, 12, 0], [0, 0, 0]]
    // L = [[1, 0, 0], [0.5, 1, 0], [0, 0, 1]] (lower triangular)
    // C = L·L^T = [[1, 0.5, 0], [0.5, 1.25, 0], [0, 0, 1]]
    // tr(C·H) = C[0][0]*H[0][0] + C[0][1]*H[1][0] + C[1][0]*H[0][1] + C[1][1]*H[1][1]
    //         + C[0][2]*H[2][0] + ... (all zero)
    //         = 1*4 + 0.5*2 + 0.5*2 + 1.25*12 + 0 + 0 + 0 + 0 + 1*0
    //         = 4 + 1 + 1 + 15 = 21
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let row0: Vec<f64> = vec![1.0, 0.0, 0.0];
    let row1: Vec<f64> = vec![0.5, 1.0, 0.0];
    let row2: Vec<f64> = vec![0.0, 0.0, 1.0];
    let cholesky: Vec<&[f64]> = vec![&row0, &row1, &row2];

    // All 8 Rademacher z-vectors for exact result
    let signs: [f64; 2] = [1.0, -1.0];
    let mut vecs = Vec::new();
    for &s0 in &signs {
        for &s1 in &signs {
            for &s2 in &signs {
                vecs.push(vec![s0, s1, s2]);
            }
        }
    }
    let z_vecs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

    let result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);
    assert_relative_eq!(result.estimate, 21.0, epsilon = 1e-8);
}

#[test]
fn dense_stde_matches_parabolic() {
    // L=σ, dense_stde_2nd matches 2*parabolic_diffusion (½ tr(σσ^T H) = ½ * dense_stde_2nd)
    // σ = diag(2, 3) as column vectors
    // parabolic_diffusion computes ½ tr(σσ^T H)
    // dense_stde_2nd computes tr(σσ^T H) = tr(C·H)
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    // σ columns (for parabolic_diffusion)
    let c0: Vec<f64> = vec![2.0, 0.0];
    let c1: Vec<f64> = vec![0.0, 3.0];
    let cols: Vec<&[f64]> = vec![&c0, &c1];
    let (_, diffusion) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);

    // Same σ as Cholesky rows (for dense_stde_2nd)
    let row0: Vec<f64> = vec![2.0, 0.0];
    let row1: Vec<f64> = vec![0.0, 3.0];
    let cholesky: Vec<&[f64]> = vec![&row0, &row1];

    let z0: Vec<f64> = vec![1.0, 1.0];
    let z1: Vec<f64> = vec![1.0, -1.0];
    let z2: Vec<f64> = vec![-1.0, 1.0];
    let z3: Vec<f64> = vec![-1.0, -1.0];
    let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

    let dense_result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);

    // parabolic_diffusion = ½ tr(C·H), dense_stde_2nd = tr(C·H)
    assert_relative_eq!(dense_result.estimate, 2.0 * diffusion, epsilon = 1e-10);
}

#[test]
fn dense_stde_stats_populated() {
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let row0: Vec<f64> = vec![1.0, 0.0];
    let row1: Vec<f64> = vec![0.0, 1.0];
    let cholesky: Vec<&[f64]> = vec![&row0, &row1];

    let z0: Vec<f64> = vec![1.0, 1.0];
    let z1: Vec<f64> = vec![1.0, -1.0];
    let z_vecs: Vec<&[f64]> = vec![&z0, &z1];

    let result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);
    assert_eq!(result.num_samples, 2);
    assert_relative_eq!(result.value, 5.0, epsilon = 1e-10);
    // With diagonal H and identity Cholesky, variance is zero
    assert_relative_eq!(result.sample_variance, 0.0, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  21. Sparse STDE (requires stde + diffop)
// ══════════════════════════════════════════════

#[cfg(feature = "diffop")]
mod sparse_stde_tests {
    use super::*;
    use echidna::diffop::DiffOp;

    #[test]
    fn stde_sparse_full_sample_matches_exact() {
        // Full sample (all entries) should match DiffOp::eval exactly
        // Use Laplacian on sum_of_squares: exact = 4
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let op = DiffOp::<f64>::laplacian(2);
        let (_, exact) = op.eval(&tape, &x);

        let dist = op.sparse_distribution();
        let all_indices: Vec<usize> = (0..dist.len()).collect();
        let result = echidna::stde::stde_sparse(&tape, &x, &dist, &all_indices);

        assert_relative_eq!(result.estimate, exact, epsilon = 1e-6);
    }

    #[test]
    fn stde_sparse_laplacian_convergence() {
        // 1000 deterministic samples: mean should be within tolerance of exact
        let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
        let x = [1.0, 2.0, 3.0];

        let op = DiffOp::<f64>::laplacian(3);
        let (_, exact) = op.eval(&tape, &x); // 16.0

        let dist = op.sparse_distribution();

        // Generate deterministic "random" indices via simple hash
        let num_samples = 1000;
        let indices: Vec<usize> = (0..num_samples)
            .map(|i| {
                let u = ((i as u64 * 2654435761u64) % 1000) as f64 / 1000.0;
                dist.sample_index(u)
            })
            .collect();

        let result = echidna::stde::stde_sparse(&tape, &x, &dist, &indices);

        // Mean should be close to exact (within 3 standard errors)
        let error = (result.estimate - exact).abs();
        let bound = 3.0 * result.standard_error;
        assert!(
            error < bound || error < 1.0,
            "stde_sparse estimate {} too far from exact {}: error = {}, 3*SE = {}",
            result.estimate,
            exact,
            error,
            bound,
        );
    }

    #[test]
    fn stde_sparse_diagonal_4th() {
        // Biharmonic on quartic: ∂⁴(x⁴+y⁴)/∂x⁴ + ∂⁴(x⁴+y⁴)/∂y⁴ = 24 + 24 = 48
        let tape = record_fn(quartic, &[2.0, 3.0]);
        let x = [2.0, 3.0];

        let op = DiffOp::<f64>::biharmonic(2);
        let (_, exact) = op.eval(&tape, &x);
        assert_relative_eq!(exact, 48.0, epsilon = 1e-4);

        let dist = op.sparse_distribution();
        let all_indices: Vec<usize> = (0..dist.len()).collect();
        let result = echidna::stde::stde_sparse(&tape, &x, &dist, &all_indices);

        assert_relative_eq!(result.estimate, 48.0, epsilon = 1e-4);
    }

    /// f(x,y) = sin(x)*cos(y)
    fn sin_cos_2d<T: Scalar>(x: &[T]) -> T {
        x[0].sin() * x[1].cos()
    }

    #[test]
    fn stde_sparse_mixed_second_order() {
        // Test with an operator that has mixed second-order terms:
        // L = ∂²/∂x² + 2∂²/∂y² on sin(x)cos(y) at (1, 2)
        // ∂²(sin(x)cos(y))/∂x² = -sin(x)cos(y) = -sin(1)cos(2)
        // ∂²(sin(x)cos(y))/∂y² = -sin(x)cos(y) = -sin(1)cos(2)
        // L = -sin(1)cos(2) + 2*(-sin(1)cos(2)) = -3*sin(1)cos(2)
        let tape = record_fn(sin_cos_2d, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let expected = -3.0 * 1.0_f64.sin() * 2.0_f64.cos();

        let op = DiffOp::from_orders(
            2,
            &[
                (1.0, &[2, 0]), // ∂²/∂x²
                (2.0, &[0, 2]), // 2∂²/∂y²
            ],
        );
        let (_, exact) = op.eval(&tape, &x);
        assert_relative_eq!(exact, expected, epsilon = 1e-6);

        let dist = op.sparse_distribution();
        let all_indices: Vec<usize> = (0..dist.len()).collect();
        let result = echidna::stde::stde_sparse(&tape, &x, &dist, &all_indices);
        assert_relative_eq!(result.estimate, expected, epsilon = 1e-6);
    }
}

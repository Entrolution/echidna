#![cfg(feature = "stde")]

use approx::assert_relative_eq;
use echidna::{BReverse, BytecodeTape, Scalar};

fn record_fn(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record(f, x);
    tape
}

fn record_multi_fn(
    f: impl FnOnce(&[BReverse<f64>]) -> Vec<BReverse<f64>>,
    x: &[f64],
) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record_multi(f, x);
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

/// Regression: estimate_weighted SE must use effective sample size (n_eff),
/// not raw sample count. With non-uniform weights AND non-zero variance,
/// the old code (SE = sqrt(var/n)) gave a different answer than the correct
/// formula (SE = sqrt(var * w_sum2 / w_sum^2)).
#[test]
fn estimate_weighted_se_uses_effective_n() {
    // f(x,y) = x^2 + 3*y^2 has H = [[2,0],[0,6]].
    // Direction [1,0]: v^T H v = 2, sample = 2*c2 = 2
    // Direction [0,1]: v^T H v = 6, sample = 2*c2 = 6
    // Non-uniform weights [3.0, 1.0] → n_eff = (3+1)^2 / (9+1) = 1.6
    // Whereas raw n = 2.
    let tape = record_fn(
        |x: &[BReverse<f64>]| x[0] * x[0] + (x[1] * x[1]) * BReverse::constant(3.0),
        &[1.0, 2.0],
    );
    let v0: Vec<f64> = vec![1.0, 0.0];
    let v1: Vec<f64> = vec![0.0, 1.0];
    let dirs: Vec<&[f64]> = vec![&v0, &v1];
    let weights = vec![3.0, 1.0];

    let result = echidna::stde::estimate_weighted(
        &echidna::stde::Laplacian,
        &tape,
        &[1.0, 2.0],
        &dirs,
        &weights,
    );

    // Samples: 2 and 6, weights: 3 and 1
    // Weighted mean = (3*2 + 1*6) / (3+1) = 12/4 = 3.0
    assert_relative_eq!(result.estimate, 3.0, epsilon = 1e-10);
    // n_eff = (3+1)^2 / (3^2 + 1^2) = 16/10 = 1.6
    // The SE with n_eff should be sqrt(var / 1.6), NOT sqrt(var / 2)
    let n_eff = 16.0_f64 / 10.0;
    let se_with_n_eff = (result.sample_variance / n_eff).sqrt();
    assert_relative_eq!(result.standard_error, se_with_n_eff, epsilon = 1e-10);
    // And it should NOT equal the old wrong formula
    let se_wrong = (result.sample_variance / 2.0).sqrt();
    assert!(
        (result.standard_error - se_wrong).abs() > 1e-6,
        "SE should differ from wrong formula using raw n"
    );
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

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

/// f(x,y) = x^4 + y^4
fn quartic<T: Scalar>(x: &[T]) -> T {
    let x0 = x[0];
    let y0 = x[1];
    x0 * x0 * x0 * x0 + y0 * y0 * y0 * y0
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
        // Diagonal 4th-order operator on quartic: Σ_j ∂⁴/∂x_j⁴ = 24 + 24 = 48
        // Uses diagonal(n, 4) for uniform-weight stochastic test (biharmonic has
        // non-uniform weights which require importance sampling, not full enumeration).
        let tape = record_fn(quartic, &[2.0, 3.0]);
        let x = [2.0, 3.0];

        let op = DiffOp::<f64>::diagonal(2, 4);
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

// ══════════════════════════════════════════════
//  22. Indefinite Dense STDE (requires stde + nalgebra)
// ══════════════════════════════════════════════

#[cfg(feature = "nalgebra")]
mod indefinite_stde_tests {
    use super::*;

    /// PD matrix: verify result matches dense_stde_2nd with same z-vectors.
    #[test]
    fn indefinite_stde_matches_positive_definite() {
        // C = [[4, 1], [1, 3]] (positive definite)
        // Cholesky: L = [[2, 0], [0.5, sqrt(2.75)]]
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let c = nalgebra::DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);

        // Cholesky factor for comparison
        let l00 = 2.0;
        let l10 = 0.5;
        let l11 = (3.0 - 0.25_f64).sqrt(); // sqrt(2.75)
        let row0 = vec![l00, 0.0];
        let row1 = vec![l10, l11];
        let cholesky: Vec<&[f64]> = vec![&row0, &row1];

        // Use Rademacher-like z-vectors
        let z0 = vec![1.0, 1.0];
        let z1 = vec![1.0, -1.0];
        let z2 = vec![-1.0, 1.0];
        let z3 = vec![-1.0, -1.0];
        let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

        let chol_result = echidna::stde::dense_stde_2nd(&tape, &x, &cholesky, &z_vecs);
        let indef_result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);

        // H = diag(2, 2), tr(C·H) = 4*2 + 3*2 = 14
        assert_relative_eq!(chol_result.estimate, 14.0, epsilon = 1e-8);
        assert_relative_eq!(indef_result.estimate, 14.0, epsilon = 1e-8);
    }

    /// Diagonal indefinite C = diag(2, -3): verify tr(C·H) against analytical 2·H₀₀ - 3·H₁₁.
    #[test]
    fn indefinite_stde_diagonal_indefinite() {
        // f(x,y) = x² + y², H = diag(2, 2)
        // C = diag(2, -3), tr(C·H) = 2·2 + (-3)·2 = 4 - 6 = -2
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let c = nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, -3.0]);

        let z0 = vec![1.0, 1.0];
        let z1 = vec![1.0, -1.0];
        let z2 = vec![-1.0, 1.0];
        let z3 = vec![-1.0, -1.0];
        let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

        let result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);
        assert_relative_eq!(result.estimate, -2.0, epsilon = 1e-8);
    }

    /// Full symmetric indefinite C, verify against tr(C·H) computed from dense Hessian.
    #[test]
    fn indefinite_stde_full_indefinite() {
        // f(x,y,z) = x²y + y³, H at (1,2,3):
        // H = [[2y, 2x, 0], [2x, 6y, 0], [0, 0, 0]] = [[4, 2, 0], [2, 12, 0], [0, 0, 0]]
        let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
        let x = [1.0, 2.0, 3.0];

        // C = [[1, 0, -1], [0, -2, 0], [-1, 0, 3]] — indefinite
        let c = nalgebra::DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 0.0, -1.0, 0.0, -2.0, 0.0, -1.0, 0.0, 3.0],
        );

        // tr(C·H) = Σ_{ij} C_{ij} H_{ij}
        // = 1·4 + 0·2 + (-1)·0 + 0·2 + (-2)·12 + 0·0 + (-1)·0 + 0·0 + 3·0
        // = 4 - 24 = -20
        let expected = -20.0;

        // Use many z-vectors for convergence (Rademacher-like from all sign combos)
        let signs: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
            vec![1.0, -1.0, 1.0],
            vec![1.0, -1.0, -1.0],
            vec![-1.0, 1.0, 1.0],
            vec![-1.0, 1.0, -1.0],
            vec![-1.0, -1.0, 1.0],
            vec![-1.0, -1.0, -1.0],
        ];
        let z_vecs: Vec<&[f64]> = signs.iter().map(|v| v.as_slice()).collect();

        let result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);
        assert_relative_eq!(result.estimate, expected, epsilon = 1e-6);
    }

    /// All-negative eigenvalues: C = diag(-2, -3), H = diag(2, 2).
    /// tr(C·H) = -2·2 + (-3)·2 = -10.
    #[test]
    fn indefinite_stde_all_negative() {
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let c = nalgebra::DMatrix::from_row_slice(2, 2, &[-2.0, 0.0, 0.0, -3.0]);

        let z0 = vec![1.0, 1.0];
        let z1 = vec![1.0, -1.0];
        let z2 = vec![-1.0, 1.0];
        let z3 = vec![-1.0, -1.0];
        let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

        let result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);
        assert_relative_eq!(result.estimate, -10.0, epsilon = 1e-8);
    }

    /// C = 0: result should be 0.
    #[test]
    fn indefinite_stde_zero_matrix() {
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let c = nalgebra::DMatrix::zeros(2, 2);

        let z0 = vec![1.0, 1.0];
        let z1 = vec![1.0, -1.0];
        let z_vecs: Vec<&[f64]> = vec![&z0, &z1];

        let result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);
        assert_relative_eq!(result.estimate, 0.0, epsilon = 1e-10);
    }

    /// C with eigenvalue ~1e-15: verify epsilon clamping prevents sign-flip.
    #[test]
    fn indefinite_stde_near_zero_eigenvalue() {
        // C = [[1, 0], [0, 1e-15]] — the tiny eigenvalue should be clamped to zero
        let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
        let x = [1.0, 2.0];

        let c = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1e-15]);

        let z0 = vec![1.0, 1.0];
        let z1 = vec![1.0, -1.0];
        let z2 = vec![-1.0, 1.0];
        let z3 = vec![-1.0, -1.0];
        let z_vecs: Vec<&[f64]> = vec![&z0, &z1, &z2, &z3];

        // With eps_factor=1e-12, threshold = 1e-12 * 1.0 = 1e-12.
        // The eigenvalue 1e-15 < 1e-12, so it's clamped to zero.
        // Result should be tr(diag(1,0) · diag(2,2)) = 1·2 + 0·2 = 2
        let result = echidna::stde::dense_stde_2nd_indefinite(&tape, &x, &c, &z_vecs, 1e-12);
        assert_relative_eq!(result.estimate, 2.0, epsilon = 1e-8);
    }
}

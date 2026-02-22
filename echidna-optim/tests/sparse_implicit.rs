#![cfg(feature = "sparse-implicit")]

use echidna::record_multi;
use echidna_optim::linalg::lu_solve;
use echidna_optim::{
    implicit_adjoint, implicit_jacobian, implicit_tangent, implicit_adjoint_sparse,
    implicit_jacobian_sparse, implicit_tangent_sparse, SparseImplicitContext,
};

/// Simple Newton root-finder for testing: solve F(z, x) = 0 for z given fixed x.
fn newton_root_find(
    tape: &mut echidna::BytecodeTape<f64>,
    z_init: &[f64],
    x: &[f64],
    num_states: usize,
) -> Vec<f64> {
    let mut z = z_init.to_vec();
    let max_iter = 100;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let mut inputs = z.clone();
        inputs.extend_from_slice(x);

        let jac = tape.jacobian(&inputs);
        tape.forward(&inputs);
        let residual = tape.output_values();

        let norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if norm < tol {
            return z;
        }

        let f_z: Vec<Vec<f64>> = jac.iter().map(|row| row[..num_states].to_vec()).collect();
        let neg_res: Vec<f64> = residual.iter().map(|r| -r).collect();
        let delta = lu_solve(&f_z, &neg_res).expect("Singular Jacobian in Newton root-find");

        for i in 0..num_states {
            z[i] += delta[i];
        }
    }

    panic!("Newton root-finder did not converge");
}

// ============================================================
// Test 1: sparse matches dense — linear system
// ============================================================

#[test]
fn sparse_matches_dense_linear() {
    // F(z, x) = A*z + B*x where A = [[2,1],[1,3]], B = I
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            let one = x0 / x0;
            let two = one + one;
            let three = two + one;
            vec![two * z0 + z1 + x0, z0 + three * z1 + x1]
        },
        &[0.0_f64, 0.0, 1.0, 1.0],
    );

    let z_star = [-0.4, -0.2];
    let x = [1.0, 1.0];
    let num_states = 2;

    let ctx = SparseImplicitContext::new(&tape, num_states);

    let dense = implicit_jacobian(&mut tape, &z_star, &x, num_states).unwrap();
    let sparse = implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx).unwrap();

    for i in 0..num_states {
        for j in 0..x.len() {
            assert!(
                (dense[i][j] - sparse[i][j]).abs() < 1e-10,
                "dense[{}][{}] = {}, sparse[{}][{}] = {}",
                i, j, dense[i][j], i, j, sparse[i][j]
            );
        }
    }
}

// ============================================================
// Test 2: sparse matches dense — nonlinear F(z,x) = z^3 - x
// ============================================================

#[test]
fn sparse_matches_dense_nonlinear() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z * z - x]
        },
        &[2.0_f64, 8.0],
    );

    let z_star = [2.0];
    let x = [8.0];

    let ctx = SparseImplicitContext::new(&tape, 1);

    let dense = implicit_jacobian(&mut tape, &z_star, &x, 1).unwrap();
    let sparse = implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx).unwrap();

    assert!(
        (dense[0][0] - sparse[0][0]).abs() < 1e-10,
        "dense = {}, sparse = {}",
        dense[0][0],
        sparse[0][0]
    );
}

// ============================================================
// Test 3: sparse tangent matches dense tangent
// ============================================================

#[test]
fn sparse_tangent_matches_dense() {
    // F(z0, z1, x0, x1) = [z0^2 + z1 - x0, z0*z1 - x1]
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            vec![z0 * z0 + z1 - x0, z0 * z1 - x1]
        },
        &[1.0_f64, 1.0, 2.0, 1.0],
    );

    let z_star = [1.0, 1.0];
    let x = [2.0, 1.0];
    let num_states = 2;

    let ctx = SparseImplicitContext::new(&tape, num_states);

    // Test both basis directions
    for x_dot in &[[1.0, 0.0], [0.0, 1.0]] {
        let dense = implicit_tangent(&mut tape, &z_star, &x, x_dot, num_states).unwrap();
        let sparse = implicit_tangent_sparse(&mut tape, &z_star, &x, x_dot, &ctx).unwrap();

        for i in 0..num_states {
            assert!(
                (dense[i] - sparse[i]).abs() < 1e-10,
                "x_dot={:?}, dense[{}] = {}, sparse[{}] = {}",
                x_dot, i, dense[i], i, sparse[i]
            );
        }
    }
}

// ============================================================
// Test 4: sparse adjoint matches dense adjoint
// ============================================================

#[test]
fn sparse_adjoint_matches_dense() {
    // Same system as tangent test
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            vec![z0 * z0 + z1 - x0, z0 * z1 - x1]
        },
        &[1.0_f64, 1.0, 2.0, 1.0],
    );

    let z_star = [1.0, 1.0];
    let x = [2.0, 1.0];
    let num_states = 2;

    let ctx = SparseImplicitContext::new(&tape, num_states);

    // Test both basis directions
    for z_bar in &[[1.0, 0.0], [0.0, 1.0]] {
        let dense = implicit_adjoint(&mut tape, &z_star, &x, z_bar, num_states).unwrap();
        let sparse = implicit_adjoint_sparse(&mut tape, &z_star, &x, z_bar, &ctx).unwrap();

        for j in 0..x.len() {
            assert!(
                (dense[j] - sparse[j]).abs() < 1e-10,
                "z_bar={:?}, dense[{}] = {}, sparse[{}] = {}",
                z_bar, j, dense[j], j, sparse[j]
            );
        }
    }
}

// ============================================================
// Test 5: singular F_z returns None
// ============================================================

#[test]
fn sparse_singular_returns_none() {
    // F(z, x) = [z0 + z1 - x, 2*z0 + 2*z1 - 2*x]
    // F_z = [[1,1],[2,2]] which is singular
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x = v[2];
            let one = x / x;
            let two = one + one;
            vec![z0 + z1 - x, two * z0 + two * z1 - two * x]
        },
        &[0.5_f64, 0.5, 1.0],
    );

    let z_star = [0.5, 0.5];
    let x = [1.0];

    let ctx = SparseImplicitContext::new(&tape, 2);

    assert!(implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx).is_none());
    assert!(implicit_tangent_sparse(&mut tape, &z_star, &x, &[1.0], &ctx).is_none());
    assert!(implicit_adjoint_sparse(&mut tape, &z_star, &x, &[1.0, 0.0], &ctx).is_none());
}

// ============================================================
// Test 6: tridiagonal system — verifies sparsity is exploited
// ============================================================

#[test]
fn tridiagonal_system() {
    // m=20, F_z is tridiagonal: F_i = -z_{i-1} + 2*z_i - z_{i+1} - x_i
    // (discrete 1D Laplacian with parameter coupling)
    // NOTE: We avoid `v[0] / v[0]` to create constants, because that introduces
    // spurious sparsity dependencies on input 0. Instead we use z[i] + z[i]
    // which only depends on z[i].
    let m = 20;
    let (mut tape, _) = record_multi(
        |v| {
            let m = 20;
            let z = &v[..m];
            let x = &v[m..2 * m];
            let mut f = Vec::with_capacity(m);
            for i in 0..m {
                // 2*z[i] via z[i] + z[i] — no spurious dependency on other inputs
                let mut val = z[i] + z[i] - x[i];
                if i > 0 {
                    val = val - z[i - 1];
                }
                if i < m - 1 {
                    val = val - z[i + 1];
                }
                f.push(val);
            }
            f
        },
        &vec![1.0_f64; 40],
    );

    let ctx = SparseImplicitContext::new(&tape, m);

    // Verify sparsity: F_z is tridiagonal, so nnz(F_z) = 3*m - 2 << m^2
    let expected_fz_nnz = 3 * m - 2;
    assert_eq!(
        ctx.fz_nnz(),
        expected_fz_nnz,
        "F_z should be tridiagonal with nnz = {}, got {}",
        expected_fz_nnz,
        ctx.fz_nnz()
    );

    // Find z* for x = ones via Newton
    let x: Vec<f64> = vec![1.0; m];
    let z_init: Vec<f64> = vec![1.0; m];
    let z_star = newton_root_find(&mut tape, &z_init, &x, m);

    // Compare sparse vs dense
    let dense = implicit_jacobian(&mut tape, &z_star, &x, m).unwrap();
    let sparse = implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx).unwrap();

    for i in 0..m {
        for j in 0..m {
            assert!(
                (dense[i][j] - sparse[i][j]).abs() < 1e-8,
                "mismatch at [{},{}]: dense = {}, sparse = {}",
                i, j, dense[i][j], sparse[i][j]
            );
        }
    }
}

// ============================================================
// Test 7: context reuse — one context, two evaluation points
// ============================================================

#[test]
fn context_reuse() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z * z - x]
        },
        &[2.0_f64, 8.0],
    );

    let ctx = SparseImplicitContext::new(&tape, 1);

    // Point 1: x=8, z*=2
    let jac1 = implicit_jacobian_sparse(&mut tape, &[2.0], &[8.0], &ctx).unwrap();
    let expected1 = 1.0 / 12.0; // 1/(3*4)

    // Point 2: x=27, z*=3
    let jac2 = implicit_jacobian_sparse(&mut tape, &[3.0], &[27.0], &ctx).unwrap();
    let expected2 = 1.0 / 27.0; // 1/(3*9)

    assert!(
        (jac1[0][0] - expected1).abs() < 1e-10,
        "point 1: got {}, expected {}",
        jac1[0][0], expected1
    );
    assert!(
        (jac2[0][0] - expected2).abs() < 1e-10,
        "point 2: got {}, expected {}",
        jac2[0][0], expected2
    );
}

// ============================================================
// Test 8: block diagonal — two independent 2×2 blocks
// ============================================================

#[test]
fn block_diagonal() {
    // F(z0, z1, z2, z3, x0, x1) where:
    //   F0 = z0^2 - x0
    //   F1 = z1^2 - x0
    //   F2 = z2^2 - x1
    //   F3 = z3^2 - x1
    // F_z is block diagonal: [[2z0, 0, 0, 0], [0, 2z1, 0, 0], [0, 0, 2z2, 0], [0, 0, 0, 2z3]]
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let z2 = v[2];
            let z3 = v[3];
            let x0 = v[4];
            let x1 = v[5];
            vec![z0 * z0 - x0, z1 * z1 - x0, z2 * z2 - x1, z3 * z3 - x1]
        },
        &[1.0_f64, 2.0, 3.0, 4.0, 1.0, 9.0],
    );

    let z_star = [1.0, 1.0, 3.0, 3.0]; // F = [1-1, 1-1, 9-9, 9-9] = [0,0,0,0]
    let x = [1.0, 9.0];
    let m = 4;

    let ctx = SparseImplicitContext::new(&tape, m);

    // F_z is diagonal, so nnz should be exactly 4
    assert_eq!(ctx.fz_nnz(), m, "F_z should be diagonal, nnz = {}", ctx.fz_nnz());

    let dense = implicit_jacobian(&mut tape, &z_star, &x, m).unwrap();
    let sparse = implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx).unwrap();

    for i in 0..m {
        for j in 0..x.len() {
            assert!(
                (dense[i][j] - sparse[i][j]).abs() < 1e-10,
                "mismatch at [{},{}]: dense = {}, sparse = {}",
                i, j, dense[i][j], sparse[i][j]
            );
        }
    }
}

// ============================================================
// Test 9: dimension mismatch panics
// ============================================================

#[test]
#[should_panic(expected = "z_star length")]
fn dimension_mismatch_panics() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z * z - x]
        },
        &[2.0_f64, 8.0],
    );

    let ctx = SparseImplicitContext::new(&tape, 1);

    // Wrong z_star length
    let _ = implicit_jacobian_sparse(&mut tape, &[1.0, 2.0], &[8.0], &ctx);
}

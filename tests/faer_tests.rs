#![cfg(feature = "faer")]

use echidna::faer_support::{
    grad_faer, hessian_faer, hvp_faer, jacobian_faer, solve_dense_cholesky_faer,
    solve_dense_lu_faer, solve_sparse_cholesky_faer, solve_sparse_lu_faer,
    tape_hvp_faer, tape_sparse_hessian_faer,
};
use echidna::BReverse;
use faer::{Col, Mat};

fn rosenbrock_br(x: &[BReverse<f64>]) -> BReverse<f64> {
    let one = BReverse::constant(1.0);
    let hundred = BReverse::constant(100.0);
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

fn multi_br(x: &[BReverse<f64>]) -> Vec<BReverse<f64>> {
    vec![x[0] * x[1], x[1] * x[1]]
}

#[test]
fn grad_faer_rosenbrock() {
    let x = Col::from_fn(2, |i| [1.0_f64, 2.0][i]);
    let g = grad_faer(rosenbrock_br, &x);

    assert!((g[0] - (-400.0)).abs() < 1e-10, "g[0]={}", g[0]);
    assert!((g[1] - 200.0).abs() < 1e-10, "g[1]={}", g[1]);
}

#[test]
fn hessian_faer_rosenbrock() {
    let x = Col::from_fn(2, |i| [1.0_f64, 2.0][i]);
    let (val, grad, hess) = hessian_faer(rosenbrock_br, &x);

    assert!(val.is_finite());
    assert_eq!(grad.nrows(), 2);
    assert_eq!(hess.nrows(), 2);
    assert_eq!(hess.ncols(), 2);

    // Check symmetry
    assert!(
        (hess[(0, 1)] - hess[(1, 0)]).abs() < 1e-10,
        "Hessian should be symmetric"
    );
}

#[test]
fn jacobian_faer_multi() {
    let x = Col::from_fn(2, |i| [2.0_f64, 3.0][i]);
    let jac = jacobian_faer(multi_br, &x);

    assert_eq!(jac.nrows(), 2);
    assert_eq!(jac.ncols(), 2);
    assert!((jac[(0, 0)] - 3.0).abs() < 1e-10);
    assert!((jac[(0, 1)] - 2.0).abs() < 1e-10);
    assert!((jac[(1, 0)] - 0.0).abs() < 1e-10);
    assert!((jac[(1, 1)] - 6.0).abs() < 1e-10);
}

// ── Dense solver tests ──

#[test]
fn solve_dense_lu_faer_2x2() {
    // A = [[2, 1], [1, 3]], b = [5, 7] → x = [8/5, 9/5] = [1.6, 1.8]
    let a = Mat::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
    let b = Col::from_fn(2, |i| [5.0, 7.0][i]);
    let x = solve_dense_lu_faer(&a, &b);

    assert!((x[0] - 1.6).abs() < 1e-10, "x[0]={}", x[0]);
    assert!((x[1] - 1.8).abs() < 1e-10, "x[1]={}", x[1]);
}

#[test]
fn solve_dense_cholesky_faer_spd() {
    // A = [[4, 2], [2, 3]] (SPD), b = [8, 7]
    let a = Mat::from_fn(2, 2, |i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
    let b = Col::from_fn(2, |i| [8.0, 7.0][i]);
    let x = solve_dense_cholesky_faer(&a, &b).expect("should be SPD");

    // Verify A*x = b
    for i in 0..2 {
        let ax_i: f64 = (0..2).map(|j| a[(i, j)] * x[j]).sum();
        assert!((ax_i - b[i]).abs() < 1e-10, "A*x[{}]={}, b[{}]={}", i, ax_i, i, b[i]);
    }
}

#[test]
fn solve_dense_cholesky_faer_not_pd() {
    // A = [[1, 2], [2, 1]] — not positive-definite (eigenvalues: 3, -1)
    let a = Mat::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 1.0]][i][j]);
    let b = Col::from_fn(2, |i| [1.0, 1.0][i]);
    let result = solve_dense_cholesky_faer(&a, &b);

    assert!(result.is_none(), "should return None for non-PD matrix");
}

// ── Sparse solver tests ──

#[test]
fn solve_sparse_cholesky_faer_rosenbrock() {
    // Compute sparse Hessian at the minimum (1,1) where H is PD
    let x = Col::from_fn(2, |i| [1.0_f64, 1.0][i]);
    let xs: Vec<f64> = vec![1.0, 1.0];
    let (_, _, pattern, values) = echidna::sparse_hessian(rosenbrock_br, &xs);

    let b = Col::from_fn(2, |i| [1.0, 1.0][i]);
    let result = solve_sparse_cholesky_faer(&pattern, &values, &b);

    // At the minimum (1,1), H = [[802, -400], [-400, 200]] which is PD
    assert!(result.is_some(), "Hessian at minimum should be PD");
    let x_sol = result.unwrap();
    assert_eq!(x_sol.nrows(), 2);
    // Verify we got a finite answer
    assert!(x_sol[0].is_finite() && x_sol[1].is_finite());

    // Cross-check with dense LU
    let (_, _, hess) = hessian_faer(rosenbrock_br, &x);
    let x_dense = solve_dense_lu_faer(&hess, &b);
    assert!((x_sol[0] - x_dense[0]).abs() < 1e-8, "mismatch [0]");
    assert!((x_sol[1] - x_dense[1]).abs() < 1e-8, "mismatch [1]");
}

#[test]
fn solve_sparse_lu_faer_rosenbrock() {
    let xs: Vec<f64> = vec![1.0, 1.0];
    let (_, _, pattern, values) = echidna::sparse_hessian(rosenbrock_br, &xs);

    let b = Col::from_fn(2, |i| [1.0, 1.0][i]);
    let result = solve_sparse_lu_faer(&pattern, &values, &b);

    assert!(result.is_some(), "sparse LU should succeed");
    let x_sol = result.unwrap();
    assert_eq!(x_sol.nrows(), 2);

    // Cross-check with Cholesky
    let x_chol = solve_sparse_cholesky_faer(&pattern, &values, &b).unwrap();
    assert!((x_sol[0] - x_chol[0]).abs() < 1e-10, "LU vs Cholesky [0]");
    assert!((x_sol[1] - x_chol[1]).abs() < 1e-10, "LU vs Cholesky [1]");
}

// ── HVP tests ──

#[test]
fn hvp_faer_rosenbrock() {
    let x = Col::from_fn(2, |i| [1.0_f64, 2.0][i]);
    let v = Col::from_fn(2, |i| [1.0, 0.0][i]);
    let (grad, hvp) = hvp_faer(rosenbrock_br, &x, &v);

    assert_eq!(grad.nrows(), 2);
    assert_eq!(hvp.nrows(), 2);
    assert!((grad[0] - (-400.0)).abs() < 1e-10, "grad[0]={}", grad[0]);
    assert!((grad[1] - 200.0).abs() < 1e-10, "grad[1]={}", grad[1]);
    // HVP = H * [1, 0] = first column of Hessian
    // H[0,0] = 2 + 1200*x^2 - 400*y = 2 + 1200 - 800 = 402
    assert!((hvp[0] - 402.0).abs() < 1e-8, "hvp[0]={}", hvp[0]);
    // H[1,0] = -400*x = -400
    assert!((hvp[1] - (-400.0)).abs() < 1e-8, "hvp[1]={}", hvp[1]);
}

#[test]
fn tape_hvp_faer_rosenbrock() {
    let x = Col::from_fn(2, |i| [1.0_f64, 2.0][i]);
    let v = Col::from_fn(2, |i| [1.0, 0.0][i]);
    let xs: Vec<f64> = vec![1.0, 2.0];
    let (tape, _) = echidna::record(rosenbrock_br, &xs);

    let (grad_t, hvp_t) = tape_hvp_faer(&tape, &x, &v);
    let (grad_d, hvp_d) = hvp_faer(rosenbrock_br, &x, &v);

    for i in 0..2 {
        assert!(
            (grad_t[i] - grad_d[i]).abs() < 1e-12,
            "grad mismatch at {}",
            i
        );
        assert!(
            (hvp_t[i] - hvp_d[i]).abs() < 1e-12,
            "hvp mismatch at {}",
            i
        );
    }
}

#[test]
fn tape_sparse_hessian_faer_rosenbrock() {
    let x = Col::from_fn(2, |i| [1.0_f64, 2.0][i]);
    let xs: Vec<f64> = vec![1.0, 2.0];
    let (tape, _) = echidna::record(rosenbrock_br, &xs);

    let (val, grad, pattern, values) = tape_sparse_hessian_faer(&tape, &x);

    assert!(val.is_finite());
    assert_eq!(grad.nrows(), 2);
    assert!(pattern.nnz() > 0);
    assert_eq!(pattern.nnz(), values.len());
}

//! faer adapters for echidna's bytecode tape AD.
//!
//! Thin wrappers accepting `faer::Col<f64>` and returning `faer::Col<f64>` / `faer::Mat<f64>`.

use faer::{Col, Mat};

use crate::bytecode_tape::BytecodeTape;
use crate::BReverse;

/// Record a function and compute its gradient, returning a `Col<f64>`.
pub fn grad_faer(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &Col<f64>) -> Col<f64> {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (mut tape, _) = crate::api::record(f, &xs);
    let g = tape.gradient(&xs);
    Col::from_fn(g.len(), |i| g[i])
}

/// Record a function, compute value and gradient.
pub fn grad_faer_val(
    f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    x: &Col<f64>,
) -> (f64, Col<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (mut tape, val) = crate::api::record(f, &xs);
    let g = tape.gradient(&xs);
    (val, Col::from_fn(g.len(), |i| g[i]))
}

/// Record and compute the Hessian, returning `(value, gradient, hessian)`.
pub fn hessian_faer(
    f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    x: &Col<f64>,
) -> (f64, Col<f64>, Mat<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (tape, _) = crate::api::record(f, &xs);
    let (val, grad, hess) = tape.hessian(&xs);
    let n = xs.len();
    let g = Col::from_fn(n, |i| grad[i]);
    let h = Mat::from_fn(n, n, |i, j| hess[i][j]);
    (val, g, h)
}

/// Compute the Jacobian of a multi-output function, returning `Mat<f64>`.
pub fn jacobian_faer(
    f: impl FnOnce(&[BReverse<f64>]) -> Vec<BReverse<f64>>,
    x: &Col<f64>,
) -> Mat<f64> {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (mut tape, _) = crate::api::record_multi(f, &xs);
    let jac = tape.jacobian(&xs);
    let m = jac.len();
    let n = if m > 0 { jac[0].len() } else { xs.len() };
    Mat::from_fn(m, n, |i, j| jac[i][j])
}

/// Evaluate gradient on a pre-recorded tape, accepting and returning faer types.
pub fn tape_gradient_faer(tape: &mut BytecodeTape<f64>, x: &Col<f64>) -> Col<f64> {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let g = tape.gradient(&xs);
    Col::from_fn(g.len(), |i| g[i])
}

/// Evaluate Hessian on a pre-recorded tape, accepting and returning faer types.
pub fn tape_hessian_faer(tape: &BytecodeTape<f64>, x: &Col<f64>) -> (f64, Col<f64>, Mat<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (val, grad, hess) = tape.hessian(&xs);
    let n = xs.len();
    let g = Col::from_fn(n, |i| grad[i]);
    let h = Mat::from_fn(n, n, |i, j| hess[i][j]);
    (val, g, h)
}

// ══════════════════════════════════════════════
//  HVP and sparse wrappers
// ══════════════════════════════════════════════

/// Compute the Hessian-vector product, returning `(gradient, hvp)` as `Col<f64>`.
pub fn hvp_faer(
    f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    x: &Col<f64>,
    v: &Col<f64>,
) -> (Col<f64>, Col<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let vs: Vec<f64> = (0..v.nrows()).map(|i| v[i]).collect();
    let (grad, hvp) = crate::api::hvp(f, &xs, &vs);
    (
        Col::from_fn(grad.len(), |i| grad[i]),
        Col::from_fn(hvp.len(), |i| hvp[i]),
    )
}

/// Compute the Hessian-vector product on a pre-recorded tape.
pub fn tape_hvp_faer(tape: &BytecodeTape<f64>, x: &Col<f64>, v: &Col<f64>) -> (Col<f64>, Col<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let vs: Vec<f64> = (0..v.nrows()).map(|i| v[i]).collect();
    let (grad, hvp) = tape.hvp(&xs, &vs);
    (
        Col::from_fn(grad.len(), |i| grad[i]),
        Col::from_fn(hvp.len(), |i| hvp[i]),
    )
}

/// Compute the sparse Hessian, returning `(value, gradient, pattern, values)`.
pub fn sparse_hessian_faer(
    f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    x: &Col<f64>,
) -> (f64, Col<f64>, crate::sparse::SparsityPattern, Vec<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (val, grad, pattern, values) = crate::api::sparse_hessian(f, &xs);
    let g = Col::from_fn(grad.len(), |i| grad[i]);
    (val, g, pattern, values)
}

/// Compute the sparse Hessian on a pre-recorded tape.
pub fn tape_sparse_hessian_faer(
    tape: &BytecodeTape<f64>,
    x: &Col<f64>,
) -> (f64, Col<f64>, crate::sparse::SparsityPattern, Vec<f64>) {
    let xs: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
    let (val, grad, pattern, values) = tape.sparse_hessian(&xs);
    let g = Col::from_fn(grad.len(), |i| grad[i]);
    (val, g, pattern, values)
}

// ══════════════════════════════════════════════
//  Dense solver wrappers
// ══════════════════════════════════════════════

/// Solve `A * x = b` via dense partial-pivoting LU decomposition.
pub fn solve_dense_lu_faer(a: &Mat<f64>, b: &Col<f64>) -> Col<f64> {
    use faer::linalg::solvers::SpSolver;
    a.partial_piv_lu().solve(b)
}

/// Solve `A * x = b` via dense Cholesky decomposition.
///
/// Returns `None` if `A` is not positive-definite.
pub fn solve_dense_cholesky_faer(a: &Mat<f64>, b: &Col<f64>) -> Option<Col<f64>> {
    use faer::linalg::solvers::SpSolver;
    Some(a.cholesky(faer::Side::Lower).ok()?.solve(b))
}

// ══════════════════════════════════════════════
//  Sparse solver wrappers
// ══════════════════════════════════════════════

/// Convert a [`crate::SparsityPattern`] (lower-triangle COO) plus values into a full
/// symmetric `SparseColMat` suitable for faer's sparse solvers.
///
/// Returns `None` if the triplet construction fails.
pub fn sparsity_to_faer_symmetric(
    pattern: &crate::sparse::SparsityPattern,
    values: &[f64],
) -> Option<faer::sparse::SparseColMat<usize, f64>> {
    assert_eq!(pattern.nnz(), values.len());
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(pattern.nnz() * 2);
    for ((&row, &col), &v) in pattern.rows.iter().zip(&pattern.cols).zip(values) {
        let r = row as usize;
        let c = col as usize;
        triplets.push((r, c, v));
        if r != c {
            triplets.push((c, r, v));
        }
    }
    faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
        pattern.dim,
        pattern.dim,
        &triplets,
    )
    .ok()
}

/// Solve a symmetric linear system `H * x = b` via sparse Cholesky, where `H` is
/// given by a [`crate::SparsityPattern`] and its values (lower-triangle COO from `sparse_hessian`).
///
/// Returns `None` if the matrix is not positive-definite or construction fails.
pub fn solve_sparse_cholesky_faer(
    pattern: &crate::sparse::SparsityPattern,
    values: &[f64],
    b: &Col<f64>,
) -> Option<Col<f64>> {
    use faer::linalg::solvers::SpSolver;
    let mat = sparsity_to_faer_symmetric(pattern, values)?;
    let chol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mat.sp_cholesky(faer::Side::Lower)
    }))
    .ok()?
    .ok()?;
    Some(chol.solve(b))
}

/// Solve a symmetric linear system `H * x = b` via sparse LU, where `H` is
/// given by a [`crate::SparsityPattern`] and its values.
///
/// Returns `None` if the matrix is singular or construction fails.
pub fn solve_sparse_lu_faer(
    pattern: &crate::sparse::SparsityPattern,
    values: &[f64],
    b: &Col<f64>,
) -> Option<Col<f64>> {
    use faer::linalg::solvers::SpSolver;
    let mat = sparsity_to_faer_symmetric(pattern, values)?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mat.sp_lu().ok()))
        .ok()
        .flatten()
        .map(|lu| lu.solve(b))
}

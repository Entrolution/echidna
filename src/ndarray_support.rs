//! ndarray adapters for echidna's bytecode tape AD.
//!
//! Thin wrappers accepting `Array1<F>` and returning `Array1<F>` / `Array2<F>`.
//!
//! All functions here accept non-contiguous arrays (slices, transposed
//! views, stepped views). Input data is copied element-wise via
//! `iter().copied()` before being passed to the tape; the previous
//! `.as_slice().unwrap()` path panicked on any non-C-contiguous layout,
//! which was inconsistent with the faer/nalgebra adapters.

use ndarray::{Array1, Array2};

use crate::bytecode_tape::{BtapeThreadLocal, BytecodeTape};
use crate::float::Float;
use crate::BReverse;

/// Copy an `Array1<F>` into a plain `Vec<F>` regardless of memory layout.
#[inline]
fn to_vec<F: Copy>(x: &Array1<F>) -> Vec<F> {
    x.iter().copied().collect()
}

/// Record a function and compute its gradient, returning an `Array1`.
pub fn grad_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> Array1<F> {
    let xs = to_vec(x);
    let (mut tape, _) = crate::api::record(f, &xs);
    let g = tape.gradient(&xs);
    Array1::from_vec(g)
}

/// Record a function, compute value and gradient, returning `(value, Array1)`.
pub fn grad_ndarray_val<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>) {
    let xs = to_vec(x);
    let (mut tape, val) = crate::api::record(f, &xs);
    let g = tape.gradient(&xs);
    (val, Array1::from_vec(g))
}

/// Record and compute the Hessian, returning `(value, gradient, hessian)`.
pub fn hessian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, Array2<F>) {
    let xs = to_vec(x);
    let (tape, _) = crate::api::record(f, &xs);
    let (val, grad, hess) = tape.hessian(&xs);
    let n = xs.len();
    let hess_flat: Vec<F> = hess.into_iter().flat_map(|row| row.into_iter()).collect();
    (
        val,
        Array1::from_vec(grad),
        Array2::from_shape_vec((n, n), hess_flat).unwrap(),
    )
}

/// Compute the Jacobian of a multi-output function, returning `Array2<F>`.
///
/// Returns `J[i][j] = ∂f_i/∂x_j`.
pub fn jacobian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> Vec<BReverse<F>>,
    x: &Array1<F>,
) -> Array2<F> {
    let xs = to_vec(x);
    let (mut tape, _) = crate::api::record_multi(f, &xs);
    let jac = tape.jacobian(&xs);
    let m = jac.len();
    let n = if m > 0 { jac[0].len() } else { xs.len() };
    let flat: Vec<F> = jac.into_iter().flat_map(|row| row.into_iter()).collect();
    Array2::from_shape_vec((m, n), flat).unwrap()
}

/// Evaluate gradient on a pre-recorded tape, accepting and returning ndarray types.
pub fn tape_gradient_ndarray<F: Float>(tape: &mut BytecodeTape<F>, x: &Array1<F>) -> Array1<F> {
    let xs = to_vec(x);
    let g = tape.gradient(&xs);
    Array1::from_vec(g)
}

/// Evaluate Hessian on a pre-recorded tape, accepting and returning ndarray types.
#[must_use]
pub fn tape_hessian_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, Array2<F>) {
    let xs = to_vec(x);
    let (val, grad, hess) = tape.hessian(&xs);
    let n = xs.len();
    let hess_flat: Vec<F> = hess.into_iter().flat_map(|row| row.into_iter()).collect();
    (
        val,
        Array1::from_vec(grad),
        Array2::from_shape_vec((n, n), hess_flat).unwrap(),
    )
}

/// Compute the Hessian-vector product, returning `(gradient, hvp)` as `Array1`.
pub fn hvp_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
    v: &Array1<F>,
) -> (Array1<F>, Array1<F>) {
    let xs = to_vec(x);
    let vs = to_vec(v);
    let (grad, hvp) = crate::api::hvp(f, &xs, &vs);
    (Array1::from_vec(grad), Array1::from_vec(hvp))
}

/// Compute the Hessian-vector product on a pre-recorded tape, returning `(gradient, hvp)`.
#[must_use]
pub fn tape_hvp_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
    v: &Array1<F>,
) -> (Array1<F>, Array1<F>) {
    let xs = to_vec(x);
    let vs = to_vec(v);
    let (grad, hvp) = tape.hvp(&xs, &vs);
    (Array1::from_vec(grad), Array1::from_vec(hvp))
}

/// Compute the sparse Hessian, returning `(value, gradient, pattern, values)`.
///
/// The `values` array contains non-zero Hessian entries in the order defined by `pattern`.
pub fn sparse_hessian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, crate::sparse::SparsityPattern, Array1<F>) {
    let xs = to_vec(x);
    let (val, grad, pattern, values) = crate::api::sparse_hessian(f, &xs);
    (
        val,
        Array1::from_vec(grad),
        pattern,
        Array1::from_vec(values),
    )
}

/// Compute the sparse Hessian on a pre-recorded tape.
#[must_use]
pub fn tape_sparse_hessian_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, crate::sparse::SparsityPattern, Array1<F>) {
    let xs = to_vec(x);
    let (val, grad, pattern, values) = tape.sparse_hessian(&xs);
    (
        val,
        Array1::from_vec(grad),
        pattern,
        Array1::from_vec(values),
    )
}

/// Compute the sparse Jacobian, returning `(outputs, pattern, values)`.
///
/// `outputs` is `f(x)` as an `Array1<F>`. `values` contains the non-zero
/// Jacobian entries in the order defined by `pattern` (a flat vector, not
/// per-column arrays — the old "column_vectors" description was wrong
/// and the old signature discarded `outputs` entirely).
pub fn sparse_jacobian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> Vec<BReverse<F>>,
    x: &Array1<F>,
) -> (Array1<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
    let xs = to_vec(x);
    let (outputs, pattern, values) = crate::api::sparse_jacobian(f, &xs);
    (Array1::from_vec(outputs), pattern, values)
}

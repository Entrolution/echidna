//! ndarray adapters for echidna's bytecode tape AD.
//!
//! Thin wrappers accepting `Array1<F>` and returning `Array1<F>` / `Array2<F>`.

use ndarray::{Array1, Array2};

use crate::bytecode_tape::{BtapeThreadLocal, BytecodeTape};
use crate::float::Float;
#[allow(unused_imports)]
use crate::BReverse;

/// Record a function and compute its gradient, returning an `Array1`.
pub fn grad_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> Array1<F> {
    let (mut tape, _) = crate::api::record(f, x.as_slice().unwrap());
    let g = tape.gradient(x.as_slice().unwrap());
    Array1::from_vec(g)
}

/// Record a function, compute value and gradient, returning `(value, Array1)`.
pub fn grad_ndarray_val<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>) {
    let (mut tape, val) = crate::api::record(f, x.as_slice().unwrap());
    let g = tape.gradient(x.as_slice().unwrap());
    (val, Array1::from_vec(g))
}

/// Record and compute the Hessian, returning `(value, gradient, hessian)`.
pub fn hessian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, Array2<F>) {
    let xs = x.as_slice().unwrap();
    let (tape, _) = crate::api::record(f, xs);
    let (val, grad, hess) = tape.hessian(xs);
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
    let xs = x.as_slice().unwrap();
    let (mut tape, _) = crate::api::record_multi(f, xs);
    let jac = tape.jacobian(xs);
    let m = jac.len();
    let n = if m > 0 { jac[0].len() } else { xs.len() };
    let flat: Vec<F> = jac.into_iter().flat_map(|row| row.into_iter()).collect();
    Array2::from_shape_vec((m, n), flat).unwrap()
}

/// Evaluate gradient on a pre-recorded tape, accepting and returning ndarray types.
pub fn tape_gradient_ndarray<F: Float>(tape: &mut BytecodeTape<F>, x: &Array1<F>) -> Array1<F> {
    let g = tape.gradient(x.as_slice().unwrap());
    Array1::from_vec(g)
}

/// Evaluate Hessian on a pre-recorded tape, accepting and returning ndarray types.
pub fn tape_hessian_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, Array2<F>) {
    let xs = x.as_slice().unwrap();
    let (val, grad, hess) = tape.hessian(xs);
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
    let xs = x.as_slice().unwrap();
    let vs = v.as_slice().unwrap();
    let (grad, hvp) = crate::api::hvp(f, xs, vs);
    (Array1::from_vec(grad), Array1::from_vec(hvp))
}

/// Compute the Hessian-vector product on a pre-recorded tape, returning `(gradient, hvp)`.
pub fn tape_hvp_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
    v: &Array1<F>,
) -> (Array1<F>, Array1<F>) {
    let xs = x.as_slice().unwrap();
    let vs = v.as_slice().unwrap();
    let (grad, hvp) = tape.hvp(xs, vs);
    (Array1::from_vec(grad), Array1::from_vec(hvp))
}

/// Compute the sparse Hessian, returning `(value, gradient, pattern, values)`.
///
/// The `values` array contains non-zero Hessian entries in the order defined by `pattern`.
pub fn sparse_hessian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, crate::sparse::SparsityPattern, Array1<F>) {
    let xs = x.as_slice().unwrap();
    let (val, grad, pattern, values) = crate::api::sparse_hessian(f, xs);
    (
        val,
        Array1::from_vec(grad),
        pattern,
        Array1::from_vec(values),
    )
}

/// Compute the sparse Hessian on a pre-recorded tape.
pub fn tape_sparse_hessian_ndarray<F: Float>(
    tape: &BytecodeTape<F>,
    x: &Array1<F>,
) -> (F, Array1<F>, crate::sparse::SparsityPattern, Array1<F>) {
    let xs = x.as_slice().unwrap();
    let (val, grad, pattern, values) = tape.sparse_hessian(xs);
    (
        val,
        Array1::from_vec(grad),
        pattern,
        Array1::from_vec(values),
    )
}

/// Compute the sparse Jacobian, returning `(pattern, column_vectors)`.
///
/// Each `Array1` in the returned `Vec` contains one column of sparse Jacobian values.
pub fn sparse_jacobian_ndarray<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> Vec<BReverse<F>>,
    x: &Array1<F>,
) -> (crate::sparse::JacobianSparsityPattern, Vec<F>) {
    let xs = x.as_slice().unwrap();
    let (_outputs, pattern, values) = crate::api::sparse_jacobian(f, xs);
    (pattern, values)
}

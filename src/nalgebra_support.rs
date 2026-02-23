//! nalgebra adapters for echidna's bytecode tape AD.
//!
//! Thin wrappers accepting `DVector<F>` and returning `DVector<F>` / `DMatrix<F>`.

use nalgebra::{DMatrix, DVector};

use crate::bytecode_tape::{BtapeThreadLocal, BytecodeTape};
use crate::float::Float;
#[allow(unused_imports)]
use crate::BReverse;

/// Record a function and compute its gradient, returning a `DVector`.
pub fn grad_nalgebra<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> DVector<F> {
    let xs = x.as_slice();
    let (mut tape, _) = crate::api::record(f, xs);
    let g = tape.gradient(xs);
    DVector::from_vec(g)
}

/// Record a function, compute value and gradient, returning `(value, DVector)`.
pub fn grad_nalgebra_val<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> (F, DVector<F>) {
    let xs = x.as_slice();
    let (mut tape, val) = crate::api::record(f, xs);
    let g = tape.gradient(xs);
    (val, DVector::from_vec(g))
}

/// Record and compute the Hessian, returning `(value, gradient, hessian)`.
pub fn hessian_nalgebra<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> (F, DVector<F>, DMatrix<F>) {
    let xs = x.as_slice();
    let (tape, _) = crate::api::record(f, xs);
    let (val, grad, hess) = tape.hessian(xs);
    let n = xs.len();
    let hess_flat: Vec<F> = hess.into_iter().flat_map(|row| row.into_iter()).collect();
    (
        val,
        DVector::from_vec(grad),
        DMatrix::from_row_slice(n, n, &hess_flat),
    )
}

/// Compute the Jacobian of a multi-output function, returning `DMatrix<F>`.
///
/// Returns `J[i][j] = ∂f_i/∂x_j`.
pub fn jacobian_nalgebra<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> Vec<BReverse<F>>,
    x: &DVector<F>,
) -> DMatrix<F> {
    let xs = x.as_slice();
    let (mut tape, _) = crate::api::record_multi(f, xs);
    let jac = tape.jacobian(xs);
    let m = jac.len();
    let n = if m > 0 { jac[0].len() } else { xs.len() };
    let flat: Vec<F> = jac.into_iter().flat_map(|row| row.into_iter()).collect();
    DMatrix::from_row_slice(m, n, &flat)
}

/// Evaluate gradient on a pre-recorded tape, accepting and returning nalgebra types.
pub fn tape_gradient_nalgebra<F: Float>(tape: &mut BytecodeTape<F>, x: &DVector<F>) -> DVector<F> {
    let g = tape.gradient(x.as_slice());
    DVector::from_vec(g)
}

/// Evaluate Hessian on a pre-recorded tape, accepting and returning nalgebra types.
pub fn tape_hessian_nalgebra<F: Float>(
    tape: &BytecodeTape<F>,
    x: &DVector<F>,
) -> (F, DVector<F>, DMatrix<F>) {
    let xs = x.as_slice();
    let (val, grad, hess) = tape.hessian(xs);
    let n = xs.len();
    let hess_flat: Vec<F> = hess.into_iter().flat_map(|row| row.into_iter()).collect();
    (
        val,
        DVector::from_vec(grad),
        DMatrix::from_row_slice(n, n, &hess_flat),
    )
}

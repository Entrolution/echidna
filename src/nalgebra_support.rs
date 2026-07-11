//! nalgebra adapters for echidna's bytecode tape AD.
//!
//! Thin wrappers accepting `DVector<F>` and returning `DVector<F>` / `DMatrix<F>`.

use nalgebra::{DMatrix, DVector};

use crate::bytecode_tape::{BtapeThreadLocal, BytecodeTape};
use crate::float::Float;
use crate::BReverse;

/// Record a function and compute its gradient, returning a `DVector`.
pub fn grad_nalgebra<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> DVector<F> {
    let (mut tape, _) = crate::api::record(f, x.as_slice());
    tape_gradient_nalgebra(&mut tape, x)
}

/// Record a function, compute value and gradient, returning `(value, DVector)`.
pub fn grad_nalgebra_val<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> (F, DVector<F>) {
    let (mut tape, val) = crate::api::record(f, x.as_slice());
    (val, tape_gradient_nalgebra(&mut tape, x))
}

/// Record and compute the Hessian, returning `(value, gradient, hessian)`.
pub fn hessian_nalgebra<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &DVector<F>,
) -> (F, DVector<F>, DMatrix<F>) {
    let (tape, _) = crate::api::record(f, x.as_slice());
    tape_hessian_nalgebra(&tape, x)
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
    DMatrix::from_fn(m, n, |i, j| jac[i][j])
}

/// Evaluate gradient on a pre-recorded tape, accepting and returning nalgebra types.
pub fn tape_gradient_nalgebra<F: Float>(tape: &mut BytecodeTape<F>, x: &DVector<F>) -> DVector<F> {
    let g = tape.gradient(x.as_slice());
    DVector::from_vec(g)
}

/// Evaluate Hessian on a pre-recorded tape, accepting and returning nalgebra types.
#[must_use]
pub fn tape_hessian_nalgebra<F: Float>(
    tape: &BytecodeTape<F>,
    x: &DVector<F>,
) -> (F, DVector<F>, DMatrix<F>) {
    let xs = x.as_slice();
    let (val, grad, hess) = tape.hessian(xs);
    let n = xs.len();
    (
        val,
        DVector::from_vec(grad),
        DMatrix::from_fn(n, n, |i, j| hess[i][j]),
    )
}

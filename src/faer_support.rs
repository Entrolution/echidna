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

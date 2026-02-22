#![cfg(feature = "faer")]

use echidna::faer_support::{grad_faer, hessian_faer, jacobian_faer};
use echidna::BReverse;
use faer::Col;

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

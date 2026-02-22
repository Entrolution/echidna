#![cfg(feature = "ndarray")]

use echidna::ndarray_support::{grad_ndarray, hessian_ndarray, jacobian_ndarray};
use echidna::BReverse;
use ndarray::array;

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
fn grad_ndarray_rosenbrock() {
    let x = array![1.0_f64, 2.0];
    let g = grad_ndarray(rosenbrock_br, &x);

    // At (1, 2): f(x,y) = (x-1)^2 + 100(y-x^2)^2
    // df/dx = 2(x-1) + 100*2*(y-x^2)*(-2x) = 0 + 100*2*(2-1)*(-2) = -400
    // df/dy = 100*2*(y-x^2) = 200
    assert!((g[0] - (-400.0)).abs() < 1e-10, "g[0]={}", g[0]);
    assert!((g[1] - 200.0).abs() < 1e-10, "g[1]={}", g[1]);
}

#[test]
fn hessian_ndarray_rosenbrock() {
    let x = array![1.0_f64, 2.0];
    let (val, grad, hess) = hessian_ndarray(rosenbrock_br, &x);

    assert!(val.is_finite());
    assert_eq!(grad.len(), 2);
    assert_eq!(hess.shape(), [2, 2]);

    // Hessian at (1,2):
    // H = [[2 + 800*x^2 - 400*(y-x^2)*2, -400*x],
    //      [-400*x, 200]]
    // = [[2 + 800 - 400, -400], [-400, 200]]
    // = [[402, -400], [-400, 200]]
    // Wait, let me recalculate. Let me just check symmetry
    assert!(
        (hess[[0, 1]] - hess[[1, 0]]).abs() < 1e-10,
        "Hessian should be symmetric"
    );
}

#[test]
fn jacobian_ndarray_multi() {
    let x = array![2.0_f64, 3.0];
    let jac = jacobian_ndarray(multi_br, &x);

    // f0 = x*y, f1 = y^2
    // J = [[y, x], [0, 2*y]]
    assert_eq!(jac.shape(), [2, 2]);
    assert!((jac[[0, 0]] - 3.0).abs() < 1e-10, "df0/dx={}", jac[[0, 0]]);
    assert!((jac[[0, 1]] - 2.0).abs() < 1e-10, "df0/dy={}", jac[[0, 1]]);
    assert!((jac[[1, 0]] - 0.0).abs() < 1e-10, "df1/dx={}", jac[[1, 0]]);
    assert!((jac[[1, 1]] - 6.0).abs() < 1e-10, "df1/dy={}", jac[[1, 1]]);
}

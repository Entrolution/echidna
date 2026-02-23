#![cfg(feature = "nalgebra")]

use echidna::nalgebra_support::{
    grad_nalgebra, grad_nalgebra_val, hessian_nalgebra, jacobian_nalgebra, tape_gradient_nalgebra,
    tape_hessian_nalgebra,
};
use echidna::BReverse;
use nalgebra::DVector;

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
fn grad_nalgebra_rosenbrock() {
    let x = DVector::from_vec(vec![1.0_f64, 2.0]);
    let g = grad_nalgebra(rosenbrock_br, &x);

    assert!((g[0] - (-400.0)).abs() < 1e-10, "g[0]={}", g[0]);
    assert!((g[1] - 200.0).abs() < 1e-10, "g[1]={}", g[1]);
}

#[test]
fn grad_nalgebra_val_rosenbrock() {
    let x = DVector::from_vec(vec![1.0_f64, 2.0]);
    let (val, g) = grad_nalgebra_val(rosenbrock_br, &x);

    // f(1,2) = 0 + 100*(2-1)^2 = 100
    assert!((val - 100.0).abs() < 1e-10, "val={}", val);
    assert!((g[0] - (-400.0)).abs() < 1e-10, "g[0]={}", g[0]);
    assert!((g[1] - 200.0).abs() < 1e-10, "g[1]={}", g[1]);
}

#[test]
fn hessian_nalgebra_rosenbrock() {
    let x = DVector::from_vec(vec![1.0_f64, 2.0]);
    let (val, grad, hess) = hessian_nalgebra(rosenbrock_br, &x);

    assert!(val.is_finite());
    assert_eq!(grad.len(), 2);
    assert_eq!(hess.nrows(), 2);
    assert_eq!(hess.ncols(), 2);

    // Check symmetry
    assert!(
        (hess[(0, 1)] - hess[(1, 0)]).abs() < 1e-10,
        "Hessian should be symmetric"
    );
}

#[test]
fn jacobian_nalgebra_multi() {
    let x = DVector::from_vec(vec![2.0_f64, 3.0]);
    let jac = jacobian_nalgebra(multi_br, &x);

    // f0 = x*y, f1 = y^2
    // J = [[y, x], [0, 2*y]]
    assert_eq!(jac.nrows(), 2);
    assert_eq!(jac.ncols(), 2);
    assert!((jac[(0, 0)] - 3.0).abs() < 1e-10, "df0/dx={}", jac[(0, 0)]);
    assert!((jac[(0, 1)] - 2.0).abs() < 1e-10, "df0/dy={}", jac[(0, 1)]);
    assert!((jac[(1, 0)] - 0.0).abs() < 1e-10, "df1/dx={}", jac[(1, 0)]);
    assert!((jac[(1, 1)] - 6.0).abs() < 1e-10, "df1/dy={}", jac[(1, 1)]);
}

#[test]
fn tape_gradient_nalgebra_reuse() {
    let x = DVector::from_vec(vec![1.0_f64, 2.0]);
    let (mut tape, _) = echidna::record(rosenbrock_br, x.as_slice());

    let g1 = tape_gradient_nalgebra(&mut tape, &x);
    let g2 = tape_gradient_nalgebra(&mut tape, &x);

    for i in 0..g1.len() {
        assert!((g1[i] - g2[i]).abs() < 1e-14, "gradient mismatch at {}", i);
    }
}

#[test]
fn tape_hessian_nalgebra_reuse() {
    let x = DVector::from_vec(vec![1.0_f64, 2.0]);
    let (tape, _) = echidna::record(rosenbrock_br, x.as_slice());

    let (v1, g1, h1) = tape_hessian_nalgebra(&tape, &x);
    let (v2, g2, h2) = tape_hessian_nalgebra(&tape, &x);

    assert!((v1 - v2).abs() < 1e-14);
    for i in 0..g1.len() {
        assert!((g1[i] - g2[i]).abs() < 1e-14);
    }
    for i in 0..h1.nrows() {
        for j in 0..h1.ncols() {
            assert!((h1[(i, j)] - h2[(i, j)]).abs() < 1e-14);
        }
    }
}

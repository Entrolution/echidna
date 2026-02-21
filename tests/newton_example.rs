//! Newton step example using the Hessian from forward-over-reverse.
//!
//! Demonstrates using `tape.hessian()` + nalgebra to solve `H·δ = -∇f`
//! for Newton's method on the Rosenbrock function.

#![cfg(feature = "bytecode")]

use echidna::Scalar;

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

#[test]
fn newton_step_rosenbrock() {
    let n = 2;
    let (tape, _) = echidna::record(|x| rosenbrock(x), &[0.0_f64, 0.0]);

    let mut x = vec![0.0_f64, 0.0];
    for _ in 0..50 {
        let (_, grad, hess) = tape.hessian(&x);

        let h = nalgebra::DMatrix::from_fn(n, n, |i, j| hess[i][j]);
        let g = nalgebra::DVector::from_column_slice(&grad);

        let delta = h.lu().solve(&(-&g)).expect("Hessian should be invertible");

        for i in 0..n {
            x[i] += delta[i];
        }
    }

    assert!(
        (x[0] - 1.0_f64).abs() < 1e-8,
        "x[0] should converge to 1.0, got {}",
        x[0]
    );
    assert!(
        (x[1] - 1.0_f64).abs() < 1e-8,
        "x[1] should converge to 1.0, got {}",
        x[1]
    );
}

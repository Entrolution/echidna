use approx::assert_relative_eq;
use echidna::{grad, jacobian, jvp, vjp, Scalar};
use num_traits::Float;

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let a = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let b = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = a - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + b * t2 * t2;
    }
    sum
}

// ── grad ──

#[test]
fn grad_x_squared() {
    let g = grad(|x| x[0] * x[0], &[3.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
}

#[test]
fn grad_sum_of_squares() {
    let g = grad(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);
    assert_relative_eq!(g[0], 6.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 8.0, max_relative = 1e-12);
}

#[test]
fn grad_rosenbrock_2d() {
    // Rosenbrock f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // df/dx = -2(1-x) - 400x(y-x²)
    // df/dy = 200(y-x²)
    let x = [1.5_f64, 2.0];
    let g = grad(|v| rosenbrock(v), &x);
    let expected_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
    let expected_dy = 200.0 * (x[1] - x[0] * x[0]);
    assert_relative_eq!(g[0], expected_dx, max_relative = 1e-10);
    assert_relative_eq!(g[1], expected_dy, max_relative = 1e-10);
}

#[test]
fn grad_rosenbrock_at_minimum() {
    // At the minimum (1,1,...,1), gradient should be zero.
    let x = vec![1.0; 10];
    let g = grad(|v| rosenbrock(v), &x);
    for gi in &g {
        assert!(
            gi.abs() < 1e-12,
            "gradient at minimum should be zero, got {}",
            gi
        );
    }
}

#[test]
fn grad_rosenbrock_100d() {
    // Smoke test: gradient of 100-dimensional Rosenbrock.
    let x: Vec<f64> = (0..100).map(|i| 0.5 + 0.01 * i as f64).collect();
    let g = grad(|v| rosenbrock(v), &x);
    assert_eq!(g.len(), 100);
    // Just check it doesn't panic and produces finite values.
    for gi in &g {
        assert!(gi.is_finite(), "gradient should be finite");
    }
}

// ── jvp ──

#[test]
fn jvp_linear() {
    // f(x) = [2*x[0] + x[1], x[0] - x[1]]
    // J = [[2, 1], [1, -1]]
    // J * v for v = [1, 0] = [2, 1]
    let (vals, tangents) = jvp(
        |x| vec![x[0] + x[0] + x[1], x[0] - x[1]],
        &[3.0, 4.0],
        &[1.0, 0.0],
    );
    assert_relative_eq!(vals[0], 10.0, max_relative = 1e-12);
    assert_relative_eq!(vals[1], -1.0, max_relative = 1e-12);
    assert_relative_eq!(tangents[0], 2.0, max_relative = 1e-12);
    assert_relative_eq!(tangents[1], 1.0, max_relative = 1e-12);
}

// ── vjp ──

#[test]
fn vjp_linear() {
    // Same function, w = [1, 0]
    // wᵀ J = [1, 0] * [[2, 1], [1, -1]] = [2, 1]
    let (vals, grad) = vjp(
        |x| vec![x[0] + x[0] + x[1], x[0] - x[1]],
        &[3.0, 4.0],
        &[1.0, 0.0],
    );
    assert_relative_eq!(vals[0], 10.0, max_relative = 1e-12);
    assert_relative_eq!(vals[1], -1.0, max_relative = 1e-12);
    assert_relative_eq!(grad[0], 2.0, max_relative = 1e-12);
    assert_relative_eq!(grad[1], 1.0, max_relative = 1e-12);
}

#[test]
fn jvp_vjp_transpose_consistency() {
    // For any f, x, v, w: <jvp(f,x,v), w> == <v, vjp(f,x,w)>
    let x = [1.5, 2.0];
    let v = [0.7, -0.3];
    let w = [1.2, 0.5];

    let f_fwd = |inp: &[echidna::Dual<f64>]| -> Vec<echidna::Dual<f64>> {
        vec![inp[0] * inp[1], inp[0].sin() + inp[1].exp()]
    };
    let f_rev = |inp: &[echidna::Reverse<f64>]| -> Vec<echidna::Reverse<f64>> {
        vec![inp[0] * inp[1], inp[0].sin() + inp[1].exp()]
    };

    let (_, tangents) = jvp(f_fwd, &x, &v);
    let (_, grad) = vjp(f_rev, &x, &w);

    // <tangents, w> should equal <v, grad>
    let lhs: f64 = tangents.iter().zip(w.iter()).map(|(t, wi)| t * wi).sum();
    let rhs: f64 = v.iter().zip(grad.iter()).map(|(vi, gi)| vi * gi).sum();
    assert_relative_eq!(lhs, rhs, max_relative = 1e-10);
}

// ── jacobian ──

#[test]
fn jacobian_linear() {
    // f(x) = [2*x[0] + x[1], x[0] - x[1]]
    // J = [[2, 1], [1, -1]]
    let (vals, jac) = jacobian(|x| vec![x[0] + x[0] + x[1], x[0] - x[1]], &[3.0, 4.0]);
    assert_relative_eq!(vals[0], 10.0, max_relative = 1e-12);
    assert_relative_eq!(vals[1], -1.0, max_relative = 1e-12);
    assert_relative_eq!(jac[0][0], 2.0, max_relative = 1e-12);
    assert_relative_eq!(jac[0][1], 1.0, max_relative = 1e-12);
    assert_relative_eq!(jac[1][0], 1.0, max_relative = 1e-12);
    assert_relative_eq!(jac[1][1], -1.0, max_relative = 1e-12);
}

#[test]
fn jacobian_nonlinear() {
    // f(x) = [x[0]², x[0]*x[1]]
    // J = [[2*x[0], 0], [x[1], x[0]]]
    let (_, jac) = jacobian(|x| vec![x[0] * x[0], x[0] * x[1]], &[3.0, 4.0]);
    assert_relative_eq!(jac[0][0], 6.0, max_relative = 1e-12);
    assert_relative_eq!(jac[0][1], 0.0, epsilon = 1e-14);
    assert_relative_eq!(jac[1][0], 4.0, max_relative = 1e-12);
    assert_relative_eq!(jac[1][1], 3.0, max_relative = 1e-12);
}

// ── Scalar trait generic code ──

#[test]
fn scalar_generic_function() {
    fn square<T: Scalar>(x: T) -> T {
        x * x
    }

    // Works with f64
    assert_relative_eq!(square(3.0_f64), 9.0);

    // Works with Dual
    let d = square(echidna::Dual::variable(3.0));
    assert_relative_eq!(d.re, 9.0);
    assert_relative_eq!(d.eps, 6.0);
}

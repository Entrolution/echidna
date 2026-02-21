//! Tests for AD types inside nalgebra matrices.

#![cfg(feature = "simba")]

use approx::assert_relative_eq;
use echidna::{Dual64, Reverse64};
use nalgebra::{Matrix3, Vector3};
use num_traits::Float;

// ── Dual<f64> in nalgebra ──

#[test]
fn dual_vector3_dot_product() {
    let a = Vector3::new(
        Dual64::variable(1.0),
        Dual64::constant(2.0),
        Dual64::constant(3.0),
    );
    let b = Vector3::new(
        Dual64::constant(4.0),
        Dual64::constant(5.0),
        Dual64::constant(6.0),
    );
    let dot = a.dot(&b);
    // dot = 1*4 + 2*5 + 3*6 = 32
    assert_relative_eq!(dot.re, 32.0, max_relative = 1e-12);
    // d(dot)/d(a[0]) = b[0] = 4
    assert_relative_eq!(dot.eps, 4.0, max_relative = 1e-12);
}

#[test]
fn dual_vector3_norm() {
    // v = [x, 2, 3], norm = sqrt(x² + 4 + 9)
    // d(norm)/dx = x / sqrt(x² + 13)
    let v = Vector3::new(
        Dual64::variable(3.0),
        Dual64::constant(2.0),
        Dual64::constant(3.0),
    );
    let n = v.norm();
    let expected_norm = (9.0 + 4.0 + 9.0_f64).sqrt();
    assert_relative_eq!(n.re, expected_norm, max_relative = 1e-12);
    let expected_deriv = 3.0 / expected_norm;
    assert_relative_eq!(n.eps, expected_deriv, max_relative = 1e-10);
}

#[test]
fn dual_matrix_vector_product() {
    // M * v where v[0] is the variable
    let m = Matrix3::new(
        Dual64::constant(1.0),
        Dual64::constant(2.0),
        Dual64::constant(3.0),
        Dual64::constant(4.0),
        Dual64::constant(5.0),
        Dual64::constant(6.0),
        Dual64::constant(7.0),
        Dual64::constant(8.0),
        Dual64::constant(9.0),
    );
    let v = Vector3::new(
        Dual64::variable(1.0),
        Dual64::constant(0.0),
        Dual64::constant(0.0),
    );
    let result = m * v;
    // result = [1, 4, 7]
    assert_relative_eq!(result[0].re, 1.0, max_relative = 1e-12);
    assert_relative_eq!(result[1].re, 4.0, max_relative = 1e-12);
    assert_relative_eq!(result[2].re, 7.0, max_relative = 1e-12);
    // d/dx: derivatives are first column of M
    assert_relative_eq!(result[0].eps, 1.0, max_relative = 1e-12);
    assert_relative_eq!(result[1].eps, 4.0, max_relative = 1e-12);
    assert_relative_eq!(result[2].eps, 7.0, max_relative = 1e-12);
}

// ── Reverse<f64> in nalgebra ──

#[test]
fn reverse_vector3_dot_product() {
    let g = echidna::grad(
        |x| {
            let a = Vector3::new(x[0], x[1], x[2]);
            let b = Vector3::new(
                Reverse64::constant(4.0),
                Reverse64::constant(5.0),
                Reverse64::constant(6.0),
            );
            a.dot(&b)
        },
        &[1.0, 2.0, 3.0],
    );
    // d(a·b)/d(a) = b
    assert_relative_eq!(g[0], 4.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 5.0, max_relative = 1e-12);
    assert_relative_eq!(g[2], 6.0, max_relative = 1e-12);
}

#[test]
fn reverse_vector3_norm_squared() {
    // f(x) = ||x||² = x₀² + x₁² + x₂²
    // ∇f = [2x₀, 2x₁, 2x₂]
    let g = echidna::grad(
        |x| {
            let v = Vector3::new(x[0], x[1], x[2]);
            v.dot(&v)
        },
        &[1.0, 2.0, 3.0],
    );
    assert_relative_eq!(g[0], 2.0, max_relative = 1e-12);
    assert_relative_eq!(g[1], 4.0, max_relative = 1e-12);
    assert_relative_eq!(g[2], 6.0, max_relative = 1e-12);
}

#[test]
fn grad_of_nalgebra_function() {
    // f(x) = sin(||x||)
    let g = echidna::grad(
        |x| {
            let v = Vector3::new(x[0], x[1], x[2]);
            let n: Reverse64 = v.norm();
            n.sin()
        },
        &[1.0, 2.0, 3.0],
    );
    // ∂f/∂xᵢ = cos(||x||) * xᵢ / ||x||
    let norm = (1.0 + 4.0 + 9.0_f64).sqrt();
    for (i, &xi) in [1.0, 2.0, 3.0].iter().enumerate() {
        let expected = norm.cos() * xi / norm;
        assert_relative_eq!(g[i], expected, max_relative = 1e-10);
    }
}

// ── Matrix inverse (exercises ComplexField / RealField deeply) ──

#[test]
fn dual_matrix3_try_inverse() {
    // A 3×3 matrix of constants — test that try_inverse compiles and runs.
    let m = Matrix3::new(
        Dual64::constant(2.0),
        Dual64::constant(1.0),
        Dual64::constant(0.0),
        Dual64::constant(1.0),
        Dual64::constant(3.0),
        Dual64::constant(1.0),
        Dual64::constant(0.0),
        Dual64::constant(1.0),
        Dual64::constant(2.0),
    );
    let inv = m.try_inverse().expect("matrix should be invertible");
    let identity = m * inv;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(identity[(i, j)].re, expected, max_relative = 1e-10);
        }
    }
}

#[test]
fn reverse_matrix3_try_inverse() {
    // Same test for Reverse — validates the full ComplexField/RealField chain.
    let g = echidna::grad(
        |x| {
            let m = Matrix3::new(
                x[0],
                Reverse64::constant(1.0),
                Reverse64::constant(0.0),
                Reverse64::constant(1.0),
                Reverse64::constant(3.0),
                Reverse64::constant(1.0),
                Reverse64::constant(0.0),
                Reverse64::constant(1.0),
                Reverse64::constant(2.0),
            );
            let inv = m.try_inverse().expect("invertible");
            // Return trace of inverse (scalar output)
            inv[(0, 0)] + inv[(1, 1)] + inv[(2, 2)]
        },
        &[2.0],
    );
    // Verify finite-diff agreement.
    let h = 1e-7;
    let f = |a: f64| {
        let m = Matrix3::new(a, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0);
        let inv = m.try_inverse().unwrap();
        inv[(0, 0)] + inv[(1, 1)] + inv[(2, 2)]
    };
    let fd = (f(2.0 + h) - f(2.0 - h)) / (2.0 * h);
    assert_relative_eq!(g[0], fd, max_relative = 1e-4);
}

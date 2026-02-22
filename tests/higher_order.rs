#![cfg(feature = "bytecode")]

use echidna::{record, Scalar};

fn simple_quadratic<T: Scalar>(x: &[T]) -> T {
    // f(x, y) = x^2 * y + y^3
    x[0] * x[0] * x[1] + x[1] * x[1] * x[1]
}

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

fn cubic<T: Scalar>(x: &[T]) -> T {
    // f(x) = x^3
    x[0] * x[0] * x[0]
}

#[test]
fn third_order_hvvp_simple_quadratic() {
    // f(x, y) = x^2 * y + y^3
    // gradient = [2xy, x^2 + 3y^2]
    // Hessian = [[2y, 2x], [2x, 6y]]
    // H*v1 where v1 = [1, 0]: [2y, 2x]
    // d/dv2(H*v1) where v2 = [0, 1]:
    //   d/dy[2y] = 2, d/dy[2x] = 0 → [2, 0]
    let x = [3.0_f64, 2.0];
    let (tape, _) = record(|v| simple_quadratic(v), &x);

    let v1 = [1.0, 0.0];
    let v2 = [0.0, 1.0];
    let (gradient, hvp, third) = tape.third_order_hvvp(&x, &v1, &v2);

    // gradient = [2*3*2, 3^2 + 3*2^2] = [12, 21]
    assert!(
        (gradient[0] - 12.0).abs() < 1e-10,
        "grad[0]={}, expected=12",
        gradient[0]
    );
    assert!(
        (gradient[1] - 21.0).abs() < 1e-10,
        "grad[1]={}, expected=21",
        gradient[1]
    );

    // H*v1 = [2*2, 2*3] = [4, 6]
    assert!(
        (hvp[0] - 4.0).abs() < 1e-10,
        "hvp[0]={}, expected=4",
        hvp[0]
    );
    assert!(
        (hvp[1] - 6.0).abs() < 1e-10,
        "hvp[1]={}, expected=6",
        hvp[1]
    );

    // d/dv2(H*v1) = [2, 0]
    assert!(
        (third[0] - 2.0).abs() < 1e-10,
        "third[0]={}, expected=2",
        third[0]
    );
    assert!(
        (third[1] - 0.0).abs() < 1e-10,
        "third[1]={}, expected=0",
        third[1]
    );
}

#[test]
fn third_order_hvvp_same_direction() {
    // f(x, y) = x^2 * y + y^3
    // v1 = v2 = [1, 1]
    // H*v1 = [2y + 2x, 2x + 6y]
    // d/d[1,1](H*v1) = d/dx(H*v1) + d/dy(H*v1)
    //   d/dx[2y + 2x] = 2, d/dy[2y + 2x] = 2 → 4
    //   d/dx[2x + 6y] = 2, d/dy[2x + 6y] = 6 → 8
    let x = [3.0_f64, 2.0];
    let (tape, _) = record(|v| simple_quadratic(v), &x);

    let v = [1.0, 1.0];
    let (_, hvp, third) = tape.third_order_hvvp(&x, &v, &v);

    // H*v = [2*2 + 2*3, 2*3 + 6*2] = [10, 18]
    assert!(
        (hvp[0] - 10.0).abs() < 1e-10,
        "hvp[0]={}, expected=10",
        hvp[0]
    );
    assert!(
        (hvp[1] - 18.0).abs() < 1e-10,
        "hvp[1]={}, expected=18",
        hvp[1]
    );

    // d/d[1,1](H*v) = [4, 8]
    assert!(
        (third[0] - 4.0).abs() < 1e-10,
        "third[0]={}, expected=4",
        third[0]
    );
    assert!(
        (third[1] - 8.0).abs() < 1e-10,
        "third[1]={}, expected=8",
        third[1]
    );
}

#[test]
fn third_order_cubic() {
    // f(x) = x^3
    // f' = 3x^2, f'' = 6x, f''' = 6
    // H*v1 = [6x * v1[0]] = [6*2*1] = [12] at x=2
    // d/dv2(H*v1) = [6 * v1[0] * v2[0]] = [6] at v1=v2=[1]
    let x = [2.0_f64];
    let (tape, _) = record(|v| cubic(v), &x);

    let v = [1.0];
    let (gradient, hvp, third) = tape.third_order_hvvp(&x, &v, &v);

    assert!(
        (gradient[0] - 12.0).abs() < 1e-10,
        "grad={}, expected=12",
        gradient[0]
    );
    assert!(
        (hvp[0] - 12.0).abs() < 1e-10,
        "hvp={}, expected=12",
        hvp[0]
    );
    assert!(
        (third[0] - 6.0).abs() < 1e-10,
        "third={}, expected=6",
        third[0]
    );
}

#[test]
fn third_order_hvvp_gradient_matches_standard() {
    // Verify that the gradient from third_order_hvvp matches the standard gradient
    let x = [1.5_f64, 2.5];
    let (mut tape, _) = record(|v| rosenbrock(v), &x);

    let standard_grad = tape.gradient(&x);
    let (third_grad, _, _) = tape.third_order_hvvp(&x, &[1.0, 0.0], &[1.0, 0.0]);

    for (s, t) in standard_grad.iter().zip(third_grad.iter()) {
        assert!(
            (s - t).abs() < 1e-10,
            "standard={}, third_order={}",
            s,
            t
        );
    }
}

#[test]
fn third_order_hvvp_matches_hvp() {
    // Verify that the HVP from third_order_hvvp matches the standard HVP
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let v1 = [1.0, 0.0];
    let (_, standard_hvp) = tape.hvp(&x, &v1);
    let (_, third_hvp, _) = tape.third_order_hvvp(&x, &v1, &[1.0, 0.0]);

    for (s, t) in standard_hvp.iter().zip(third_hvp.iter()) {
        assert!(
            (s - t).abs() < 1e-10,
            "standard hvp={}, third_order hvp={}",
            s,
            t
        );
    }
}

#[test]
fn third_order_zero_directions() {
    // With zero directions, HVP and third-order should be zero
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let zero = [0.0, 0.0];
    let (gradient, hvp, third) = tape.third_order_hvvp(&x, &zero, &zero);

    // Gradient is still correct
    let (mut tape2, _) = record(|v| rosenbrock(v), &x);
    let expected_grad = tape2.gradient(&x);
    for (g, e) in gradient.iter().zip(expected_grad.iter()) {
        assert!((g - e).abs() < 1e-10);
    }

    // HVP and third-order are zero
    for &h in &hvp {
        assert!(h.abs() < 1e-10, "hvp should be zero, got {}", h);
    }
    for &t in &third {
        assert!(t.abs() < 1e-10, "third should be zero, got {}", t);
    }
}

//! Tests for second-order derivatives: forward-over-reverse (HVP, Hessian)
//! and forward-over-forward (Dual<Dual<f64>>).

use echidna::{Dual, Scalar};

#[cfg(feature = "bytecode")]
use echidna::record;

// ══════════════════════════════════════════════
//  Forward-over-reverse: Hessian via tape
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod forward_over_reverse {
    use super::*;
    use num_traits::Float;

    // ── Known analytic Hessians (tape method) ──

    #[test]
    fn hessian_sphere() {
        // f(x,y) = x² + y²  →  H = [[2,0],[0,2]]
        let (tape, _) = record(|x| x[0] * x[0] + x[1] * x[1], &[3.0_f64, 4.0]);
        let (val, grad, hess) = tape.hessian(&[3.0, 4.0]);

        assert!((val - 25.0_f64).abs() < 1e-10);
        assert!((grad[0] - 6.0_f64).abs() < 1e-10);
        assert!((grad[1] - 8.0_f64).abs() < 1e-10);
        assert!((hess[0][0] - 2.0_f64).abs() < 1e-10);
        assert!(hess[0][1].abs() < 1e-10);
        assert!(hess[1][0].abs() < 1e-10);
        assert!((hess[1][1] - 2.0_f64).abs() < 1e-10);
    }

    #[test]
    fn hessian_cross_term() {
        // f(x,y) = x*y  →  H = [[0,1],[1,0]]
        let (tape, _) = record(|x| x[0] * x[1], &[2.0_f64, 3.0]);
        let (val, grad, hess) = tape.hessian(&[2.0, 3.0]);

        assert!((val - 6.0_f64).abs() < 1e-10);
        assert!((grad[0] - 3.0_f64).abs() < 1e-10);
        assert!((grad[1] - 2.0_f64).abs() < 1e-10);
        assert!(hess[0][0].abs() < 1e-10);
        assert!((hess[0][1] - 1.0_f64).abs() < 1e-10);
        assert!((hess[1][0] - 1.0_f64).abs() < 1e-10);
        assert!(hess[1][1].abs() < 1e-10);
    }

    #[test]
    fn hessian_cubic_mixed() {
        // f(x,y) = x²*y + y³
        // grad = [2xy, x² + 3y²]
        // H = [[2y, 2x], [2x, 6y]]
        let x = 1.5_f64;
        let y = 2.0_f64;
        let (tape, _) = record(|v| v[0] * v[0] * v[1] + v[1] * v[1] * v[1], &[x, y]);
        let (val, grad, hess) = tape.hessian(&[x, y]);

        let expected_val = x * x * y + y * y * y;
        assert!((val - expected_val).abs() < 1e-10);
        assert!((grad[0] - 2.0 * x * y).abs() < 1e-10);
        assert!((grad[1] - (x * x + 3.0 * y * y)).abs() < 1e-10);
        assert!((hess[0][0] - 2.0 * y).abs() < 1e-10);
        assert!((hess[0][1] - 2.0 * x).abs() < 1e-10);
        assert!((hess[1][0] - 2.0 * x).abs() < 1e-10);
        assert!((hess[1][1] - 6.0 * y).abs() < 1e-10);
    }

    #[test]
    fn hessian_sin_1d() {
        // f(x) = sin(x)  →  H = [[-sin(x)]]
        let x = 1.0_f64;
        let (tape, _) = record(|v| v[0].sin(), &[x]);
        let (val, grad, hess) = tape.hessian(&[x]);

        assert!((val - x.sin()).abs() < 1e-10);
        assert!((grad[0] - x.cos()).abs() < 1e-10);
        assert!((hess[0][0] - (-x.sin())).abs() < 1e-10);
    }

    #[test]
    fn hessian_sin_exp() {
        // f(x,y) = sin(x) * exp(y)
        // ∂²f/∂x² = -sin(x)*exp(y), ∂²f/∂x∂y = cos(x)*exp(y)
        // ∂²f/∂y∂x = cos(x)*exp(y), ∂²f/∂y² = sin(x)*exp(y)
        let x = 0.7_f64;
        let y = 0.3_f64;
        let (tape, _) = record(|v| v[0].sin() * v[1].exp(), &[x, y]);
        let (val, grad, hess) = tape.hessian(&[x, y]);

        let ey = y.exp();
        assert!((val - x.sin() * ey).abs() < 1e-10);
        assert!((grad[0] - x.cos() * ey).abs() < 1e-10);
        assert!((grad[1] - x.sin() * ey).abs() < 1e-10);
        assert!((hess[0][0] - (-x.sin() * ey)).abs() < 1e-10);
        assert!((hess[0][1] - x.cos() * ey).abs() < 1e-10);
        assert!((hess[1][0] - x.cos() * ey).abs() < 1e-10);
        assert!((hess[1][1] - x.sin() * ey).abs() < 1e-10);
    }

    // ── Known analytic Hessians (top-level API) ──

    #[test]
    fn api_hessian_sphere() {
        let (val, grad, hess) =
            echidna::hessian(|x| x[0] * x[0] + x[1] * x[1], &[3.0_f64, 4.0]);

        assert!((val - 25.0_f64).abs() < 1e-10);
        assert!((grad[0] - 6.0_f64).abs() < 1e-10);
        assert!((grad[1] - 8.0_f64).abs() < 1e-10);
        assert!((hess[0][0] - 2.0_f64).abs() < 1e-10);
        assert!((hess[1][1] - 2.0_f64).abs() < 1e-10);
    }

    #[test]
    fn api_hvp_cross_term() {
        let (grad, hv) =
            echidna::hvp(|x| x[0] * x[1], &[2.0_f64, 3.0], &[1.0_f64, 0.0]);

        // H = [[0,1],[1,0]], v = [1,0] → H·v = [0, 1]
        assert!((grad[0] - 3.0_f64).abs() < 1e-10);
        assert!((grad[1] - 2.0_f64).abs() < 1e-10);
        assert!(hv[0].abs() < 1e-10);
        assert!((hv[1] - 1.0_f64).abs() < 1e-10);
    }

    // ── HVP against finite-difference gradient ──

    fn finite_diff_hvp(
        tape: &mut echidna::BytecodeTape<f64>,
        x: &[f64],
        v: &[f64],
        h: f64,
    ) -> Vec<f64> {
        let n = x.len();
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        for i in 0..n {
            xp[i] = x[i] + h * v[i];
            xm[i] = x[i] - h * v[i];
        }
        let gp = tape.gradient(&xp);
        let gm = tape.gradient(&xm);
        (0..n).map(|i| (gp[i] - gm[i]) / (2.0 * h)).collect()
    }

    #[test]
    fn hvp_vs_finite_diff_cubic() {
        let x = [1.5_f64, 2.0];
        let v = [0.7_f64, -0.3];
        let (mut tape, _) = record(|v| v[0] * v[0] * v[1] + v[1] * v[1] * v[1], &x);

        let (_, analytic_hv) = tape.hvp(&x, &v);
        let fd_hv = finite_diff_hvp(&mut tape, &x, &v, 1e-5);

        for i in 0..x.len() {
            assert!(
                (analytic_hv[i] - fd_hv[i]).abs() < 1e-4,
                "hvp vs fd, component {}: analytic={}, fd={}",
                i,
                analytic_hv[i],
                fd_hv[i]
            );
        }
    }

    #[test]
    fn hvp_vs_finite_diff_sin_exp() {
        let x = [0.7_f64, 0.3];
        let v = [1.0_f64, 1.0];
        let (mut tape, _) = record(|v| v[0].sin() * v[1].exp(), &x);

        let (_, analytic_hv) = tape.hvp(&x, &v);
        let fd_hv = finite_diff_hvp(&mut tape, &x, &v, 1e-5);

        for i in 0..x.len() {
            assert!(
                (analytic_hv[i] - fd_hv[i]).abs() < 1e-4,
                "hvp vs fd, component {}: analytic={}, fd={}",
                i,
                analytic_hv[i],
                fd_hv[i]
            );
        }
    }

    // ── Hessian symmetry ──

    fn rosenbrock<T: Scalar>(x: &[T]) -> T {
        let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
        let hundred =
            T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
        let mut sum = T::zero();
        for i in 0..x.len() - 1 {
            let t1 = one - x[i];
            let t2 = x[i + 1] - x[i] * x[i];
            sum = sum + t1 * t1 + hundred * t2 * t2;
        }
        sum
    }

    fn check_symmetry(hess: &[Vec<f64>], label: &str) {
        let n = hess.len();
        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (hess[i][j] - hess[j][i]).abs() < 1e-10,
                    "{} symmetry: H[{}][{}]={}, H[{}][{}]={}",
                    label,
                    i,
                    j,
                    hess[i][j],
                    j,
                    i,
                    hess[j][i]
                );
            }
        }
    }

    #[test]
    fn hessian_symmetry_cubic() {
        let x = [1.5_f64, 2.0];
        let (tape, _) = record(|v| v[0] * v[0] * v[1] + v[1] * v[1] * v[1], &x);
        let (_, _, hess) = tape.hessian(&x);
        check_symmetry(&hess, "cubic");
    }

    #[test]
    fn hessian_symmetry_sin_exp() {
        let x = [0.7_f64, 0.3];
        let (tape, _) = record(|v| v[0].sin() * v[1].exp(), &x);
        let (_, _, hess) = tape.hessian(&x);
        check_symmetry(&hess, "sin*exp");
    }

    #[test]
    fn hessian_symmetry_rosenbrock() {
        let x = [1.5_f64, 2.0];
        let (tape, _) = record(|v| rosenbrock(v), &x);
        let (_, _, hess) = tape.hessian(&x);
        check_symmetry(&hess, "rosenbrock");
    }

    // ── Rosenbrock Hessian (analytic) ──

    #[test]
    fn hessian_rosenbrock_analytic() {
        let x = 1.5_f64;
        let y = 2.0_f64;
        let (tape, _) = record(|v| rosenbrock(v), &[x, y]);
        let (_, _, hess) = tape.hessian(&[x, y]);

        // H[0][0] = 2 - 400*y + 1200*x²
        let h00 = 2.0 - 400.0 * y + 1200.0 * x * x;
        // H[0][1] = H[1][0] = -400*x
        let h01 = -400.0 * x;
        // H[1][1] = 200
        let h11 = 200.0;

        assert!((hess[0][0] - h00).abs() < 1e-10, "H[0][0]: {} vs {}", hess[0][0], h00);
        assert!((hess[0][1] - h01).abs() < 1e-10, "H[0][1]: {} vs {}", hess[0][1], h01);
        assert!((hess[1][0] - h01).abs() < 1e-10, "H[1][0]: {} vs {}", hess[1][0], h01);
        assert!((hess[1][1] - h11).abs() < 1e-10, "H[1][1]: {} vs {}", hess[1][1], h11);
    }

    // ── Tape reuse ──

    #[test]
    fn tape_reuse_hvp() {
        let (tape, _) =
            record(|v| v[0] * v[0] * v[1] + v[1] * v[1] * v[1], &[1.0_f64, 1.0]);

        // HVP with different directions using the same tape.
        let (g1, hv1) = tape.hvp(&[1.5_f64, 2.0], &[1.0_f64, 0.0]);
        let (g2, hv2) = tape.hvp(&[1.5_f64, 2.0], &[0.0_f64, 1.0]);

        // Both should yield the same gradient.
        for i in 0..2 {
            assert!(
                (g1[i] - g2[i]).abs() < 1e-10,
                "gradient mismatch at {}: {} vs {}",
                i,
                g1[i],
                g2[i]
            );
        }

        // hv1 = H·[1,0] = first column of H
        // hv2 = H·[0,1] = second column of H
        // H = [[2y, 2x],[2x, 6y]] at (1.5, 2.0)
        assert!((hv1[0] - 4.0_f64).abs() < 1e-10); // 2*2.0
        assert!((hv1[1] - 3.0_f64).abs() < 1e-10); // 2*1.5
        assert!((hv2[0] - 3.0_f64).abs() < 1e-10); // 2*1.5
        assert!((hv2[1] - 12.0_f64).abs() < 1e-10); // 6*2.0
    }

    // ── Gradient consistency ──

    #[test]
    fn gradient_from_hvp_matches_tape_gradient() {
        let x = [1.5_f64, 2.0];
        let (mut tape, _) = record(|v| rosenbrock(v), &x);

        let tape_grad = tape.gradient(&x);
        let (hvp_grad, _) = tape.hvp(&x, &[1.0_f64, 0.0]);

        for i in 0..x.len() {
            assert!(
                (tape_grad[i] - hvp_grad[i]).abs() < 1e-10,
                "gradient mismatch at {}: tape={}, hvp={}",
                i,
                tape_grad[i],
                hvp_grad[i]
            );
        }
    }

    // ── hvp_with_buf matches hvp ──

    #[test]
    fn hvp_with_buf_matches_hvp() {
        let x = [1.5_f64, 2.0];
        let v = [0.7_f64, -0.3];
        let (tape, _) = record(|v| rosenbrock(v), &x);

        let (grad1, hv1) = tape.hvp(&x, &v);

        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let (grad2, hv2) = tape.hvp_with_buf(&x, &v, &mut dual_vals_buf, &mut adjoint_buf);

        for i in 0..x.len() {
            assert!((grad1[i] - grad2[i]).abs() < 1e-15);
            assert!((hv1[i] - hv2[i]).abs() < 1e-15);
        }
    }
}

// ══════════════════════════════════════════════
//  Forward-over-forward: Dual<Dual<f64>>
// ══════════════════════════════════════════════

mod forward_over_forward {
    use super::*;

    #[test]
    fn second_derivative_cubic() {
        // f(x) = x³, f'(x) = 3x², f''(x) = 6x
        // At x=2: f''(2) = 12
        // Outer variable() sets eps = One = Dual::constant(1.0) = Dual(1,0).
        let x = Dual::<Dual<f64>>::variable(Dual::variable(2.0));
        let y = x * x * x;

        assert!((y.re.re - 8.0).abs() < 1e-10);
        assert!((y.re.eps - 12.0).abs() < 1e-10);
        assert!((y.eps.eps - 12.0).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_sin() {
        // f(x) = sin(x), f''(x) = -sin(x)
        let x_val = 1.0_f64;
        let x = Dual::<Dual<f64>>::variable(Dual::variable(x_val));
        let y = x.sin();

        assert!((y.re.re - x_val.sin()).abs() < 1e-10);
        assert!((y.re.eps - x_val.cos()).abs() < 1e-10);
        assert!((y.eps.eps - (-x_val.sin())).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_exp() {
        // f(x) = exp(x), f''(x) = exp(x)
        let x_val = 0.5_f64;
        let x = Dual::<Dual<f64>>::variable(Dual::variable(x_val));
        let y = x.exp();

        let e = x_val.exp();
        assert!((y.re.re - e).abs() < 1e-10);
        assert!((y.re.eps - e).abs() < 1e-10);
        assert!((y.eps.eps - e).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_ln() {
        // f(x) = ln(x), f'(x) = 1/x, f''(x) = -1/x²
        let x_val = 2.0_f64;
        let x = Dual::<Dual<f64>>::variable(Dual::variable(x_val));
        let y = x.ln();

        assert!((y.re.re - x_val.ln()).abs() < 1e-10);
        assert!((y.re.eps - 1.0 / x_val).abs() < 1e-10);
        assert!((y.eps.eps - (-1.0 / (x_val * x_val))).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_sqrt() {
        // f(x) = sqrt(x), f'(x) = 1/(2√x), f''(x) = -1/(4x^(3/2))
        let x_val = 4.0_f64;
        let x = Dual::<Dual<f64>>::variable(Dual::variable(x_val));
        let y = x.sqrt();

        assert!((y.re.re - 2.0).abs() < 1e-10);
        assert!((y.re.eps - 0.25).abs() < 1e-10);
        let expected_f2 = -1.0 / (4.0 * x_val.powf(1.5));
        assert!((y.eps.eps - expected_f2).abs() < 1e-10);
    }

    #[test]
    fn nested_dual_composition() {
        // f(x) = sin(exp(x))
        // f''(x) = exp(x) * (cos(exp(x)) - exp(x)*sin(exp(x)))
        let x_val = 0.5_f64;
        let x = Dual::<Dual<f64>>::variable(Dual::variable(x_val));
        let y = x.exp().sin();

        let ex = x_val.exp();
        let expected_f0 = ex.sin();
        let expected_f1 = ex.cos() * ex;
        let expected_f2 = ex * (ex.cos() - ex * ex.sin());

        assert!((y.re.re - expected_f0).abs() < 1e-10);
        assert!((y.re.eps - expected_f1).abs() < 1e-10);
        assert!((y.eps.eps - expected_f2).abs() < 1e-8);
    }

    #[test]
    fn scalar_trait_with_nested_dual() {
        fn generic_square<T: Scalar>(x: T) -> T {
            x * x
        }

        let x = Dual::<Dual<f64>>::variable(Dual::variable(3.0));
        let y = generic_square(x);

        // f(x) = x², f'(x) = 2x, f''(x) = 2
        assert!((y.re.re - 9.0).abs() < 1e-10);
        assert!((y.re.eps - 6.0).abs() < 1e-10);
        assert!((y.eps.eps - 2.0).abs() < 1e-10);
    }
}

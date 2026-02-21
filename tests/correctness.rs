use echidna::{grad, Dual, Scalar};

#[cfg(feature = "bytecode")]
use echidna::{record, BReverse};

/// Central finite difference gradient.
fn finite_diff_grad(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    grad
}

/// Forward-mode gradient (one pass per variable).
fn forward_grad(f: impl Fn(&[Dual<f64>]) -> Dual<f64>, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let inputs: Vec<Dual<f64>> = x
            .iter()
            .enumerate()
            .map(|(k, &xi)| {
                if k == i {
                    Dual::variable(xi)
                } else {
                    Dual::constant(xi)
                }
            })
            .collect();
        grad[i] = f(&inputs).eps;
    }
    grad
}

/// Rosenbrock function, generic over scalar type.
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

/// Beale function: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
fn beale<T: Scalar>(x: &[T]) -> T {
    let x0 = x[0];
    let x1 = x[1];
    let c1 = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.5).unwrap());
    let c2 = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.25).unwrap());
    let c3 = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.625).unwrap());
    let t1 = c1 - x0 + x0 * x1;
    let t2 = c2 - x0 + x0 * x1 * x1;
    let t3 = c3 - x0 + x0 * x1 * x1 * x1;
    t1 * t1 + t2 * t2 + t3 * t3
}

/// Sphere function: f(x) = sum(x_i²)
fn sphere<T: Scalar>(x: &[T]) -> T {
    let mut sum = T::zero();
    for &xi in x {
        sum = sum + xi * xi;
    }
    sum
}

/// Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
fn booth<T: Scalar>(x: &[T]) -> T {
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    let five = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(5.0).unwrap());
    let seven = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(7.0).unwrap());
    let t1 = x[0] + two * x[1] - seven;
    let t2 = two * x[0] + x[1] - five;
    t1 * t1 + t2 * t2
}

/// Cross-validate forward, reverse, and finite differences.
fn cross_validate(
    f_dual: impl Fn(&[Dual<f64>]) -> Dual<f64>,
    f_rev: impl FnOnce(&[echidna::Reverse<f64>]) -> echidna::Reverse<f64>,
    f_f64: impl Fn(&[f64]) -> f64,
    x: &[f64],
    label: &str,
) {
    let fwd_grad = forward_grad(&f_dual, x);
    let rev_grad = grad(f_rev, x);
    let fd_grad = finite_diff_grad(&f_f64, x, 1e-7);

    // Forward vs reverse: should match to machine precision.
    for i in 0..x.len() {
        assert!(
            (fwd_grad[i] - rev_grad[i]).abs() <= 1e-10 * fwd_grad[i].abs().max(1e-12),
            "{} fwd vs rev, component {}: fwd={}, rev={}",
            label,
            i,
            fwd_grad[i],
            rev_grad[i]
        );
    }

    // Forward vs finite diff: should match to ~1e-4.
    for i in 0..x.len() {
        let scale = fwd_grad[i].abs().max(1.0);
        assert!(
            (fwd_grad[i] - fd_grad[i]).abs() <= 1e-4 * scale,
            "{} fwd vs fd, component {}: fwd={}, fd={}",
            label,
            i,
            fwd_grad[i],
            fd_grad[i]
        );
    }
}

#[test]
fn cross_validate_rosenbrock_2d() {
    let x = [1.5, 2.0];
    cross_validate(
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        &x,
        "rosenbrock-2d",
    );
}

#[test]
fn cross_validate_rosenbrock_10d() {
    let x: Vec<f64> = (0..10).map(|i| 0.5 + 0.1 * i as f64).collect();
    cross_validate(
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        &x,
        "rosenbrock-10d",
    );
}

#[test]
fn cross_validate_beale() {
    let x = [1.0, 0.5];
    cross_validate(|v| beale(v), |v| beale(v), |v| beale(v), &x, "beale");
}

#[test]
fn cross_validate_sphere_5d() {
    let x = [1.0, -2.0, 3.0, -0.5, 0.7];
    cross_validate(|v| sphere(v), |v| sphere(v), |v| sphere(v), &x, "sphere-5d");
}

#[test]
fn cross_validate_booth() {
    let x = [2.0, 3.0];
    cross_validate(|v| booth(v), |v| booth(v), |v| booth(v), &x, "booth");
}

/// Test transcendental-heavy function.
fn trig_mix<T: Scalar>(x: &[T]) -> T {
    x[0].sin() * x[1].exp() + x[0].cos() * x[1].ln()
}

#[test]
fn cross_validate_trig_mix() {
    let x = [1.0, 2.0];
    cross_validate(
        |v| trig_mix(v),
        |v| trig_mix(v),
        |v| trig_mix(v),
        &x,
        "trig-mix",
    );
}

/// Ackley function (2D): lots of trig and exp.
fn ackley<T: Scalar>(x: &[T]) -> T {
    let a = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(20.0).unwrap());
    let b = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(0.2).unwrap());
    let c = T::from_f(
        <T::Float as num_traits::FromPrimitive>::from_f64(std::f64::consts::TAU).unwrap(),
    );
    let half = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(0.5).unwrap());
    let e_const =
        T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(std::f64::consts::E).unwrap());

    let sum_sq = x[0] * x[0] + x[1] * x[1];
    let sum_cos = (c * x[0]).cos() + (c * x[1]).cos();

    let term1 = (-(b) * (half * sum_sq).sqrt()).exp();
    let term2 = (half * sum_cos).exp();

    -a * term1 - term2 + a + e_const
}

#[test]
fn cross_validate_ackley() {
    let x = [0.5, -0.3];
    cross_validate(|v| ackley(v), |v| ackley(v), |v| ackley(v), &x, "ackley");
}

/// Test: forward and reverse agree on a deeply nested function.
fn deep_nest<T: Scalar>(x: &[T]) -> T {
    let mut y = x[0];
    for _ in 0..10 {
        y = y.sin().exp().ln_1p();
    }
    y
}

#[test]
fn cross_validate_deep_nest() {
    let x = [0.5];
    cross_validate(
        |v| deep_nest(v),
        |v| deep_nest(v),
        |v| deep_nest(v),
        &x,
        "deep-nest",
    );
}

/// Sum of pairwise products — exercises fan-out heavily.
fn pairwise_products<T: Scalar>(x: &[T]) -> T {
    let mut sum = T::zero();
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            sum = sum + x[i] * x[j];
        }
    }
    sum
}

#[test]
fn cross_validate_pairwise_products() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    cross_validate(
        |v| pairwise_products(v),
        |v| pairwise_products(v),
        |v| pairwise_products(v),
        &x,
        "pairwise-products",
    );
}

/// Logistic sigmoid chain.
fn logistic_chain<T: Scalar>(x: &[T]) -> T {
    let one = T::one();
    let mut y = x[0];
    for _ in 0..5 {
        y = one / (one + (-y).exp());
    }
    y
}

#[test]
fn cross_validate_logistic_chain() {
    let x = [0.5];
    cross_validate(
        |v| logistic_chain(v),
        |v| logistic_chain(v),
        |v| logistic_chain(v),
        &x,
        "logistic-chain",
    );
}

// ══════════════════════════════════════════════
//  Bytecode tape cross-validation
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
fn bytecode_grad(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> Vec<f64> {
    let (mut tape, _) = record(f, x);
    tape.gradient(x)
}

/// Cross-validate forward, reverse (Adept), and bytecode reverse.
#[cfg(feature = "bytecode")]
fn cross_validate_all(
    f_dual: impl Fn(&[Dual<f64>]) -> Dual<f64>,
    f_rev: impl FnOnce(&[echidna::Reverse<f64>]) -> echidna::Reverse<f64>,
    f_brev: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    f_f64: impl Fn(&[f64]) -> f64,
    x: &[f64],
    label: &str,
) {
    let fwd_grad = forward_grad(&f_dual, x);
    let rev_grad = grad(f_rev, x);
    let btape_grad = bytecode_grad(f_brev, x);
    let fd_grad = finite_diff_grad(&f_f64, x, 1e-7);

    for i in 0..x.len() {
        // Forward vs Adept reverse.
        assert!(
            (fwd_grad[i] - rev_grad[i]).abs() <= 1e-10 * fwd_grad[i].abs().max(1e-12),
            "{} fwd vs rev, component {}: fwd={}, rev={}",
            label,
            i,
            fwd_grad[i],
            rev_grad[i]
        );
        // Adept reverse vs bytecode reverse.
        assert!(
            (rev_grad[i] - btape_grad[i]).abs() <= 1e-10 * rev_grad[i].abs().max(1e-12),
            "{} rev vs btape, component {}: rev={}, btape={}",
            label,
            i,
            rev_grad[i],
            btape_grad[i]
        );
        // Forward vs finite diff.
        let scale = fwd_grad[i].abs().max(1.0);
        assert!(
            (fwd_grad[i] - fd_grad[i]).abs() <= 1e-4 * scale,
            "{} fwd vs fd, component {}: fwd={}, fd={}",
            label,
            i,
            fwd_grad[i],
            fd_grad[i]
        );
    }
}

#[cfg(feature = "bytecode")]
#[test]
fn cross_validate_all_rosenbrock_2d() {
    let x = [1.5, 2.0];
    cross_validate_all(
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        |v| rosenbrock(v),
        &x,
        "all-rosenbrock-2d",
    );
}

#[cfg(feature = "bytecode")]
#[test]
fn cross_validate_all_trig_mix() {
    let x = [1.0, 2.0];
    cross_validate_all(
        |v| trig_mix(v),
        |v| trig_mix(v),
        |v| trig_mix(v),
        |v| trig_mix(v),
        &x,
        "all-trig-mix",
    );
}

#[cfg(feature = "bytecode")]
#[test]
fn cross_validate_all_deep_nest() {
    let x = [0.5];
    cross_validate_all(
        |v| deep_nest(v),
        |v| deep_nest(v),
        |v| deep_nest(v),
        |v| deep_nest(v),
        &x,
        "all-deep-nest",
    );
}

#[cfg(feature = "bytecode")]
#[test]
fn cross_validate_all_logistic_chain() {
    let x = [0.5];
    cross_validate_all(
        |v| logistic_chain(v),
        |v| logistic_chain(v),
        |v| logistic_chain(v),
        |v| logistic_chain(v),
        &x,
        "all-logistic-chain",
    );
}

use echidna::{Dual, DualVec, Scalar};

#[test]
fn dualvec_1_matches_dual() {
    let x_dual = Dual::new(2.0_f64, 1.0);
    let x_vec = DualVec::<f64, 1>::new(2.0, [1.0]);

    // Unary ops
    let ops: Vec<(&str, Dual<f64>, DualVec<f64, 1>)> = vec![
        ("sin", x_dual.sin(), x_vec.sin()),
        ("cos", x_dual.cos(), x_vec.cos()),
        ("tan", x_dual.tan(), x_vec.tan()),
        ("exp", x_dual.exp(), x_vec.exp()),
        ("ln", x_dual.ln(), x_vec.ln()),
        ("sqrt", x_dual.sqrt(), x_vec.sqrt()),
        ("cbrt", x_dual.cbrt(), x_vec.cbrt()),
        ("recip", x_dual.recip(), x_vec.recip()),
        ("asin", Dual::new(0.5, 1.0).asin(), DualVec::<f64, 1>::new(0.5, [1.0]).asin()),
        ("acos", Dual::new(0.5, 1.0).acos(), DualVec::<f64, 1>::new(0.5, [1.0]).acos()),
        ("atan", x_dual.atan(), x_vec.atan()),
        ("sinh", x_dual.sinh(), x_vec.sinh()),
        ("cosh", x_dual.cosh(), x_vec.cosh()),
        ("tanh", x_dual.tanh(), x_vec.tanh()),
        ("asinh", x_dual.asinh(), x_vec.asinh()),
        ("acosh", x_dual.acosh(), x_vec.acosh()),
        ("exp2", x_dual.exp2(), x_vec.exp2()),
        ("exp_m1", x_dual.exp_m1(), x_vec.exp_m1()),
        ("log2", x_dual.log2(), x_vec.log2()),
        ("log10", x_dual.log10(), x_vec.log10()),
        ("ln_1p", x_dual.ln_1p(), x_vec.ln_1p()),
        ("abs", Dual::new(-2.0, 1.0).abs(), DualVec::<f64, 1>::new(-2.0, [1.0]).abs()),
        ("powi3", x_dual.powi(3), x_vec.powi(3)),
    ];

    for (name, dual, dvec) in ops {
        assert!(
            (dual.re - dvec.re).abs() < 1e-15,
            "{}: re mismatch: {} vs {}",
            name,
            dual.re,
            dvec.re
        );
        assert!(
            (dual.eps - dvec.eps[0]).abs() < 1e-15,
            "{}: eps mismatch: {} vs {}",
            name,
            dual.eps,
            dvec.eps[0]
        );
    }
}

#[test]
fn dualvec_binary_ops_match_dual() {
    let a_dual = Dual::new(2.0_f64, 1.0);
    let b_dual = Dual::new(3.0_f64, 0.5);
    let a_vec = DualVec::<f64, 1>::new(2.0, [1.0]);
    let b_vec = DualVec::<f64, 1>::new(3.0, [0.5]);

    let cases: Vec<(&str, Dual<f64>, DualVec<f64, 1>)> = vec![
        ("add", a_dual + b_dual, a_vec + b_vec),
        ("sub", a_dual - b_dual, a_vec - b_vec),
        ("mul", a_dual * b_dual, a_vec * b_vec),
        ("div", a_dual / b_dual, a_vec / b_vec),
        ("powf", a_dual.powf(b_dual), a_vec.powf(b_vec)),
        ("atan2", a_dual.atan2(b_dual), a_vec.atan2(b_vec)),
        ("hypot", a_dual.hypot(b_dual), a_vec.hypot(b_vec)),
    ];

    for (name, dual, dvec) in cases {
        assert!(
            (dual.re - dvec.re).abs() < 1e-14,
            "{}: re mismatch: {} vs {}",
            name,
            dual.re,
            dvec.re
        );
        assert!(
            (dual.eps - dvec.eps[0]).abs() < 1e-14,
            "{}: eps mismatch: {} vs {}",
            name,
            dual.eps,
            dvec.eps[0]
        );
    }
}

#[test]
fn multi_lane_independence() {
    let x = DualVec::<f64, 3>::new(2.0, [1.0, 0.0, 0.0]);
    let y = DualVec::<f64, 3>::new(3.0, [0.0, 1.0, 0.0]);
    let z = DualVec::<f64, 3>::new(4.0, [0.0, 0.0, 1.0]);

    let result = x * y + z.sin();
    assert!((result.eps[0] - 3.0).abs() < 1e-10); // d/dx(xy + sin(z)) = y
    assert!((result.eps[1] - 2.0).abs() < 1e-10); // d/dy = x
    assert!((result.eps[2] - 4.0_f64.cos()).abs() < 1e-10); // d/dz = cos(z)
}

#[test]
fn sin_cos_consistency() {
    let x = DualVec::<f64, 2>::new(1.5, [1.0, 0.0]);
    let (s, c) = x.sin_cos();
    let s2 = x.sin();
    let c2 = x.cos();

    assert!((s.re - s2.re).abs() < 1e-15);
    assert!((c.re - c2.re).abs() < 1e-15);
    assert!((s.eps[0] - s2.eps[0]).abs() < 1e-15);
    assert!((c.eps[0] - c2.eps[0]).abs() < 1e-15);
}

#[test]
fn scalar_trait_dualvec() {
    fn generic_square<T: Scalar>(x: T) -> T {
        x * x
    }

    let x = DualVec::<f64, 2>::new(3.0, [1.0, 0.0]);
    let y = generic_square(x);
    assert!((y.re - 9.0).abs() < 1e-10);
    assert!((y.eps[0] - 6.0).abs() < 1e-10);
    assert!(y.eps[1].abs() < 1e-10);
}

#[test]
fn display_format() {
    let x = DualVec::<f64, 2>::new(3.0, [1.0, 2.0]);
    let s = format!("{}", x);
    assert!(s.contains("3"));
}

#[test]
fn default_is_zero() {
    let x = DualVec::<f64, 4>::default();
    assert_eq!(x.re, 0.0);
    for e in &x.eps {
        assert_eq!(*e, 0.0);
    }
}

#[test]
fn with_tangent() {
    let x = DualVec::<f64, 3>::with_tangent(5.0, 1);
    assert_eq!(x.re, 5.0);
    assert_eq!(x.eps[0], 0.0);
    assert_eq!(x.eps[1], 1.0);
    assert_eq!(x.eps[2], 0.0);
}

#[test]
fn scalar_ops() {
    let x = DualVec::<f64, 1>::new(3.0, [1.0]);

    let r1 = x + 2.0;
    assert!((r1.re - 5.0).abs() < 1e-15);
    assert!((r1.eps[0] - 1.0).abs() < 1e-15);

    let r2 = 2.0 + x;
    assert!((r2.re - 5.0).abs() < 1e-15);

    let r3 = x * 3.0;
    assert!((r3.re - 9.0).abs() < 1e-15);
    assert!((r3.eps[0] - 3.0).abs() < 1e-15);

    let r4 = x / 2.0;
    assert!((r4.re - 1.5).abs() < 1e-15);
    assert!((r4.eps[0] - 0.5).abs() < 1e-15);

    let r5 = 6.0 / x;
    assert!((r5.re - 2.0).abs() < 1e-15);
    assert!((r5.eps[0] - (-6.0 / 9.0)).abs() < 1e-15);
}

#[cfg(feature = "bytecode")]
mod bytecode_tests {
    use echidna::{record, Scalar};
    use num_traits::Float;

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
    fn hessian_vec_matches_hessian_n2() {
        let x = vec![0.5_f64, 0.51];
        let (tape, _) = record(|v| rosenbrock(v), &x);

        let (val1, grad1, hess1) = tape.hessian(&x);
        let (val2, grad2, hess2) = tape.hessian_vec::<4>(&x);

        assert!((val1 - val2).abs() < 1e-10);
        for i in 0..2 {
            assert!((grad1[i] - grad2[i]).abs() < 1e-10);
            for j in 0..2 {
                assert!(
                    (hess1[i][j] - hess2[i][j]).abs() < 1e-10,
                    "Hessian mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    hess1[i][j],
                    hess2[i][j]
                );
            }
        }
    }

    #[test]
    fn hessian_vec_padding() {
        // hessian_vec::<4> on a 3-variable problem (tests zero-padding)
        let x = vec![1.0_f64, 2.0, 3.0];
        let (tape, _) = record(|v| v[0] * v[1] + v[2] * v[2], &x);

        let (_, _, hess1) = tape.hessian(&x);
        let (_, _, hess2) = tape.hessian_vec::<4>(&x);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (hess1[i][j] - hess2[i][j]).abs() < 1e-10,
                    "Hessian mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    hess1[i][j],
                    hess2[i][j]
                );
            }
        }
    }

    #[test]
    fn api_hessian_vec() {
        let x = vec![1.5_f64, 2.0];
        let (val1, grad1, hess1) = echidna::hessian(|v| rosenbrock(v), &x);
        let (val2, grad2, hess2) = echidna::hessian_vec::<_, 4>(|v| rosenbrock(v), &x);

        assert!((val1 - val2).abs() < 1e-10);
        for i in 0..2 {
            assert!((grad1[i] - grad2[i]).abs() < 1e-10);
            for j in 0..2 {
                assert!((hess1[i][j] - hess2[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn hessian_vec_various_sizes() {
        for n in [2, 3, 5] {
            let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
            let (tape, _) = record(|v| rosenbrock(v), &x);
            let (_, _, hess1) = tape.hessian(&x);

            for batch_size in ["1", "2", "4", "8"] {
                let (_, _, hess2) = match batch_size {
                    "1" => tape.hessian_vec::<1>(&x),
                    "2" => tape.hessian_vec::<2>(&x),
                    "4" => tape.hessian_vec::<4>(&x),
                    "8" => tape.hessian_vec::<8>(&x),
                    _ => unreachable!(),
                };

                for i in 0..n {
                    for j in 0..n {
                        assert!(
                            (hess1[i][j] - hess2[i][j]).abs() < 1e-10,
                            "n={}, batch={}, [{},{}]: {} vs {}",
                            n,
                            batch_size,
                            i,
                            j,
                            hess1[i][j],
                            hess2[i][j]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hessian_vec_sin_exp() {
        // f(x,y) = sin(x) * exp(y)
        let x = 0.7_f64;
        let y = 0.3_f64;
        let (tape, _) = record(|v| v[0].sin() * v[1].exp(), &[x, y]);

        let (_, _, hess1) = tape.hessian(&[x, y]);
        let (_, _, hess2) = tape.hessian_vec::<2>(&[x, y]);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (hess1[i][j] - hess2[i][j]).abs() < 1e-10,
                    "Hessian mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    hess1[i][j],
                    hess2[i][j]
                );
            }
        }
    }
}

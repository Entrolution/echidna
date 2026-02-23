//! Tests for composable mode nesting (R12).
//!
//! Forward-wrapping-reverse compositions: `Dual<BReverse<f64>>`, `Dual<Reverse<f64>>`,
//! `DualVec<BReverse<f64>, N>`, `Taylor<BReverse<f64>, K>`, and triple nesting.

// ══════════════════════════════════════════════
//  Dual<BReverse<f64>> — forward-over-bytecode-reverse
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod dual_breverse {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, Dual, Scalar};

    /// f(x) = x³ → f'(x) = 3x², f''(x) = 6x
    #[test]
    fn single_variable_cubic() {
        let x_val = 2.0_f64;
        let v_val = 1.0_f64;

        let mut tape = BytecodeTape::with_capacity(100);
        let idx = tape.new_input(x_val);
        let re = BReverse::from_tape(x_val, idx);
        let eps = BReverse::constant(v_val);
        let x: Dual<BReverse<f64>> = Dual::new(re, eps);

        let _guard = BtapeGuard::new(&mut tape);
        let y = x * x * x; // x³

        let primal_index = y.re.index();
        let tangent_index = y.eps.index();

        // f(2) = 8
        assert!((Scalar::value(&y.re) - 8.0).abs() < 1e-10);

        // f'(2) = 3*4 = 12 (tangent output = derivative * v = 12 * 1 = 12)
        assert!((Scalar::value(&y.eps) - 12.0).abs() < 1e-10);

        // Reverse from primal → gradient = f'(x) = 3x² = 12
        let primal_adjoints = tape.reverse(primal_index);
        assert!((primal_adjoints[0] - 12.0).abs() < 1e-10);

        // Reverse from tangent → HVP = f''(x)*v = 6x*1 = 12
        let tangent_adjoints = tape.reverse(tangent_index);
        assert!((tangent_adjoints[0] - 12.0).abs() < 1e-10);
    }

    /// 2D Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    /// Compare Dual<BReverse<f64>> HVP against tape-based hvp().
    #[test]
    fn rosenbrock_hvp() {
        let x = [1.5_f64, 2.0];
        let v = [0.3, -0.7];

        // Reference: tape-based hvp
        let (ref_grad, ref_hvp) = echidna::hvp(
            |x| {
                let a = BReverse::constant(1.0) - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + BReverse::constant(100.0) * b * b
            },
            &x,
            &v,
        );

        // Composed: Dual<BReverse<f64>>
        let (val, grad, hvp) = echidna::composed_hvp(
            |x| {
                let one: Dual<BReverse<f64>> = Scalar::from_f(BReverse::constant(1.0));
                let hundred: Dual<BReverse<f64>> = Scalar::from_f(BReverse::constant(100.0));
                let a = one - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + hundred * b * b
            },
            &x,
            &v,
        );

        // Check function value: (1-1.5)² + 100(2 - 2.25)² = 0.25 + 6.25 = 6.5
        assert!((val - 6.5).abs() < 1e-10);

        // Check gradient matches tape-based
        for i in 0..2 {
            assert!(
                (grad[i] - ref_grad[i]).abs() < 1e-10,
                "grad[{}]: composed={}, tape={}",
                i,
                grad[i],
                ref_grad[i]
            );
        }

        // Check HVP matches tape-based
        for i in 0..2 {
            assert!(
                (hvp[i] - ref_hvp[i]).abs() < 1e-10,
                "hvp[{}]: composed={}, tape={}",
                i,
                hvp[i],
                ref_hvp[i]
            );
        }
    }

    /// sin(x)*cos(y) + exp(x*y) — compare against tape-based hvp().
    #[test]
    fn trig_exp_hvp() {
        use num_traits::Float;

        let x = [0.5_f64, 1.2];
        let v = [1.0, 0.0];

        let (ref_grad, ref_hvp) =
            echidna::hvp(|x| x[0].sin() * x[1].cos() + (x[0] * x[1]).exp(), &x, &v);

        let (_, grad, hvp) = echidna::composed_hvp(
            |x: &[Dual<BReverse<f64>>]| x[0].sin() * x[1].cos() + (x[0] * x[1]).exp(),
            &x,
            &v,
        );

        for i in 0..2 {
            assert!((grad[i] - ref_grad[i]).abs() < 1e-8, "grad[{}] mismatch", i);
            assert!((hvp[i] - ref_hvp[i]).abs() < 1e-8, "hvp[{}] mismatch", i);
        }
    }

    /// Test the composed_hvp convenience function at multiple points.
    #[test]
    fn composed_hvp_api() {
        let points: &[([f64; 2], [f64; 2])] = &[([1.0, 1.0], [1.0, 0.0]), ([0.0, 0.0], [0.0, 1.0])];

        for (x, v) in points {
            let (ref_grad, ref_hvp) = echidna::hvp(
                |x| {
                    let a = BReverse::constant(1.0) - x[0];
                    let b = x[1] - x[0] * x[0];
                    a * a + BReverse::constant(100.0) * b * b
                },
                x,
                v,
            );

            let (_, grad, hvp) = echidna::composed_hvp(
                |x| {
                    let one: Dual<BReverse<f64>> = Scalar::from_f(BReverse::constant(1.0));
                    let hundred: Dual<BReverse<f64>> = Scalar::from_f(BReverse::constant(100.0));
                    let a = one - x[0];
                    let b = x[1] - x[0] * x[0];
                    a * a + hundred * b * b
                },
                x,
                v,
            );

            for i in 0..2 {
                assert!(
                    (grad[i] - ref_grad[i]).abs() < 1e-10,
                    "point {:?}: grad[{}] mismatch",
                    x,
                    i
                );
                assert!(
                    (hvp[i] - ref_hvp[i]).abs() < 1e-10,
                    "point {:?}: hvp[{}] mismatch",
                    x,
                    i
                );
            }
        }
    }
}

// ══════════════════════════════════════════════
//  Dual<Reverse<f64>> — forward-over-adept-reverse
// ══════════════════════════════════════════════

mod dual_reverse {
    use echidna::tape::{Tape, TapeGuard};
    use echidna::{Dual, Reverse, Scalar};

    /// f(x) = x³ → f''(x) = 6x via Dual<Reverse<f64>>.
    #[test]
    fn second_derivative_cubic() {
        let x_val = 3.0_f64;
        let v_val = 1.0_f64;

        let mut tape = Tape::with_capacity(100);
        let (idx, _) = tape.new_variable(x_val);
        let re = Reverse::from_tape(x_val, idx);
        let eps = Reverse::constant(v_val);
        let x: Dual<Reverse<f64>> = Dual::new(re, eps);

        let _guard = TapeGuard::new(&mut tape);
        let y = x * x * x;

        // f(3) = 27
        assert!((Scalar::value(&y.re) - 27.0).abs() < 1e-10);

        // tangent = f'(3) * v = 27 * 1 = 27
        assert!((Scalar::value(&y.eps) - 27.0).abs() < 1e-10);

        // Reverse from primal → f'(x) = 3x² = 27
        let primal_adjoints = tape.reverse(y.re.index());
        assert!((primal_adjoints[0] - 27.0).abs() < 1e-10);

        // Reverse from tangent → f''(x)*v = 6x = 18
        let tangent_adjoints = tape.reverse(y.eps.index());
        assert!((tangent_adjoints[0] - 18.0).abs() < 1e-10);
    }

    /// 2D Rosenbrock via Dual<Reverse<f64>>.
    #[test]
    fn rosenbrock() {
        let x_vals = [1.0_f64, 1.0];
        let v_vals = [1.0_f64, 0.0];

        let mut tape = Tape::with_capacity(200);
        let inputs: Vec<Dual<Reverse<f64>>> = x_vals
            .iter()
            .zip(v_vals.iter())
            .map(|(&xi, &vi)| {
                let (idx, _) = tape.new_variable(xi);
                let re = Reverse::from_tape(xi, idx);
                let eps = Reverse::constant(vi);
                Dual::new(re, eps)
            })
            .collect();

        let _guard = TapeGuard::new(&mut tape);

        let one: Dual<Reverse<f64>> = Dual::constant(Reverse::constant(1.0));
        let hundred: Dual<Reverse<f64>> = Dual::constant(Reverse::constant(100.0));
        let a = one - inputs[0];
        let b = inputs[1] - inputs[0] * inputs[0];
        let y = a * a + hundred * b * b;

        // At (1,1): f = 0, gradient = (0, 0)
        assert!(Scalar::value(&y.re).abs() < 1e-10);

        // Reverse from primal → gradient
        let grad = tape.reverse(y.re.index());
        assert!(grad[0].abs() < 1e-10);
        assert!(grad[1].abs() < 1e-10);

        // Reverse from tangent → HVP in direction v=(1,0)
        // H at (1,1) = [[802, -400], [-400, 200]]
        // H·(1,0) = (802, -400)
        let hvp = tape.reverse(y.eps.index());
        assert!(
            (hvp[0] - 802.0).abs() < 1e-8,
            "hvp[0] = {}, expected 802",
            hvp[0]
        );
        assert!(
            (hvp[1] - (-400.0)).abs() < 1e-8,
            "hvp[1] = {}, expected -400",
            hvp[1]
        );
    }
}

// ══════════════════════════════════════════════
//  Taylor<BReverse<f64>, K> — Taylor-over-bytecode-reverse
// ══════════════════════════════════════════════

#[cfg(all(feature = "bytecode", feature = "taylor"))]
mod taylor_breverse {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, Scalar, Taylor};

    /// Taylor<BReverse<f64>, 4> on x³.
    /// Taylor coefficients: [x³, 3x², 3x, 1] (for input x(t) = x + t).
    /// Reverse from each coefficient → derivatives of that coefficient wrt x.
    #[test]
    fn cubic_taylor_coefficients() {
        let x_val = 2.0_f64;

        let mut tape = BytecodeTape::with_capacity(200);
        let idx = tape.new_input(x_val);
        let re = BReverse::from_tape(x_val, idx);

        // Taylor input: x(t) = x + t → coefficients [x, 1, 0, 0]
        let coeffs = [
            re,
            BReverse::constant(1.0),
            BReverse::constant(0.0),
            BReverse::constant(0.0),
        ];
        let xt: Taylor<BReverse<f64>, 4> = Taylor::new(coeffs);

        let _guard = BtapeGuard::new(&mut tape);
        let y = xt * xt * xt; // x(t)³

        // y(t) = (x + t)³ = x³ + 3x²t + 3xt² + t³
        // Coefficients: [8, 12, 6, 1]
        assert!((Scalar::value(&y.coeffs[0]) - 8.0).abs() < 1e-10);
        assert!((Scalar::value(&y.coeffs[1]) - 12.0).abs() < 1e-10);
        assert!((Scalar::value(&y.coeffs[2]) - 6.0).abs() < 1e-10);
        assert!((Scalar::value(&y.coeffs[3]) - 1.0).abs() < 1e-10);

        // Reverse from coeff[0] (= x³) → d(x³)/dx = 3x² = 12
        let adj0 = tape.reverse(y.coeffs[0].index());
        assert!((adj0[0] - 12.0).abs() < 1e-10);

        // Reverse from coeff[1] (= 3x²) → d(3x²)/dx = 6x = 12
        let adj1 = tape.reverse(y.coeffs[1].index());
        assert!((adj1[0] - 12.0).abs() < 1e-10);

        // Reverse from coeff[2] (= 3x) → d(3x)/dx = 3
        let adj2 = tape.reverse(y.coeffs[2].index());
        assert!((adj2[0] - 3.0).abs() < 1e-10);

        // Reverse from coeff[3] (= 1) → constant, adjoint = 0
        if y.coeffs[3].index() != echidna::bytecode_tape::CONSTANT {
            let adj3 = tape.reverse(y.coeffs[3].index());
            assert!(adj3[0].abs() < 1e-10);
        }
    }
}

// ══════════════════════════════════════════════
//  DualVec<BReverse<f64>, N> — batched tangent-over-reverse
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod dualvec_breverse {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, DualVec, Scalar};

    /// 2 tangent directions simultaneously via DualVec<BReverse<f64>, 2>.
    /// f(x,y) = x²y → f'_x = 2xy, f'_y = x²
    /// H = [[2y, 2x], [2x, 0]]
    /// At (2,3): H = [[6,4],[4,0]]
    /// H·(1,0) = (6,4), H·(0,1) = (4,0)
    #[test]
    fn batched_two_directions() {
        let x_val = 2.0_f64;
        let y_val = 3.0_f64;

        let mut tape = BytecodeTape::with_capacity(500);
        let idx0 = tape.new_input(x_val);
        let idx1 = tape.new_input(y_val);

        let x_re = BReverse::from_tape(x_val, idx0);
        let y_re = BReverse::from_tape(y_val, idx1);

        let x: DualVec<BReverse<f64>, 2> = DualVec {
            re: x_re,
            eps: [BReverse::constant(1.0), BReverse::constant(0.0)],
        };
        let y: DualVec<BReverse<f64>, 2> = DualVec {
            re: y_re,
            eps: [BReverse::constant(0.0), BReverse::constant(1.0)],
        };

        let _guard = BtapeGuard::new(&mut tape);
        let out = x * x * y; // x²y

        // f(2,3) = 12
        assert!((Scalar::value(&out.re) - 12.0).abs() < 1e-10);

        // tangent[0] = ∇f · v1 = 2xy*1 + x²*0 = 12
        assert!((Scalar::value(&out.eps[0]) - 12.0).abs() < 1e-10);
        // tangent[1] = ∇f · v2 = 2xy*0 + x²*1 = 4
        assert!((Scalar::value(&out.eps[1]) - 4.0).abs() < 1e-10);

        // Reverse from primal → gradient = (2xy, x²) = (12, 4)
        let grad = tape.reverse(out.re.index());
        assert!((grad[0] - 12.0).abs() < 1e-10);
        assert!((grad[1] - 4.0).abs() < 1e-10);

        // Reverse from tangent[0] → H·v1 = (6, 4)
        let hvp0 = tape.reverse(out.eps[0].index());
        assert!((hvp0[0] - 6.0).abs() < 1e-10, "hvp0[0]={}", hvp0[0]);
        assert!((hvp0[1] - 4.0).abs() < 1e-10, "hvp0[1]={}", hvp0[1]);

        // Reverse from tangent[1] → H·v2 = (4, 0)
        let hvp1 = tape.reverse(out.eps[1].index());
        assert!((hvp1[0] - 4.0).abs() < 1e-10, "hvp1[0]={}", hvp1[0]);
        assert!(hvp1[1].abs() < 1e-10, "hvp1[1]={}", hvp1[1]);
    }
}

// ══════════════════════════════════════════════
//  Triple nesting: Dual<Dual<BReverse<f64>>>
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod triple_nesting {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, Dual, Scalar};

    /// Dual<Dual<BReverse<f64>>> on x³ → third derivative = 6.
    #[test]
    fn dual_dual_breverse_cubic() {
        let x_val = 2.0_f64;

        let mut tape = BytecodeTape::with_capacity(500);
        let idx = tape.new_input(x_val);
        let base = BReverse::from_tape(x_val, idx);

        // Inner Dual: tracks first tangent direction (v1 = 1)
        let inner: Dual<BReverse<f64>> = Dual::new(base, BReverse::constant(1.0));

        // Outer Dual: tracks second tangent direction (v2 = 1)
        let x: Dual<Dual<BReverse<f64>>> =
            Dual::new(inner, Dual::constant(BReverse::constant(1.0)));

        let _guard = BtapeGuard::new(&mut tape);
        let y = x * x * x;

        // y.re.re = f(x) = x³ = 8 (BReverse, tracked on tape)
        assert!((Scalar::value(&y.re.re) - 8.0).abs() < 1e-10);

        // y.re.eps = f'(x)*v1 = 3x²*1 = 12 (BReverse, tracked)
        assert!((Scalar::value(&y.re.eps) - 12.0).abs() < 1e-10);

        // y.eps.re = f'(x)*v2 = 3x²*1 = 12 (BReverse, tracked)
        assert!((Scalar::value(&y.eps.re) - 12.0).abs() < 1e-10);

        // y.eps.eps = f''(x)*v1*v2 = 6x*1*1 = 12 (BReverse, tracked)
        assert!((Scalar::value(&y.eps.eps) - 12.0).abs() < 1e-10);

        // Reverse from y.eps.eps → d(f''(x))/dx = 6 (third derivative!)
        let adj = tape.reverse(y.eps.eps.index());
        assert!(
            (adj[0] - 6.0).abs() < 1e-10,
            "third derivative: got {}, expected 6",
            adj[0]
        );
    }
}

// ══════════════════════════════════════════════
//  Scalar type chain tests
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod scalar_chain {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, Dual, Scalar};

    /// Verify Scalar::from_f and Scalar::value for Dual<BReverse<f64>>.
    /// Scalar::Float for Dual<BReverse<f64>> is BReverse<f64> (one level up).
    #[test]
    fn scalar_from_f_chain() {
        let mut tape = BytecodeTape::<f64>::with_capacity(50);
        let idx = tape.new_input(5.0);
        let base = BReverse::from_tape(5.0, idx);

        let _guard = BtapeGuard::new(&mut tape);

        // Scalar::from_f for Dual<BReverse<f64>> takes BReverse<f64>
        let x: Dual<BReverse<f64>> = Scalar::from_f(base);

        // from_f creates a constant Dual (eps = zero)
        assert!((Scalar::value(&x.re) - 5.0).abs() < 1e-10);
        assert!((Scalar::value(&x.eps) - 0.0).abs() < 1e-10);

        // Scalar::value returns BReverse<f64>
        let val: BReverse<f64> = x.value();
        assert!((Scalar::value(&val) - 5.0).abs() < 1e-10);
    }

    /// Same Scalar-generic Rosenbrock evaluated with f64, Dual<f64>,
    /// BReverse<f64>, and Dual<BReverse<f64>>.
    #[test]
    fn scalar_generic_all_modes() {
        fn rosenbrock<T: Scalar>(x: &[T]) -> T {
            let one = T::one();
            let hundred = T::from(100.0).unwrap();
            let a = one - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + hundred * b * b
        }

        // f64
        let val_f64 = rosenbrock(&[1.5_f64, 2.0]);
        assert!((val_f64 - 6.5).abs() < 1e-10);

        // Dual<f64>
        let val_dual = rosenbrock(&[Dual::constant(1.5_f64), Dual::constant(2.0)]);
        assert!((val_dual.re - 6.5).abs() < 1e-10);

        // BReverse<f64> — needs tape
        {
            let mut tape = BytecodeTape::with_capacity(100);
            let i0 = tape.new_input(1.5);
            let i1 = tape.new_input(2.0);
            let inputs = [BReverse::from_tape(1.5, i0), BReverse::from_tape(2.0, i1)];
            let _guard = BtapeGuard::new(&mut tape);
            let val_br = rosenbrock(&inputs);
            assert!((Scalar::value(&val_br) - 6.5_f64).abs() < 1e-10);
        }

        // Dual<BReverse<f64>> — needs tape
        {
            let mut tape = BytecodeTape::with_capacity(200);
            let i0 = tape.new_input(1.5);
            let i1 = tape.new_input(2.0);
            let inputs: [Dual<BReverse<f64>>; 2] = [
                Dual::new(BReverse::from_tape(1.5, i0), BReverse::constant(0.0)),
                Dual::new(BReverse::from_tape(2.0, i1), BReverse::constant(0.0)),
            ];
            let _guard = BtapeGuard::new(&mut tape);
            let val_composed = rosenbrock(&inputs);
            assert!((Scalar::value(&val_composed.re) - 6.5).abs() < 1e-10);
        }
    }
}

// ══════════════════════════════════════════════
//  Tape lifecycle
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod tape_lifecycle {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
    use echidna::{BReverse, Dual, Scalar};

    /// Verify the full lifecycle: create tape → guard → build inputs →
    /// evaluate → extract indices → reverse → drop guard → no panic.
    #[test]
    fn tape_lifecycle_composed() {
        let mut tape = BytecodeTape::<f64>::with_capacity(200);

        let idx0 = tape.new_input(3.0);
        let idx1 = tape.new_input(4.0);

        let x0: Dual<BReverse<f64>> =
            Dual::new(BReverse::from_tape(3.0, idx0), BReverse::constant(1.0));
        let x1: Dual<BReverse<f64>> =
            Dual::new(BReverse::from_tape(4.0, idx1), BReverse::constant(0.0));

        {
            let _guard = BtapeGuard::new(&mut tape);
            let y = x0 * x0 + x1 * x1; // x² + y²

            let primal_idx = y.re.index();
            let tangent_idx = y.eps.index();

            // f(3,4) = 25
            assert!((Scalar::value(&y.re) - 25.0).abs() < 1e-10);

            // tangent = 2*3*1 + 2*4*0 = 6
            assert!((Scalar::value(&y.eps) - 6.0).abs() < 1e-10);

            // Reverse from primal → gradient = (6, 8)
            let grad = tape.reverse(primal_idx);
            assert!((grad[0] - 6.0).abs() < 1e-10);
            assert!((grad[1] - 8.0).abs() < 1e-10);

            // Reverse from tangent → HVP = H·(1,0) = (2, 0) since H = 2I
            let hvp = tape.reverse(tangent_idx);
            assert!((hvp[0] - 2.0).abs() < 1e-10);
            assert!(hvp[1].abs() < 1e-10);
        }
        // Guard dropped — no panic. Tape still alive but deactivated.
        assert!(tape.num_inputs() == 2);
    }
}

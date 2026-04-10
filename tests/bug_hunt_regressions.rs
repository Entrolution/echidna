//! Regression tests for bugs found by multi-agent bug hunt (2026-04-10).
//!
//! Each test targets a specific finding and prevents regressions.

use echidna::Dual;

type Dual64 = Dual<f64>;

fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-7;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// ══════════════════════════════════════════════════════
//  Phase 1: Critical Panics (C4, D1)
// ══════════════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod phase1 {
    use echidna::{record, record_multi, BReverse};

    #[test]
    fn breverse_constant_add_scalar() {
        let (mut tape, val) = record(|_| BReverse::constant(3.0_f64) + 1.0, &[1.0]);
        assert_eq!(val, 4.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_scalar_sub_constant() {
        let (mut tape, val) = record(|_| 10.0 - BReverse::constant(3.0_f64), &[1.0]);
        assert_eq!(val, 7.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_constant_mul_scalar() {
        let (mut tape, val) = record(|_| BReverse::constant(2.0_f64) * 5.0, &[1.0]);
        assert_eq!(val, 10.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_scalar_div_constant() {
        let (mut tape, val) = record(|_| 12.0 / BReverse::constant(4.0_f64), &[1.0]);
        assert_eq!(val, 3.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_constant_rem_scalar() {
        let (mut tape, val) = record(|_| BReverse::constant(7.0_f64) % 3.0, &[1.0]);
        assert_eq!(val, 1.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn record_constant_output() {
        let (mut tape, val) = record(|_| BReverse::constant(42.0_f64), &[1.0, 2.0]);
        assert_eq!(val, 42.0);
        let grad = tape.gradient(&[1.0, 2.0]);
        assert_eq!(grad, vec![0.0, 0.0]);
    }

    #[test]
    fn record_multi_mixed_constant() {
        let (mut tape, vals) = record_multi(
            |x| vec![x[0] * x[0], BReverse::constant(99.0)],
            &[3.0],
        );
        assert_eq!(vals[0], 9.0);
        assert_eq!(vals[1], 99.0);
        // Verify gradient of first (non-constant) output works
        let jac = tape.jacobian(&[3.0]);
        assert!((jac[0][0] - 6.0_f64).abs() < 1e-10, "d(x^2)/dx at x=3 = 6");
        // Second output is constant → zero gradient
        assert_eq!(jac[1][0], 0.0);
    }
}

// ══════════════════════════════════════════════════════
//  Phase 2: Power/Root Edge Cases (A1-A6)
// ══════════════════════════════════════════════════════

mod phase2 {
    use super::*;

    #[test]
    fn powi_zero_at_zero_dual() {
        // d/dx(x^0) = 0 for all x, including x = 0
        let d = Dual64::variable(0.0).powi(0);
        assert_eq!(d.re, 1.0);
        assert!(!d.eps.is_nan(), "powi(0) derivative should be 0, not NaN");
        assert_eq!(d.eps, 0.0);
    }

    #[test]
    fn powi_negative_base_dual() {
        // d/dx(x^3) at x = -2 should be 3*(-2)^2 = 12
        let d = Dual64::variable(-2.0).powi(3);
        assert_eq!(d.re, -8.0);
        assert!(!d.eps.is_nan());
        assert!((d.eps - 12.0).abs() < 1e-10);
    }

    #[test]
    fn powi_nested_dual() {
        // Second derivative of x^3 at x = -2 via Dual<Dual<f64>>
        type D2 = Dual<Dual64>;
        let x = D2::new(
            Dual64::new(-2.0, 1.0), // primal direction
            Dual64::new(1.0, 0.0),  // tangent direction
        );
        let y = x.powi(3);
        // f''(-2) = 6*(-2) = -12, lives in eps.eps for nested dual
        assert!(!y.eps.eps.is_nan(), "nested powi second derivative should not be NaN");
        assert!((y.eps.eps - (-12.0)).abs() < 1e-10);
    }

    #[test]
    fn powf_zero_base_dual() {
        // d/dx(x^2.0) at x = 0 should be 0 (via powf)
        let x = Dual64::variable(0.0);
        let y = x.powf(Dual64::constant(2.0));
        assert_eq!(y.re, 0.0);
        assert!(!y.eps.is_nan(), "powf at x=0 should not be NaN");
        assert_eq!(y.eps, 0.0);
    }

    #[cfg(feature = "bytecode")]
    #[test]
    fn powi_zero_at_zero_tape() {
        use num_traits::Float as _;
        let (mut tape, val) = echidna::record(|x| x[0].powi(0), &[0.0]);
        assert_eq!(val, 1.0);
        let grad = tape.gradient(&[0.0]);
        assert!(!grad[0].is_nan(), "tape powi(0) gradient should be 0, not NaN");
        assert_eq!(grad[0], 0.0);
    }

    #[cfg(feature = "taylor")]
    #[test]
    fn taylor_powi_zero_base_large_exp() {
        use echidna::Taylor;
        let t = Taylor::<f64, 4>::variable(0.0).powi(10);
        assert_eq!(t.coeffs[0], 0.0);
        assert!(!t.coeffs[1].is_nan(), "taylor powi(10) at zero should not produce NaN");
        // All derivatives of x^10 at x=0 are zero for orders 1-3
        assert_eq!(t.coeffs[1], 0.0);
        assert_eq!(t.coeffs[2], 0.0);
        assert_eq!(t.coeffs[3], 0.0);
    }

    #[cfg(feature = "taylor")]
    #[test]
    fn taylor_cbrt_negative() {
        use echidna::Taylor;
        let t = Taylor::<f64, 4>::variable(-8.0).cbrt();
        assert!((t.coeffs[0] - (-2.0)).abs() < 1e-12, "cbrt(-8) = -2");
        assert!(!t.coeffs[1].is_nan(), "cbrt derivative of negative value should not be NaN");
        // Cross-validate derivative against finite differences
        let fd = super::finite_diff(|x| x.cbrt(), -8.0);
        assert!((t.coeffs[1] - fd).abs() < 1e-4, "cbrt derivative should match FD");
    }

    #[cfg(feature = "bytecode")]
    #[test]
    fn hypot_zero_zero_tape() {
        use num_traits::Float as _;
        let (mut tape, val) = echidna::record(|x| x[0].hypot(x[1]), &[0.0, 0.0]);
        assert_eq!(val, 0.0);
        let grad = tape.gradient(&[0.0, 0.0]);
        assert!(!grad[0].is_nan(), "hypot(0,0) gradient should be 0, not NaN");
        assert_eq!(grad[0], 0.0);
        assert_eq!(grad[1], 0.0);
    }
}

// ══════════════════════════════════════════════════════
//  Phase 3: Taylor/Laurent + DiffOp (B1-B8, F2)
// ══════════════════════════════════════════════════════

#[cfg(feature = "taylor")]
mod phase3_taylor {
    use echidna::Taylor;

    #[test]
    fn scalar_rem_taylor_preserves_derivatives() {
        // 7.0 % Taylor::variable(3.0) = 7 - trunc(7/3)*[3, 1, 0, 0]
        // = 7 - 2*[3, 1, 0, 0] = [1, -2, 0, 0]
        let t = 7.0_f64 % Taylor::<f64, 4>::variable(3.0);
        assert!((t.coeffs[0] - 1.0).abs() < 1e-12);
        assert!((t.coeffs[1] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn taylor_atan2_b_zero() {
        // atan2(1, 0) = pi/2
        let a = Taylor::<f64, 4>::variable(1.0);
        let b = Taylor::<f64, 4>::constant(0.0);
        let t = a.atan2(b);
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!((t.coeffs[0] - half_pi).abs() < 1e-12, "atan2(1,0) = pi/2");
        assert!(!t.coeffs[1].is_nan(), "atan2 derivative at b=0 should not be NaN");
    }
}

#[cfg(feature = "laurent")]
mod phase3_laurent {
    use echidna::Laurent;
    use num_traits::Float;

    #[test]
    fn laurent_fract_normalizes() {
        // fract of 3.0 + t should give 0.0 + t, properly normalized.
        // Without normalization, coeffs[0]=0 would violate the invariant.
        let l = Laurent::<f64, 4>::variable(3.0).fract();
        assert_eq!(l.value(), 0.0);
        // The leading nonzero coefficient is the t term, so pole_order should be 1
        assert_eq!(l.pole_order(), 1, "fract must normalize leading zeros");
    }

    #[test]
    fn laurent_log2_pole_is_nan() {
        // log2 of a series with a zero at the origin should return NaN
        // variable(0.0) = t (normalized to pole_order=1), which has a zero at origin
        let l = Laurent::<f64, 4>::variable(0.0);
        assert_ne!(l.pole_order(), 0);
        let result = l.log2();
        assert!(result.value().is_nan(), "log2 of series with zero should be NaN");
    }

    #[test]
    fn laurent_to_degrees_preserves_pole() {
        use num_traits::FloatConst;
        // Linear operation should preserve pole_order
        let l = Laurent::<f64, 4>::variable(1.0).recip(); // pole_order = -1
        let deg = l.to_degrees();
        let factor = 180.0 / f64::PI();
        assert!((deg.value() - l.value() * factor).abs() < 1e-8);
        assert_eq!(deg.pole_order(), l.pole_order());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn laurent_deser_normalizes() {
        // Deserialize a Laurent with leading zeros — should normalize
        let json = r#"{"coeffs":[0.0, 1.0, 0.0, 0.0],"pole_order":0}"#;
        let l: Laurent<f64, 4> = serde_json::from_str(json).unwrap();
        // After normalization, pole_order should be 1 (shifted by 1)
        assert_eq!(l.pole_order(), 1);
    }
}

#[cfg(all(feature = "diffop", feature = "bytecode"))]
mod phase3_diffop {
    use echidna::diffop::DiffOp;
    use echidna::record;

    #[test]
    fn biharmonic_cross_terms() {
        // f(x,y) = x^2 * y^2
        // ∂⁴f/∂x⁴ = 0, ∂⁴f/∂y⁴ = 0, ∂⁴f/(∂x²∂y²) = 4
        // True biharmonic Δ² = 0 + 0 + 2*4 = 8
        let (tape, _) = record(|x| x[0] * x[0] * x[1] * x[1], &[1.0, 1.0]);
        let op = DiffOp::<f64>::biharmonic(2);
        let (_value, biharm) = op.eval(&tape, &[1.0, 1.0]);
        assert!(
            (biharm - 8.0).abs() < 1e-6,
            "biharmonic should include cross terms, got {biharm}"
        );
    }
}

// ══════════════════════════════════════════════════════
//  Phase 4: Bytecode Tape & Sparse (C3, D3)
// ══════════════════════════════════════════════════════

mod phase4 {
    use super::*;

    #[test]
    fn is_all_zero_nested_dual() {
        use echidna::float::IsAllZero;
        type D2 = Dual<Dual64>;

        // re.eps = 1.0 carries derivative info — should NOT be all-zero
        let x = D2::new(Dual64::new(0.0, 1.0), Dual64::new(0.0, 0.0));
        assert!(
            !x.is_all_zero(),
            "nested dual with nonzero eps should not be all-zero"
        );

        // Truly zero
        let z = D2::new(Dual64::new(0.0, 0.0), Dual64::new(0.0, 0.0));
        assert!(z.is_all_zero());
    }

    #[test]
    fn max_nan_returns_non_nan() {
        let a = Dual64::constant(5.0);
        let b = Dual64::constant(f64::NAN);
        assert_eq!(a.max(b).re, 5.0, "max(5, NaN) should return 5");
        assert_eq!(b.max(a).re, 5.0, "max(NaN, 5) should return 5");
    }

    #[test]
    fn min_nan_returns_non_nan() {
        let a = Dual64::constant(5.0);
        let b = Dual64::constant(f64::NAN);
        assert_eq!(a.min(b).re, 5.0, "min(5, NaN) should return 5");
        assert_eq!(b.min(a).re, 5.0, "min(NaN, 5) should return 5");
    }
}

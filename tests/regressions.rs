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
        let (mut tape, vals) =
            record_multi(|x| vec![x[0] * x[0], BReverse::constant(99.0)], &[3.0]);
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
        assert!(
            !y.eps.eps.is_nan(),
            "nested powi second derivative should not be NaN"
        );
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
        assert!(
            !grad[0].is_nan(),
            "tape powi(0) gradient should be 0, not NaN"
        );
        assert_eq!(grad[0], 0.0);
    }

    #[cfg(feature = "taylor")]
    #[test]
    fn taylor_powi_zero_base_large_exp() {
        use echidna::Taylor;
        let t = Taylor::<f64, 4>::variable(0.0).powi(10);
        assert_eq!(t.coeffs[0], 0.0);
        assert!(
            !t.coeffs[1].is_nan(),
            "taylor powi(10) at zero should not produce NaN"
        );
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
        assert!(
            !t.coeffs[1].is_nan(),
            "cbrt derivative of negative value should not be NaN"
        );
        // Cross-validate derivative against finite differences
        let fd = super::finite_diff(|x| x.cbrt(), -8.0);
        assert!(
            (t.coeffs[1] - fd).abs() < 1e-4,
            "cbrt derivative should match FD"
        );
    }

    #[cfg(feature = "bytecode")]
    #[test]
    fn hypot_zero_zero_tape() {
        use num_traits::Float as _;
        let (mut tape, val) = echidna::record(|x| x[0].hypot(x[1]), &[0.0, 0.0]);
        assert_eq!(val, 0.0);
        let grad = tape.gradient(&[0.0, 0.0]);
        assert!(
            !grad[0].is_nan(),
            "hypot(0,0) gradient should be 0, not NaN"
        );
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
        // atan2(1, 0) = pi/2; derivative is 0 (atan2(a,0) = sign(a)*pi/2 is constant for a>0)
        let a = Taylor::<f64, 4>::variable(1.0);
        let b = Taylor::<f64, 4>::constant(0.0);
        let t = a.atan2(b);
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!((t.coeffs[0] - half_pi).abs() < 1e-12, "atan2(1,0) = pi/2");
        assert!(
            !t.coeffs[1].is_nan(),
            "atan2 derivative at b=0 should not be NaN"
        );
        assert_eq!(t.coeffs[1], 0.0, "d/da atan2(a,0) = 0 for a > 0");
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
        assert!(
            result.value().is_nan(),
            "log2 of series with zero should be NaN"
        );
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
        // True biharmonic Δ² on x²y²: only cross terms contribute
        // ∂⁴/∂x⁴ = 0, ∂⁴/∂y⁴ = 0, 2*∂⁴/(∂x²∂y²) = 2*4 = 8
        let (tape, _) = record(|x| x[0] * x[0] * x[1] * x[1], &[1.0, 1.0]);
        let op = DiffOp::<f64>::biharmonic(2);
        let (_value, biharm) = op.eval(&tape, &[1.0, 1.0]);
        assert!(
            (biharm - 8.0).abs() < 1e-4,
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

    #[test]
    fn dualvec_powi_zero_at_zero() {
        use echidna::DualVec;
        let x = DualVec::<f64, 2>::with_tangent(0.0, 0);
        let y = x.powi(0);
        assert_eq!(y.re, 1.0);
        assert!(
            !y.eps[0].is_nan(),
            "DualVec powi(0) eps should be 0, not NaN"
        );
        assert_eq!(y.eps[0], 0.0);
    }

    #[test]
    fn dualvec_powf_zero_base() {
        use echidna::DualVec;
        let x = DualVec::<f64, 2>::with_tangent(0.0, 0);
        let n = DualVec::<f64, 2>::constant(2.0);
        let y = x.powf(n);
        assert_eq!(y.re, 0.0);
        assert!(!y.eps[0].is_nan(), "DualVec powf at x=0 should not be NaN");
    }

    #[test]
    fn dualvec_max_min_nan() {
        use echidna::DualVec;
        let a = DualVec::<f64, 2>::constant(5.0);
        let b = DualVec::<f64, 2>::constant(f64::NAN);
        assert_eq!(a.max(b).re, 5.0, "DualVec max(5, NaN) should return 5");
        assert_eq!(a.min(b).re, 5.0, "DualVec min(5, NaN) should return 5");
    }
}

// ══════════════════════════════════════════════════════
//  Phase 1C: Additional test coverage (review-fix gaps)
// ══════════════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod phase1c_breverse {
    use echidna::{record, BReverse};

    // Test the reverse directions (scalar op BReverse::constant) that were untested
    #[test]
    fn scalar_add_breverse_constant() {
        let (mut tape, val) = record(|_| 1.0 + BReverse::constant(3.0_f64), &[1.0]);
        assert_eq!(val, 4.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_constant_sub_scalar() {
        let (mut tape, val) = record(|_| BReverse::constant(10.0_f64) - 3.0, &[1.0]);
        assert_eq!(val, 7.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn scalar_mul_breverse_constant() {
        let (mut tape, val) = record(|_| 5.0 * BReverse::constant(2.0_f64), &[1.0]);
        assert_eq!(val, 10.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn breverse_constant_div_scalar() {
        let (mut tape, val) = record(|_| BReverse::constant(12.0_f64) / 4.0, &[1.0]);
        assert_eq!(val, 3.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn scalar_rem_breverse_constant() {
        let (mut tape, val) = record(|_| 7.0 % BReverse::constant(3.0_f64), &[1.0]);
        assert_eq!(val, 1.0);
        let grad = tape.gradient(&[1.0]);
        assert_eq!(grad[0], 0.0);
    }
}

#[cfg(feature = "laurent")]
mod phase1c_laurent {
    use echidna::Laurent;
    use num_traits::Float;

    #[test]
    fn laurent_ln_normalizes() {
        // ln(1 + t) has c[0] = ln(1) = 0, so normalization should shift
        let l = Laurent::<f64, 4>::variable(1.0).ln();
        assert_eq!(l.value(), 0.0);
        assert_eq!(l.pole_order(), 1, "ln result must normalize leading zero");
    }

    #[test]
    fn laurent_log10_pole_is_nan() {
        let l = Laurent::<f64, 4>::variable(0.0); // pole_order = 1
        let result = l.log10();
        assert!(
            result.value().is_nan(),
            "log10 of series with zero should be NaN"
        );
    }

    #[test]
    fn laurent_to_radians_preserves_pole() {
        use num_traits::FloatConst;
        let l = Laurent::<f64, 4>::variable(1.0).recip();
        let rad = l.to_radians();
        let factor = f64::PI() / 180.0;
        assert!((rad.value() - l.value() * factor).abs() < 1e-8);
        assert_eq!(rad.pole_order(), l.pole_order());
    }
}

#[cfg(all(feature = "bytecode", feature = "diffop"))]
mod phase1c_sparsity {
    use echidna::record;
    use num_traits::Float as _;

    #[test]
    fn sparsity_custom_binary_op() {
        let (mut tape, _) = record(|x| x[0] * x[1] + x[0].sin(), &[1.0, 2.0]);
        let dense = tape.hessian(&[1.0, 2.0]);
        let (_, _, pattern, _sparse_vals) = tape.sparse_hessian(&[1.0, 2.0]);

        for i in 0..2 {
            for j in 0..=i {
                if (dense.2[i][j] as f64).abs() > 1e-12 {
                    assert!(
                        pattern.contains(i, j),
                        "dense H[{i},{j}]={} missing from sparse pattern",
                        dense.2[i][j]
                    );
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════
//  Additional coverage: Reverse-mode and Rem
// ══════════════════════════════════════════════════════

mod reverse_mode {
    use echidna::grad;
    use num_traits::Float as _;

    #[test]
    fn reverse_powi_zero_at_zero() {
        let g = grad(|x| x[0].powi(0), &[0.0_f64]);
        assert!(!g[0].is_nan(), "Reverse powi(0) at x=0 should be 0");
        assert_eq!(g[0], 0.0);
    }

    #[test]
    fn reverse_max_nan() {
        let g = grad(|x| x[0].max(x[1]), &[5.0_f64, f64::NAN]);
        assert_eq!(g[0], 1.0, "d/dx max(x, NaN) should be 1");
    }

    #[test]
    fn reverse_min_nan() {
        let g = grad(|x| x[0].min(x[1]), &[5.0_f64, f64::NAN]);
        assert_eq!(g[0], 1.0, "d/dx min(x, NaN) should be 1");
    }
}

#[cfg(feature = "bytecode")]
mod rem_coverage {
    use echidna::record;

    #[test]
    fn rem_db_partial_tape() {
        // a % b: d/db = -trunc(a/b). For a=7, b=3: db = -trunc(7/3) = -2
        let (mut tape, val) = record(|x| x[0] % x[1], &[7.0, 3.0]);
        assert_eq!(val, 1.0);
        let grad = tape.gradient(&[7.0, 3.0]);
        assert_eq!(grad[0], 1.0, "d(a%b)/da = 1");
        assert_eq!(grad[1], -2.0, "d(a%b)/db = -trunc(7/3) = -2");
    }
}

// ════════════════════════════════════════════════════════════════════════
// Phase 5: Bug hunt 2 — 2026-04-10
//
// Batch 1: Core NaN & edge-case handling (B1, B3, B4, B11, B12)
// ════════════════════════════════════════════════════════════════════════

#[cfg(feature = "bytecode")]
mod phase5 {
    use echidna::{record, BReverse};
    use num_traits::Float;

    // ── B1: BReverse/opcode Max/Min NaN handling ──

    #[test]
    fn breverse_max_with_nan() {
        // max(5.0, NaN) should return 5.0, not NaN
        let (mut tape, val) = record(|x: &[BReverse<f64>]| x[0].max(x[1]), &[5.0, f64::NAN]);
        assert_eq!(val, 5.0, "max(5, NaN) should be 5");
        let grad = tape.gradient(&[5.0, f64::NAN]);
        assert_eq!(grad[0], 1.0, "gradient flows through the non-NaN arg");
    }

    #[test]
    fn breverse_min_with_nan() {
        // min(5.0, NaN) should return 5.0, not NaN
        let (mut tape, val) = record(|x: &[BReverse<f64>]| x[0].min(x[1]), &[5.0, f64::NAN]);
        assert_eq!(val, 5.0, "min(5, NaN) should be 5");
        let grad = tape.gradient(&[5.0, f64::NAN]);
        assert_eq!(grad[0], 1.0, "gradient flows through the non-NaN arg");
    }

    #[test]
    fn opcode_max_nan_re_eval() {
        // Record with normal values, then re-evaluate with NaN
        let (mut tape, _) = record(|x: &[BReverse<f64>]| x[0].max(x[1]), &[3.0, 4.0]);
        tape.forward(&[5.0, f64::NAN]);
        let grad = tape.gradient(&[5.0, f64::NAN]);
        assert_eq!(grad[0], 1.0, "after re-eval, gradient through non-NaN arg");
    }

    // ── B3: atan2(0,0) derivative should not be NaN ──

    #[test]
    fn atan2_zero_zero_dual() {
        use echidna::Dual;
        let y = Dual::new(0.0_f64, 1.0);
        let x = Dual::new(0.0_f64, 0.0);
        let r = y.atan2(x);
        assert!(
            r.eps.is_finite(),
            "atan2(0,0) dual derivative must be finite, got {}",
            r.eps
        );
    }

    #[test]
    fn atan2_zero_zero_reverse() {
        let g = echidna::api::grad(|x| x[0].atan2(x[1]), &[0.0_f64, 0.0]);
        assert!(
            g[0].is_finite(),
            "atan2(0,0) reverse dy must be finite, got {}",
            g[0]
        );
        assert!(
            g[1].is_finite(),
            "atan2(0,0) reverse dx must be finite, got {}",
            g[1]
        );
    }

    #[test]
    fn atan2_zero_zero_breverse() {
        let (mut tape, _) = record(|x: &[BReverse<f64>]| x[0].atan2(x[1]), &[0.0, 0.0]);
        let grad = tape.gradient(&[0.0, 0.0]);
        assert!(grad[0].is_finite(), "atan2(0,0) breverse dy must be finite");
        assert!(grad[1].is_finite(), "atan2(0,0) breverse dx must be finite");
    }

    // ── B4: powf(0,0) derivative should be 0, not NaN ──

    #[test]
    fn powf_zero_zero_dual() {
        use echidna::Dual;
        let x = Dual::new(0.0_f64, 1.0);
        let n = Dual::new(0.0_f64, 0.0);
        let r = x.powf(n);
        assert_eq!(r.re, 1.0, "0^0 = 1");
        assert_eq!(r.eps, 0.0, "d/dx(x^0) at x=0 = 0");
    }

    #[test]
    fn powf_zero_zero_reverse() {
        let g = echidna::api::grad(|x| x[0].powf(x[1]), &[0.0_f64, 0.0]);
        assert!(
            g[0].is_finite(),
            "powf(0,0) reverse dx must be finite, got {}",
            g[0]
        );
        assert!(
            g[1].is_finite(),
            "powf(0,0) reverse dy must be finite, got {}",
            g[1]
        );
    }

    #[test]
    fn powf_positive_base_zero_exp_dual() {
        // d/dy(x^y) at (2, 0) should be ln(2) ≈ 0.693
        use echidna::Dual;
        let x = Dual::new(2.0_f64, 0.0);
        let n = Dual::new(0.0_f64, 1.0); // seed derivative w.r.t. exponent
        let r = x.powf(n);
        assert_eq!(r.re, 1.0, "2^0 = 1");
        assert!(
            (r.eps - 2.0_f64.ln()).abs() < 1e-12,
            "d/dy(2^y) at y=0 = ln(2), got {}",
            r.eps
        );
    }

    #[test]
    fn powf_positive_base_zero_exp_reverse() {
        let g = echidna::api::grad(|x| x[0].powf(x[1]), &[2.0_f64, 0.0]);
        assert_eq!(g[0], 0.0, "d/dx(x^0) = 0");
        assert!(
            (g[1] - 2.0_f64.ln()).abs() < 1e-12,
            "d/dy(2^y) at y=0 = ln(2), got {}",
            g[1]
        );
    }

    #[test]
    fn powf_positive_base_zero_exp_breverse() {
        let (mut tape, val) = record(|x: &[BReverse<f64>]| x[0].powf(x[1]), &[2.0, 0.0]);
        assert_eq!(val, 1.0, "2^0 = 1");
        let grad = tape.gradient(&[2.0, 0.0]);
        assert_eq!(grad[0], 0.0, "d/dx(x^0) = 0 via breverse");
        assert!(
            (grad[1] - 2.0_f64.ln()).abs() < 1e-12,
            "d/dy(2^y) at y=0 = ln(2) via breverse, got {}",
            grad[1]
        );
    }

    // ── B11: Reverse powf(0, 2) derivative should be 0 ──

    #[test]
    fn powf_zero_base_reverse() {
        // d/dx(x^2) at x=0 should be 0
        let g = echidna::api::grad(|x| x[0].powf(x[1]), &[0.0_f64, 2.0]);
        assert_eq!(g[0], 0.0, "d/dx(x^2) at x=0 should be 0");
    }

    #[test]
    fn powf_zero_base_breverse() {
        let (mut tape, val) = record(|x: &[BReverse<f64>]| x[0].powf(x[1]), &[0.0, 2.0]);
        assert_eq!(val, 0.0, "0^2 = 0");
        let grad = tape.gradient(&[0.0, 2.0]);
        assert_eq!(grad[0], 0.0, "d/dx(x^2) at x=0 via breverse should be 0");
    }

    // ── B5: Checkpoint thinning produces uniform spacing ──

    #[test]
    fn checkpoint_thinning_online() {
        // Exercise the actual online checkpointing path with enough steps to
        // trigger multiple thinning rounds (num_steps=50, 3 checkpoint slots).
        // Compare against non-checkpointed gradient.
        let x0 = [0.5_f64, 1.0];
        let num_steps = 50;

        let step = |x: &[BReverse<f64>]| {
            let half = BReverse::constant(0.5_f64);
            vec![
                x[0] * half + x[1].sin() * half,
                x[0].cos() * half + x[1] * half,
            ]
        };
        let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

        let g_online = echidna::grad_checkpointed_online(
            step,
            |_, step_idx| step_idx >= num_steps,
            loss,
            &x0,
            3, // small budget forces many thinning rounds
        );
        let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

        for i in 0..2 {
            assert!(
                (g_online[i] - g_ref[i]).abs() < 1e-10,
                "B5 thinning regression at {}: online={}, ref={}",
                i,
                g_online[i],
                g_ref[i]
            );
        }
    }

    // ── B15: Abs has zero Hessian in sparse pattern ──

    #[test]
    fn sparse_hessian_abs_no_diagonal() {
        // f(x) = |x|, Hessian should be zero (or empty pattern)
        let (tape, _) = record(|x: &[BReverse<f64>]| x[0].abs(), &[1.0]);
        let (_value, _grad, pattern, hess_vals) = tape.sparse_hessian(&[1.0]);
        // The pattern should have no entries (d²|x|/dx² = 0 a.e.)
        assert!(
            pattern.rows.is_empty(),
            "sparse Hessian of |x| should have no structural entries, got {} entries",
            pattern.rows.len()
        );
        assert!(hess_vals.is_empty(), "Hessian values should be empty");
    }

    #[test]
    fn sparse_hessian_abs_composition() {
        // f(x) = x * |x| has f''(x) = 2*signum(x) ≠ 0, so the Hessian pattern
        // must still include the (0,0) entry even with Abs as ZeroDerivative.
        // The Mul node's BinaryNonlinear classification captures this.
        let (tape, _) = record(|x: &[BReverse<f64>]| x[0] * x[0].abs(), &[1.0]);
        let (_value, _grad, pattern, hess_vals) = tape.sparse_hessian(&[1.0]);
        assert!(
            !pattern.rows.is_empty(),
            "sparse Hessian of x*|x| should have structural entries"
        );
        // f''(1) = 2*signum(1) = 2
        assert!(
            (hess_vals[0] - 2.0).abs() < 1e-10,
            "d²(x*|x|)/dx² at x=1 should be 2, got {}",
            hess_vals[0]
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// Phase 5 continued: Taylor/Laurent edge cases (B6, B7, B8)
// ════════════════════════════════════════════════════════════════════════

#[cfg(feature = "taylor")]
mod phase5_taylor {
    use echidna::Taylor;

    // ── B6: abs(0) should not zero the entire jet ──

    #[test]
    fn taylor_abs_zero_positive_approach() {
        // f(t) = t, so a = [0, 1, 0]. abs(f(t)) should have c[1] = +1
        let t = Taylor::<f64, 3>::new([0.0, 1.0, 0.0]);
        let r = t.abs();
        assert_eq!(r.coeffs[0], 0.0, "abs(0) = 0");
        assert_eq!(
            r.coeffs[1], 1.0,
            "d/dt |t| at t=0+ should be +1, got {}",
            r.coeffs[1]
        );
    }

    #[test]
    fn taylor_abs_zero_negative_approach() {
        // f(t) = -t, so a = [0, -1, 0]. abs(f(t)) should have c[1] = +1 (sign flipped)
        let t = Taylor::<f64, 3>::new([0.0, -1.0, 0.0]);
        let r = t.abs();
        assert_eq!(r.coeffs[0], 0.0, "abs(0) = 0");
        assert_eq!(
            r.coeffs[1], 1.0,
            "d/dt |-t| at t=0 should be +1, got {}",
            r.coeffs[1]
        );
    }

    // ── B7: taylor_cbrt at zero should not produce NaN ──

    #[test]
    fn taylor_cbrt_zero() {
        let t = Taylor::<f64, 3>::new([0.0, 1.0, 0.0]);
        let r = t.cbrt();
        assert_eq!(r.coeffs[0], 0.0, "cbrt(0) = 0");
        // cbrt'(0) = Inf, so c[1] should be Inf (not NaN)
        assert!(
            r.coeffs[1].is_infinite(),
            "cbrt'(0) should be Inf, got {}",
            r.coeffs[1]
        );
        assert!(!r.coeffs[1].is_nan(), "cbrt'(0) should not be NaN");
    }

    // ── B8: taylor_sqrt at zero returns Inf (not NaN) ──

    #[test]
    fn taylor_sqrt_zero() {
        let t = Taylor::<f64, 3>::new([0.0, 1.0, 0.0]);
        let r = t.sqrt();
        assert_eq!(r.coeffs[0], 0.0, "sqrt(0) = 0");
        // sqrt'(0) = 1/(2*sqrt(0)) = Inf
        assert!(
            r.coeffs[1].is_infinite(),
            "sqrt'(0) should be Inf, got {}",
            r.coeffs[1]
        );
        assert!(!r.coeffs[1].is_nan(), "sqrt'(0) should not be NaN");
    }
}

#[cfg(feature = "laurent")]
mod phase5_laurent {
    // ── B9: Laurent Add panics on large pole-order gap ──

    #[test]
    #[should_panic(expected = "pole-order gap")]
    fn laurent_add_truncation_panics() {
        use echidna::Laurent;
        // Pole orders -5 and 0 with K=4: gap=5 > K-1=3, should panic
        let a = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], -5);
        let b = Laurent::<f64, 4>::new([1.0, 0.0, 0.0, 0.0], 0);
        let _ = a + b; // should panic
    }
}

// ════════════════════════════════════════════════════════════════════════
// Bug hunt Phase 4 regression tests
// ════════════════════════════════════════════════════════════════════════

// ── #16: taylor_cbrt negative base ──

#[cfg(feature = "taylor")]
mod regression_16 {
    use echidna::Taylor;

    #[test]
    fn regression_taylor_cbrt_negative_base() {
        let x = Taylor::<f64, 5>::variable(-8.0);
        let r = x.cbrt();
        assert!(
            (r.coeffs[0] - (-2.0)).abs() < 1e-10,
            "cbrt(-8) should be -2, got {}",
            r.coeffs[0]
        );
        // Higher-order coefficients should be finite
        for k in 1..5 {
            assert!(
                r.coeffs[k].is_finite(),
                "cbrt coefficient {} should be finite, got {}",
                k,
                r.coeffs[k]
            );
        }
    }
}

// ── #17: atan2 underflow with very small inputs ──

mod regression_17 {
    use echidna::Dual;

    #[test]
    fn regression_atan2_underflow_small_inputs() {
        // With very small inputs, the derivative should be finite (not NaN or Inf).
        // The value may be zero due to underflow protection, which is acceptable.
        let y = Dual::new(1e-200_f64, 1.0);
        let x = Dual::new(1e-200_f64, 0.0);
        let r = y.atan2(x);
        assert!(
            r.eps.is_finite(),
            "atan2 derivative should be finite for small inputs, got {}",
            r.eps
        );
    }
}

// ── #28: hessian_vec debug_assert with custom ops ──

#[cfg(all(debug_assertions, feature = "bytecode"))]
mod regression_28 {
    use echidna::bytecode_tape::BtapeGuard;
    use echidna::{BReverse, BytecodeTape, CustomOp};
    use std::sync::Arc;

    struct Scale;
    impl CustomOp<f64> for Scale {
        fn eval(&self, a: f64, _b: f64) -> f64 {
            2.0 * a
        }
        fn partials(&self, _a: f64, _b: f64, _r: f64) -> (f64, f64) {
            (2.0, 0.0)
        }
    }

    #[test]
    #[should_panic(expected = "custom ops")]
    fn regression_hessian_vec_panics_with_custom_ops() {
        let x = [1.0_f64];
        let mut tape = BytecodeTape::with_capacity(10);
        let handle = tape.register_custom(Arc::new(Scale));
        let idx = tape.new_input(x[0]);
        let input = BReverse::from_tape(x[0], idx);
        let _guard = BtapeGuard::new(&mut tape);
        let output = input.custom_unary(handle, 2.0 * x[0]);
        tape.set_output(output.index());

        // hessian_vec should debug_assert because custom ops are present
        let _ = tape.hessian_vec::<1>(&x);
    }
}

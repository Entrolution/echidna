//! Regression tests for bugs found by multi-agent bug hunt (2026-04-10).
//!
//! Each test targets a specific finding and prevents regressions.

#[cfg(feature = "taylor")]
use echidna::taylor::Taylor;
#[cfg(feature = "taylor")]
use echidna::taylor_dyn::{TaylorDyn, TaylorDynGuard};
use echidna::Dual;

type Dual64 = Dual<f64>;

#[cfg(feature = "taylor")]
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
        let (tape, _) = record(|x| x[0] * x[1] + x[0].sin(), &[1.0, 2.0]);
        let dense = tape.hessian(&[1.0, 2.0]);
        let (_, _, pattern, _sparse_vals) = tape.sparse_hessian(&[1.0, 2.0]);

        for i in 0..2 {
            for j in 0..=i {
                if dense.2[i][j].abs() > 1e-12 {
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
        let output = {
            let _guard = BtapeGuard::new(&mut tape);
            input.custom_unary(handle, 2.0 * x[0])
        };
        tape.set_output(output.index());

        // hessian_vec should assert because custom ops are present
        let _ = tape.hessian_vec::<1>(&x);
    }
}

// ═════════════════════════════════���════════════════════════
//  Boundary-value derivative regression tests (PR #49 fixes)
// ══════════════════════════════════════════════════════════

// d/dx asin(x) = 1/sqrt(1-x²). At x = 1 - 1e-15, the naive formula 1 - x*x
// loses ~15 digits. The (1-x)(1+x) formulation preserves precision.
mod boundary_asin {
    use echidna::Dual;

    #[test]
    fn asin_near_boundary_dual() {
        let x_val = 1.0_f64 - 1e-15;
        let d = Dual::new(x_val, 1.0);
        let r = d.asin();
        // Analytical: 1/sqrt((1-x)(1+x)) = 1/sqrt(1e-15 * (2 - 1e-15))
        let expected = 1.0 / ((1.0 - x_val) * (1.0 + x_val)).sqrt();
        let rel_err = ((r.eps - expected) / expected).abs();
        assert!(
            rel_err < 1e-6,
            "asin derivative near boundary: got {}, expected {}, rel_err={}",
            r.eps,
            expected,
            rel_err
        );
    }

    #[test]
    fn acos_near_boundary_dual() {
        let x_val = 1.0_f64 - 1e-15;
        let d = Dual::new(x_val, 1.0);
        let r = d.acos();
        let expected = -1.0 / ((1.0 - x_val) * (1.0 + x_val)).sqrt();
        let rel_err = ((r.eps - expected) / expected).abs();
        assert!(
            rel_err < 1e-6,
            "acos derivative near boundary: got {}, expected {}, rel_err={}",
            r.eps,
            expected,
            rel_err
        );
    }

    #[test]
    fn atanh_near_boundary_dual() {
        let x_val = 1.0_f64 - 1e-15;
        let d = Dual::new(x_val, 1.0);
        let r = d.atanh();
        let expected = 1.0 / ((1.0 - x_val) * (1.0 + x_val));
        let rel_err = ((r.eps - expected) / expected).abs();
        assert!(
            rel_err < 1e-6,
            "atanh derivative near boundary: got {}, expected {}, rel_err={}",
            r.eps,
            expected,
            rel_err
        );
    }
}

#[cfg(feature = "bytecode")]
mod boundary_bytecode {
    use num_traits::Float;

    fn breverse_grad(
        f: impl FnOnce(&[echidna::BReverse<f64>]) -> echidna::BReverse<f64>,
        x: &[f64],
    ) -> Vec<f64> {
        let (mut tape, _) = echidna::record(f, x);
        tape.gradient(x)
    }

    #[test]
    fn asin_near_boundary_breverse() {
        let x_val = 1.0_f64 - 1e-15;
        let g = breverse_grad(|x| x[0].asin(), &[x_val]);
        let expected = 1.0 / ((1.0 - x_val) * (1.0 + x_val)).sqrt();
        let rel_err = ((g[0] - expected) / expected).abs();
        assert!(
            rel_err < 1e-6,
            "BReverse asin near boundary: rel_err={}",
            rel_err
        );
    }

    #[test]
    fn atan2_large_inputs_breverse() {
        let g = breverse_grad(|x| x[0].atan2(x[1]), &[1e200, 1e200]);
        // At (a,a), d/da atan2(a,b) = b/(a²+b²) = 1/(2a) ≈ 5e-201
        // This is subnormal for f64, so it may flush to zero on some platforms.
        // The key property: it must be finite (not NaN or Inf).
        assert!(
            g[0].is_finite(),
            "atan2 da gradient should be finite for large inputs, got {}",
            g[0]
        );
        assert!(
            g[1].is_finite(),
            "atan2 db gradient should be finite for large inputs, got {}",
            g[1]
        );
    }

    #[test]
    fn div_small_denominator_breverse() {
        let x_val = 1e-200_f64;
        let g = breverse_grad(|x| x[0].recip(), &[x_val]);
        // d/dx(1/x) = -1/x² = -1e400 → Inf for f64. That's the correct IEEE result.
        // The key is it should NOT be NaN.
        assert!(
            !g[0].is_nan(),
            "recip derivative should not be NaN for small x"
        );
    }
}

#[cfg(feature = "taylor")]
mod boundary_taylor {
    use echidna::Taylor;

    #[test]
    fn taylor_hypot_large_inputs() {
        // hypot(a, b) at a₀=1e200, b₀=1e200 with direction (1, 0)
        let a = Taylor::<f64, 3>::new([1e200, 1.0, 0.0]);
        let b = Taylor::<f64, 3>::constant(1e200);
        let r = a.hypot(b);
        assert!(r.coeffs[0].is_finite(), "hypot primal should be finite");
        assert!(
            r.coeffs[1] != 0.0 && r.coeffs[1].is_finite(),
            "hypot first derivative should be non-zero and finite, got {}",
            r.coeffs[1]
        );
    }

    #[test]
    fn taylor_hypot_small_inputs() {
        let a = Taylor::<f64, 3>::new([1e-200, 1.0, 0.0]);
        let b = Taylor::<f64, 3>::constant(1e-200);
        let r = a.hypot(b);
        assert!(r.coeffs[0].is_finite(), "hypot primal should be finite");
        assert!(
            r.coeffs[1].is_finite(),
            "hypot first derivative should be finite, got {}",
            r.coeffs[1]
        );
    }

    #[test]
    fn taylor_asin_near_boundary() {
        // asin at x₀ = 1 - 1e-10 — derivative should be large but finite
        let x = Taylor::<f64, 3>::new([1.0 - 1e-10, 1.0, 0.0]);
        let r = x.asin();
        assert!(r.coeffs[0].is_finite(), "asin primal should be finite");
        assert!(
            r.coeffs[1].is_finite() && r.coeffs[1] > 0.0,
            "asin first Taylor coefficient should be positive and finite, got {}",
            r.coeffs[1]
        );
    }
}

// ══════════════════════════════════════════════════════
//  Cycle 5 Phase 1: Correctness fixes
// ══════════════════════════════════════════════════════

#[cfg(feature = "taylor")]
#[test]
fn taylor_max_nan_guard() {
    let valid = Taylor::<f64, 3>::new([5.0, 1.0, 0.0]);
    let nan = Taylor::<f64, 3>::new([f64::NAN, 1.0, 0.0]);

    // max(valid, NaN) should return valid
    let r = valid.max(nan);
    assert_eq!(r.coeffs[0], 5.0, "max(valid, NaN) should return valid");

    // max(NaN, valid) should return valid
    let r = nan.max(valid);
    assert_eq!(r.coeffs[0], 5.0, "max(NaN, valid) should return valid");

    // min(valid, NaN) should return valid
    let r = valid.min(nan);
    assert_eq!(r.coeffs[0], 5.0, "min(valid, NaN) should return valid");

    // min(NaN, valid) should return valid
    let r = nan.min(valid);
    assert_eq!(r.coeffs[0], 5.0, "min(NaN, valid) should return valid");
}

#[cfg(feature = "taylor")]
#[test]
fn taylor_dyn_max_nan_guard() {
    let _guard = TaylorDynGuard::<f64>::new(3);

    let valid = TaylorDyn::variable(5.0);
    let nan = TaylorDyn::constant(f64::NAN);

    let r = valid.max(nan);
    assert_eq!(
        r.value(),
        5.0,
        "TaylorDyn max(valid, NaN) should return valid"
    );

    let r = nan.max(valid);
    assert_eq!(
        r.value(),
        5.0,
        "TaylorDyn max(NaN, valid) should return valid"
    );

    let r = valid.min(nan);
    assert_eq!(
        r.value(),
        5.0,
        "TaylorDyn min(valid, NaN) should return valid"
    );

    let r = nan.min(valid);
    assert_eq!(
        r.value(),
        5.0,
        "TaylorDyn min(NaN, valid) should return valid"
    );
}

#[cfg(feature = "taylor")]
#[test]
fn taylor_acosh_near_domain_boundary() {
    // acosh at x₀ = 1 + 1e-10 — cancellation-safe form should preserve precision
    let x = Taylor::<f64, 3>::new([1.0 + 1e-10, 1.0, 0.0]);
    let r = x.acosh();
    assert!(
        r.coeffs[0].is_finite(),
        "acosh primal should be finite near x=1"
    );
    assert!(
        r.coeffs[1].is_finite() && r.coeffs[1] > 0.0,
        "acosh first Taylor coeff should be positive and finite near x=1, got {}",
        r.coeffs[1]
    );
    // Compare with asin at the equivalent point for similar precision
    let y = Taylor::<f64, 3>::new([1.0 - 1e-10, 1.0, 0.0]);
    let asin_r = y.asin();
    // Both should have similar magnitudes of first coefficient
    assert!(
        (r.coeffs[1].ln() - asin_r.coeffs[1].ln()).abs() < 2.0,
        "acosh and asin should have similar-magnitude derivatives near their boundaries"
    );
}

#[test]
fn div_forward_partial_small_denominator() {
    // d/db(a/b) at b = 1e-155 should not overflow to inf
    let a = Dual64::new(1e-308, 0.0);
    let b = Dual64::new(1e-155, 1.0);
    let r = a / b;
    // d/db(a/b) = -a/b² = -1e-308 / 1e-310 = -1e2 (approximately)
    // With the old formula -a * (1/b)² the intermediate (1/b)² overflows
    assert!(
        r.eps.is_finite(),
        "d/db(a/b) should be finite for small b when a is also small, got {}",
        r.eps
    );
}

#[cfg(feature = "bytecode")]
#[test]
fn div_reverse_partial_via_tape() {
    // Same test through bytecode tape reverse mode
    use echidna::{record, BReverse};
    let (mut tape, _) = record(|x: &[BReverse<f64>]| x[0] / x[1], &[1e-308, 1e-155]);
    let grad = tape.gradient(&[1e-308, 1e-155]);
    assert!(
        grad[1].is_finite(),
        "tape gradient d/db(a/b) should be finite, got {}",
        grad[1]
    );
    // Should be approximately -a/b² = -1e-308/1e-310 ≈ -100
    assert!(
        (grad[1] + 100.0).abs() < 10.0,
        "tape gradient d/db should be ≈ -100, got {}",
        grad[1]
    );
}

#[test]
fn powf_forward_partial_underflow() {
    // d/da(a^2) at a = 1e-200: r = a² underflows to 0, but derivative 2a = 2e-200 is nonzero
    let a = Dual64::new(1e-200, 1.0);
    let r = a.powf(Dual64::new(2.0, 0.0));
    assert!(
        r.eps != 0.0,
        "d/da(a^2) at a=1e-200 should be nonzero, got {}",
        r.eps
    );
    assert!(
        (r.eps - 2e-200).abs() < 1e-210,
        "d/da(a^2) at a=1e-200 should be ≈ 2e-200, got {}",
        r.eps
    );
}

#[cfg(feature = "bytecode")]
#[test]
fn powf_reverse_partial_underflow_tape() {
    use echidna::{record, BReverse};
    use num_traits::Float; // powf is on the Float trait
    let two_const = 2.0f64;
    let (mut tape, _) = record(
        |x: &[BReverse<f64>]| x[0].powf(BReverse::constant(two_const)),
        &[1e-200],
    );
    let grad = tape.gradient(&[1e-200]);
    assert!(
        grad[0] != 0.0,
        "tape gradient d/da(a^2) at a=1e-200 should be nonzero, got {}",
        grad[0]
    );
}

// ══════════════════════════════════════════════════════
//  Nested-dual tangent guards, structural-zero convention,
//  and division primal rounding
// ══════════════════════════════════════════════════════

/// Second-order forward-over-forward: guards that test whether a tangent is
/// zero must inspect the WHOLE tangent (`is_all_zero`), not just its primal —
/// `PartialEq` on `Dual` compares only `.re`, so a tangent with zero
/// first-order but live second-order component would otherwise be dropped.
mod nested_second_order {
    use echidna::{Dual, DualVec};

    type D2 = Dual<Dual<f64>>;

    /// Forward-over-forward seed: inner eps tracks d/dx, outer eps re-tracks it.
    fn var(x0: f64) -> D2 {
        Dual::new(Dual::new(x0, 1.0), Dual::new(1.0, 0.0))
    }
    fn cst(c: f64) -> D2 {
        Dual::new(Dual::new(c, 0.0), Dual::new(0.0, 0.0))
    }

    #[test]
    fn recip_keeps_second_order_through_zero_first_derivative() {
        // f(x) = 1/(x²+1): f'(0) = 0 but f''(0) = -2. At x = 0 the inner
        // tangent of g = x²+1 has primal 0 with a live second-order part.
        let x = var(0.0);
        let g = x * x + cst(1.0);
        let y = g.recip();
        assert_eq!(y.eps.eps, -2.0, "d²/dx² 1/(x²+1) at 0 via recip");
        // The equivalent spelling 1/g must agree.
        let y2 = cst(1.0) / g;
        assert_eq!(y2.eps.eps, -2.0, "d²/dx² 1/(x²+1) at 0 via Div");
    }

    #[test]
    fn dual_vec_recip_keeps_second_order_per_lane() {
        type Dv = DualVec<Dual<f64>, 1>;
        let x: Dv = DualVec::new(Dual::new(0.0, 1.0), [Dual::new(1.0, 0.0)]);
        let one: Dv = DualVec::constant(Dual::new(1.0, 0.0));
        let g = x * x + one;
        let y = g.recip();
        assert_eq!(y.eps[0].eps, -2.0);
    }

    #[test]
    fn powf_constant_integer_fast_path_keeps_exponent_second_order() {
        // Base 3 constant; exponent h with h(x0) = 2, h'(x0) = 0, h''(x0) = 1.
        // d²/dx²(3^h) = 3^h·(ln3)²·h'² + 3^h·ln3·h'' = 9·ln3.
        // The seed h' = 0, h'' = 1 is deliberate: the exponent tangent's
        // primal is zero, so a primal-only "is the exponent constant?" test
        // would wrongly dispatch to powi and drop the ln-weighted term.
        let a = cst(3.0);
        let h: D2 = Dual::new(Dual::new(2.0, 0.0), Dual::new(0.0, 1.0));
        let f = a.powf(h);
        let expected = 9.0 * 3.0_f64.ln();
        assert!(
            (f.eps.eps - expected).abs() < 1e-12,
            "d²/dx² 3^h(x): expected {expected}, got {}",
            f.eps.eps
        );
    }

    #[test]
    fn dual_vec_powf_fast_path_keeps_exponent_second_order() {
        type Dv = DualVec<Dual<f64>, 1>;
        let a: Dv = DualVec::constant(Dual::new(3.0, 0.0));
        let h: Dv = DualVec::new(Dual::new(2.0, 0.0), [Dual::new(0.0, 1.0)]);
        let f = a.powf(h);
        let expected = 9.0 * 3.0_f64.ln();
        assert!((f.eps[0].eps - expected).abs() < 1e-12);
    }
}

/// Elementals with unbounded derivatives at reachable finite points: a
/// structurally-zero (constant) tangent stays exactly 0 there instead of
/// IEEE 0×Inf = NaN, while a live tangent keeps the non-finite derivative.
/// Matches the hypot-origin convention and the reverse-sweep zero-adjoint
/// skip.
mod structural_zero_tangent {
    use echidna::{Dual, DualVec};

    macro_rules! zero_tangent_stays_zero {
        ($name:ident, $method:ident, $p:expr, $sign:expr) => {
            #[test]
            fn $name() {
                let p: f64 = $p;
                let sign: f64 = $sign;
                let c = Dual::<f64>::constant(p).$method();
                assert_eq!(
                    c.eps,
                    0.0,
                    "constant through {} at {p}: tangent must be exactly 0, got {}",
                    stringify!($method),
                    c.eps
                );
                let v = Dual::<f64>::variable(p).$method();
                assert!(
                    v.eps.is_infinite() && v.eps.signum() == sign,
                    "variable through {} at {p}: tangent must be {}Inf, got {}",
                    stringify!($method),
                    if sign > 0.0 { "+" } else { "-" },
                    v.eps
                );
                let dv = DualVec::<f64, 2>::new(p, [0.0, 1.0]).$method();
                assert_eq!(dv.eps[0], 0.0, "zero lane must stay 0");
                assert!(
                    dv.eps[1].is_infinite() && dv.eps[1].signum() == sign,
                    "live lane must keep the signed Inf"
                );
            }
        };
    }

    zero_tangent_stays_zero!(sqrt_at_zero, sqrt, 0.0, 1.0);
    zero_tangent_stays_zero!(cbrt_at_zero, cbrt, 0.0, 1.0);
    zero_tangent_stays_zero!(recip_at_zero, recip, 0.0, -1.0);
    zero_tangent_stays_zero!(ln_at_zero, ln, 0.0, 1.0);
    zero_tangent_stays_zero!(log2_at_zero, log2, 0.0, 1.0);
    zero_tangent_stays_zero!(log10_at_zero, log10, 0.0, 1.0);
    zero_tangent_stays_zero!(asin_at_one, asin, 1.0, 1.0);
    zero_tangent_stays_zero!(acos_at_one, acos, 1.0, -1.0);
    zero_tangent_stays_zero!(acosh_at_one, acosh, 1.0, 1.0);
    zero_tangent_stays_zero!(atanh_at_one, atanh, 1.0, 1.0);

    #[test]
    fn ln_1p_at_boundary() {
        let c = Dual::<f64>::constant(-1.0).ln_1p();
        assert_eq!(c.eps, 0.0);
        let v = Dual::<f64>::variable(-1.0).ln_1p();
        assert!(v.eps.is_infinite() && v.eps > 0.0);
    }

    #[test]
    fn powi_negative_exponent_at_zero() {
        // 1/x² at x = 0: same singularity class as recip. The derivative
        // -2·x^(-3) approaches -Inf from the right.
        let c = Dual::<f64>::constant(0.0).powi(-2);
        assert_eq!(c.eps, 0.0);
        let v = Dual::<f64>::variable(0.0).powi(-2);
        assert!(v.eps.is_infinite() && v.eps < 0.0);
    }

    #[test]
    fn l2_norm_at_origin_matches_hypot() {
        // sqrt(x²+y²) and hypot(x,y) at the origin must agree: the tangent of
        // x²+y² is exactly zero there (2x·ẋ = 0), so the structural-zero
        // convention gives 0 through sqrt — the same value hypot's origin
        // short-circuit produces.
        let x = Dual::<f64>::variable(0.0);
        let y = Dual::<f64>::constant(0.0);
        let via_sqrt = (x * x + y * y).sqrt();
        let via_hypot = x.hypot(y);
        assert_eq!(via_sqrt.eps, 0.0);
        assert_eq!(via_hypot.eps, 0.0);
        assert_eq!(via_sqrt.eps, via_hypot.eps);
    }
}

/// powf at non-finite / boundary points: the exponent-direction term must not
/// poison a finite base-direction derivative, in either forward or reverse
/// mode.
mod powf_nonfinite_edges {
    use echidna::{grad, Dual, DualVec, Reverse};
    use num_traits::Float as NumFloat;

    #[test]
    fn overflowed_primal_keeps_finite_base_derivative() {
        // f(x) = x^2.5 at x = 1e200: the primal overflows to Inf but the true
        // derivative 2.5·x^1.5 ≈ 2.5e300 is representable. The exponent term
        // (Inf·ln(x)·0) must not turn it into NaN.
        let x = Dual::<f64>::new(1e200, 1.0);
        let r = x.powf(Dual::constant(2.5));
        assert!(r.re.is_infinite());
        let expected = 2.5e300;
        assert!(
            ((r.eps - expected) / expected).abs() < 1e-10,
            "expected ≈{expected}, got {}",
            r.eps
        );
    }

    #[test]
    fn reverse_overflowed_primal_keeps_finite_base_derivative() {
        let g = grad(
            |x: &[Reverse<f64>]| NumFloat::powf(x[0], Reverse::constant(2.5)),
            &[1e200],
        );
        let expected = 2.5e300;
        assert!(
            ((g[0] - expected) / expected).abs() < 1e-10,
            "reverse d/dx x^2.5 at 1e200: expected ≈{expected}, got {}",
            g[0]
        );
    }

    #[test]
    fn infinite_base_gives_infinite_derivative_not_nan() {
        let x = Dual::<f64>::new(f64::INFINITY, 1.0);
        let r = x.powf(Dual::constant(2.5));
        assert!(r.re.is_infinite() && r.re > 0.0);
        assert!(
            r.eps.is_infinite() && r.eps > 0.0,
            "d/dx x^2.5 at +Inf must be +Inf, got {}",
            r.eps
        );
        let g = grad(
            |x: &[Reverse<f64>]| NumFloat::powf(x[0], Reverse::constant(2.5)),
            &[f64::INFINITY],
        );
        assert!(
            g[0].is_infinite() && g[0] > 0.0,
            "reverse d/dx x^2.5 at +Inf must be +Inf, got {}",
            g[0]
        );
    }

    // Non-discriminating regression guard: the constant-integer fast path
    // already routes Inf^0 through powi (result {1, 0}).
    #[test]
    fn infinite_base_zero_constant_exponent() {
        let x = Dual::<f64>::new(f64::INFINITY, 1.0);
        let r = x.powf(Dual::constant(0.0));
        assert_eq!(r.re, 1.0);
        assert_eq!(r.eps, 0.0);
    }

    #[test]
    fn fractional_exponent_at_zero_base() {
        // powf's base-direction path (distinct code from sqrt's chain):
        // 0^0.5 with a constant base must not produce 0.5·0^(-0.5)·0 = NaN.
        let c = Dual::<f64>::constant(0.0).powf(Dual::constant(0.5));
        assert_eq!(c.re, 0.0);
        assert_eq!(c.eps, 0.0);
        let v = Dual::<f64>::variable(0.0).powf(Dual::constant(0.5));
        assert!(v.eps.is_infinite() && v.eps > 0.0);
    }

    #[test]
    fn dual_vec_mixed_lanes_fractional_exponent_at_zero_base() {
        let x = DualVec::<f64, 2>::new(0.0, [0.0, 1.0]);
        let n = DualVec::<f64, 2>::constant(0.5);
        let r = x.powf(n);
        assert_eq!(r.eps[0], 0.0, "constant lane must stay 0");
        assert!(
            r.eps[1].is_infinite() && r.eps[1] > 0.0,
            "live lane must be +Inf, got {}",
            r.eps[1]
        );
    }

    #[test]
    fn dual_vec_infinite_base_mixed_lanes() {
        // Both direction factors are non-finite at an infinite base
        // (dx_factor = n·Inf^{n-1} = Inf, dy_factor = Inf·ln(Inf) = Inf):
        // each constant lane must contribute exactly 0 in its direction,
        // each live lane must stay +Inf — otherwise Inf·0 = NaN leaks in.
        let x = DualVec::<f64, 2>::new(f64::INFINITY, [1.0, 0.0]);
        let n = DualVec::<f64, 2>::new(2.5, [0.0, 1.0]);
        let r = x.powf(n);
        assert!(
            r.eps[0].is_infinite() && r.eps[0] > 0.0,
            "base-live lane must be +Inf, got {}",
            r.eps[0]
        );
        assert!(
            r.eps[1].is_infinite() && r.eps[1] > 0.0,
            "exponent-live lane must be +Inf, got {}",
            r.eps[1]
        );
    }

    #[test]
    fn dual_vec_infinite_base_zero_exponent_mixed_lanes() {
        // Exponent primal exactly 0 with one live lane bypasses the
        // all-constant powi fast path: dy = ln(Inf) = +Inf, so the constant
        // lane needs the structural-zero short-circuit to stay 0.
        let x = DualVec::<f64, 2>::new(f64::INFINITY, [0.0, 0.0]);
        let n = DualVec::<f64, 2>::new(0.0, [0.0, 1.0]);
        let r = x.powf(n);
        assert_eq!(r.re, 1.0);
        assert_eq!(r.eps[0], 0.0, "constant lane must stay exactly 0");
        assert!(
            r.eps[1].is_infinite() && r.eps[1] > 0.0,
            "live lane must be +Inf, got {}",
            r.eps[1]
        );
    }

    #[test]
    fn f32_overflowed_primal_keeps_finite_base_derivative() {
        // The f32 overflow threshold (~3.4e38) is ~270 orders of magnitude
        // below f64's: x^2.5 at x = 2e25 overflows the f32 primal while the
        // derivative 2.5·x^1.5 ≈ 2.2e38 is still representable.
        let x = Dual::<f32>::new(2e25, 1.0);
        let r = x.powf(Dual::constant(2.5));
        assert!(r.re.is_infinite());
        assert!(
            r.eps.is_finite() && r.eps > 0.0,
            "f32 derivative must stay finite, got {}",
            r.eps
        );
    }
}

/// Division primal must be the single correctly-rounded IEEE quotient —
/// bit-identical to `f64` division and to the bytecode tape — not the
/// double-rounded `a·(1/b)`.
mod div_primal_rounding {
    use echidna::{Dual, DualVec};

    #[test]
    fn dual_div_primal_is_correctly_rounded() {
        let cases: [(f64, f64); 6] = [
            (3.0, 10.0),
            (7.0, 3.0),
            (1.0, 49.0),
            (1e-3, 7.0),
            (5.5, 1.1),
            (-3.0, 10.0),
        ];
        for (a, b) in cases {
            let exact = (a / b).to_bits();
            let q = (Dual::<f64>::variable(a) / Dual::variable(b)).re;
            assert_eq!(q.to_bits(), exact, "Dual/Dual {a}/{b}");
            let q = (Dual::<f64>::variable(a) / b).re;
            assert_eq!(q.to_bits(), exact, "Dual/f64 {a}/{b}");
            let q = (a / Dual::<f64>::variable(b)).re;
            assert_eq!(q.to_bits(), exact, "f64/Dual {a}/{b}");
            let q = (DualVec::<f64, 1>::with_tangent(a, 0) / DualVec::with_tangent(b, 0)).re;
            assert_eq!(q.to_bits(), exact, "DualVec/DualVec {a}/{b}");
            let q = (DualVec::<f64, 1>::with_tangent(a, 0) / b).re;
            assert_eq!(q.to_bits(), exact, "DualVec/f64 {a}/{b}");
            let q = (a / DualVec::<f64, 1>::with_tangent(b, 0)).re;
            assert_eq!(q.to_bits(), exact, "f64/DualVec {a}/{b}");
        }
    }

    #[test]
    fn reverse_div_primal_is_correctly_rounded() {
        use echidna::{vjp, Reverse};
        let (out, g) = vjp(|x: &[Reverse<f64>]| vec![x[0] / x[1]], &[3.0, 10.0], &[1.0]);
        assert_eq!(out[0].to_bits(), (3.0_f64 / 10.0).to_bits());
        // Adjoints: d/da (a/b) = 1/b = 0.1, d/db (a/b) = -a/b² = -0.03.
        assert!((g[0] - 0.1).abs() < 1e-16);
        assert!((g[1] - (-0.03)).abs() < 1e-16);
    }

    #[test]
    fn f32_div_primal_is_correctly_rounded() {
        // Pairs chosen so that a·(1/b) genuinely double-rounds in f32
        // (3/7: 0x3edb6db7 direct vs 0x3edb6db8 via reciprocal); pairs like
        // 3/10 happen to agree in f32 and would not discriminate.
        for (a, b) in [(3.0_f32, 7.0_f32), (7.0_f32, 3.0_f32)] {
            let q = (Dual::<f32>::variable(a) / Dual::variable(b)).re;
            assert_eq!(q.to_bits(), (a / b).to_bits(), "{a}/{b}");
        }
    }

    #[cfg(feature = "bytecode")]
    #[test]
    fn taped_div_agrees_bitwise_with_f64() {
        use echidna::{record, BReverse};
        let (mut tape, val) = record(|x: &[BReverse<f64>]| x[0] / x[1], &[3.0, 10.0]);
        assert_eq!(val.to_bits(), (3.0_f64 / 10.0).to_bits());
        let g = tape.gradient(&[3.0, 10.0]);
        assert!((g[0] - 0.1).abs() < 1e-16);
        assert!((g[1] - (-0.03)).abs() < 1e-16);
    }
}

/// Dropping BtapeGuards out of LIFO order would install a stale (possibly
/// dangling) tape pointer in the thread-local cell — the contract violation
/// must be a deterministic panic in every build profile, not a debug-only
/// check.
#[cfg(feature = "bytecode")]
mod btape_guard_lifo {
    use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};

    #[test]
    #[should_panic(expected = "LIFO")]
    fn out_of_order_guard_drop_panics() {
        let mut t1 = BytecodeTape::<f64>::new();
        let mut t2 = BytecodeTape::<f64>::new();
        let g1 = BtapeGuard::new(&mut t1);
        let g2 = BtapeGuard::new(&mut t2);
        drop(g1);
        drop(g2);
    }
}

/// BReverse values are bound to the recording that produced them; using one
/// while a different tape is active must panic in debug builds (release
/// builds omit the tag and the check).
#[cfg(all(feature = "bytecode", debug_assertions))]
mod breverse_tape_identity {
    use echidna::{record, BReverse};

    #[test]
    #[should_panic(expected = "another recording")]
    fn stashed_value_across_recordings_panics() {
        let stash = std::cell::Cell::new(None);
        let _ = record(
            |x: &[BReverse<f64>]| {
                stash.set(Some(x[0]));
                x[0] * 2.0
            },
            &[1.0],
        );
        let stale = stash.get().unwrap();
        let _ = record(move |x: &[BReverse<f64>]| x[0] + stale, &[1.0]);
    }

    #[test]
    #[should_panic(expected = "another recording")]
    fn nested_recording_capturing_outer_variable_panics() {
        let _ = record(
            |x: &[BReverse<f64>]| {
                let outer = x[0];
                let _ = record(move |y: &[BReverse<f64>]| y[0] * outer, &[2.0]);
                x[0]
            },
            &[1.0],
        );
    }
}

//! Phase 8 Commit 2 regressions — opcode semantics.
//!
//! Covers L14 (domain-restricted ops propagate NaN partial strictly outside
//! their valid interval; boundary values keep the ±Inf one-sided limit) and
//! L15 (Abs at 0 uses symmetric-midpoint subgradient, sign-bit independent).
//!
//! L16 from the plan (unreachable!() for reverse_partials(Powi, ..)) was
//! dropped during implementation: `reverse_partials` is a public API that
//! the existing `tests/opcode_edge_cases.rs::reverse_powi` test relies on.
//! The elaborate Powi branch is therefore not dead code — the runtime tape
//! dispatchers specialise it inline for performance, but the standalone
//! function remains a legitimate entry point.

#![cfg(feature = "bytecode")]

use echidna::opcode::{reverse_partials, OpCode};

// L14: Ln / Log2 / Log10 / Ln1p / Atanh emit (NaN, 0) strictly outside
// their valid domain. At the boundary (a == 0 for Ln etc.) the original
// IEEE `1/0 = ±Inf` limit is preserved, matching CPU-side expectations.
#[test]
fn l14_ln_nan_partial_outside_domain() {
    let (da, _) = reverse_partials::<f64>(OpCode::Ln, -1.0, 0.0, f64::NAN);
    assert!(da.is_nan(), "Ln at a=-1 must emit NaN partial, got {}", da);

    let (da, _) = reverse_partials::<f64>(OpCode::Log2, -0.5, 0.0, f64::NAN);
    assert!(da.is_nan(), "Log2 at a<0 must emit NaN partial");

    let (da, _) = reverse_partials::<f64>(OpCode::Log10, -0.25, 0.0, f64::NAN);
    assert!(da.is_nan(), "Log10 at a<0 must emit NaN partial");

    let (da, _) = reverse_partials::<f64>(OpCode::Ln1p, -2.0, 0.0, f64::NAN);
    assert!(da.is_nan(), "Ln1p at a<-1 must emit NaN partial");

    let (da, _) = reverse_partials::<f64>(OpCode::Atanh, 1.5, 0.0, f64::NAN);
    assert!(da.is_nan(), "Atanh at |a|>1 must emit NaN partial");

    let (da, _) = reverse_partials::<f64>(OpCode::Atanh, -2.0, 0.0, f64::NAN);
    assert!(da.is_nan(), "Atanh at a<-1 must emit NaN partial");
}

#[test]
fn l14_ln_boundary_preserves_inf_limit() {
    // Ln at a=0: IEEE `1/0 = +Inf` must survive the domain guard so the
    // one-sided limit of the derivative is preserved.
    let (da, _) = reverse_partials::<f64>(OpCode::Ln, 0.0, 0.0, f64::NEG_INFINITY);
    assert!(
        da.is_infinite() && da > 0.0,
        "Ln at a=0 should give +Inf, got {}",
        da
    );
}

#[test]
fn l14_ln_finite_partial_inside_domain() {
    let (da, _) = reverse_partials::<f64>(OpCode::Ln, 2.0, 0.0, 2.0_f64.ln());
    assert!(
        (da - 0.5).abs() < 1e-15,
        "Ln at 2 must give 1/2, got {}",
        da
    );

    let (da, _) = reverse_partials::<f64>(OpCode::Ln1p, 3.0, 0.0, 4.0_f64.ln());
    assert!(
        (da - 0.25).abs() < 1e-15,
        "Ln1p at 3 must give 1/(1+3), got {}",
        da
    );
}

// L15: Abs subgradient at ±0 is 0 (symmetric midpoint), not ±0 inherited
// from the sign bit.
#[test]
fn l15_abs_at_zero_returns_zero_regardless_of_sign_bit() {
    let (da, _) = reverse_partials::<f64>(OpCode::Abs, 0.0_f64, 0.0, 0.0);
    assert_eq!(da, 0.0);

    // Construct an actual negative zero via bit-pattern so the test really
    // exercises sign-bit-independence. `(-1.0).copysign(0.0)` is a trap: it
    // returns +1.0 (magnitude of -1, sign of +0), not negative zero.
    let neg_zero = f64::from_bits(0x8000_0000_0000_0000u64);
    assert!(neg_zero.is_sign_negative());
    let (da, _) = reverse_partials::<f64>(OpCode::Abs, neg_zero, 0.0, 0.0);
    assert_eq!(da, 0.0, "Abs subgradient at -0.0 must be 0, not -0.0");
    assert!(!da.is_sign_negative(), "result must be +0.0, not -0.0");
}

#[test]
fn l15_abs_nonzero_unchanged() {
    let (da, _) = reverse_partials::<f64>(OpCode::Abs, 2.0, 0.0, 2.0);
    assert_eq!(da, 1.0);
    let (da, _) = reverse_partials::<f64>(OpCode::Abs, -3.0, 0.0, 3.0);
    assert_eq!(da, -1.0);
}

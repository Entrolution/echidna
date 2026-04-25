//! Tests for Laurent<F, K> — singularity analysis via Laurent series arithmetic.

#![cfg(feature = "laurent")]

use approx::assert_relative_eq;
use echidna::Laurent;
use num_traits::Float;

type L4 = Laurent<f64, 4>;

// ══════════════════════════════════════════════
//  Construction and basic properties
// ══════════════════════════════════════════════

#[test]
fn constant_and_variable() {
    let c: L4 = Laurent::constant(3.0);
    assert_relative_eq!(c.value(), 3.0);
    assert_eq!(c.pole_order(), 0);
    assert!(!c.has_pole());

    let v: L4 = Laurent::variable(2.0);
    assert_relative_eq!(v.value(), 2.0);
    assert_eq!(v.pole_order(), 0);
    assert_relative_eq!(v.coeff(0), 2.0);
    assert_relative_eq!(v.coeff(1), 1.0);
}

// ══════════════════════════════════════════════
//  Regular arithmetic (should match Taylor)
// ══════════════════════════════════════════════

#[test]
fn add_sub_regular() {
    let a: L4 = Laurent::variable(1.0); // 1 + t
    let b: L4 = Laurent::constant(2.0); // 2
    let sum = a + b;
    assert_relative_eq!(sum.value(), 3.0);
    assert_relative_eq!(sum.coeff(1), 1.0); // t coefficient preserved

    let diff = a - b;
    assert_relative_eq!(diff.value(), -1.0);
    assert_relative_eq!(diff.coeff(1), 1.0);
}

#[test]
fn mul_regular() {
    let a: L4 = Laurent::variable(2.0); // 2 + t
    let b: L4 = Laurent::variable(3.0); // 3 + t
    let prod = a * b;
    // (2 + t)(3 + t) = 6 + 5t + t²
    assert_relative_eq!(prod.value(), 6.0);
    assert_relative_eq!(prod.coeff(1), 5.0);
    assert_relative_eq!(prod.coeff(2), 1.0);
}

// ══════════════════════════════════════════════
//  Pole creation and arithmetic
// ══════════════════════════════════════════════

#[test]
fn div_creates_pole() {
    // 1 / t → pole_order = -1
    let one: L4 = Laurent::constant(1.0);
    let t: L4 = Laurent::variable(0.0); // 0 + t (zero at origin, pole_order becomes 1 after normalize)
    let result = one / t;
    assert_eq!(result.pole_order(), -1);
    assert_relative_eq!(result.leading_coefficient(), 1.0);
    assert!(result.has_pole());
    assert!(result.value().is_infinite());
}

#[test]
fn pole_cancellation() {
    // t * (1/t) = 1
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let product = t * inv_t;
    assert_eq!(product.pole_order(), 0);
    assert_relative_eq!(product.value(), 1.0, max_relative = 1e-12);
}

#[test]
fn pole_arithmetic() {
    // (1/t²) + (1/t) — different pole orders must align
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t; // pole_order = -1
    let inv_t2 = inv_t * inv_t; // pole_order = -2

    let sum = inv_t2 + inv_t;
    // 1/t² + 1/t → pole_order = -2, coeffs[0] = 1 (t^{-2}), coeffs[1] = 1 (t^{-1})
    assert_eq!(sum.pole_order(), -2);
    assert_relative_eq!(sum.coeff(-2), 1.0);
    assert_relative_eq!(sum.coeff(-1), 1.0);
}

#[test]
fn residue_extraction() {
    // Build a Laurent series with a specific residue at t^{-1}
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(3.0) / t; // 3/t → residue = 3
    assert_relative_eq!(inv_t.residue(), 3.0);

    // 1/t² has residue 0 (no t^{-1} term)
    let inv_t2 = Laurent::constant(1.0) / (t * t);
    assert_relative_eq!(inv_t2.residue(), 0.0);
}

// ══════════════════════════════════════════════
//  Transcendentals with poles
// ══════════════════════════════════════════════

#[test]
fn sqrt_even_pole() {
    // sqrt(1/t²) = 1/t (when pole_order is -2, even → ok)
    let t: L4 = Laurent::variable(0.0);
    let inv_t2 = Laurent::constant(1.0) / (t * t);
    let result = inv_t2.sqrt();
    assert_eq!(result.pole_order(), -1);
    assert_relative_eq!(result.leading_coefficient(), 1.0, max_relative = 1e-10);
}

#[test]
fn exp_of_pole_is_nan() {
    // exp(1/t) → essential singularity → NaN
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let result = inv_t.exp();
    assert!(result.value().is_nan());
}

#[test]
fn ln_of_zero_is_nan() {
    // ln(t) where t→0 → logarithmic singularity → NaN
    let t: L4 = Laurent::variable(0.0);
    let result = t.ln();
    assert!(result.value().is_nan());
}

#[test]
fn powi_with_poles() {
    // (1/t)^3 → pole_order = -3
    let t: L4 = Laurent::variable(0.0);
    let inv_t = Laurent::constant(1.0) / t;
    let result = inv_t.powi(3);
    assert_eq!(result.pole_order(), -3);
    assert_relative_eq!(result.leading_coefficient(), 1.0, max_relative = 1e-10);
}

// ══════════════════════════════════════════════
//  Regular matches Taylor
// ══════════════════════════════════════════════

#[test]
fn regular_matches_taylor() {
    use echidna::Taylor;

    // For pole_order=0, Laurent arithmetic should match Taylor exactly
    type T4 = Taylor<f64, 4>;

    let lt: L4 = Laurent::variable(2.0);
    let tt: T4 = Taylor::variable(2.0);

    // exp
    let le = lt.exp();
    let te = tt.exp();
    for k in 0..4 {
        assert_relative_eq!(le.coeff(k as i32), te.coeff(k), max_relative = 1e-12);
    }

    // sin
    let ls = lt.sin();
    let ts = tt.sin();
    for k in 0..4 {
        assert_relative_eq!(ls.coeff(k as i32), ts.coeff(k), max_relative = 1e-12);
    }
}

// ══════════════════════════════════════════════
//  Tape integration
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
#[test]
fn tape_forward_tangent() {
    use echidna::record;

    // f(x) = x² + sin(x)
    let (tape, _) = record(|x| x[0] * x[0] + x[0].sin(), &[1.0]);

    let x: L4 = Laurent::variable(1.0);
    let mut buf = Vec::new();
    tape.forward_tangent(&[x], &mut buf);
    let result = buf[tape.output_index()];

    // value should be 1 + sin(1)
    assert_relative_eq!(result.value(), 1.0 + 1.0_f64.sin(), max_relative = 1e-12);
    assert_eq!(result.pole_order(), 0);
}

// ══════════════════════════════════════════════
//  Singularity detection
// ══════════════════════════════════════════════

#[cfg(feature = "bytecode")]
#[test]
fn singularity_detection_reciprocal() {
    use echidna::record;

    // f(x) = 1/x — has a pole at x = 0
    let (tape, _) = record(|x| x[0].recip(), &[1.0]);

    // Evaluate at x = t (variable with value 0) to detect the singularity
    let x: L4 = Laurent::variable(0.0);
    let mut buf = Vec::new();
    tape.forward_tangent(&[x], &mut buf);
    let result = buf[tape.output_index()];

    // Should detect a pole: pole_order < 0
    assert!(result.has_pole());
    assert_eq!(result.pole_order(), -1);
}

// ══════════════════════════════════════════════
//  Normalization
// ══════════════════════════════════════════════

#[test]
fn normalization_strips_zeros() {
    // Create a Laurent with leading zeros: [0, 0, 1, 0] at pole_order = -2
    // After normalization: [1, 0] at pole_order = 0
    let l: L4 = Laurent::new([0.0, 0.0, 1.0, 0.0], -2);
    assert_eq!(l.pole_order(), 0);
    assert_relative_eq!(l.leading_coefficient(), 1.0);
    assert_relative_eq!(l.value(), 1.0);
}

// ══════════════════════════════════════════════
//  Bug hunt regression tests
// ══════════════════════════════════════════════

type L3 = Laurent<f64, 3>;

// ── #12: Laurent Sub pole-order gap panic ──

#[test]
#[should_panic(expected = "pole-order gap")]
fn regression_12_laurent_sub_pole_order_gap_panics() {
    let a = L3::new([1.0, 0.0, 0.0], -5);
    let b = L3::new([1.0, 0.0, 0.0], 0);
    let _ = a - b; // pole-order gap = 5 > K-1 = 2, should panic
}

// ── #13: Laurent is_zero semantics ──

#[test]
fn regression_13_laurent_nonzero_with_positive_pole_order_is_not_zero() {
    use num_traits::Zero;
    let l = L3::new([1.0, 0.0, 0.0], 1);
    assert!(
        !l.is_zero(),
        "Laurent with nonzero coefficients and pole_order>0 should not be zero"
    );
}

// ── #21: Laurent max/min NaN ──

#[test]
fn regression_21_laurent_max_nan_returns_non_nan() {
    let valid = L3::constant(5.0);
    let nan = L3::constant(f64::NAN);
    let r1 = valid.max(nan);
    assert!(!r1.value().is_nan(), "max(valid, nan) should return valid");
    let r2 = nan.max(valid);
    assert!(!r2.value().is_nan(), "max(nan, valid) should return valid");
}

#[test]
fn regression_21_laurent_min_nan_returns_non_nan() {
    let valid = L3::constant(5.0);
    let nan = L3::constant(f64::NAN);
    let r1 = valid.min(nan);
    assert!(!r1.value().is_nan(), "min(valid, nan) should return valid");
    let r2 = nan.min(valid);
    assert!(!r2.value().is_nan(), "min(nan, valid) should return valid");
}

// ── #22: Laurent powi pole_order overflow ──
//
// Historical behaviour: `Laurent::powi` panicked on `pole_order * n` i32
// overflow. That violates the `num_traits::Float::powi` contract (which
// must not panic on domain edges). Current behaviour: return `nan_laurent()`.

#[test]
fn regression_22_laurent_powi_pole_order_overflow_returns_nan() {
    let l = L3::new([1.0, 0.0, 0.0], i32::MAX / 2 + 1);
    let r = l.powi(3); // pole_order * 3 overflows i32
    assert!(
        r.coeff(r.pole_order()).is_nan(),
        "overflow should yield NaN Laurent, not panic"
    );
}

// ══════════════════════════════════════════════
//  WS8 — Laurent::hypot kernel migration
// ══════════════════════════════════════════════

// Post-WS8, `Laurent::hypot` delegates the rescale / sum-of-squares /
// sqrt coefficient math to `taylor_ops::taylor_hypot` (the shared CPU
// HYPOT kernel also used by `Taylor::hypot` and `TaylorDyn::hypot`).
// These tests pin the public `Laurent::hypot` behaviour end-to-end.

/// Baseline: hypot(3, 4) = 5. Pole_order = 0 on both. Pin primal and
/// first-order derivative (should be 0 for constants).
#[test]
fn hypot_basic_3_4() {
    let a: L4 = Laurent::constant(3.0);
    let b: L4 = Laurent::constant(4.0);
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), 0);
    assert_relative_eq!(r.coeff(0), 5.0);
    assert_relative_eq!(r.coeff(1), 0.0);
    assert_relative_eq!(r.coeff(2), 0.0);
}

/// Rescale check: leading coefficients at 1e200. Without the
/// leading-coefficient rescale, `a² + b²` overflows to Inf and the
/// sqrt chain NaN-propagates. With rescale, everything stays finite.
#[test]
fn hypot_large_magnitude_rescale() {
    let a: L4 = Laurent::constant(1e200);
    let b: L4 = Laurent::constant(1e200);
    let r = a.hypot(b);
    assert!(
        r.coeff(0).is_finite(),
        "rescale must keep primal finite, got {}",
        r.coeff(0)
    );
    let expected = 2.0_f64.sqrt() * 1e200;
    assert_relative_eq!(r.coeff(0), expected, max_relative = 1e-12);
}

/// Reverse-direction rescale guard: coefficients at 1e-200. The
/// rescale keeps the intermediate `a/s = b/s ≈ 1.0` so the squaring
/// doesn't underflow to zero.
#[test]
fn hypot_small_magnitude_no_underflow() {
    let a: L4 = Laurent::constant(1e-200);
    let b: L4 = Laurent::constant(1e-200);
    let r = a.hypot(b);
    assert!(
        r.coeff(0) > 0.0,
        "rescale must keep primal non-zero, got {}",
        r.coeff(0)
    );
    let expected = 2.0_f64.sqrt() * 1e-200;
    assert_relative_eq!(r.coeff(0), expected, max_relative = 1e-12);
}

/// Matched negative pole_order (both -2). The rebase is a no-op
/// because the pole orders already agree; delegates directly to
/// `taylor_hypot` on the raw coefficient arrays. Result pole_order
/// equals the common input pole order.
#[test]
fn hypot_matched_pole_negative() {
    // a = 3·t^-2 + 4·t^-1  (pole_order = -2, coeffs = [3, 4, 0, 0])
    // b = 4·t^-2 + 3·t^-1  (pole_order = -2, coeffs = [4, 3, 0, 0])
    // hypot leading: sqrt(3² + 4²)·t^-2 = 5·t^-2
    let a: L4 = Laurent::new([3.0, 4.0, 0.0, 0.0], -2);
    let b: L4 = Laurent::new([4.0, 3.0, 0.0, 0.0], -2);
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), -2);
    assert_relative_eq!(r.coeff(-2), 5.0);
}

/// Mismatched pole_order: rebase-to-min alignment. Golden-value
/// regression pin — the expected output was traced by hand using
/// the polynomial-sqrt recurrence on the rebased+rescaled inputs,
/// and cross-checks against the analytical Taylor expansion of
/// `sqrt(9 + 24t + 17t²)`. See the plan file's trace for the
/// derivation.
#[test]
fn hypot_mismatched_pole_rebase() {
    // a = 3 + 4·t at pole_order = 0
    let a: L4 = Laurent::new([3.0, 4.0, 0.0, 0.0], 0);
    // b = t at pole_order = 1 (represents 1·t^1 = t)
    let b: L4 = Laurent::new([1.0, 0.0, 0.0, 0.0], 1);
    // hypot(3+4t, t) = sqrt((3+4t)² + t²) = sqrt(9 + 24t + 17t²).
    // Expected expansion: [3, 4, 1/6, -2/9] at pole_order = 0.
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), 0);
    assert_relative_eq!(r.coeff(0), 3.0);
    assert_relative_eq!(r.coeff(1), 4.0);
    assert_relative_eq!(r.coeff(2), 1.0 / 6.0, max_relative = 1e-12);
    assert_relative_eq!(r.coeff(3), -2.0 / 9.0, max_relative = 1e-12);
}

/// Zero-at-origin inputs with non-zero higher-order seeds. Each
/// operand represents `c·t` (leading zero in t^0 is absorbed into
/// pole_order via `Laurent::new`'s normalization, producing
/// pole_order=1 with non-zero leading). Post-rebase both share
/// pole_order=1, the kernel runs its normal scale>0 path on
/// `[3, 0, 0, 0]` vs `[4, 0, 0, 0]`, and the result represents
/// `hypot(3t, 4t) = 5·t` near origin.
///
/// Note on the kernel's `scale == 0` recursive shift-and-square
/// branch: that path only fires when both kernel-level
/// coefficient arrays have `[0] == 0`. Laurent's normalization
/// prevents that for any single non-zero operand, and the
/// both-identically-zero case is short-circuited in
/// `Laurent::hypot` before the kernel call. So the recursion
/// branch is unreachable from Laurent in practice — this test
/// exercises the normal pole-shift path that makes it so.
#[test]
fn hypot_zero_origin_via_pole_shift() {
    let a: L4 = Laurent::new([0.0, 3.0, 0.0, 0.0], 0);
    let b: L4 = Laurent::new([0.0, 4.0, 0.0, 0.0], 0);
    // Both normalize to pole_order=1 with leading coefficient = 3
    // / 4 respectively.
    assert_eq!(a.pole_order(), 1);
    assert_eq!(b.pole_order(), 1);
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), 1);
    assert_relative_eq!(r.coeff(1), 5.0, max_relative = 1e-12);
    assert_relative_eq!(r.coeff(2), 0.0);
    assert_relative_eq!(r.coeff(3), 0.0);
}

/// Cone-point singularity: both operands identically zero.
/// `Laurent::hypot` short-circuits here (ahead of the shared
/// kernel call) to preserve the invariant
/// `Laurent::zero().hypot(Laurent::zero()) == Laurent::zero()`.
///
/// Without the short-circuit, the underlying `taylor_hypot`
/// kernel's "singular-derivative convention at a true zero"
/// would return `[0, Inf, Inf, ...]` as the coefficient array,
/// and Laurent's `normalize()` would then strip the leading zero
/// and bump `pole_order`, producing a nonsense "Laurent pole of
/// order 1 with infinite derivative" representation. The short-
/// circuit keeps the zero Laurent clean.
#[test]
fn hypot_both_identically_zero() {
    let a: L4 = Laurent::zero();
    let b: L4 = Laurent::zero();
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), 0);
    for k in 0..4 {
        assert_eq!(
            r.coeff(k),
            0.0,
            "hypot(zero, zero) must stay all-zero; coeff({k}) = {}",
            r.coeff(k)
        );
    }
}

/// Extreme pole-order mismatch: when the delta exceeds the Laurent
/// window `K`, the higher-pole operand rebases to an all-zero array
/// (everything "falls off" the end). The kernel then sees
/// `hypot(a_rebased, 0)` and returns `|a_rebased|` at the primal,
/// with the higher-order coefficients of `|a|` propagating through.
///
/// Pins the `rebase_to` early-return-on-`delta >= K` branch — any
/// future refactor that drops it would crash via `copy_from_slice`
/// out-of-bounds instead of silently handling the truncation.
#[test]
fn hypot_extreme_pole_mismatch_truncates_operand() {
    // a at pole_order = 0, b at pole_order = K = 4. The rebase
    // delta (4) is >= K, so b_rebased becomes [0; 4].
    let a: L4 = Laurent::new([3.0, 0.0, 0.0, 0.0], 0);
    let b: L4 = Laurent::new([5.0, 0.0, 0.0, 0.0], 4);
    let r = a.hypot(b);
    // Result should be hypot(3, 0) = 3 at pole_order=0; higher-
    // order coefficients all zero (hypot of a constant is a
    // constant).
    assert_eq!(r.pole_order(), 0);
    assert_relative_eq!(r.coeff(0), 3.0);
    assert_relative_eq!(r.coeff(1), 0.0);
    assert_relative_eq!(r.coeff(2), 0.0);
    assert_relative_eq!(r.coeff(3), 0.0);
}

/// Pole-order near `i32::MAX` with a second operand at a much
/// smaller pole_order. The `saturating_sub` in `rebase_to` must
/// avoid i32 wraparound; the `delta >= K` guard then truncates
/// the high-pole operand to zero. Pins this defensive path.
#[test]
fn hypot_pole_i32_max_saturates() {
    let a: L4 = Laurent::new([2.0, 0.0, 0.0, 0.0], 0);
    let b: L4 = Laurent::new([1.0, 0.0, 0.0, 0.0], i32::MAX);
    let r = a.hypot(b);
    // Expected: b truncates (delta saturates at MAX - 0 = MAX,
    // cast to usize >> K), so hypot(a, 0) = a.
    assert_eq!(r.pole_order(), 0);
    assert_relative_eq!(r.coeff(0), 2.0);
}

/// Rescale-pivot difference pin: pre-WS8, `Laurent::hypot` rescaled
/// by `max over ALL coefficients`; `taylor_hypot` (the post-WS8
/// kernel) rescales only by `max(|a[0]|, |b[0]|)` — leading
/// coefficients only. Both produce mathematically equivalent
/// results (rescaling is a numerical-stability device, not a
/// semantic operation), but specific intermediate values can
/// differ for pathological inputs where a higher-order coefficient
/// exceeds the leading.
///
/// Analytical reference for `hypot(1 + 1e10·t, 2)` expanded around
/// `t = 0`:
///     P(t) = (1 + 1e10·t)² + 4 = 5 + 2·10^10·t + 10^20·t²
///     h(t) = √P(t). Polynomial-sqrt recurrence:
///     h[0] = √5
///     h[1] = 10^10 / √5 = 2·10^9·√5
///     h[2] = (10^20 - h[1]²) / (2·√5) = 8·10^18·√5
///     h[3] = -2·h[1]·h[2] / (2·√5) = -1.6·10^28·√5
#[test]
fn hypot_rescale_pivot_pin() {
    let a: L4 = Laurent::new([1.0, 1e10, 0.0, 0.0], 0);
    let b: L4 = Laurent::constant(2.0);
    let r = a.hypot(b);
    assert_eq!(r.pole_order(), 0);
    let sqrt5 = 5.0_f64.sqrt();
    assert_relative_eq!(r.coeff(0), sqrt5, max_relative = 1e-12);
    assert_relative_eq!(r.coeff(1), 2e9 * sqrt5, max_relative = 1e-10);
    assert_relative_eq!(r.coeff(2), 8e18 * sqrt5, max_relative = 1e-10);
    assert_relative_eq!(r.coeff(3), -1.6e28 * sqrt5, max_relative = 1e-10);
}

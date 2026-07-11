//! Per-opcode math kernels (single source of truth for formulas).
//!
//! Each opcode's numerical formula is easy to duplicate across the CPU AD
//! types (`Dual`, `DualVec`, `Reverse`, `BReverse`, `Laurent`) and the
//! bytecode-tape opcode dispatcher (`opcode.rs`), with the GPU side — WGSL
//! shaders and the CUDA NVRTC kernel — carrying yet another copy in their
//! respective source-level languages. A change to one copy can drift silently
//! from the others; such drift has produced real bugs (atan large-|a|, div
//! small-|b|, hypot Inf handling, out-of-domain log/atanh partials).
//!
//! Any numerical formula shared between the CPU AD types **and** the opcode
//! dispatcher should live here as a single generic function over
//! `num_traits::Float`; each AD type delegates to the helper so CPU drift
//! becomes impossible, and GPU drift is caught by `tests/gpu_cpu_parity.rs`
//! (and `tests/domain_nan_convention.rs` for the out-of-domain convention).
//! Currently extracted: `hypot_partials`, `atan2_partials`, `atan_deriv`,
//! `asinh_deriv`, `acosh_deriv`, and the domain-guarded log / `atanh`
//! derivatives below. Every new extraction must come with a call-site refactor
//! so we don't add abstraction for its own sake.
//!
//! # The structural-zero convention
//!
//! Zero tangents, zero partials, and singular points are handled by ONE
//! convention, applied symmetrically in both AD directions:
//!
//! 1. **Forward mode — zero tangents.** A structurally zero tangent
//!    ([`IsAllZero`](crate::float::IsAllZero), every component exactly
//!    zero) through any unary elemental stays exactly zero, even where the
//!    derivative is unbounded (`sqrt`/`ln` at 0, `atanh` at ±1),
//!    overflowed (`exp`/`sinh`/`cosh` at large arguments), or NaN
//!    (out-of-domain primals). A constant is a constant; IEEE `0 × Inf =
//!    NaN` must not leak into lanes that carry no derivative. Implemented
//!    in `Dual`/`DualVec::chain` and, on the GPU, as blanket unary guards
//!    in the tangent kernels.
//!
//! 2. **Reverse mode — zero multipliers.** An exactly-zero partial
//!    (multiplier) absorbs ANY adjoint, including Inf/NaN adjoints from a
//!    chained singularity downstream: a zero partial means this input does
//!    not move the output locally, so the adjoint's magnitude is
//!    irrelevant. Implemented as accumulation guards in every reverse
//!    sweep (bytecode, eager, tangent-carrying, WGSL, CUDA). Note the
//!    guard necessarily fires for every exact-zero partial — singular-point
//!    zeros (hypot/atan2 at the origin), kink losing branches (max/min),
//!    and regular zeros (mul by a zero operand, signum away from 0) are
//!    indistinguishable at sweep time, and the convention is correct for
//!    all of them. This completes the long-documented zero-ADJOINT skip
//!    (the JAX-style `0 × NaN → 0` trade-off noted in `tape.rs`). A
//!    corollary shared by both skips: reverse mode does not preserve a
//!    derivative's NEGATIVE zero — adjoint accumulation starts at `+0.0`
//!    and IEEE `+0.0 + (-0.0) = +0.0` — while forward mode's direct
//!    products do. Signed zeros of derivatives are a forward-mode-only
//!    guarantee.
//!
//! 3. **Singular-primal partials.** Kernels return a defined convention at
//!    non-differentiable points rather than raw IEEE fallout: `hypot` and
//!    `atan2` return exact `(0, 0)` at the origin; `ln`/`log2`/`log10`
//!    derivatives follow the IEEE reciprocal sign at ±0 (`1/-0 = -Inf`);
//!    `sqrt`/`cbrt` derivatives are `+Inf` at 0; out-of-domain points give
//!    NaN partials (which are NOT zero, so rule 2 does not absorb them —
//!    NaN still propagates to signal the domain error).
//!
//! 4. **Two mechanisms for origin short-circuits.** The eager single-use
//!    tape (`Reverse`) may short-circuit a singular-point op to a tape-free
//!    constant (`hypot`/`atan2` at the origin) because that tape is never
//!    re-evaluated. The re-evaluable bytecode tape must NOT do this — a
//!    recording-point constant would be stale on replay at other inputs —
//!    so it keeps the node and relies on rule 2 at sweep time. Both
//!    mechanisms produce the same gradients; do not "unify" them, the
//!    asymmetry is load-bearing.
//!
//! A live (nonzero) tangent or adjoint through a singular point keeps the
//! non-finite result — the convention protects constants, never masks
//! singularities.

use num_traits::Float;

/// Partial derivatives of `hypot(a, b) = sqrt(a² + b²)`.
///
/// `r` is the primal value `hypot(a, b)`. Returns `(∂r/∂a, ∂r/∂b)`.
///
/// At the origin (`r == 0`) the gradient is mathematically undefined
/// but we return `(0, 0)` to match the JAX / PyTorch convention and
/// to avoid emitting NaN into downstream adjoint chains.
#[inline]
pub fn hypot_partials<T: Float>(a: T, b: T, r: T) -> (T, T) {
    if r == T::zero() {
        (T::zero(), T::zero())
    } else {
        (a / r, b / r)
    }
}

/// Partial derivatives of `atan2(a, b)`.
///
/// Formula: `∂/∂a = b/(a²+b²)`, `∂/∂b = -a/(a²+b²)`. Factored as
/// `(b/h)/h` and `-(a/h)/h` where `h = hypot(a, b)` so the
/// intermediate never forms `a²+b²` directly — that would overflow
/// for `|a|, |b| > sqrt(MAX)` and underflow for values below
/// `sqrt(MIN_POSITIVE)`.
///
/// At the origin (`h == 0`) the gradient is mathematically undefined
/// and we return `(0, 0)`.
#[inline]
pub fn atan2_partials<T: Float>(a: T, b: T) -> (T, T) {
    let h = a.hypot(b);
    if h == T::zero() {
        (T::zero(), T::zero())
    } else {
        // Unary `-x` (via `Float: Neg<Output = Self>`) preserves the IEEE
        // signed-zero invariant `-(+0.0) = -0.0`. `T::zero() - x` would
        // flatten to `+0.0` at `x = +0.0` under round-to-nearest, silently
        // changing sign-bit semantics observable by downstream `copysign`
        // / `is_sign_negative` consumers.
        (b / h / h, -a / h / h)
    }
}

/// Derivative of `atan(a)` with an overflow-safe large-|a| path.
///
/// For `|a| ≤ 1e8`, returns `1/(1+a²)`. For `|a| > 1e8`, reformulates
/// via `u = 1/a` so `1/(1+a²) = u²/(1+u²)`, keeping every intermediate
/// in-range even at `|a| ≈ 1e19` where `a² overflows in f32.
#[inline]
pub fn atan_deriv<T: Float>(a: T) -> T {
    let one = T::one();
    if a.abs() > T::from(1e8).unwrap() {
        let inv = one / a;
        inv * inv / (one + inv * inv)
    } else {
        one / (one + a * a)
    }
}

/// Derivative of `asinh(a) = ln(a + sqrt(1+a²))` with a large-|a|
/// overflow-safe path.
///
/// For `|a| ≤ 1e8`, returns `1/sqrt(1+a²)`. For `|a| > 1e8`, uses
/// `u = 1/a` and `|u|/sqrt(1+u²)` so `1+a²` can't overflow.
#[inline]
pub fn asinh_deriv<T: Float>(a: T) -> T {
    let one = T::one();
    if a.abs() > T::from(1e8).unwrap() {
        let inv = one / a;
        inv.abs() / (one + inv * inv).sqrt()
    } else {
        one / (a * a + one).sqrt()
    }
}

/// Derivative of `acosh(a) = ln(a + sqrt(a²-1))`, guarded to the domain `a >= 1`
/// with a large-`a` overflow-safe path.
///
/// Strictly outside the domain (`a < 1`) returns NaN, matching [`ln_deriv`] /
/// [`atanh_deriv`]: the factored `(a-1)·(a+1)` self-guards only for `-1 < a < 1`
/// (sqrt of a negative), but `a <= -1` leaves it `>= 0` → a finite but meaningless
/// value (and `-Inf` at `a = -1` via `1/sqrt(-0.0)`), as does the overflow branch,
/// so the guard is explicit. The boundary `a = 1` keeps the IEEE `1/sqrt(0) = +Inf`
/// one-sided limit.
///
/// For `1 <= a <= 1e8`, returns `1/sqrt((a-1)·(a+1))`. The factored form (vs naive
/// `a*a - 1`) avoids catastrophic cancellation near `a = 1`: at `a = 1 + ε`, `a*a`
/// rounds to `1 + 2ε` and `a*a - 1 = 2ε` loses the `ε²` contribution, while
/// `(a-1)·(a+1) = ε·(2 + ε)` retains it. For `a > 1e8`, uses `u = 1/a` and
/// `|u|/sqrt(1-u²)`.
///
/// Every CPU AD mode (`Dual`, `DualVec`, `Reverse`) and the bytecode `OpCode`
/// dispatcher delegate here, so the factored form and the domain guard share a
/// single source of truth. The GPU kernels carry the same factored form and guard
/// in their own languages: the wgpu shaders (`reverse.wgsl`, `tangent_forward.wgsl`,
/// `tangent_reverse.wgsl` — both HVP phases) and the CUDA `tape_eval.cu` at all four
/// derivative sweeps (reverse, tangent-forward, and both HVP phases). Pinned by
/// `acosh_deriv_factored_form_keeps_precision_near_one` (factoring) and
/// `tests/domain_nan_convention.rs` (the guard, cross-mode + GPU).
#[inline]
pub fn acosh_deriv<T: Float>(a: T) -> T {
    let one = T::one();
    // Domain guard first (see doc): a < 1 is strictly out of domain. After it,
    // `a >= 1`, so the large-|a| test can use `a` directly instead of `a.abs()`.
    if a < one {
        return T::nan();
    }
    if a > T::from(1e8).unwrap() {
        let inv = one / a;
        inv.abs() / (one - inv * inv).sqrt()
    } else {
        one / ((a - one) * (a + one)).sqrt()
    }
}

/// Derivative of `ln(a) = 1/a`, guarded to the real domain.
///
/// Domain-restricted logs emit a NaN partial *strictly* outside their valid
/// interval (`a < 0`) so a caller that supplied an out-of-domain input sees NaN
/// rather than a finite-but-meaningless value (`1/-2 = -0.5`). The boundary
/// `a = 0` is left to IEEE arithmetic, sign of zero included: `1/(+0) = +Inf`
/// (the one-sided derivative limit) and `1/(-0) = -Inf`. Unlike `abs_deriv`,
/// ±0 is deliberately NOT collapsed to one value here — every path (all CPU AD
/// modes and both GPU backends) computes the same `1/a` formula, with the ±0
/// result following each backend's IEEE division, so special-casing `-0` would
/// require touching every one of them for an input whose primal `ln(-0) = -Inf`
/// already signals the degenerate limit. Every AD mode and the bytecode
/// `OpCode` dispatcher delegate here so the convention has a single source of
/// truth; the wgpu and CUDA kernels carry the same guard in their own languages
/// (reverse, forward-tangent, and HVP sweeps), pinned by
/// `tests/domain_nan_convention.rs`.
#[inline]
pub fn ln_deriv<T: Float>(a: T) -> T {
    if a >= T::zero() {
        T::one() / a
    } else {
        T::nan()
    }
}

/// Derivative of `log2(a) = 1/(a·ln 2)`, guarded to `a >= 0` (see [`ln_deriv`]).
#[inline]
pub fn log2_deriv<T: Float>(a: T) -> T {
    if a >= T::zero() {
        T::one() / (a * T::from(std::f64::consts::LN_2).unwrap())
    } else {
        T::nan()
    }
}

/// Derivative of `log10(a) = 1/(a·ln 10)`, guarded to `a >= 0` (see [`ln_deriv`]).
#[inline]
pub fn log10_deriv<T: Float>(a: T) -> T {
    if a >= T::zero() {
        T::one() / (a * T::from(std::f64::consts::LN_10).unwrap())
    } else {
        T::nan()
    }
}

/// Derivative of `ln(1+a) = 1/(1+a)`, guarded to `a >= -1`.
///
/// The boundary `a = -1` yields IEEE `1/0 = +Inf`; `a < -1` yields NaN.
#[inline]
pub fn ln_1p_deriv<T: Float>(a: T) -> T {
    if a >= -T::one() {
        T::one() / (T::one() + a)
    } else {
        T::nan()
    }
}

/// Derivative of `atanh(a) = 1/((1-a)(1+a))`, guarded to `|a| <= 1`.
///
/// Unlike `asin`/`acos` (whose `sqrt(neg)` self-guards to NaN), `atanh`'s
/// derivative is a bare reciprocal that stays finite for `|a| > 1`
/// (`1/((1-1.5)(1+1.5)) = -0.8`), so it needs an explicit domain guard. The
/// boundary `|a| = 1` yields IEEE `1/0 = +Inf`.
#[inline]
pub fn atanh_deriv<T: Float>(a: T) -> T {
    if a >= -T::one() && a <= T::one() {
        T::one() / ((T::one() - a) * (T::one() + a))
    } else {
        T::nan()
    }
}

/// Derivative (subgradient) of `|a|`, unified across every AD mode and both GPU
/// backends: `0` at the kink `a = 0`, `sign(a)` elsewhere, `NaN` at `NaN`.
///
/// `0` is the minimal-norm element of the Clarke subdifferential `[-1, 1]` at the
/// kink — the convention PyTorch / JAX / TensorFlow use. Crucially it is *value*-
/// based: `a == 0` catches both `+0.0` and `-0.0`, so the result never depends on
/// the sign bit of a zero (which is reachable via `0*-1`, `x - x`, underflow…).
/// Relying on `a.signum()` alone would leak that sign bit — `signum(+0.0) = +1`,
/// `signum(-0.0) = -1` in Rust — making algebraically equivalent points report
/// different subgradients. `NaN` flows through via `signum(NaN) = NaN`.
///
/// Every eager AD type (`Dual`, `DualVec`, `Reverse`) and the bytecode `OpCode`
/// dispatcher delegate here; the wgpu and CUDA kernels carry the same
/// `a == 0 -> 0` guard. Sharp/limiting subgradients (Clarke, `jacobian_limiting`)
/// still force `±1` explicitly via `forced_reverse_partials`, independent of this
/// smooth default.
#[inline]
pub fn abs_deriv<T: Float>(a: T) -> T {
    if a == T::zero() {
        T::zero()
    } else {
        a.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_restricted_derivs_guard_outside_and_keep_boundary() {
        // Out of domain → NaN (not finite garbage).
        assert!(ln_deriv(-2.0_f64).is_nan());
        assert!(log2_deriv(-2.0_f64).is_nan());
        assert!(log10_deriv(-2.0_f64).is_nan());
        assert!(ln_1p_deriv(-2.0_f64).is_nan());
        assert!(atanh_deriv(1.5_f64).is_nan());
        assert!(atanh_deriv(-1.5_f64).is_nan());

        // Boundary → IEEE +Inf one-sided limit.
        assert!(ln_deriv(0.0_f64).is_infinite() && ln_deriv(0.0_f64) > 0.0);
        assert!(log2_deriv(0.0_f64).is_infinite() && log2_deriv(0.0_f64) > 0.0);
        assert!(log10_deriv(0.0_f64).is_infinite() && log10_deriv(0.0_f64) > 0.0);

        // Signed zero: the sign of the zero flows through the reciprocal
        // (deliberately NOT collapsed like `abs_deriv` — see the `ln_deriv`
        // doc). 1/(-0.0) = -Inf on the CPU paths; the GPU shaders share the
        // same `1/a` formula, with the ±0 sign following each backend's
        // IEEE division.
        assert!(ln_deriv(-0.0_f64).is_infinite() && ln_deriv(-0.0_f64) < 0.0);
        assert!(log2_deriv(-0.0_f64).is_infinite() && log2_deriv(-0.0_f64) < 0.0);
        assert!(log10_deriv(-0.0_f64).is_infinite() && log10_deriv(-0.0_f64) < 0.0);
        assert!(ln_1p_deriv(-1.0_f64).is_infinite() && ln_1p_deriv(-1.0_f64) > 0.0);
        assert!(atanh_deriv(1.0_f64).is_infinite() && atanh_deriv(1.0_f64) > 0.0);
        assert!(atanh_deriv(-1.0_f64).is_infinite() && atanh_deriv(-1.0_f64) > 0.0);

        // In domain → known values.
        assert!((ln_deriv(2.0_f64) - 0.5).abs() < 1e-15);
        assert!((log2_deriv(2.0_f64) - 1.0 / (2.0 * 2.0_f64.ln())).abs() < 1e-15);
        assert!((log10_deriv(2.0_f64) - 1.0 / (2.0 * 10.0_f64.ln())).abs() < 1e-15);
        assert!((ln_1p_deriv(3.0_f64) - 0.25).abs() < 1e-15);
        assert!((atanh_deriv(0.5_f64) - 1.0 / (0.75)).abs() < 1e-15);

        // acosh: domain is a >= 1. Strictly outside → NaN. The factored form
        // self-guards for -1 < a < 1 (sqrt of a negative), but a <= -1 leaves
        // (a-1)(a+1) >= 0 → finite garbage (and -Inf at a=-1 via 1/sqrt(-0.0)),
        // and the large-|a| overflow branch stays finite for a <= -1e8, so an
        // explicit guard is required for those legs.
        assert!(acosh_deriv(-1.5_f64).is_nan());
        assert!(acosh_deriv(-2.0_f64).is_nan());
        assert!(acosh_deriv(0.5_f64).is_nan());
        assert!(acosh_deriv(-1.0_f64).is_nan());
        assert!(acosh_deriv(-1e9_f64).is_nan()); // overflow-branch leg
                                                 // Boundary a=1 → +Inf (1/sqrt(0)); in domain a=2 → 1/sqrt(3).
        assert!(acosh_deriv(1.0_f64).is_infinite() && acosh_deriv(1.0_f64) > 0.0);
        assert!((acosh_deriv(2.0_f64) - 1.0 / 3.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn abs_deriv_is_zero_at_the_kink_regardless_of_sign_bit() {
        // The kink returns 0 (minimal-norm subgradient), value-based so +0 and -0
        // agree; sign(a) elsewhere; NaN -> NaN.
        assert_eq!(abs_deriv(0.0_f64), 0.0);
        assert_eq!(abs_deriv(-0.0_f64), 0.0);
        assert_eq!(abs_deriv(2.0_f64), 1.0);
        assert_eq!(abs_deriv(-3.0_f64), -1.0);
        assert!(abs_deriv(f64::NAN).is_nan());
    }

    /// Pin the factored-form precision near `a = 1`. The factored form
    /// `(a-1)·(a+1)` and the unfactored form `a*a - 1` are mathematically
    /// equal but produce measurably different f64 results when `a` is
    /// extremely close to 1: the `a*a` computation rounds away terms of
    /// order `(a-1)²` because they fall below the f64 precision floor at
    /// `a ≈ 1`. Tests that the kernel returns a result distinguishable
    /// from the unfactored form. A swap back to `a*a - 1` would make
    /// the two sides equal and trip this test.
    #[test]
    fn acosh_deriv_factored_form_keeps_precision_near_one() {
        // Unfactored reference: what `kernels::acosh_deriv` would return
        // before the factored-form upgrade.
        fn acosh_deriv_unfactored(a: f64) -> f64 {
            1.0 / (a * a - 1.0).sqrt()
        }

        // At ε = 1e-12, `a*a` in f64 loses the ε² = 1e-24 term because
        // f64 precision around 1 is ~2.2e-16 — way coarser than 1e-24.
        // The factored product `(a-1)·(a+1) = ε·(2+ε)` keeps the term.
        let a = 1.0 + 1e-12_f64;
        let factored = acosh_deriv::<f64>(a);
        let unfactored = acosh_deriv_unfactored(a);

        let rel_diff = (factored - unfactored).abs() / factored.max(unfactored);
        // Theory: factored vs unfactored differ by ~ε/(2·(2+ε)) ≈ ε/4
        // relative — at ε = 1e-12 that's ~2.5e-13, comfortably above the
        // f64 precision floor (~1e-16). Threshold 1e-13 passes with the
        // factored form and fails if anyone reverts to `a*a - 1`.
        assert!(
            rel_diff > 1e-13,
            "kernel must use factored (a-1)·(a+1) form: factored={factored}, unfactored={unfactored}, rel_diff={rel_diff:e} (a swap back to a*a-1 would make them equal)"
        );
    }
}

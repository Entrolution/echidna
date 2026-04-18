//! Per-opcode math kernels (single source of truth for formulas).
//!
//! Historical context: before this module, each opcode's numerical
//! formula was duplicated across several CPU AD types (`Dual`,
//! `DualVec`, `Reverse`, `BReverse`, `Laurent`) and the bytecode-tape
//! opcode dispatcher (`opcode.rs`). The GPU side — WGSL shaders and the
//! CUDA NVRTC kernel — carry yet another copy in their respective
//! source-level languages. A change to one copy could drift silently
//! from the others; Phase 7 of Cycle 6 found multiple such drifts
//! (atan large-|a|, div small-|b|, hypot Inf handling).
//!
//! Going forward, any numerical formula shared between the CPU AD
//! types **and** the opcode dispatcher should live here as a single
//! generic function over `num_traits::Float`. Each AD type then
//! delegates to the helper, so CPU drift becomes impossible. GPU
//! drift is caught by `tests/gpu_cpu_parity.rs`.
//!
//! Scope note: This module is intentionally minimal at the start.
//! Only the formulas most prone to recent drift (hypot partials,
//! atan large-|a|) are extracted; the rest remain inline in the AD
//! types and opcode dispatcher. Every extraction here must come with
//! a call-site refactor so we don't add abstraction for its own sake.

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

/// Derivative of `acosh(a) = ln(a + sqrt(a²-1))` with a large-|a|
/// overflow-safe path.
///
/// For `|a| ≤ 1e8`, returns `1/sqrt(a²-1)` (defined for |a| ≥ 1).
/// For `|a| > 1e8`, uses `u = 1/a` and `|u|/sqrt(1-u²)`.
#[inline]
pub fn acosh_deriv<T: Float>(a: T) -> T {
    let one = T::one();
    if a.abs() > T::from(1e8).unwrap() {
        let inv = one / a;
        inv.abs() / (one - inv * inv).sqrt()
    } else {
        one / (a * a - one).sqrt()
    }
}

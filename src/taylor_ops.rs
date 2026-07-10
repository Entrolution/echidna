//! Shared Taylor coefficient propagation functions.
//!
//! Convention: `c[k] = f^(k)(t₀) / k!` (scaled/normalized Taylor coefficients).
//! All functions operate on slices `&[F]` (inputs) and `&mut [F]` (outputs),
//! where `F: num_traits::Float`. The degree (number of coefficients) is
//! determined by the slice lengths.
//!
//! Used by both `Taylor<F, K>` (stack arrays) and `TaylorDyn<F>` (arena slices).

use num_traits::Float;

// ══════════════════════════════════════════════
//  Arithmetic
// ══════════════════════════════════════════════

/// `c = a + b`
#[inline]
pub fn taylor_add<F: Float>(a: &[F], b: &[F], c: &mut [F]) {
    for k in 0..c.len() {
        c[k] = a[k] + b[k];
    }
}

/// `c = a - b`
#[inline]
pub fn taylor_sub<F: Float>(a: &[F], b: &[F], c: &mut [F]) {
    for k in 0..c.len() {
        c[k] = a[k] - b[k];
    }
}

/// `c = -a`
#[inline]
pub fn taylor_neg<F: Float>(a: &[F], c: &mut [F]) {
    for k in 0..c.len() {
        c[k] = -a[k];
    }
}

/// `c = s * a` where `s` is a scalar.
#[inline]
pub fn taylor_scale<F: Float>(a: &[F], s: F, c: &mut [F]) {
    for k in 0..c.len() {
        c[k] = s * a[k];
    }
}

/// `c = a * b` — Cauchy product.
///
/// `c[k] = Σ_{j=0}^{k} a[j] * b[k-j]`
#[inline]
pub fn taylor_mul<F: Float>(a: &[F], b: &[F], c: &mut [F]) {
    let n = c.len();
    for k in 0..n {
        let mut sum = F::zero();
        for j in 0..=k {
            sum = sum + a[j] * b[k - j];
        }
        c[k] = sum;
    }
}

/// `c = a / b` — recursive Taylor division.
///
/// `c[k] = (a[k] - Σ_{j=1}^{k} b[j] * c[k-j]) / b[0]`
#[inline]
pub fn taylor_div<F: Float>(a: &[F], b: &[F], c: &mut [F]) {
    let n = c.len();
    // Primal: one correctly-rounded division (a·(1/b) double-rounds).
    // Higher coefficients keep the reciprocal-multiply recurrence — one
    // multiply per term, at most 1 ULP from the divide form.
    c[0] = a[0] / b[0];
    let inv_b0 = F::one() / b[0];
    for k in 1..n {
        let mut sum = a[k];
        for j in 1..=k {
            sum = sum - b[j] * c[k - j];
        }
        c[k] = sum * inv_b0;
    }
}

/// `c = 1/a` — reciprocal via recursive division.
///
/// Special case of div with numerator = [1, 0, ..., 0].
#[inline]
pub fn taylor_recip<F: Float>(a: &[F], c: &mut [F]) {
    let n = c.len();
    let inv_a0 = F::one() / a[0];
    c[0] = inv_a0;
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + a[j] * c[k - j];
        }
        c[k] = -sum * inv_a0;
    }
}

// ══════════════════════════════════════════════
//  Transcendentals (Griewank Ch 13 logarithmic derivative technique)
// ══════════════════════════════════════════════

/// `c = exp(a)`
///
/// `c[0] = exp(a[0])`
/// `c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * c[k-j]`
#[inline]
pub fn taylor_exp<F: Float>(a: &[F], c: &mut [F]) {
    let n = c.len();
    c[0] = a[0].exp();
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * c[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

/// `c = ln(a)`
///
/// `c[0] = ln(a[0])`
/// `c[k] = (a[k] - (1/k) * Σ_{j=1}^{k-1} j * c[j] * a[k-j]) / a[0]`
#[inline]
pub fn taylor_ln<F: Float>(a: &[F], c: &mut [F]) {
    let n = c.len();
    if a[0] < F::zero() {
        // `ln` is undefined on the negatives; emit an all-NaN jet rather than a
        // NaN primal beside finite higher coefficients computed from `1/a[0]`.
        // The `a[0] == 0` branch point is left to the IEEE singularity
        // (`c[0] = -Inf`), matching `taylor_sqrt` and the scalar convention.
        // This also covers `taylor_log2`/`taylor_log10`/`taylor_ln_1p` (which
        // delegate here) and `taylor_powf`'s non-integer negative-base path.
        nan_jet(c);
        return;
    }
    let inv_a0 = F::one() / a[0];
    c[0] = a[0].ln();
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..k {
            sum = sum + F::from(j).unwrap() * c[j] * a[k - j];
        }
        c[k] = (a[k] - sum / F::from(k).unwrap()) * inv_a0;
    }
}

/// Fill `c` with the all-NaN jet: the out-of-domain convention, so callers
/// never see a NaN primal beside finite higher coefficients that a recurrence
/// happened to compute.
#[inline]
fn nan_jet<F: Float>(c: &mut [F]) {
    for ci in c.iter_mut() {
        *ci = F::nan();
    }
}

/// Fill `c` with the vertical-tangent jet at a zero base: zero primal, `+Inf`
/// every higher coefficient (the sqrt/cbrt convention at `x = 0`).
#[inline]
fn vertical_tangent_jet<F: Float>(c: &mut [F]) {
    c[0] = F::zero();
    for ci in c.iter_mut().skip(1) {
        *ci = F::infinity();
    }
}

/// `c = sqrt(a)`
///
/// `c[0] = sqrt(a[0])`
/// `c[k] = (a[k] - Σ_{j=1}^{k-1} c[j] * c[k-j]) / (2 * c[0])`
///
/// When `a[0] == 0`, returns `c[0] = 0` and `c[k] = Inf` for `k >= 1` (the
/// derivative is singular at a branch point). Use the `Laurent` type for
/// functions with branch points at the expansion point.
#[inline]
pub fn taylor_sqrt<F: Float>(a: &[F], c: &mut [F]) {
    let n = c.len();
    if a[0] == F::zero() {
        // sqrt(0) = 0, but sqrt'(0) = 1/(2*sqrt(0)) = Inf (vertical tangent).
        vertical_tangent_jet(c);
        return;
    }
    if a[0] < F::zero() {
        // sqrt is undefined on the negative reals. Relying on `.sqrt()`'s
        // silent NaN propagates the NaN into c[0] but leaves higher-order
        // coefficients computed from division by `2 * NaN` in a state that
        // looks like a normal recurrence. Make the degeneracy explicit so
        // downstream callers don't accidentally consume a mix of finite
        // and non-finite coefficients.
        nan_jet(c);
        return;
    }
    c[0] = a[0].sqrt();
    let two_c0 = F::from(2.0).unwrap() * c[0];
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..k {
            sum = sum + c[j] * c[k - j];
        }
        c[k] = (a[k] - sum) / two_c0;
    }
}

/// `(s, co) = sin_cos(a)` — coupled recurrence.
///
/// `s[0] = sin(a[0])`, `co[0] = cos(a[0])`
/// `s[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * co[k-j]`
/// `co[k] = -(1/k) * Σ_{j=1}^{k} j * a[j] * s[k-j]`
#[inline]
pub fn taylor_sin_cos<F: Float>(a: &[F], s: &mut [F], co: &mut [F]) {
    let n = s.len();
    let (s0, c0) = a[0].sin_cos();
    s[0] = s0;
    co[0] = c0;
    for k in 1..n {
        let inv_k = F::one() / F::from(k).unwrap();
        let mut sum_s = F::zero();
        let mut sum_c = F::zero();
        for j in 1..=k {
            let jf = F::from(j).unwrap();
            sum_s = sum_s + jf * a[j] * co[k - j];
            sum_c = sum_c + jf * a[j] * s[k - j];
        }
        s[k] = sum_s * inv_k;
        co[k] = -sum_c * inv_k;
    }
}

/// `(sh, ch) = sinh_cosh(a)` — coupled recurrence (positive signs).
///
/// `sh[0] = sinh(a[0])`, `ch[0] = cosh(a[0])`
/// `sh[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * ch[k-j]`
/// `ch[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * sh[k-j]`
#[inline]
pub fn taylor_sinh_cosh<F: Float>(a: &[F], sh: &mut [F], ch: &mut [F]) {
    let n = sh.len();
    sh[0] = a[0].sinh();
    ch[0] = a[0].cosh();
    for k in 1..n {
        let inv_k = F::one() / F::from(k).unwrap();
        let mut sum_sh = F::zero();
        let mut sum_ch = F::zero();
        for j in 1..=k {
            let jf = F::from(j).unwrap();
            sum_sh = sum_sh + jf * a[j] * ch[k - j];
            sum_ch = sum_ch + jf * a[j] * sh[k - j];
        }
        sh[k] = sum_sh * inv_k;
        ch[k] = sum_ch * inv_k;
    }
}

/// Integrate `c' = a' · g` term by term:
/// `c[k] = (1/k) · Σ_{j=1}^{k} j · a[j] · g[k-j]`.
///
/// The shared antiderivative step of the inverse-function kernels (atan,
/// asin, asinh, acosh, atanh), which differ only in the derivative factor
/// `g` they construct first; `c[0]` is written by the caller. tan/tanh
/// interleave their `g` update with this recurrence (self-referential) and
/// deliberately keep their own loops.
#[inline]
fn integrate_from_deriv<F: Float>(a: &[F], g: &[F], c: &mut [F]) {
    for k in 1..c.len() {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * g[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

/// `c = atan(a)` — via `c' = a' / (1 + a²)`, then integrate.
///
/// Uses `scratch` for the `1 + a²` denominator.
#[inline]
pub fn taylor_atan<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = 1 + a²
    scratch2[..n].copy_from_slice(&scratch1[..n]);
    scratch2[0] = F::one() + scratch1[0];
    // c[0] = atan(a[0])
    c[0] = a[0].atan();
    // c' = a' / (1 + a²). With g = 1/(1 + a²) this is c' = a' * g, and
    // integrating the Cauchy product gives
    //   c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * g[k-j]
    // Reuse scratch1 for g = recip(1 + a²):
    taylor_recip(scratch2, scratch1);
    integrate_from_deriv(a, scratch1, c);
}

/// `c = asin(a)` — via `c' = a' / sqrt(1 - a²)`, then integrate.
///
/// Uses `scratch1` and `scratch2` as work space.
#[inline]
pub fn taylor_asin<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    c[0] = a[0].asin();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = 1 - a²  (use (1-a₀)(1+a₀) to avoid cancellation near |a₀|→1)
    scratch2[0] = (F::one() - a[0]) * (F::one() + a[0]);
    for k in 1..n {
        scratch2[k] = -scratch1[k];
    }
    // scratch1 = sqrt(1 - a²)
    taylor_sqrt(scratch2, scratch1);
    // scratch2 = 1/sqrt(1 - a²)
    taylor_recip(scratch1, scratch2);
    integrate_from_deriv(a, scratch2, c);
}

/// `c = acos(a) = π/2 - asin(a)`
#[inline]
pub fn taylor_acos<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    taylor_asin(a, c, scratch1, scratch2);
    c[0] = a[0].acos();
    for ck in &mut c[1..] {
        *ck = -*ck;
    }
}

/// `c = tan(a)` — via `c' = a' * (1 + tan²(a))` = `a' * (1 + c²)`.
///
/// Uses `scratch` for `1 + c²`.
#[inline]
pub fn taylor_tan<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let n = c.len();
    c[0] = a[0].tan();
    // Let s = 1 + c², so c' = a' * s. Integrating the Cauchy product:
    //   c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * s[k-j]
    // reads only s[0..k-1] (j >= 1), all already known; then
    //   s[k] = Σ_{j=0}^{k} c[j] * c[k-j]
    // uses the fresh c[k], so the two recurrences interleave without
    // circularity.

    // scratch = s (1 + c²)
    scratch[0] = F::one() + c[0] * c[0];
    for k in 1..n {
        // First compute c[k] using s[0..k-1]
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
        // Now update scratch[k] = s[k] = Σ_{j=0}^{k} c[j]*c[k-j]
        let mut s_k = F::zero();
        for j in 0..=k {
            s_k = s_k + c[j] * c[k - j];
        }
        scratch[k] = s_k;
    }
}

/// `c = tanh(a)` — via `c' = a' * (1 - tanh²(a))` = `a' * (1 - c²)`.
///
/// Uses `scratch` for `1 - c²`.
#[inline]
pub fn taylor_tanh<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let n = c.len();
    c[0] = a[0].tanh();
    // scratch = s = 1 - c²
    scratch[0] = F::one() - c[0] * c[0];
    for k in 1..n {
        // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch[k-j]
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
        // scratch[k] = -Σ_{j=0}^{k} c[j]*c[k-j]
        let mut s_k = F::zero();
        for j in 0..=k {
            s_k = s_k + c[j] * c[k - j];
        }
        scratch[k] = -s_k;
    }
}

/// `c = asinh(a)` — via `c' = a' / sqrt(1 + a²)`.
#[inline]
pub fn taylor_asinh<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    c[0] = a[0].asinh();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = 1 + a²
    scratch2[..n].copy_from_slice(&scratch1[..n]);
    scratch2[0] = F::one() + scratch1[0];
    // scratch1 = sqrt(1 + a²)
    taylor_sqrt(scratch2, scratch1);
    // scratch2 = 1/sqrt(1 + a²)
    taylor_recip(scratch1, scratch2);
    integrate_from_deriv(a, scratch2, c);
}

/// `c = acosh(a)` — via `c' = a' / sqrt(a² - 1)`.
#[inline]
pub fn taylor_acosh<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    if a[0] < F::one() {
        // `acosh` domain is `a >= 1`. For `-1 < a[0] < 1` the `a²-1 < 0` sqrt
        // already yields an all-NaN jet, but `a[0] <= -1` leaves `a²-1 >= 0`, so
        // the recurrence produces finite higher coefficients beside a NaN
        // primal. Emit an all-NaN jet across the whole out-of-domain range.
        nan_jet(c);
        return;
    }
    c[0] = a[0].acosh();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = a² - 1  (factored form avoids cancellation near a[0]=1)
    scratch2[..n].copy_from_slice(&scratch1[..n]);
    scratch2[0] = (a[0] - F::one()) * (a[0] + F::one());
    // scratch1 = sqrt(a² - 1)
    taylor_sqrt(scratch2, scratch1);
    // scratch2 = 1/sqrt(a² - 1)
    taylor_recip(scratch1, scratch2);
    integrate_from_deriv(a, scratch2, c);
}

/// `c = atanh(a)` — via `c' = a' / (1 - a²)`.
#[inline]
pub fn taylor_atanh<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    if a[0] < -F::one() || a[0] > F::one() {
        // `atanh` domain is `|a| <= 1`. Outside it, `1 - a²` is finite so the
        // recurrence would produce finite higher coefficients beside a NaN
        // primal; emit an all-NaN jet instead. (`|a[0]| == 1` is left to the
        // IEEE `±Inf` singularity, matching the scalar boundary convention.)
        nan_jet(c);
        return;
    }
    c[0] = a[0].atanh();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = 1 - a²  (use (1-a₀)(1+a₀) to avoid cancellation near |a₀|→1)
    scratch2[0] = (F::one() - a[0]) * (F::one() + a[0]);
    for k in 1..n {
        scratch2[k] = -scratch1[k];
    }
    // scratch1 = 1/(1 - a²)
    taylor_recip(scratch2, scratch1);
    integrate_from_deriv(a, scratch1, c);
}

// ══════════════════════════════════════════════
//  Derived functions
// ══════════════════════════════════════════════

/// `c = a^b` (powf) = `exp(b * ln(a))`.
///
/// Uses `scratch1` for `ln(a)` and `scratch2` for `b * ln(a)`.
#[inline]
pub fn taylor_powf<F: Float>(
    a: &[F],
    b: &[F],
    c: &mut [F],
    scratch1: &mut [F],
    scratch2: &mut [F],
) {
    // Constant integer exponent fast path: if `b` is a plain scalar (higher
    // coefficients are zero) and that scalar is an integer, route to
    // `taylor_powi`. Otherwise `taylor_ln(a)` yields an all-NaN jet for
    // `a[0] < 0` (and an Inf-singular jet at `a[0] == 0`), poisoning the entire
    // result — even for negative-base integer powers that have well-defined
    // Taylor coefficients.
    if b[1..].iter().all(|&bk| bk == F::zero()) {
        let b0 = b[0];
        if let Some(ni) = b0.to_i32().filter(|&ni| F::from(ni).unwrap() == b0) {
            taylor_powi(a, ni, c, scratch1, scratch2);
            return;
        }
    }
    if a[0] == F::zero() {
        // Branch point at a zero base. A CONSTANT integer exponent already
        // took the powi fast path above, so the exponent here is either
        // non-integer or live.
        let b0 = b[0];
        let b0_is_integer = b0.to_i32().is_some_and(|ni| F::from(ni).unwrap() == b0);
        if b0_is_integer {
            // Live integer exponent at a zero base: the true jet mixes
            // finite entries (k ≤ b0) with ln(0)-driven unbounded ones
            // (k > b0). Emit a consistent all-NaN jet, matching the
            // negative-base arm below, rather than a finite primal beside
            // garbage derivatives.
            nan_jet(c);
            return;
        }
        // Non-integer exponent: the k-th derivative of x^b0 at 0 vanishes
        // for k < b0 and is unbounded for k > b0 (the exponent jet's
        // ln(0)-driven terms vanish at the same x^b0·ln x → 0 rate, so the
        // rule also covers live exponents). Mirrors taylor_sqrt/taylor_cbrt's
        // [0, Inf, ...] convention (b0 = 1/2, 1/3 are the k > b0 case) and,
        // like those, assumes the generic vertical-tangent case (a[1] ≠ 0).
        c[0] = a[0].powf(b0);
        for (k, ck) in c.iter_mut().enumerate().skip(1) {
            *ck = if F::from(k).unwrap() < b0 {
                F::zero()
            } else {
                F::infinity()
            };
        }
        return;
    }
    // scratch1 = ln(a)
    taylor_ln(a, scratch1);
    // scratch2 = b * ln(a)
    taylor_mul(b, scratch1, scratch2);
    // c = exp(b * ln(a)); taylor_exp reads scratch2 and writes c, so
    // scratch1 is free from here on.
    taylor_exp(scratch2, c);
    if a[0] < F::zero() {
        // Negative base with a LIVE exponent (the constant-integer fast path
        // above already returned). `a(t)^b(t)` for a varying — hence, for
        // t != 0, non-integer — exponent is complex, so the whole jet is
        // undefined: `taylor_ln(a)` already produced an all-NaN `c[1..]`.
        // Return a consistent all-NaN jet rather than a finite primal
        // (`a[0].powf(b[0])`, finite only when `b[0]` is an integer) beside
        // NaN derivative coefficients. Matches `taylor_ln`/`taylor_sqrt`.
        c[0] = F::nan();
    } else {
        // Fix c[0] for better primal accuracy (direct powf vs exp(b*ln(a))).
        // Higher coefficients c[1..] used the exp-ln path's c[0], which may
        // differ from the patched value by sub-ULP rounding — an intentional
        // precision tradeoff that does not affect derivative correctness.
        c[0] = a[0].powf(b[0]);
    }
}

/// `c = a^n` (powi) — integer power.
///
/// Dispatches between two strategies:
/// - **Repeated squaring** (binary exponentiation via `taylor_mul`): used when
///   `a[0] <= 0` (`ln` is singular there: NaN for a negative base, `-Inf`
///   at zero) or `|n| <= 8` (at most 3
///   multiplications, competitive with exp-ln).
/// - **exp(n * ln(a))**: used for positive base with large exponents.
#[inline]
pub fn taylor_powi<F: Float>(a: &[F], n: i32, c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let deg = c.len();
    if n == 0 {
        c[0] = F::one();
        for ck in &mut c[1..deg] {
            *ck = F::zero();
        }
        return;
    }
    if n == 1 {
        c.copy_from_slice(a);
        return;
    }
    if n == -1 {
        taylor_recip(a, c);
        return;
    }
    if a[0] <= F::zero() || n.unsigned_abs() <= 8 {
        taylor_powi_squaring(a, n, c, scratch1, scratch2);
    } else {
        // scratch1 = ln(a)
        taylor_ln(a, scratch1);
        // scratch2 = n * ln(a)
        let nf = F::from(n).unwrap();
        taylor_scale(scratch1, nf, scratch2);
        // c = exp(n * ln(a))
        taylor_exp(scratch2, c);
        c[0] = a[0].powi(n);
    }
}

/// Integer power via binary exponentiation on Taylor coefficient arrays.
///
/// Computes `a^n` using repeated squaring with `taylor_mul`. Works correctly
/// for negative base values (unlike the exp-ln path). For negative `n`,
/// computes `a^|n|` then takes the reciprocal.
fn taylor_powi_squaring<F: Float>(
    a: &[F],
    n: i32,
    c: &mut [F],
    scratch1: &mut [F],
    scratch2: &mut [F],
) {
    let deg = c.len();
    let abs_n = n.unsigned_abs();

    // result (c) = 1
    c[0] = F::one();
    for ck in &mut c[1..deg] {
        *ck = F::zero();
    }

    // base (scratch1) = a
    scratch1[..deg].copy_from_slice(&a[..deg]);

    let mut power = abs_n;
    while power > 0 {
        if power & 1 == 1 {
            // result = result * base
            taylor_mul(c, &*scratch1, scratch2);
            c[..deg].copy_from_slice(&scratch2[..deg]);
        }
        power >>= 1;
        if power > 0 {
            // base = base * base
            let base_ref: &[F] = &*scratch1;
            // Inline squaring to avoid borrow conflict (scratch1 is both source and dest)
            for k in 0..deg {
                let mut sum = F::zero();
                for j in 0..=k {
                    sum = sum + base_ref[j] * base_ref[k - j];
                }
                scratch2[k] = sum;
            }
            scratch1[..deg].copy_from_slice(&scratch2[..deg]);
        }
    }

    if n < 0 {
        // c = 1/c: copy c into scratch1, then compute recip into c
        scratch1[..deg].copy_from_slice(&c[..deg]);
        taylor_recip(scratch1, c);
    }
}

/// `c = cbrt(a) = a^(1/3)`.
///
/// Uses `scratch1` and `scratch2`.
#[inline]
pub fn taylor_cbrt<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let deg = c.len();
    debug_assert_eq!(a.len(), c.len());
    if a[0] == F::zero() {
        // cbrt(0) = 0, but cbrt'(0) = 1/(3*cbrt(0)^2) = Inf (vertical tangent).
        vertical_tangent_jet(c);
        return;
    }
    if a[0] < F::zero() {
        // cbrt(-x) = -cbrt(x): negate input, compute cbrt on positive, negate output.
        // Use c as temporary for negated input (safe: taylor_ln reads before writing).
        for i in 0..deg {
            c[i] = -a[i];
        }
        let three = F::from(3.0).unwrap();
        let third = F::one() / three;
        taylor_ln(c, scratch1);
        taylor_scale(scratch1, third, scratch2);
        taylor_exp(scratch2, c);
        // Same O(ULP) primal-patch tradeoff as taylor_powf (see comment there).
        // The negate-all-coefficients approach uses cbrt(a) = -cbrt(-a), which is exact.
        c[0] = a[0].cbrt();
        for ci in c.iter_mut().skip(1) {
            *ci = -*ci;
        }
    } else {
        let three = F::from(3.0).unwrap();
        let third = F::one() / three;
        taylor_ln(a, scratch1);
        taylor_scale(scratch1, third, scratch2);
        taylor_exp(scratch2, c);
        // Primal patch: same O(ULP) tradeoff as taylor_powf (see comment there).
        c[0] = a[0].cbrt();
    }
}

/// `c = exp2(a) = 2^a = exp(a * ln(2))`.
#[inline]
pub fn taylor_exp2<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let ln2 = F::from(2.0).unwrap().ln();
    taylor_scale(a, ln2, scratch);
    taylor_exp(scratch, c);
    // Primal patch: same O(ULP) tradeoff as taylor_powf (see comment there).
    c[0] = a[0].exp2();
}

/// `c = exp(a) - 1` (exp_m1).
///
/// Runs `taylor_exp` with `c[0] = exp(a[0])` during the recurrence —
/// `d/dx[exp(x) - 1] = exp(x)`, so the higher-order coefficients of
/// `exp(x) - 1` equal those of `exp(x)` — then patches `c[0]` to
/// `exp_m1(a[0])` for the cancellation-free primal.
#[inline]
pub fn taylor_exp_m1<F: Float>(a: &[F], c: &mut [F]) {
    taylor_exp(a, c);
    c[0] = a[0].exp_m1();
}

/// `c = log2(a) = ln(a) / ln(2)`.
#[inline]
pub fn taylor_log2<F: Float>(a: &[F], c: &mut [F]) {
    taylor_ln(a, c);
    let inv_ln2 = F::one() / F::from(2.0).unwrap().ln();
    // Primal patch: same O(ULP) tradeoff as taylor_powf (see comment there).
    c[0] = a[0].log2();
    for ck in &mut c[1..] {
        *ck = *ck * inv_ln2;
    }
}

/// `c = log10(a) = ln(a) / ln(10)`.
#[inline]
pub fn taylor_log10<F: Float>(a: &[F], c: &mut [F]) {
    taylor_ln(a, c);
    let inv_ln10 = F::one() / F::from(10.0).unwrap().ln();
    // Primal patch: same O(ULP) tradeoff as taylor_powf (see comment there).
    c[0] = a[0].log10();
    for ck in &mut c[1..] {
        *ck = *ck * inv_ln10;
    }
}

/// `c = ln(1 + a)`.
///
/// Uses `scratch` for `1 + a`.
#[inline]
pub fn taylor_ln_1p<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let n = c.len();
    // All call sites are already guarded by the compile-time `K >= 1` const
    // assert in bytecode_tape/taylor.rs, but `taylor_ln_1p` is also callable
    // directly. `c[0] = a[0].ln_1p()` below unconditionally indexes [0],
    // so n=0 would panic; surface it as a debug assert with an actionable
    // message rather than a raw slice panic.
    debug_assert!(n >= 1, "taylor_ln_1p requires c.len() >= 1");
    scratch[1..n].copy_from_slice(&a[1..n]);
    scratch[0] = F::one() + a[0];
    taylor_ln(scratch, c);
    // Primal patch: same O(ULP) tradeoff as taylor_powf (see comment there).
    c[0] = a[0].ln_1p();
}

/// `c = hypot(a, b) = sqrt(a² + b²)`.
///
/// Shared CPU HYPOT kernel for jet-coefficient arrays. Used by
/// [`crate::Taylor::hypot`], [`crate::TaylorDyn::hypot`], and
/// [`crate::Laurent::hypot`]. The Laurent caller first rebases
/// operands to a common pole order so the coefficient arrays
/// become directly comparable, then calls through here.
///
/// The scratch buffers stage the rescaled operands and squared terms
/// (exact roles rotate; the step comments in the body annotate each).
/// Rescales inputs by `max(|a[m]|, |b[m]|)`, with `m` the first order
/// carrying signal (0 in the ordinary case), to avoid overflow/underflow
/// in the intermediate a²+b². Shared leading zeros peel away in one shot:
/// hypot(a, b) near t = 0 with both operands zero through order m-1
/// equals `|t|^m · hypot(a(t)/t^m, b(t)/t^m)`, so the body computes the
/// shifted series and returns it behind a zero prefix. Mirrors
/// `Taylor::abs` at the function-domain boundary.
#[inline]
pub fn taylor_hypot<F: Float>(
    a: &[F],
    b: &[F],
    c: &mut [F],
    scratch1: &mut [F],
    scratch2: &mut [F],
) {
    let n = c.len();
    let m = if a[0].abs().max(b[0].abs()) == F::zero() {
        // IEEE maxNum drops NaN (`max(NaN, 0) == 0`), so a NaN leading
        // coefficient lands in this zero-scale branch rather than on the
        // general rescale path. Propagate it to every coefficient
        // (hypot(NaN, ·) = NaN), exactly as the general path does when the
        // co-operand is non-zero — without this, the peel and all-zero arms
        // below would silently swallow the NaN.
        if a[0].is_nan() || b[0].is_nan() {
            nan_jet(c);
            return;
        }
        // Both leading primals are zero. If some later order carries signal,
        // the composite t ↦ hypot(a(t), b(t)) is smoothly
        // `|t|^m · hypot(a(t)/t^m, b(t)/t^m)` near t = 0, with m the first
        // signal order: peel all m zero orders at once by running the
        // general path on the m-shifted series and shifting the result
        // back. This mirrors CPU `Taylor::abs` and gives the true Taylor
        // expansion rather than the `log(0)·exp` path's NaN/Inf.
        let Some(m) = (1..n).find(|&k| a[k] != F::zero() || b[k] != F::zero()) else {
            // Both series are identically zero: t ↦ hypot(a(t), b(t)) is the
            // constant 0, so every Taylor coefficient is zero. Unlike sqrt at
            // a genuine simple zero there is no branch point here — emitting
            // the singular [0, Inf, …] jet would poison downstream sweeps
            // with a spurious pole. Matches `Laurent::hypot`'s all-zero
            // guard, so the three hypot surfaces (Taylor, TaylorDyn, Laurent)
            // agree on this input.
            for ck in c.iter_mut() {
                *ck = F::zero();
            }
            return;
        };
        // A NaN at the first signal order is what the shifted general path
        // would meet as its leading coefficient; emit the peeled result
        // directly rather than relying on `Inf · 0` propagation to build it.
        // The primal slot still goes through `hypot`, which is NaN except
        // for the IEEE override `hypot(±Inf, NaN) = +Inf`.
        if a[m].is_nan() || b[m].is_nan() {
            for ck in c[..m].iter_mut() {
                *ck = F::zero();
            }
            c[m] = a[m].hypot(b[m]);
            for ck in c[m + 1..].iter_mut() {
                *ck = F::nan();
            }
            return;
        }
        m
    } else {
        0
    };

    // General path on the m-shifted, zero-padded series (m == 0 is the
    // ordinary case); the operand reads — the two rescale loops and the
    // final primal patch — shift by pure index arithmetic, no staging
    // buffers.
    let scale = a[m].abs().max(b[m].abs());
    let inv_scale = F::one() / scale;
    // scratch1 = shifted (a/scale)
    for k in 0..n {
        scratch1[k] = if k + m < n {
            a[k + m] * inv_scale
        } else {
            F::zero()
        };
    }
    // scratch2 = shifted (b/scale)
    for k in 0..n {
        scratch2[k] = if k + m < n {
            b[k + m] * inv_scale
        } else {
            F::zero()
        };
    }
    // c = (a/scale)²  -- reuse c as temp
    taylor_mul(scratch1, scratch1, c);
    // scratch1 = (b/scale)²  -- reuse scratch1
    taylor_mul(scratch2, scratch2, scratch1);
    // c = (a/scale)² + (b/scale)²
    for k in 0..n {
        c[k] = c[k] + scratch1[k];
    }
    // scratch1 = sqrt((a/scale)² + (b/scale)²)
    taylor_sqrt(c, scratch1);
    // c = scale * sqrt(...)  — undo rescaling
    for k in 0..n {
        c[k] = scratch1[k] * scale;
    }
    c[0] = a[m].hypot(b[m]);
    if m > 0 {
        // The peeled |t|^m factor returns as a zero prefix.
        c.copy_within(0..n - m, m);
        for ck in c[..m].iter_mut() {
            *ck = F::zero();
        }
    }
}

/// `c = atan2(a, b)` = atan(a/b) with quadrant handling.
///
/// Uses scratch arrays for intermediate computation.
#[inline]
pub fn taylor_atan2<F: Float>(
    a: &[F],
    b: &[F],
    c: &mut [F],
    scratch1: &mut [F],
    scratch2: &mut [F],
    scratch3: &mut [F],
) {
    if b[0] != F::zero() {
        // Standard path: atan(a/b) with quadrant correction on c[0].
        // Derivatives of atan2(a,b) and atan(a/b) are identical where both defined.
        taylor_div(a, b, scratch1);
        taylor_atan(scratch1, c, scratch2, scratch3);
        c[0] = a[0].atan2(b[0]);
    } else if a[0] != F::zero() {
        // b[0]==0, a[0]!=0: use atan2(a,b) = sign(a)*pi/2 - atan(b/a)
        taylor_div(b, a, scratch1);
        taylor_atan(scratch1, c, scratch2, scratch3);
        // c = -atan(b/a)
        for ck in c.iter_mut() {
            *ck = -*ck;
        }
        // Fix c[0] to the correct atan2 value
        c[0] = a[0].atan2(b[0]);
    } else {
        // Both zero: mathematically undefined; return discontinuous zero
        taylor_discontinuous(F::zero(), c);
    }
}

/// `n!` computed by direct product.
///
/// Exact through `18!` in f64 (the largest factorial below 2^53; f32
/// degrades from `14!`); IEEE-saturating to `+inf` far beyond that.
/// Callers guard `k <= 18` (the STDE diagonal estimators), document
/// saturation as the contract (diffop's extraction prefactor), or divide
/// small slot factorials where the same bound holds structurally (diffop's
/// `1/slot!` jet seeds). NOT a substitute for the interleaved `k! * c[k]`
/// form in `Taylor::derivative`/`TaylorDyn::derivative`, which avoids the
/// standalone overflow entirely.
#[cfg(any(feature = "stde", feature = "diffop"))]
pub(crate) fn factorial<F: Float>(n: usize) -> F {
    let mut f = F::one();
    for i in 2..=n {
        f = f * F::from(i).unwrap();
    }
    f
}

/// Discontinuous function: `c[0] = f(a[0])`, `c[k>=1] = 0`.
#[inline]
pub fn taylor_discontinuous<F: Float>(val: F, c: &mut [F]) {
    c[0] = val;
    for ck in &mut c[1..] {
        *ck = F::zero();
    }
}

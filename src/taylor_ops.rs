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
    let inv_b0 = F::one() / b[0];
    for k in 0..n {
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

/// `c = sqrt(a)`
///
/// `c[0] = sqrt(a[0])`
/// `c[k] = (a[k] - Σ_{j=1}^{k-1} c[j] * c[k-j]) / (2 * c[0])`
#[inline]
pub fn taylor_sqrt<F: Float>(a: &[F], c: &mut [F]) {
    let n = c.len();
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
    // c' = a' / (1 + a²), so c[k] via the division-integration recurrence:
    // Let d = 1/(1+a²). Then c'[k-1] = Σ d[j] * a'[k-j-1]... but simpler approach:
    // Since (1+a²) * c' = a', we have:
    // k * (1+a²)[0] * c[k] + Σ_{j=1}^{k-1} (1+a²)[j] * (k-j) * c[k-j]...
    // Actually the integration approach: if g = 1/(1+a²), then c' = a' * g
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * g[k-j]
    // First compute g = 1/(1+a²):
    // Reuse scratch1 for g = recip(1+a²)
    taylor_recip(scratch2, scratch1);
    // Now c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch1[k-j]
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch1[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
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
    // scratch2 = 1 - a²
    scratch2[0] = F::one() - scratch1[0];
    for k in 1..n {
        scratch2[k] = -scratch1[k];
    }
    // scratch1 = sqrt(1 - a²)
    taylor_sqrt(scratch2, scratch1);
    // scratch2 = 1/sqrt(1 - a²)
    taylor_recip(scratch1, scratch2);
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch2[k-j]
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch2[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

/// `c = acos(a) = π/2 - asin(a)`
#[inline]
pub fn taylor_acos<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    taylor_asin(a, c, scratch1, scratch2);
    c[0] = a[0].acos();
    for ck in c[1..].iter_mut() {
        *ck = -*ck;
    }
}

/// `c = tan(a)` — via `c' = a' * (1 + tan²(a))` = `a' * (1 + c²)`.
///
/// Uses `scratch` for `c²`.
#[inline]
pub fn taylor_tan<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let n = c.len();
    c[0] = a[0].tan();
    // Recurrence: c' = a' * (1 + c²)
    // c[k] = a[k] + (1/k) * Σ_{j=1}^{k} j * a[j] * scratch[k-j]
    // where scratch holds running c² (partial)
    // Actually let's do it step by step: after computing c[0..k], update scratch = c[0..k]²
    // But this is circular. Better approach:
    // Let s = 1 + c². Then c' = a' * s.
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * s[k-j]
    // s[0] = 1 + c[0]², and for k>=1: s[k] = Σ_{j=0}^{k} c[j]*c[k-j]
    // But s[k] depends on c[k], which depends on s...
    // Expand: s[k] = 2*c[0]*c[k] + Σ_{j=1}^{k-1} c[j]*c[k-j]
    // And c[k] = (1/k) * [k*a[k]*s[0] + Σ_{j=1}^{k-1} j*a[j]*s[k-j]]
    // So substitute and solve for c[k]:
    // c[k] = a[k]*s[0] + (1/k) * Σ_{j=1}^{k-1} j*a[j]*s[k-j]  ... (*)
    // Then s[k] = 2*c[0]*c[k] + Σ_{j=1}^{k-1} c[j]*c[k-j]
    // Wait, (*) only uses s[0..k-1] which are known. So this works!

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
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch2[k-j]
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch2[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

/// `c = acosh(a)` — via `c' = a' / sqrt(a² - 1)`.
#[inline]
pub fn taylor_acosh<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    c[0] = a[0].acosh();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = a² - 1
    scratch2[..n].copy_from_slice(&scratch1[..n]);
    scratch2[0] = scratch1[0] - F::one();
    // scratch1 = sqrt(a² - 1)
    taylor_sqrt(scratch2, scratch1);
    // scratch2 = 1/sqrt(a² - 1)
    taylor_recip(scratch1, scratch2);
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch2[k-j]
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch2[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

/// `c = atanh(a)` — via `c' = a' / (1 - a²)`.
#[inline]
pub fn taylor_atanh<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    c[0] = a[0].atanh();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = 1 - a²
    scratch2[0] = F::one() - scratch1[0];
    for k in 1..n {
        scratch2[k] = -scratch1[k];
    }
    // scratch1 = 1/(1 - a²)
    taylor_recip(scratch2, scratch1);
    // c[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * scratch1[k-j]
    for k in 1..n {
        let mut sum = F::zero();
        for j in 1..=k {
            sum = sum + F::from(j).unwrap() * a[j] * scratch1[k - j];
        }
        c[k] = sum / F::from(k).unwrap();
    }
}

// ══════════════════════════════════════════════
//  Derived functions
// ══════════════════════════════════════════════

/// `c = a^b` (powf) = `exp(b * ln(a))`.
///
/// Uses `scratch1` for `ln(a)` and `scratch2` for `b * ln(a)`.
#[inline]
pub fn taylor_powf<F: Float>(a: &[F], b: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let n = c.len();
    // scratch1 = ln(a)
    taylor_ln(a, scratch1);
    // scratch2 = b * ln(a)
    taylor_mul(b, scratch1, scratch2);
    // c = exp(b * ln(a))
    // But we can't use scratch1 anymore since taylor_exp needs its own output...
    // Actually taylor_exp writes to c, so we just need scratch2 as input.
    taylor_exp(scratch2, c);
    // Fix c[0] for better accuracy (use direct powf instead of exp(b*ln(a)))
    c[0] = a[0].powf(b[0]);
    // Recompute higher coefficients if n > 1 to account for the c[0] fixup
    // Actually for consistency the recurrence from taylor_exp is fine since
    // exp(b[0]*ln(a[0])) = a[0]^b[0]. The difference is just floating point.
    // Let's keep the exp result as-is for mathematical consistency.
    _ = n;
}

/// `c = a^n` (powi) — integer power.
///
/// Uses `scratch1` for `ln(a)` and `scratch2` for `n * ln(a)`.
#[inline]
pub fn taylor_powi<F: Float>(a: &[F], n: i32, c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let deg = c.len();
    // For small integer powers, direct computation may be more efficient,
    // but for generality use exp(n * ln(a)).
    if n == 0 {
        c[0] = F::one();
        for ck in c[1..deg].iter_mut() {
            *ck = F::zero();
        }
        return;
    }
    if n == 1 {
        c.copy_from_slice(a);
        return;
    }
    // scratch1 = ln(a)
    taylor_ln(a, scratch1);
    // scratch2 = n * ln(a)
    let nf = F::from(n).unwrap();
    taylor_scale(scratch1, nf, scratch2);
    // c = exp(n * ln(a))
    taylor_exp(scratch2, c);
    c[0] = a[0].powi(n);
}

/// `c = cbrt(a) = a^(1/3)`.
///
/// Uses `scratch1` and `scratch2`.
#[inline]
pub fn taylor_cbrt<F: Float>(a: &[F], c: &mut [F], scratch1: &mut [F], scratch2: &mut [F]) {
    let three = F::from(3.0).unwrap();
    let third = F::one() / three;
    // scratch1 = ln(a)
    taylor_ln(a, scratch1);
    // scratch2 = (1/3) * ln(a)
    taylor_scale(scratch1, third, scratch2);
    // c = exp((1/3) * ln(a))
    taylor_exp(scratch2, c);
    c[0] = a[0].cbrt();
}

/// `c = exp2(a) = 2^a = exp(a * ln(2))`.
#[inline]
pub fn taylor_exp2<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let ln2 = F::from(2.0).unwrap().ln();
    taylor_scale(a, ln2, scratch);
    taylor_exp(scratch, c);
    c[0] = a[0].exp2();
}

/// `c = exp(a) - 1` (exp_m1).
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
    c[0] = a[0].log2();
    for ck in c[1..].iter_mut() {
        *ck = *ck * inv_ln2;
    }
}

/// `c = log10(a) = ln(a) / ln(10)`.
#[inline]
pub fn taylor_log10<F: Float>(a: &[F], c: &mut [F]) {
    taylor_ln(a, c);
    let inv_ln10 = F::one() / F::from(10.0).unwrap().ln();
    c[0] = a[0].log10();
    for ck in c[1..].iter_mut() {
        *ck = *ck * inv_ln10;
    }
}

/// `c = ln(1 + a)`.
///
/// Uses `scratch` for `1 + a`.
#[inline]
pub fn taylor_ln_1p<F: Float>(a: &[F], c: &mut [F], scratch: &mut [F]) {
    let n = c.len();
    scratch[1..n].copy_from_slice(&a[1..n]);
    scratch[0] = F::one() + a[0];
    taylor_ln(scratch, c);
    c[0] = a[0].ln_1p();
}

/// `c = hypot(a, b) = sqrt(a² + b²)`.
///
/// Uses `scratch1` for a², `scratch2` for b², and the result scratch for a²+b².
#[inline]
pub fn taylor_hypot<F: Float>(
    a: &[F],
    b: &[F],
    c: &mut [F],
    scratch1: &mut [F],
    scratch2: &mut [F],
) {
    let n = c.len();
    // scratch1 = a²
    taylor_mul(a, a, scratch1);
    // scratch2 = b²
    taylor_mul(b, b, scratch2);
    // scratch1 = a² + b² (reuse scratch1)
    for k in 0..n {
        scratch1[k] = scratch1[k] + scratch2[k];
    }
    // c = sqrt(a² + b²)
    taylor_sqrt(scratch1, c);
    c[0] = a[0].hypot(b[0]);
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
    // c = atan(a/b) with correct quadrant
    // First compute ratio = a/b
    taylor_div(a, b, scratch1);
    // Then atan(ratio)
    taylor_atan(scratch1, c, scratch2, scratch3);
    // Fix c[0] for quadrant correctness
    c[0] = a[0].atan2(b[0]);
}

/// Discontinuous function: `c[0] = f(a[0])`, `c[k>=1] = 0`.
#[inline]
pub fn taylor_discontinuous<F: Float>(val: F, c: &mut [F]) {
    c[0] = val;
    for ck in c[1..].iter_mut() {
        *ck = F::zero();
    }
}

//! Const-generic Taylor coefficient type: `Taylor<F, K>`.
//!
//! `K` is the total number of coefficients. `coeffs[0]` is the primal value,
//! `coeffs[k]` = f^(k)(t₀) / k! (scaled Taylor coefficient).
//!
//! Stack-allocated, `Copy`. Implements `Float` + `Scalar`, so it flows through
//! any AD-generic function and through `BytecodeTape::forward_tangent`.

use std::fmt::{self, Display};

use crate::taylor_ops;
use crate::Float;

/// Stack-allocated Taylor coefficient vector.
///
/// `K` = total coefficient count. `coeffs[0]` = primal value.
/// `coeffs[k]` = f^(k)(t₀) / k! for k ≥ 1.
#[derive(Clone, Copy, Debug)]
pub struct Taylor<F: Float, const K: usize> {
    /// Raw coefficient array: `coeffs[k]` = f^(k)(t0) / k!.
    pub coeffs: [F; K],
}

impl<F: Float, const K: usize> Default for Taylor<F, K> {
    fn default() -> Self {
        Taylor {
            coeffs: [F::zero(); K],
        }
    }
}

impl<F: Float, const K: usize> Display for Taylor<F, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.coeffs[0])?;
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            write!(f, " + {}·t^{}", c, i)?;
        }
        Ok(())
    }
}

impl<F: Float, const K: usize> From<F> for Taylor<F, K> {
    #[inline]
    fn from(val: F) -> Self {
        Taylor::constant(val)
    }
}

impl<F: Float, const K: usize> Taylor<F, K> {
    /// Create a Taylor number from raw coefficients.
    #[inline]
    pub fn new(coeffs: [F; K]) -> Self {
        Taylor { coeffs }
    }

    /// Create a constant (zero higher-order coefficients).
    #[inline]
    pub fn constant(val: F) -> Self {
        let mut coeffs = [F::zero(); K];
        coeffs[0] = val;
        Taylor { coeffs }
    }

    /// Create a variable: c₀ = val, c₁ = 1, rest zero.
    ///
    /// Represents the identity function `t ↦ val + (t - t₀)`.
    #[inline]
    pub fn variable(val: F) -> Self {
        let mut coeffs = [F::zero(); K];
        coeffs[0] = val;
        if K > 1 {
            coeffs[1] = F::one();
        }
        Taylor { coeffs }
    }

    /// Primal value (coefficient 0).
    #[inline]
    pub fn value(&self) -> F {
        self.coeffs[0]
    }

    /// Get the k-th Taylor coefficient (scaled: f^(k)/k!).
    #[inline]
    pub fn coeff(&self, k: usize) -> F {
        self.coeffs[k]
    }

    /// Get the k-th derivative: `k! × coeffs[k]`.
    #[inline]
    pub fn derivative(&self, k: usize) -> F {
        let mut factorial = F::one();
        for i in 2..=k {
            factorial = factorial * F::from(i).unwrap();
        }
        self.coeffs[k] * factorial
    }

    /// Evaluate the Taylor polynomial at point `h` via Horner's method.
    ///
    /// Computes `Σ_{k=0}^{K-1} coeffs[k] · h^k`.
    #[inline]
    pub fn eval_at(&self, h: F) -> F {
        let mut val = self.coeffs[K - 1];
        for k in (0..K - 1).rev() {
            val = val * h + self.coeffs[k];
        }
        val
    }

    // ── Elemental methods ──
    // Each delegates to taylor_ops with stack arrays as scratch.

    /// Reciprocal (1/x).
    #[inline]
    pub fn recip(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_recip(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_sqrt(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Cube root.
    #[inline]
    pub fn cbrt(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_cbrt(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Integer power.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_powi(&self.coeffs, n, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_powf(&self.coeffs, &n.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Natural exponential (e^x).
    #[inline]
    pub fn exp(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_exp(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Base-2 exponential (2^x).
    #[inline]
    pub fn exp2(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s = [F::zero(); K];
        taylor_ops::taylor_exp2(&self.coeffs, &mut c, &mut s);
        Taylor { coeffs: c }
    }

    /// e^x - 1, accurate near zero.
    #[inline]
    pub fn exp_m1(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_exp_m1(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_ln(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Base-2 logarithm.
    #[inline]
    pub fn log2(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_log2(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// Base-10 logarithm.
    #[inline]
    pub fn log10(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_log10(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }

    /// ln(1+x), accurate near zero.
    #[inline]
    pub fn ln_1p(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s = [F::zero(); K];
        taylor_ops::taylor_ln_1p(&self.coeffs, &mut c, &mut s);
        Taylor { coeffs: c }
    }

    /// Logarithm with given base.
    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    /// Sine.
    #[inline]
    pub fn sin(self) -> Self {
        let mut s = [F::zero(); K];
        let mut co = [F::zero(); K];
        taylor_ops::taylor_sin_cos(&self.coeffs, &mut s, &mut co);
        Taylor { coeffs: s }
    }

    /// Cosine.
    #[inline]
    pub fn cos(self) -> Self {
        let mut s = [F::zero(); K];
        let mut co = [F::zero(); K];
        taylor_ops::taylor_sin_cos(&self.coeffs, &mut s, &mut co);
        Taylor { coeffs: co }
    }

    /// Simultaneous sine and cosine.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let mut s = [F::zero(); K];
        let mut co = [F::zero(); K];
        taylor_ops::taylor_sin_cos(&self.coeffs, &mut s, &mut co);
        (Taylor { coeffs: s }, Taylor { coeffs: co })
    }

    /// Tangent.
    #[inline]
    pub fn tan(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s = [F::zero(); K];
        taylor_ops::taylor_tan(&self.coeffs, &mut c, &mut s);
        Taylor { coeffs: c }
    }

    /// Arcsine.
    #[inline]
    pub fn asin(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_asin(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Arccosine.
    #[inline]
    pub fn acos(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_acos(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Arctangent.
    #[inline]
    pub fn atan(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_atan(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        let mut s3 = [F::zero(); K];
        taylor_ops::taylor_atan2(
            &self.coeffs,
            &other.coeffs,
            &mut c,
            &mut s1,
            &mut s2,
            &mut s3,
        );
        Taylor { coeffs: c }
    }

    /// Hyperbolic sine.
    #[inline]
    pub fn sinh(self) -> Self {
        let mut sh = [F::zero(); K];
        let mut ch = [F::zero(); K];
        taylor_ops::taylor_sinh_cosh(&self.coeffs, &mut sh, &mut ch);
        Taylor { coeffs: sh }
    }

    /// Hyperbolic cosine.
    #[inline]
    pub fn cosh(self) -> Self {
        let mut sh = [F::zero(); K];
        let mut ch = [F::zero(); K];
        taylor_ops::taylor_sinh_cosh(&self.coeffs, &mut sh, &mut ch);
        Taylor { coeffs: ch }
    }

    /// Hyperbolic tangent.
    #[inline]
    pub fn tanh(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s = [F::zero(); K];
        taylor_ops::taylor_tanh(&self.coeffs, &mut c, &mut s);
        Taylor { coeffs: c }
    }

    /// Inverse hyperbolic sine.
    #[inline]
    pub fn asinh(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_asinh(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Inverse hyperbolic cosine.
    #[inline]
    pub fn acosh(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_acosh(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Inverse hyperbolic tangent.
    #[inline]
    pub fn atanh(self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_atanh(&self.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        let mut coeffs = self.coeffs;
        // Use first nonzero coefficient's sign to determine the branch direction
        // at zero, avoiding signum(+0.0) = 0 which would annihilate the jet.
        let sign = if self.coeffs[0] != F::zero() {
            self.coeffs[0].signum()
        } else if let Some(k) = (1..K).find(|&k| self.coeffs[k] != F::zero()) {
            self.coeffs[k].signum()
        } else {
            F::one()
        };
        for c in &mut coeffs {
            *c = *c * sign;
        }
        Taylor { coeffs }
    }

    /// Sign function (zero derivative).
    #[inline]
    pub fn signum(self) -> Self {
        Taylor::constant(self.coeffs[0].signum())
    }

    /// Floor (zero derivative).
    #[inline]
    pub fn floor(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_discontinuous(self.coeffs[0].floor(), &mut c);
        Taylor { coeffs: c }
    }

    /// Ceiling (zero derivative).
    #[inline]
    pub fn ceil(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_discontinuous(self.coeffs[0].ceil(), &mut c);
        Taylor { coeffs: c }
    }

    /// Round to nearest integer (zero derivative).
    #[inline]
    pub fn round(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_discontinuous(self.coeffs[0].round(), &mut c);
        Taylor { coeffs: c }
    }

    /// Truncate toward zero (zero derivative).
    #[inline]
    pub fn trunc(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_discontinuous(self.coeffs[0].trunc(), &mut c);
        Taylor { coeffs: c }
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        let mut coeffs = self.coeffs;
        coeffs[0] = self.coeffs[0].fract();
        Taylor { coeffs }
    }

    /// Fused multiply-add: self * a + b.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    /// Euclidean distance: sqrt(self^2 + other^2).
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_hypot(&self.coeffs, &other.coeffs, &mut c, &mut s1, &mut s2);
        Taylor { coeffs: c }
    }

    /// Maximum of two values.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        // NaN guard: return the non-NaN argument (IEEE 754 fmax semantics)
        if self.coeffs[0] >= other.coeffs[0] || other.coeffs[0].is_nan() {
            self
        } else {
            other
        }
    }

    /// Minimum of two values.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        // NaN guard: return the non-NaN argument (IEEE 754 fmin semantics)
        if self.coeffs[0] <= other.coeffs[0] || other.coeffs[0].is_nan() {
            self
        } else {
            other
        }
    }
}

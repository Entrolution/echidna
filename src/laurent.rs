//! Const-generic Laurent coefficient type: `Laurent<F, K>`.
//!
//! A Laurent series extends Taylor series to allow negative powers of t:
//! `f(t) = Σ_{k=p}^{p+K-1} c_{k-p} · t^k`, where `p = pole_order`.
//!
//! - `pole_order < 0`: pole (singularity), e.g. `1/t` has `pole_order = -1`
//! - `pole_order = 0`: regular (identical to Taylor series)
//! - `pole_order > 0`: zero at origin
//!
//! Stack-allocated, `Copy`. Implements `Float` + `Scalar`, so it flows through
//! any AD-generic function and through `BytecodeTape::forward_tangent`.
//!
//! Arithmetic reuses `taylor_ops` functions for coefficient propagation.
//! Only `pole_order` tracking is additional.

use std::fmt::{self, Display};

use crate::taylor_ops;
use crate::Float;

/// Stack-allocated Laurent coefficient vector.
///
/// `K` = total coefficient count. `coeffs[i]` = coefficient of `t^(pole_order + i)`.
/// Always normalized: `coeffs[0] != 0` (or all zero).
#[derive(Clone, Copy, Debug)]
pub struct Laurent<F: Float, const K: usize> {
    coeffs: [F; K],
    pole_order: i32,
}

impl<F: Float, const K: usize> Default for Laurent<F, K> {
    fn default() -> Self {
        Laurent::zero()
    }
}

impl<F: Float, const K: usize> Display for Laurent<F, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == F::zero() {
                continue;
            }
            let power = self.pole_order + i as i32;
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            if power == 0 {
                write!(f, "{}", c)?;
            } else {
                write!(f, "{}·t^{}", c, power)?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl<F: Float, const K: usize> From<F> for Laurent<F, K> {
    #[inline]
    fn from(val: F) -> Self {
        Laurent::constant(val)
    }
}

impl<F: Float, const K: usize> Laurent<F, K> {
    /// Create a Laurent number from raw coefficients and pole order.
    ///
    /// Normalizes to ensure `coeffs[0] != 0` (or all zero).
    #[inline]
    pub fn new(coeffs: [F; K], pole_order: i32) -> Self {
        let mut l = Laurent { coeffs, pole_order };
        l.normalize();
        l
    }

    /// Create a constant (zero higher-order coefficients, pole_order = 0).
    #[inline]
    pub fn constant(val: F) -> Self {
        let mut coeffs = [F::zero(); K];
        coeffs[0] = val;
        Laurent {
            coeffs,
            pole_order: 0,
        }
    }

    /// Create a variable: c₀ = val, c₁ = 1, rest zero.
    ///
    /// Represents `val + t`. If `val == 0`, normalizes to `pole_order = 1`
    /// (a zero at the origin).
    #[inline]
    pub fn variable(val: F) -> Self {
        let mut coeffs = [F::zero(); K];
        coeffs[0] = val;
        if K > 1 {
            coeffs[1] = F::one();
        }
        let mut l = Laurent {
            coeffs,
            pole_order: 0,
        };
        l.normalize();
        l
    }

    /// The zero Laurent value.
    #[inline]
    pub fn zero() -> Self {
        Laurent {
            coeffs: [F::zero(); K],
            pole_order: 0,
        }
    }

    /// The one Laurent value.
    #[inline]
    pub fn one() -> Self {
        Self::constant(F::one())
    }

    /// Pole order (power of t for `coeffs[0]`).
    #[inline]
    pub fn pole_order(&self) -> i32 {
        self.pole_order
    }

    /// True if this Laurent series has a pole (pole_order < 0).
    #[inline]
    pub fn has_pole(&self) -> bool {
        self.pole_order < 0
    }

    /// Leading coefficient (should be nonzero if normalized, unless all-zero).
    #[inline]
    pub fn leading_coefficient(&self) -> F {
        self.coeffs[0]
    }

    /// Residue: coefficient of t^{-1}.
    ///
    /// Returns zero if the series has no t^{-1} term.
    #[inline]
    pub fn residue(&self) -> F {
        self.coeff(-1)
    }

    /// Coefficient of t^k.
    ///
    /// Returns zero if k is outside the stored range.
    #[inline]
    pub fn coeff(&self, k: i32) -> F {
        let idx = k - self.pole_order;
        if idx >= 0 && (idx as usize) < K {
            self.coeffs[idx as usize]
        } else {
            F::zero()
        }
    }

    /// Function value at t=0.
    ///
    /// - `pole_order < 0` → `±infinity` (sign from leading coefficient)
    /// - `pole_order == 0` → `coeffs[0]`
    /// - `pole_order > 0` → `F::zero()`
    #[inline]
    pub fn value(&self) -> F {
        if self.pole_order < 0 {
            if self.coeffs[0].is_sign_negative() {
                F::neg_infinity()
            } else {
                F::infinity()
            }
        } else if self.pole_order == 0 {
            self.coeffs[0]
        } else {
            F::zero()
        }
    }

    /// Normalize: strip leading zero coefficients and adjust pole_order.
    ///
    /// Ensures `coeffs[0] != 0` (or the value is all-zero with `pole_order = 0`).
    fn normalize(&mut self) {
        let mut shift = 0usize;
        while shift < K && self.coeffs[shift] == F::zero() {
            shift += 1;
        }
        if shift == K {
            // All zero.
            self.pole_order = 0;
            return;
        }
        if shift > 0 {
            self.pole_order += shift as i32;
            for i in 0..K {
                self.coeffs[i] = if i + shift < K {
                    self.coeffs[i + shift]
                } else {
                    F::zero()
                };
            }
        }
    }

    /// NaN-filled Laurent (for essential singularities like exp(pole)).
    fn nan_laurent() -> Self {
        Laurent {
            coeffs: [F::nan(); K],
            pole_order: 0,
        }
    }

    /// Convert to Taylor-compatible coefficient array for regular values (pole_order >= 0).
    ///
    /// Zero-pads the front for `pole_order > 0` (terms beyond K are truncated).
    /// Panics if `pole_order < 0`.
    fn as_taylor_coeffs(&self) -> [F; K] {
        debug_assert!(self.pole_order >= 0, "as_taylor_coeffs called on pole");
        let shift = self.pole_order as usize;
        std::array::from_fn(|i| {
            if i < shift {
                F::zero()
            } else if i - shift < K {
                self.coeffs[i - shift]
            } else {
                F::zero()
            }
        })
    }

    /// Is this Laurent value all-zero?
    fn is_all_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == F::zero())
    }

    /// Public accessor for is_all_zero (used by trait impls).
    #[inline]
    pub(crate) fn is_all_zero_pub(&self) -> bool {
        self.is_all_zero()
    }

    /// Public NaN constructor (used by trait impls).
    #[inline]
    pub(crate) fn nan_pub() -> Self {
        Self::nan_laurent()
    }

    /// Raw coefficient array (for Mul/Div that operate directly on coefficients).
    #[inline]
    pub(crate) fn leading_coeffs(&self) -> [F; K] {
        self.coeffs
    }

    // ── Elemental methods ──
    // Three-case pattern: pole → NaN/special, zero-at-origin → taylor via as_taylor_coeffs, regular → taylor on coeffs.

    #[inline]
    pub fn recip(self) -> Self {
        if self.is_all_zero() {
            return Self::nan_laurent();
        }
        let mut c = [F::zero(); K];
        taylor_ops::taylor_recip(&self.coeffs, &mut c);
        let mut l = Laurent {
            coeffs: c,
            pole_order: -self.pole_order,
        };
        l.normalize();
        l
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        if self.is_all_zero() {
            return Self::zero();
        }
        if self.pole_order < 0 {
            if self.pole_order % 2 != 0 {
                return Self::nan_laurent(); // Odd pole order → no clean sqrt
            }
            let mut c = [F::zero(); K];
            taylor_ops::taylor_sqrt(&self.coeffs, &mut c);
            Laurent {
                coeffs: c,
                pole_order: self.pole_order / 2,
            }
        } else if self.pole_order > 0 {
            let tc = self.as_taylor_coeffs();
            let mut c = [F::zero(); K];
            taylor_ops::taylor_sqrt(&tc, &mut c);
            let mut l = Laurent {
                coeffs: c,
                pole_order: 0,
            };
            l.normalize();
            l
        } else {
            let mut c = [F::zero(); K];
            taylor_ops::taylor_sqrt(&self.coeffs, &mut c);
            Laurent {
                coeffs: c,
                pole_order: 0,
            }
        }
    }

    #[inline]
    pub fn cbrt(self) -> Self {
        if self.is_all_zero() {
            return Self::zero();
        }
        if self.pole_order < 0 {
            if self.pole_order % 3 != 0 {
                return Self::nan_laurent();
            }
            let mut c = [F::zero(); K];
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_cbrt(&self.coeffs, &mut c, &mut s1, &mut s2);
            Laurent {
                coeffs: c,
                pole_order: self.pole_order / 3,
            }
        } else if self.pole_order > 0 {
            let tc = self.as_taylor_coeffs();
            let mut c = [F::zero(); K];
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_cbrt(&tc, &mut c, &mut s1, &mut s2);
            let mut l = Laurent {
                coeffs: c,
                pole_order: 0,
            };
            l.normalize();
            l
        } else {
            let mut c = [F::zero(); K];
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_cbrt(&self.coeffs, &mut c, &mut s1, &mut s2);
            Laurent {
                coeffs: c,
                pole_order: 0,
            }
        }
    }

    #[inline]
    pub fn powi(self, n: i32) -> Self {
        if self.is_all_zero() {
            return if n > 0 { Self::zero() } else { Self::nan_laurent() };
        }
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        taylor_ops::taylor_powi(&self.coeffs, n, &mut c, &mut s1, &mut s2);
        let mut l = Laurent {
            coeffs: c,
            pole_order: self.pole_order * n,
        };
        l.normalize();
        l
    }

    #[inline]
    pub fn powf(self, n: Self) -> Self {
        // a^b = exp(b * ln(a))
        (n * self.ln()).exp()
    }

    /// Apply a transcendental that requires a regular (non-pole) input.
    /// Returns NaN for poles, delegates to taylor_ops for regular values.
    fn apply_regular<G>(self, apply: G) -> Self
    where
        G: FnOnce(&[F; K], &mut [F; K]),
    {
        if self.pole_order < 0 {
            return Self::nan_laurent();
        }
        let mut c = [F::zero(); K];
        if self.pole_order > 0 {
            let tc = self.as_taylor_coeffs();
            apply(&tc, &mut c);
        } else {
            apply(&self.coeffs, &mut c);
        }
        let mut l = Laurent {
            coeffs: c,
            pole_order: 0,
        };
        l.normalize();
        l
    }

    #[inline]
    pub fn exp(self) -> Self {
        self.apply_regular(|a, c| taylor_ops::taylor_exp(a, c))
    }

    #[inline]
    pub fn exp2(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s = [F::zero(); K];
            taylor_ops::taylor_exp2(a, c, &mut s);
        })
    }

    #[inline]
    pub fn exp_m1(self) -> Self {
        self.apply_regular(|a, c| taylor_ops::taylor_exp_m1(a, c))
    }

    #[inline]
    pub fn ln(self) -> Self {
        if self.is_all_zero() {
            return Self::nan_laurent(); // ln(0)
        }
        if self.pole_order != 0 {
            // ln of something with a pole or zero at origin → NaN
            return Self::nan_laurent();
        }
        if self.coeffs[0] <= F::zero() {
            return Self::nan_laurent();
        }
        let mut c = [F::zero(); K];
        taylor_ops::taylor_ln(&self.coeffs, &mut c);
        Laurent {
            coeffs: c,
            pole_order: 0,
        }
    }

    #[inline]
    pub fn log2(self) -> Self {
        self.apply_regular(|a, c| taylor_ops::taylor_log2(a, c))
    }

    #[inline]
    pub fn log10(self) -> Self {
        self.apply_regular(|a, c| taylor_ops::taylor_log10(a, c))
    }

    #[inline]
    pub fn ln_1p(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s = [F::zero(); K];
            taylor_ops::taylor_ln_1p(a, c, &mut s);
        })
    }

    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    pub fn sin(self) -> Self {
        self.apply_regular(|a, c| {
            let mut co = [F::zero(); K];
            taylor_ops::taylor_sin_cos(a, c, &mut co);
        })
    }

    #[inline]
    pub fn cos(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s = [F::zero(); K];
            taylor_ops::taylor_sin_cos(a, &mut s, c);
        })
    }

    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        if self.pole_order < 0 {
            return (Self::nan_laurent(), Self::nan_laurent());
        }
        let mut s = [F::zero(); K];
        let mut co = [F::zero(); K];
        if self.pole_order > 0 {
            let tc = self.as_taylor_coeffs();
            taylor_ops::taylor_sin_cos(&tc, &mut s, &mut co);
        } else {
            taylor_ops::taylor_sin_cos(&self.coeffs, &mut s, &mut co);
        }
        let mut ls = Laurent { coeffs: s, pole_order: 0 };
        let mut lc = Laurent { coeffs: co, pole_order: 0 };
        ls.normalize();
        lc.normalize();
        (ls, lc)
    }

    #[inline]
    pub fn tan(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s = [F::zero(); K];
            taylor_ops::taylor_tan(a, c, &mut s);
        })
    }

    #[inline]
    pub fn asin(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_asin(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn acos(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_acos(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atan(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_atan(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        // atan2 only works with regular values
        if self.pole_order != 0 || other.pole_order != 0 {
            return Self::nan_laurent();
        }
        let mut c = [F::zero(); K];
        let mut s1 = [F::zero(); K];
        let mut s2 = [F::zero(); K];
        let mut s3 = [F::zero(); K];
        taylor_ops::taylor_atan2(&self.coeffs, &other.coeffs, &mut c, &mut s1, &mut s2, &mut s3);
        Laurent {
            coeffs: c,
            pole_order: 0,
        }
    }

    #[inline]
    pub fn sinh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut ch = [F::zero(); K];
            taylor_ops::taylor_sinh_cosh(a, c, &mut ch);
        })
    }

    #[inline]
    pub fn cosh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut sh = [F::zero(); K];
            taylor_ops::taylor_sinh_cosh(a, &mut sh, c);
        })
    }

    #[inline]
    pub fn tanh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s = [F::zero(); K];
            taylor_ops::taylor_tanh(a, c, &mut s);
        })
    }

    #[inline]
    pub fn asinh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_asinh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn acosh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_acosh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atanh(self) -> Self {
        self.apply_regular(|a, c| {
            let mut s1 = [F::zero(); K];
            let mut s2 = [F::zero(); K];
            taylor_ops::taylor_atanh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn abs(self) -> Self {
        if self.is_all_zero() {
            return Self::zero();
        }
        let sign = self.coeffs[0].signum();
        let mut coeffs = self.coeffs;
        for c in &mut coeffs {
            *c = *c * sign;
        }
        Laurent {
            coeffs,
            pole_order: self.pole_order,
        }
    }

    #[inline]
    pub fn signum(self) -> Self {
        Self::constant(self.value().signum())
    }

    #[inline]
    pub fn floor(self) -> Self {
        Self::constant(self.value().floor())
    }

    #[inline]
    pub fn ceil(self) -> Self {
        Self::constant(self.value().ceil())
    }

    #[inline]
    pub fn round(self) -> Self {
        Self::constant(self.value().round())
    }

    #[inline]
    pub fn trunc(self) -> Self {
        Self::constant(self.value().trunc())
    }

    #[inline]
    pub fn fract(self) -> Self {
        if self.pole_order != 0 {
            return Self::nan_laurent();
        }
        let mut coeffs = self.coeffs;
        coeffs[0] = self.coeffs[0].fract();
        Laurent {
            coeffs,
            pole_order: 0,
        }
    }

    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.value() >= other.value() {
            self
        } else {
            other
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.value() <= other.value() {
            self
        } else {
            other
        }
    }
}

#[cfg(feature = "serde")]
mod laurent_serde {
    use super::Laurent;
    use crate::Float;
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<F: Float + Serialize, const K: usize> Serialize for Laurent<F, K> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut s = serializer.serialize_struct("Laurent", 2)?;
            s.serialize_field("coeffs", self.coeffs.as_slice())?;
            s.serialize_field("pole_order", &self.pole_order)?;
            s.end()
        }
    }

    impl<'de, F: Float + Deserialize<'de>, const K: usize> Deserialize<'de> for Laurent<F, K> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            struct LaurentData<F> {
                coeffs: Vec<F>,
                pole_order: i32,
            }

            let data = LaurentData::<F>::deserialize(deserializer)?;
            let coeffs: [F; K] = data.coeffs.try_into().map_err(|v: Vec<F>| {
                serde::de::Error::invalid_length(
                    v.len(),
                    &&*format!("array of length {K}"),
                )
            })?;
            Ok(Laurent { coeffs, pole_order: data.pole_order })
        }
    }
}

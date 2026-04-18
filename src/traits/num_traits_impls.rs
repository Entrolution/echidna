use std::num::FpCategory;

use num_traits::{
    Float as NumFloat, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::dual::Dual;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::{self, TapeThreadLocal};

// ══════════════════════════════════════════════
//  Dual<F>
// ══════════════════════════════════════════════

impl<F: Float> Zero for Dual<F> {
    #[inline]
    fn zero() -> Self {
        Dual::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero()
    }
}

impl<F: Float> One for Dual<F> {
    #[inline]
    fn one() -> Self {
        Dual::constant(F::one())
    }
}

impl<F: Float> Num for Dual<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Dual::constant)
    }
}

impl<F: Float> FromPrimitive for Dual<F> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(Dual::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(Dual::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(Dual::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(Dual::constant)
    }
}

impl<F: Float> ToPrimitive for Dual<F> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.re.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.re.to_u64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.re.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.re.to_f64()
    }
}

impl<F: Float> NumCast for Dual<F> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(Dual::constant)
    }
}

impl<F: Float> Signed for Dual<F> {
    #[inline]
    fn abs(&self) -> Self {
        Dual::abs(*self)
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.re > other.re {
            *self - *other
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        Dual::signum(*self)
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.re.is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.re.is_sign_negative()
    }
}

impl<F: Float> FloatConst for Dual<F> {
    fn E() -> Self {
        Dual::constant(F::E())
    }
    fn FRAC_1_PI() -> Self {
        Dual::constant(F::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        Dual::constant(F::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        Dual::constant(F::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Dual::constant(F::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        Dual::constant(F::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        Dual::constant(F::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        Dual::constant(F::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        Dual::constant(F::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        Dual::constant(F::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        Dual::constant(F::LN_10())
    }
    fn LN_2() -> Self {
        Dual::constant(F::LN_2())
    }
    fn LOG10_E() -> Self {
        Dual::constant(F::LOG10_E())
    }
    fn LOG2_E() -> Self {
        Dual::constant(F::LOG2_E())
    }
    fn PI() -> Self {
        Dual::constant(F::PI())
    }
    fn SQRT_2() -> Self {
        Dual::constant(F::SQRT_2())
    }
    fn TAU() -> Self {
        Dual::constant(F::TAU())
    }
    fn LOG10_2() -> Self {
        Dual::constant(F::LOG10_2())
    }
    fn LOG2_10() -> Self {
        Dual::constant(F::LOG2_10())
    }
}

impl<F: Float> NumFloat for Dual<F> {
    fn nan() -> Self {
        Dual::constant(F::nan())
    }
    fn infinity() -> Self {
        Dual::constant(F::infinity())
    }
    fn neg_infinity() -> Self {
        Dual::constant(F::neg_infinity())
    }
    fn neg_zero() -> Self {
        Dual::constant(F::neg_zero())
    }

    fn min_value() -> Self {
        Dual::constant(F::min_value())
    }
    fn min_positive_value() -> Self {
        Dual::constant(F::min_positive_value())
    }
    fn max_value() -> Self {
        Dual::constant(F::max_value())
    }
    fn epsilon() -> Self {
        Dual::constant(F::epsilon())
    }

    fn is_nan(self) -> bool {
        self.re.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.re.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.re.is_finite()
    }
    fn is_normal(self) -> bool {
        self.re.is_normal()
    }
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }
    fn classify(self) -> FpCategory {
        self.re.classify()
    }

    fn floor(self) -> Self {
        Dual::floor(self)
    }
    fn ceil(self) -> Self {
        Dual::ceil(self)
    }
    fn round(self) -> Self {
        Dual::round(self)
    }
    fn trunc(self) -> Self {
        Dual::trunc(self)
    }
    fn fract(self) -> Self {
        Dual::fract(self)
    }
    fn abs(self) -> Self {
        Dual::abs(self)
    }
    fn signum(self) -> Self {
        Dual::signum(self)
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Dual::mul_add(self, a, b)
    }

    fn recip(self) -> Self {
        Dual::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        Dual::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        Dual::powf(self, n)
    }
    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }
    fn cbrt(self) -> Self {
        Dual::cbrt(self)
    }

    fn exp(self) -> Self {
        Dual::exp(self)
    }
    fn exp2(self) -> Self {
        Dual::exp2(self)
    }
    fn exp_m1(self) -> Self {
        Dual::exp_m1(self)
    }
    fn ln(self) -> Self {
        Dual::ln(self)
    }
    fn log2(self) -> Self {
        Dual::log2(self)
    }
    fn log10(self) -> Self {
        Dual::log10(self)
    }
    fn ln_1p(self) -> Self {
        Dual::ln_1p(self)
    }
    fn log(self, base: Self) -> Self {
        Dual::log(self, base)
    }

    fn sin(self) -> Self {
        Dual::sin(self)
    }
    fn cos(self) -> Self {
        Dual::cos(self)
    }
    fn tan(self) -> Self {
        Dual::tan(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        Dual::sin_cos(self)
    }
    fn asin(self) -> Self {
        Dual::asin(self)
    }
    fn acos(self) -> Self {
        Dual::acos(self)
    }
    fn atan(self) -> Self {
        Dual::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        Dual::atan2(self, other)
    }

    fn sinh(self) -> Self {
        Dual::sinh(self)
    }
    fn cosh(self) -> Self {
        Dual::cosh(self)
    }
    fn tanh(self) -> Self {
        Dual::tanh(self)
    }
    fn asinh(self) -> Self {
        Dual::asinh(self)
    }
    fn acosh(self) -> Self {
        Dual::acosh(self)
    }
    fn atanh(self) -> Self {
        Dual::atanh(self)
    }

    fn hypot(self, other: Self) -> Self {
        Dual::hypot(self, other)
    }

    fn max(self, other: Self) -> Self {
        Dual::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        Dual::min(self, other)
    }

    fn abs_sub(self, other: Self) -> Self {
        if self.re > other.re {
            self - other
        } else {
            Self::zero()
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.re.integer_decode()
    }

    fn to_degrees(self) -> Self {
        let factor = F::from(180.0).unwrap() / F::PI();
        Dual {
            re: self.re.to_degrees(),
            eps: self.eps * factor,
        }
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        Dual {
            re: self.re.to_radians(),
            eps: self.eps * factor,
        }
    }
}

// ══════════════════════════════════════════════
//  Reverse<F>
// ══════════════════════════════════════════════

impl<F: Float + TapeThreadLocal> Zero for Reverse<F> {
    #[inline]
    fn zero() -> Self {
        Reverse::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<F: Float + TapeThreadLocal> One for Reverse<F> {
    #[inline]
    fn one() -> Self {
        Reverse::constant(F::one())
    }
}

impl<F: Float + TapeThreadLocal> Num for Reverse<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Reverse::constant)
    }
}

impl<F: Float> FromPrimitive for Reverse<F> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(Reverse::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(Reverse::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(Reverse::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(Reverse::constant)
    }
}

impl<F: Float> ToPrimitive for Reverse<F> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.value.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
}

impl<F: Float + TapeThreadLocal> NumCast for Reverse<F> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(Reverse::constant)
    }
}

impl<F: Float + TapeThreadLocal> Signed for Reverse<F> {
    #[inline]
    fn abs(&self) -> Self {
        let value = self.value.abs();
        let index = tape::with_active_tape(|t| t.push_unary(self.index, self.value.signum()));
        Reverse { value, index }
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.value > other.value {
            *self - *other
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        Reverse::constant(self.value.signum())
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.value.is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.value.is_sign_negative()
    }
}

impl<F: Float + TapeThreadLocal> FloatConst for Reverse<F> {
    fn E() -> Self {
        Reverse::constant(F::E())
    }
    fn FRAC_1_PI() -> Self {
        Reverse::constant(F::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        Reverse::constant(F::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        Reverse::constant(F::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Reverse::constant(F::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        Reverse::constant(F::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        Reverse::constant(F::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        Reverse::constant(F::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        Reverse::constant(F::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        Reverse::constant(F::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        Reverse::constant(F::LN_10())
    }
    fn LN_2() -> Self {
        Reverse::constant(F::LN_2())
    }
    fn LOG10_E() -> Self {
        Reverse::constant(F::LOG10_E())
    }
    fn LOG2_E() -> Self {
        Reverse::constant(F::LOG2_E())
    }
    fn PI() -> Self {
        Reverse::constant(F::PI())
    }
    fn SQRT_2() -> Self {
        Reverse::constant(F::SQRT_2())
    }
    fn TAU() -> Self {
        Reverse::constant(F::TAU())
    }
    fn LOG10_2() -> Self {
        Reverse::constant(F::LOG10_2())
    }
    fn LOG2_10() -> Self {
        Reverse::constant(F::LOG2_10())
    }
}

/// Helper: record a unary elemental on the active tape.
#[inline]
fn rev_unary<F: Float + TapeThreadLocal>(x: Reverse<F>, f_val: F, f_deriv: F) -> Reverse<F> {
    let index = tape::with_active_tape(|t| t.push_unary(x.index, f_deriv));
    Reverse {
        value: f_val,
        index,
    }
}

/// Helper: record a binary elemental on the active tape.
#[inline]
fn rev_binary<F: Float + TapeThreadLocal>(
    x: Reverse<F>,
    y: Reverse<F>,
    f_val: F,
    dx: F,
    dy: F,
) -> Reverse<F> {
    let index = tape::with_active_tape(|t| t.push_binary(x.index, dx, y.index, dy));
    Reverse {
        value: f_val,
        index,
    }
}

impl<F: Float + TapeThreadLocal> NumFloat for Reverse<F> {
    fn nan() -> Self {
        Reverse::constant(F::nan())
    }
    fn infinity() -> Self {
        Reverse::constant(F::infinity())
    }
    fn neg_infinity() -> Self {
        Reverse::constant(F::neg_infinity())
    }
    fn neg_zero() -> Self {
        Reverse::constant(F::neg_zero())
    }

    fn min_value() -> Self {
        Reverse::constant(F::min_value())
    }
    fn min_positive_value() -> Self {
        Reverse::constant(F::min_positive_value())
    }
    fn max_value() -> Self {
        Reverse::constant(F::max_value())
    }
    fn epsilon() -> Self {
        Reverse::constant(F::epsilon())
    }

    fn is_nan(self) -> bool {
        self.value.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.value.is_finite()
    }
    fn is_normal(self) -> bool {
        self.value.is_normal()
    }
    fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }
    fn classify(self) -> FpCategory {
        self.value.classify()
    }

    // Returning Reverse::constant() (no tape entry) is intentional: floor/ceil/round/trunc
    // are piecewise-constant with zero derivative a.e. Skipping tape recording saves space
    // without affecting gradient correctness.
    fn floor(self) -> Self {
        Reverse::constant(self.value.floor())
    }
    fn ceil(self) -> Self {
        Reverse::constant(self.value.ceil())
    }
    fn round(self) -> Self {
        Reverse::constant(self.value.round())
    }
    fn trunc(self) -> Self {
        Reverse::constant(self.value.trunc())
    }
    fn fract(self) -> Self {
        rev_unary(self, self.value.fract(), F::one())
    }
    fn abs(self) -> Self {
        rev_unary(self, self.value.abs(), self.value.signum())
    }
    fn signum(self) -> Self {
        Reverse::constant(self.value.signum())
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // d(x*a + b)/dx = a, d/da = x, d/db = 1
        // We need a ternary push; emulate as binary + unary.
        // self*a + b
        self * a + b
    }

    fn recip(self) -> Self {
        let inv = F::one() / self.value;
        rev_unary(self, inv, -inv * inv)
    }

    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return rev_unary(self, F::one(), F::zero());
        }
        let val = self.value.powi(n);
        let deriv = if n == i32::MIN {
            F::from(n).unwrap() * val / self.value
        } else {
            F::from(n).unwrap() * self.value.powi(n - 1)
        };
        rev_unary(self, val, deriv)
    }

    fn powf(self, n: Self) -> Self {
        // Constant integer exponent fast path (see `Dual::powf`): if `n` is a
        // tape constant and its value is a losslessly representable integer,
        // dispatch to `powi`. This avoids recording `dy = val * ln(self) = NaN`
        // for `self < 0` — the NaN would contaminate the constant's adjoint
        // slot, which is silently dropped on the reverse sweep for a true
        // constant, but breaks any caller that later promotes `n` to a live
        // variable by building on the same tape shape.
        if n.index == crate::tape::CONSTANT {
            if let Some(ni) = n.value.to_i32() {
                if F::from(ni).unwrap() == n.value {
                    return NumFloat::powi(self, ni);
                }
            }
        }
        if n.value == F::zero() {
            // a^0 = 1, d/da(a^0) = 0, d/db(a^b)|_{b=0} = ln(a) (for a > 0)
            let dy = if self.value > F::zero() {
                self.value.ln()
            } else {
                F::zero()
            };
            return rev_binary(self, n, F::one(), F::zero(), dy);
        }
        let val = self.value.powf(n.value);
        let dx = if self.value == F::zero() || val == F::zero() {
            // Use n*x^(n-1) form to avoid 0/0 when x=0 and to handle
            // underflow when x^n underflows to 0 but x != 0
            n.value * self.value.powf(n.value - F::one())
        } else {
            n.value * val / self.value
        };
        let dy = if val == F::zero() || self.value <= F::zero() {
            // val == 0: lim_{x→0+} x^y * ln(x) = 0 for y > 0.
            // self <= 0: ln(self) is NaN (stdlib real-valued). For self < 0
            //   with finite val, b must have been integer (otherwise val = NaN);
            //   the "derivative w.r.t. b at an integer b" is not classically
            //   defined, and 0 is the conventional choice — matches the
            //   forward-mode Dual::powf integer-dispatch fast path.
            F::zero()
        } else {
            val * self.value.ln()
        };
        rev_binary(self, n, val, dx, dy)
    }

    fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        let two = F::one() + F::one();
        rev_unary(self, s, F::one() / (two * s))
    }

    fn cbrt(self) -> Self {
        let c = self.value.cbrt();
        let three = F::from(3.0).unwrap();
        rev_unary(self, c, F::one() / (three * c * c))
    }

    fn exp(self) -> Self {
        let e = self.value.exp();
        rev_unary(self, e, e)
    }

    fn exp2(self) -> Self {
        let e = self.value.exp2();
        rev_unary(self, e, e * F::LN_2())
    }

    fn exp_m1(self) -> Self {
        rev_unary(self, self.value.exp_m1(), self.value.exp())
    }

    fn ln(self) -> Self {
        rev_unary(self, self.value.ln(), F::one() / self.value)
    }

    fn log2(self) -> Self {
        rev_unary(self, self.value.log2(), F::one() / (self.value * F::LN_2()))
    }

    fn log10(self) -> Self {
        rev_unary(
            self,
            self.value.log10(),
            F::one() / (self.value * F::LN_10()),
        )
    }

    fn ln_1p(self) -> Self {
        rev_unary(self, self.value.ln_1p(), F::one() / (F::one() + self.value))
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn sin(self) -> Self {
        rev_unary(self, self.value.sin(), self.value.cos())
    }

    fn cos(self) -> Self {
        rev_unary(self, self.value.cos(), -self.value.sin())
    }

    fn tan(self) -> Self {
        let c = self.value.cos();
        rev_unary(self, self.value.tan(), F::one() / (c * c))
    }

    // NOTE (verified correct): Two `rev_unary` calls for sin/cos are correct.
    // Each output gets its own tape index; adjoints accumulate independently
    // through both entries back to `self.index`.
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.value.sin_cos();
        (rev_unary(self, s, c), rev_unary(self, c, -s))
    }

    fn asin(self) -> Self {
        rev_unary(
            self,
            self.value.asin(),
            F::one() / ((F::one() - self.value) * (F::one() + self.value)).sqrt(),
        )
    }

    fn acos(self) -> Self {
        rev_unary(
            self,
            self.value.acos(),
            -F::one() / ((F::one() - self.value) * (F::one() + self.value)).sqrt(),
        )
    }

    fn atan(self) -> Self {
        // For large |x|, use (1/x)²/(1+(1/x)²) to avoid 1+x² overflow
        let deriv = if self.value.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.value;
            inv * inv / (F::one() + inv * inv)
        } else {
            F::one() / (F::one() + self.value * self.value)
        };
        rev_unary(self, self.value.atan(), deriv)
    }

    fn atan2(self, other: Self) -> Self {
        let h = self.value.hypot(other.value);
        if h == F::zero() {
            // At the origin, atan2 gradient is mathematically undefined.
            // Returning a constant (zero gradient) matches JAX/PyTorch convention.
            return Reverse::constant(self.value.atan2(other.value));
        }
        // Factor as (value/h)/h to avoid squaring h, which overflows for
        // |h| > sqrt(f64::MAX) and underflows for very small h even when the
        // true partial is representable. Both value/h terms are bounded by 1.
        let dx = other.value / h / h;
        let dy = -self.value / h / h;
        rev_binary(self, other, self.value.atan2(other.value), dx, dy)
    }

    fn sinh(self) -> Self {
        rev_unary(self, self.value.sinh(), self.value.cosh())
    }

    fn cosh(self) -> Self {
        rev_unary(self, self.value.cosh(), self.value.sinh())
    }

    fn tanh(self) -> Self {
        let c = self.value.cosh();
        rev_unary(self, self.value.tanh(), F::one() / (c * c))
    }

    fn asinh(self) -> Self {
        // See `Dual::asinh` for the overflow rationale.
        let deriv = if self.value.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.value;
            inv.abs() / (F::one() + inv * inv).sqrt()
        } else {
            F::one() / (self.value * self.value + F::one()).sqrt()
        };
        rev_unary(self, self.value.asinh(), deriv)
    }

    fn acosh(self) -> Self {
        let deriv = if self.value.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.value;
            inv.abs() / (F::one() - inv * inv).sqrt()
        } else {
            F::one() / (self.value * self.value - F::one()).sqrt()
        };
        rev_unary(self, self.value.acosh(), deriv)
    }

    fn atanh(self) -> Self {
        rev_unary(
            self,
            self.value.atanh(),
            F::one() / ((F::one() - self.value) * (F::one() + self.value)),
        )
    }

    fn hypot(self, other: Self) -> Self {
        let h = self.value.hypot(other.value);
        let (dx, dy) = if h == F::zero() {
            (F::zero(), F::zero())
        } else {
            (self.value / h, other.value / h)
        };
        rev_binary(self, other, h, dx, dy)
    }

    fn max(self, other: Self) -> Self {
        if self.value >= other.value || other.value.is_nan() {
            rev_unary(self, self.value, F::one())
        } else {
            rev_unary(other, other.value, F::one())
        }
    }

    fn min(self, other: Self) -> Self {
        if self.value <= other.value || other.value.is_nan() {
            rev_unary(self, self.value, F::one())
        } else {
            rev_unary(other, other.value, F::one())
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        if self.value > other.value {
            self - other
        } else {
            Self::zero()
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }

    fn to_degrees(self) -> Self {
        let factor = F::from(180.0).unwrap() / F::PI();
        rev_unary(self, self.value.to_degrees(), factor)
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        rev_unary(self, self.value.to_radians(), factor)
    }
}

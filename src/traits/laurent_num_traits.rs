//! `num_traits` implementations for `Laurent<F, K>`.
//! Filled in Step 5.

use std::num::FpCategory;

use num_traits::{
    Float as NumFloat, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::float::Float;
use crate::laurent::Laurent;

// ══════════════════════════════════════════════
//  Laurent<F, K>
// ══════════════════════════════════════════════

impl<F: Float, const K: usize> Zero for Laurent<F, K> {
    #[inline]
    fn zero() -> Self {
        Laurent::zero()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value().is_zero()
    }
}

impl<F: Float, const K: usize> One for Laurent<F, K> {
    #[inline]
    fn one() -> Self {
        Laurent::one()
    }
}

impl<F: Float, const K: usize> Num for Laurent<F, K> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Laurent::constant)
    }
}

impl<F: Float, const K: usize> FromPrimitive for Laurent<F, K> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(Laurent::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(Laurent::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(Laurent::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(Laurent::constant)
    }
}

impl<F: Float, const K: usize> ToPrimitive for Laurent<F, K> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value().to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value().to_u64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.value().to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value().to_f64()
    }
}

impl<F: Float, const K: usize> NumCast for Laurent<F, K> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(Laurent::constant)
    }
}

impl<F: Float, const K: usize> Signed for Laurent<F, K> {
    #[inline]
    fn abs(&self) -> Self {
        Laurent::abs(*self)
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.value() > other.value() {
            *self - *other
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        Laurent::signum(*self)
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.value().is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.value().is_sign_negative()
    }
}

impl<F: Float, const K: usize> FloatConst for Laurent<F, K> {
    fn E() -> Self {
        Laurent::constant(F::E())
    }
    fn FRAC_1_PI() -> Self {
        Laurent::constant(F::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        Laurent::constant(F::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        Laurent::constant(F::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Laurent::constant(F::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        Laurent::constant(F::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        Laurent::constant(F::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        Laurent::constant(F::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        Laurent::constant(F::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        Laurent::constant(F::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        Laurent::constant(F::LN_10())
    }
    fn LN_2() -> Self {
        Laurent::constant(F::LN_2())
    }
    fn LOG10_E() -> Self {
        Laurent::constant(F::LOG10_E())
    }
    fn LOG2_E() -> Self {
        Laurent::constant(F::LOG2_E())
    }
    fn PI() -> Self {
        Laurent::constant(F::PI())
    }
    fn SQRT_2() -> Self {
        Laurent::constant(F::SQRT_2())
    }
    fn TAU() -> Self {
        Laurent::constant(F::TAU())
    }
    fn LOG10_2() -> Self {
        Laurent::constant(F::LOG10_2())
    }
    fn LOG2_10() -> Self {
        Laurent::constant(F::LOG2_10())
    }
}

impl<F: Float, const K: usize> NumFloat for Laurent<F, K> {
    fn nan() -> Self {
        Laurent::constant(F::nan())
    }
    fn infinity() -> Self {
        Laurent::constant(F::infinity())
    }
    fn neg_infinity() -> Self {
        Laurent::constant(F::neg_infinity())
    }
    fn neg_zero() -> Self {
        Laurent::constant(F::neg_zero())
    }
    fn min_value() -> Self {
        Laurent::constant(F::min_value())
    }
    fn min_positive_value() -> Self {
        Laurent::constant(F::min_positive_value())
    }
    fn max_value() -> Self {
        Laurent::constant(F::max_value())
    }
    fn epsilon() -> Self {
        Laurent::constant(F::epsilon())
    }

    fn is_nan(self) -> bool {
        self.value().is_nan()
    }
    fn is_infinite(self) -> bool {
        self.value().is_infinite()
    }
    fn is_finite(self) -> bool {
        self.value().is_finite()
    }
    fn is_normal(self) -> bool {
        self.value().is_normal()
    }
    fn is_sign_positive(self) -> bool {
        self.value().is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.value().is_sign_negative()
    }
    fn classify(self) -> FpCategory {
        self.value().classify()
    }

    fn floor(self) -> Self {
        Laurent::floor(self)
    }
    fn ceil(self) -> Self {
        Laurent::ceil(self)
    }
    fn round(self) -> Self {
        Laurent::round(self)
    }
    fn trunc(self) -> Self {
        Laurent::trunc(self)
    }
    fn fract(self) -> Self {
        Laurent::fract(self)
    }
    fn abs(self) -> Self {
        Laurent::abs(self)
    }
    fn signum(self) -> Self {
        Laurent::signum(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        Laurent::mul_add(self, a, b)
    }
    fn recip(self) -> Self {
        Laurent::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        Laurent::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        Laurent::powf(self, n)
    }
    fn sqrt(self) -> Self {
        Laurent::sqrt(self)
    }
    fn cbrt(self) -> Self {
        Laurent::cbrt(self)
    }
    fn exp(self) -> Self {
        Laurent::exp(self)
    }
    fn exp2(self) -> Self {
        Laurent::exp2(self)
    }
    fn exp_m1(self) -> Self {
        Laurent::exp_m1(self)
    }
    fn ln(self) -> Self {
        Laurent::ln(self)
    }
    fn log2(self) -> Self {
        Laurent::log2(self)
    }
    fn log10(self) -> Self {
        Laurent::log10(self)
    }
    fn ln_1p(self) -> Self {
        Laurent::ln_1p(self)
    }
    fn log(self, base: Self) -> Self {
        Laurent::log(self, base)
    }
    fn sin(self) -> Self {
        Laurent::sin(self)
    }
    fn cos(self) -> Self {
        Laurent::cos(self)
    }
    fn tan(self) -> Self {
        Laurent::tan(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        Laurent::sin_cos(self)
    }
    fn asin(self) -> Self {
        Laurent::asin(self)
    }
    fn acos(self) -> Self {
        Laurent::acos(self)
    }
    fn atan(self) -> Self {
        Laurent::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        Laurent::atan2(self, other)
    }
    fn sinh(self) -> Self {
        Laurent::sinh(self)
    }
    fn cosh(self) -> Self {
        Laurent::cosh(self)
    }
    fn tanh(self) -> Self {
        Laurent::tanh(self)
    }
    fn asinh(self) -> Self {
        Laurent::asinh(self)
    }
    fn acosh(self) -> Self {
        Laurent::acosh(self)
    }
    fn atanh(self) -> Self {
        Laurent::atanh(self)
    }
    fn hypot(self, other: Self) -> Self {
        Laurent::hypot(self, other)
    }
    fn max(self, other: Self) -> Self {
        Laurent::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        Laurent::min(self, other)
    }

    fn abs_sub(self, other: Self) -> Self {
        if self.value() > other.value() {
            self - other
        } else {
            Self::zero()
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value().integer_decode()
    }

    fn to_degrees(self) -> Self {
        let factor = F::from(180.0).unwrap() / F::PI();
        if self.pole_order() != 0 {
            return Self::constant(self.value().to_degrees());
        }
        let coeffs = std::array::from_fn(|k| {
            if k == 0 {
                self.coeff(0).to_degrees()
            } else {
                self.coeff(k as i32) * factor
            }
        });
        Laurent::new(coeffs, 0)
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        if self.pole_order() != 0 {
            return Self::constant(self.value().to_radians());
        }
        let coeffs = std::array::from_fn(|k| {
            if k == 0 {
                self.coeff(0).to_radians()
            } else {
                self.coeff(k as i32) * factor
            }
        });
        Laurent::new(coeffs, 0)
    }
}

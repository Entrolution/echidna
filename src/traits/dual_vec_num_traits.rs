use std::num::FpCategory;

use num_traits::{
    Float as NumFloat, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::dual_vec::DualVec;
use crate::float::Float;

impl<F: Float, const N: usize> Zero for DualVec<F, N> {
    #[inline(always)]
    fn zero() -> Self {
        DualVec::constant(F::zero())
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re.is_zero()
    }
}

impl<F: Float, const N: usize> One for DualVec<F, N> {
    #[inline(always)]
    fn one() -> Self {
        DualVec::constant(F::one())
    }
}

impl<F: Float, const N: usize> Num for DualVec<F, N> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(DualVec::constant)
    }
}

impl<F: Float, const N: usize> FromPrimitive for DualVec<F, N> {
    #[inline(always)]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(DualVec::constant)
    }
    #[inline(always)]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(DualVec::constant)
    }
    #[inline(always)]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(DualVec::constant)
    }
    #[inline(always)]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(DualVec::constant)
    }
}

impl<F: Float, const N: usize> ToPrimitive for DualVec<F, N> {
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

impl<F: Float, const N: usize> NumCast for DualVec<F, N> {
    #[inline(always)]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(DualVec::constant)
    }
}

impl<F: Float, const N: usize> Signed for DualVec<F, N> {
    #[inline]
    fn abs(&self) -> Self {
        DualVec::abs(*self)
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
        DualVec::signum(*self)
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

impl<F: Float, const N: usize> FloatConst for DualVec<F, N> {
    fn E() -> Self {
        DualVec::constant(F::E())
    }
    fn FRAC_1_PI() -> Self {
        DualVec::constant(F::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        DualVec::constant(F::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        DualVec::constant(F::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        DualVec::constant(F::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        DualVec::constant(F::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        DualVec::constant(F::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        DualVec::constant(F::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        DualVec::constant(F::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        DualVec::constant(F::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        DualVec::constant(F::LN_10())
    }
    fn LN_2() -> Self {
        DualVec::constant(F::LN_2())
    }
    fn LOG10_E() -> Self {
        DualVec::constant(F::LOG10_E())
    }
    fn LOG2_E() -> Self {
        DualVec::constant(F::LOG2_E())
    }
    fn PI() -> Self {
        DualVec::constant(F::PI())
    }
    fn SQRT_2() -> Self {
        DualVec::constant(F::SQRT_2())
    }
    fn TAU() -> Self {
        DualVec::constant(F::TAU())
    }
    fn LOG10_2() -> Self {
        DualVec::constant(F::LOG10_2())
    }
    fn LOG2_10() -> Self {
        DualVec::constant(F::LOG2_10())
    }
}

impl<F: Float, const N: usize> NumFloat for DualVec<F, N> {
    fn nan() -> Self {
        DualVec::constant(F::nan())
    }
    fn infinity() -> Self {
        DualVec::constant(F::infinity())
    }
    fn neg_infinity() -> Self {
        DualVec::constant(F::neg_infinity())
    }
    fn neg_zero() -> Self {
        DualVec::constant(F::neg_zero())
    }

    fn min_value() -> Self {
        DualVec::constant(F::min_value())
    }
    fn min_positive_value() -> Self {
        DualVec::constant(F::min_positive_value())
    }
    fn max_value() -> Self {
        DualVec::constant(F::max_value())
    }
    fn epsilon() -> Self {
        DualVec::constant(F::epsilon())
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
        DualVec::floor(self)
    }
    fn ceil(self) -> Self {
        DualVec::ceil(self)
    }
    fn round(self) -> Self {
        DualVec::round(self)
    }
    fn trunc(self) -> Self {
        DualVec::trunc(self)
    }
    fn fract(self) -> Self {
        DualVec::fract(self)
    }
    fn abs(self) -> Self {
        DualVec::abs(self)
    }
    fn signum(self) -> Self {
        DualVec::signum(self)
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        DualVec::mul_add(self, a, b)
    }

    fn recip(self) -> Self {
        DualVec::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        DualVec::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        DualVec::powf(self, n)
    }
    fn sqrt(self) -> Self {
        DualVec::sqrt(self)
    }
    fn cbrt(self) -> Self {
        DualVec::cbrt(self)
    }

    fn exp(self) -> Self {
        DualVec::exp(self)
    }
    fn exp2(self) -> Self {
        DualVec::exp2(self)
    }
    fn exp_m1(self) -> Self {
        DualVec::exp_m1(self)
    }
    fn ln(self) -> Self {
        DualVec::ln(self)
    }
    fn log2(self) -> Self {
        DualVec::log2(self)
    }
    fn log10(self) -> Self {
        DualVec::log10(self)
    }
    fn ln_1p(self) -> Self {
        DualVec::ln_1p(self)
    }
    fn log(self, base: Self) -> Self {
        DualVec::log(self, base)
    }

    fn sin(self) -> Self {
        DualVec::sin(self)
    }
    fn cos(self) -> Self {
        DualVec::cos(self)
    }
    fn tan(self) -> Self {
        DualVec::tan(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        DualVec::sin_cos(self)
    }
    fn asin(self) -> Self {
        DualVec::asin(self)
    }
    fn acos(self) -> Self {
        DualVec::acos(self)
    }
    fn atan(self) -> Self {
        DualVec::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        DualVec::atan2(self, other)
    }

    fn sinh(self) -> Self {
        DualVec::sinh(self)
    }
    fn cosh(self) -> Self {
        DualVec::cosh(self)
    }
    fn tanh(self) -> Self {
        DualVec::tanh(self)
    }
    fn asinh(self) -> Self {
        DualVec::asinh(self)
    }
    fn acosh(self) -> Self {
        DualVec::acosh(self)
    }
    fn atanh(self) -> Self {
        DualVec::atanh(self)
    }

    fn hypot(self, other: Self) -> Self {
        DualVec::hypot(self, other)
    }

    fn max(self, other: Self) -> Self {
        DualVec::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        DualVec::min(self, other)
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
        DualVec {
            re: self.re.to_degrees(),
            eps: std::array::from_fn(|k| self.eps[k] * factor),
        }
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        DualVec {
            re: self.re.to_radians(),
            eps: std::array::from_fn(|k| self.eps[k] * factor),
        }
    }
}

//! `num_traits` implementations for `Taylor<F, K>` and `TaylorDyn<F>`.

use std::num::FpCategory;

use num_traits::{
    Float as NumFloat, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::float::Float;
use crate::taylor::Taylor;

// ══════════════════════════════════════════════
//  Taylor<F, K>
// ══════════════════════════════════════════════

impl<F: Float, const K: usize> Zero for Taylor<F, K> {
    #[inline]
    fn zero() -> Self {
        Taylor::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero()
    }
}

impl<F: Float, const K: usize> One for Taylor<F, K> {
    #[inline]
    fn one() -> Self {
        Taylor::constant(F::one())
    }
}

impl<F: Float, const K: usize> Num for Taylor<F, K> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Taylor::constant)
    }
}

impl<F: Float, const K: usize> FromPrimitive for Taylor<F, K> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(Taylor::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(Taylor::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(Taylor::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(Taylor::constant)
    }
}

impl<F: Float, const K: usize> ToPrimitive for Taylor<F, K> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.coeffs[0].to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.coeffs[0].to_u64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.coeffs[0].to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.coeffs[0].to_f64()
    }
}

impl<F: Float, const K: usize> NumCast for Taylor<F, K> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(Taylor::constant)
    }
}

impl<F: Float, const K: usize> Signed for Taylor<F, K> {
    #[inline]
    fn abs(&self) -> Self {
        Taylor::abs(*self)
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.coeffs[0] > other.coeffs[0] {
            *self - *other
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        Taylor::signum(*self)
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.coeffs[0].is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.coeffs[0].is_sign_negative()
    }
}

impl<F: Float, const K: usize> FloatConst for Taylor<F, K> {
    fn E() -> Self { Taylor::constant(F::E()) }
    fn FRAC_1_PI() -> Self { Taylor::constant(F::FRAC_1_PI()) }
    fn FRAC_1_SQRT_2() -> Self { Taylor::constant(F::FRAC_1_SQRT_2()) }
    fn FRAC_2_PI() -> Self { Taylor::constant(F::FRAC_2_PI()) }
    fn FRAC_2_SQRT_PI() -> Self { Taylor::constant(F::FRAC_2_SQRT_PI()) }
    fn FRAC_PI_2() -> Self { Taylor::constant(F::FRAC_PI_2()) }
    fn FRAC_PI_3() -> Self { Taylor::constant(F::FRAC_PI_3()) }
    fn FRAC_PI_4() -> Self { Taylor::constant(F::FRAC_PI_4()) }
    fn FRAC_PI_6() -> Self { Taylor::constant(F::FRAC_PI_6()) }
    fn FRAC_PI_8() -> Self { Taylor::constant(F::FRAC_PI_8()) }
    fn LN_10() -> Self { Taylor::constant(F::LN_10()) }
    fn LN_2() -> Self { Taylor::constant(F::LN_2()) }
    fn LOG10_E() -> Self { Taylor::constant(F::LOG10_E()) }
    fn LOG2_E() -> Self { Taylor::constant(F::LOG2_E()) }
    fn PI() -> Self { Taylor::constant(F::PI()) }
    fn SQRT_2() -> Self { Taylor::constant(F::SQRT_2()) }
    fn TAU() -> Self { Taylor::constant(F::TAU()) }
    fn LOG10_2() -> Self { Taylor::constant(F::LOG10_2()) }
    fn LOG2_10() -> Self { Taylor::constant(F::LOG2_10()) }
}

impl<F: Float, const K: usize> NumFloat for Taylor<F, K> {
    fn nan() -> Self { Taylor::constant(F::nan()) }
    fn infinity() -> Self { Taylor::constant(F::infinity()) }
    fn neg_infinity() -> Self { Taylor::constant(F::neg_infinity()) }
    fn neg_zero() -> Self { Taylor::constant(F::neg_zero()) }
    fn min_value() -> Self { Taylor::constant(F::min_value()) }
    fn min_positive_value() -> Self { Taylor::constant(F::min_positive_value()) }
    fn max_value() -> Self { Taylor::constant(F::max_value()) }
    fn epsilon() -> Self { Taylor::constant(F::epsilon()) }

    fn is_nan(self) -> bool { self.coeffs[0].is_nan() }
    fn is_infinite(self) -> bool { self.coeffs[0].is_infinite() }
    fn is_finite(self) -> bool { self.coeffs[0].is_finite() }
    fn is_normal(self) -> bool { self.coeffs[0].is_normal() }
    fn is_sign_positive(self) -> bool { self.coeffs[0].is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.coeffs[0].is_sign_negative() }
    fn classify(self) -> FpCategory { self.coeffs[0].classify() }

    fn floor(self) -> Self { Taylor::floor(self) }
    fn ceil(self) -> Self { Taylor::ceil(self) }
    fn round(self) -> Self { Taylor::round(self) }
    fn trunc(self) -> Self { Taylor::trunc(self) }
    fn fract(self) -> Self { Taylor::fract(self) }
    fn abs(self) -> Self { Taylor::abs(self) }
    fn signum(self) -> Self { Taylor::signum(self) }
    fn mul_add(self, a: Self, b: Self) -> Self { Taylor::mul_add(self, a, b) }
    fn recip(self) -> Self { Taylor::recip(self) }
    fn powi(self, n: i32) -> Self { Taylor::powi(self, n) }
    fn powf(self, n: Self) -> Self { Taylor::powf(self, n) }
    fn sqrt(self) -> Self { Taylor::sqrt(self) }
    fn cbrt(self) -> Self { Taylor::cbrt(self) }
    fn exp(self) -> Self { Taylor::exp(self) }
    fn exp2(self) -> Self { Taylor::exp2(self) }
    fn exp_m1(self) -> Self { Taylor::exp_m1(self) }
    fn ln(self) -> Self { Taylor::ln(self) }
    fn log2(self) -> Self { Taylor::log2(self) }
    fn log10(self) -> Self { Taylor::log10(self) }
    fn ln_1p(self) -> Self { Taylor::ln_1p(self) }
    fn log(self, base: Self) -> Self { Taylor::log(self, base) }
    fn sin(self) -> Self { Taylor::sin(self) }
    fn cos(self) -> Self { Taylor::cos(self) }
    fn tan(self) -> Self { Taylor::tan(self) }
    fn sin_cos(self) -> (Self, Self) { Taylor::sin_cos(self) }
    fn asin(self) -> Self { Taylor::asin(self) }
    fn acos(self) -> Self { Taylor::acos(self) }
    fn atan(self) -> Self { Taylor::atan(self) }
    fn atan2(self, other: Self) -> Self { Taylor::atan2(self, other) }
    fn sinh(self) -> Self { Taylor::sinh(self) }
    fn cosh(self) -> Self { Taylor::cosh(self) }
    fn tanh(self) -> Self { Taylor::tanh(self) }
    fn asinh(self) -> Self { Taylor::asinh(self) }
    fn acosh(self) -> Self { Taylor::acosh(self) }
    fn atanh(self) -> Self { Taylor::atanh(self) }
    fn hypot(self, other: Self) -> Self { Taylor::hypot(self, other) }
    fn max(self, other: Self) -> Self { Taylor::max(self, other) }
    fn min(self, other: Self) -> Self { Taylor::min(self, other) }

    fn abs_sub(self, other: Self) -> Self {
        if self.coeffs[0] > other.coeffs[0] {
            self - other
        } else {
            Self::zero()
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.coeffs[0].integer_decode()
    }

    fn to_degrees(self) -> Self {
        let factor = F::from(180.0).unwrap() / F::PI();
        Taylor {
            coeffs: std::array::from_fn(|k| {
                if k == 0 {
                    self.coeffs[0].to_degrees()
                } else {
                    self.coeffs[k] * factor
                }
            }),
        }
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        Taylor {
            coeffs: std::array::from_fn(|k| {
                if k == 0 {
                    self.coeffs[0].to_radians()
                } else {
                    self.coeffs[k] * factor
                }
            }),
        }
    }
}

// ══════════════════════════════════════════════
//  TaylorDyn<F>
// ══════════════════════════════════════════════

use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn};

impl<F: Float + TaylorArenaLocal> Zero for TaylorDyn<F> {
    #[inline]
    fn zero() -> Self {
        TaylorDyn::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<F: Float + TaylorArenaLocal> One for TaylorDyn<F> {
    #[inline]
    fn one() -> Self {
        TaylorDyn::constant(F::one())
    }
}

impl<F: Float + TaylorArenaLocal> Num for TaylorDyn<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(TaylorDyn::constant)
    }
}

impl<F: Float> FromPrimitive for TaylorDyn<F> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(TaylorDyn::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(TaylorDyn::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(TaylorDyn::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(TaylorDyn::constant)
    }
}

impl<F: Float> ToPrimitive for TaylorDyn<F> {
    #[inline]
    fn to_i64(&self) -> Option<i64> { self.value.to_i64() }
    #[inline]
    fn to_u64(&self) -> Option<u64> { self.value.to_u64() }
    #[inline]
    fn to_f32(&self) -> Option<f32> { self.value.to_f32() }
    #[inline]
    fn to_f64(&self) -> Option<f64> { self.value.to_f64() }
}

impl<F: Float + TaylorArenaLocal> NumCast for TaylorDyn<F> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(TaylorDyn::constant)
    }
}

impl<F: Float + TaylorArenaLocal> Signed for TaylorDyn<F> {
    #[inline]
    fn abs(&self) -> Self { TaylorDyn::abs(*self) }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.value > other.value { *self - *other } else { Self::zero() }
    }
    #[inline]
    fn signum(&self) -> Self { TaylorDyn::signum(*self) }
    #[inline]
    fn is_positive(&self) -> bool { self.value.is_sign_positive() }
    #[inline]
    fn is_negative(&self) -> bool { self.value.is_sign_negative() }
}

impl<F: Float + TaylorArenaLocal> FloatConst for TaylorDyn<F> {
    fn E() -> Self { TaylorDyn::constant(F::E()) }
    fn FRAC_1_PI() -> Self { TaylorDyn::constant(F::FRAC_1_PI()) }
    fn FRAC_1_SQRT_2() -> Self { TaylorDyn::constant(F::FRAC_1_SQRT_2()) }
    fn FRAC_2_PI() -> Self { TaylorDyn::constant(F::FRAC_2_PI()) }
    fn FRAC_2_SQRT_PI() -> Self { TaylorDyn::constant(F::FRAC_2_SQRT_PI()) }
    fn FRAC_PI_2() -> Self { TaylorDyn::constant(F::FRAC_PI_2()) }
    fn FRAC_PI_3() -> Self { TaylorDyn::constant(F::FRAC_PI_3()) }
    fn FRAC_PI_4() -> Self { TaylorDyn::constant(F::FRAC_PI_4()) }
    fn FRAC_PI_6() -> Self { TaylorDyn::constant(F::FRAC_PI_6()) }
    fn FRAC_PI_8() -> Self { TaylorDyn::constant(F::FRAC_PI_8()) }
    fn LN_10() -> Self { TaylorDyn::constant(F::LN_10()) }
    fn LN_2() -> Self { TaylorDyn::constant(F::LN_2()) }
    fn LOG10_E() -> Self { TaylorDyn::constant(F::LOG10_E()) }
    fn LOG2_E() -> Self { TaylorDyn::constant(F::LOG2_E()) }
    fn PI() -> Self { TaylorDyn::constant(F::PI()) }
    fn SQRT_2() -> Self { TaylorDyn::constant(F::SQRT_2()) }
    fn TAU() -> Self { TaylorDyn::constant(F::TAU()) }
    fn LOG10_2() -> Self { TaylorDyn::constant(F::LOG10_2()) }
    fn LOG2_10() -> Self { TaylorDyn::constant(F::LOG2_10()) }
}

impl<F: Float + TaylorArenaLocal> NumFloat for TaylorDyn<F> {
    fn nan() -> Self { TaylorDyn::constant(F::nan()) }
    fn infinity() -> Self { TaylorDyn::constant(F::infinity()) }
    fn neg_infinity() -> Self { TaylorDyn::constant(F::neg_infinity()) }
    fn neg_zero() -> Self { TaylorDyn::constant(F::neg_zero()) }
    fn min_value() -> Self { TaylorDyn::constant(F::min_value()) }
    fn min_positive_value() -> Self { TaylorDyn::constant(F::min_positive_value()) }
    fn max_value() -> Self { TaylorDyn::constant(F::max_value()) }
    fn epsilon() -> Self { TaylorDyn::constant(F::epsilon()) }

    fn is_nan(self) -> bool { self.value.is_nan() }
    fn is_infinite(self) -> bool { self.value.is_infinite() }
    fn is_finite(self) -> bool { self.value.is_finite() }
    fn is_normal(self) -> bool { self.value.is_normal() }
    fn is_sign_positive(self) -> bool { self.value.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.value.is_sign_negative() }
    fn classify(self) -> FpCategory { self.value.classify() }

    fn floor(self) -> Self { TaylorDyn::floor(self) }
    fn ceil(self) -> Self { TaylorDyn::ceil(self) }
    fn round(self) -> Self { TaylorDyn::round(self) }
    fn trunc(self) -> Self { TaylorDyn::trunc(self) }
    fn fract(self) -> Self { TaylorDyn::fract(self) }
    fn abs(self) -> Self { TaylorDyn::abs(self) }
    fn signum(self) -> Self { TaylorDyn::signum(self) }
    fn mul_add(self, a: Self, b: Self) -> Self { self * a + b }
    fn recip(self) -> Self { TaylorDyn::recip(self) }
    fn powi(self, n: i32) -> Self { TaylorDyn::powi(self, n) }
    fn powf(self, n: Self) -> Self { TaylorDyn::powf(self, n) }
    fn sqrt(self) -> Self { TaylorDyn::sqrt(self) }
    fn cbrt(self) -> Self { TaylorDyn::cbrt(self) }
    fn exp(self) -> Self { TaylorDyn::exp(self) }
    fn exp2(self) -> Self { TaylorDyn::exp2(self) }
    fn exp_m1(self) -> Self { TaylorDyn::exp_m1(self) }
    fn ln(self) -> Self { TaylorDyn::ln(self) }
    fn log2(self) -> Self { TaylorDyn::log2(self) }
    fn log10(self) -> Self { TaylorDyn::log10(self) }
    fn ln_1p(self) -> Self { TaylorDyn::ln_1p(self) }
    fn log(self, base: Self) -> Self { TaylorDyn::log(self, base) }
    fn sin(self) -> Self { TaylorDyn::sin(self) }
    fn cos(self) -> Self { TaylorDyn::cos(self) }
    fn tan(self) -> Self { TaylorDyn::tan(self) }
    fn sin_cos(self) -> (Self, Self) { TaylorDyn::sin_cos(self) }
    fn asin(self) -> Self { TaylorDyn::asin(self) }
    fn acos(self) -> Self { TaylorDyn::acos(self) }
    fn atan(self) -> Self { TaylorDyn::atan(self) }
    fn atan2(self, other: Self) -> Self { TaylorDyn::atan2(self, other) }
    fn sinh(self) -> Self { TaylorDyn::sinh(self) }
    fn cosh(self) -> Self { TaylorDyn::cosh(self) }
    fn tanh(self) -> Self { TaylorDyn::tanh(self) }
    fn asinh(self) -> Self { TaylorDyn::asinh(self) }
    fn acosh(self) -> Self { TaylorDyn::acosh(self) }
    fn atanh(self) -> Self { TaylorDyn::atanh(self) }
    fn hypot(self, other: Self) -> Self { TaylorDyn::hypot(self, other) }
    fn max(self, other: Self) -> Self { TaylorDyn::max(self, other) }
    fn min(self, other: Self) -> Self { TaylorDyn::min(self, other) }

    fn abs_sub(self, other: Self) -> Self {
        if self.value > other.value { self - other } else { Self::zero() }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }

    fn to_degrees(self) -> Self {
        let factor = F::from(180.0).unwrap() / F::PI();
        TaylorDyn::unary_op(&self, |a, c| {
            c[0] = a[0].to_degrees();
            for k in 1..c.len() {
                c[k] = a[k] * factor;
            }
        })
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        TaylorDyn::unary_op(&self, |a, c| {
            c[0] = a[0].to_radians();
            for k in 1..c.len() {
                c[k] = a[k] * factor;
            }
        })
    }
}

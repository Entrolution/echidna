//! simba trait implementations for `Dual<F>` and `Reverse<F>`.
//!
//! Enables AD types inside nalgebra matrices and solvers.

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::{Float as NumFloat, FloatConst, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};

use crate::dual::Dual;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::TapeThreadLocal;

// ══════════════════════════════════════════════
//  SimdValue — trivial scalar lane (LANES=1)
// ══════════════════════════════════════════════

impl<F: Float> SimdValue for Dual<F> {
    const LANES: usize = 1;
    type Element = Self;
    type SimdBool = bool;

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }
    #[inline(always)]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<F: Float> PrimitiveSimdValue for Dual<F> {}

impl<F: Float + TapeThreadLocal> SimdValue for Reverse<F> {
    const LANES: usize = 1;
    type Element = Self;
    type SimdBool = bool;

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }
    #[inline(always)]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<F: Float + TapeThreadLocal> PrimitiveSimdValue for Reverse<F> {}

// ══════════════════════════════════════════════
//  Field (must be explicit — no blanket impl)
// ══════════════════════════════════════════════

impl<F: Float> Field for Dual<F> {}
impl<F: Float + TapeThreadLocal> Field for Reverse<F> {}

// ══════════════════════════════════════════════
//  SubsetOf conversions
// ══════════════════════════════════════════════

// Identity: Dual<F> ⊂ Dual<F>
impl<F: Float> SubsetOf<Dual<F>> for Dual<F> {
    #[inline]
    fn to_superset(&self) -> Dual<F> {
        *self
    }
    #[inline]
    fn from_superset_unchecked(element: &Dual<F>) -> Self {
        *element
    }
    #[inline]
    fn is_in_subset(_: &Dual<F>) -> bool {
        true
    }
}

// f64 ⊂ Dual<f64>  (lossless: f64 → constant dual)
impl SubsetOf<Dual<f64>> for f64 {
    #[inline]
    fn to_superset(&self) -> Dual<f64> {
        Dual::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &Dual<f64>) -> Self {
        element.re
    }
    #[inline]
    fn is_in_subset(element: &Dual<f64>) -> bool {
        element.eps == 0.0
    }
}

// f32 ⊂ Dual<f32>  (lossless: f32 → constant dual)
impl SubsetOf<Dual<f32>> for f32 {
    #[inline]
    fn to_superset(&self) -> Dual<f32> {
        Dual::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &Dual<f32>) -> Self {
        element.re
    }
    #[inline]
    fn is_in_subset(element: &Dual<f32>) -> bool {
        element.eps == 0.0
    }
}

// f64 ⊂ Dual<f32>  (lossy: f64 → f32 → constant dual)
// Required by ComplexField: SupersetOf<f64>
impl SubsetOf<Dual<f32>> for f64 {
    #[inline]
    fn to_superset(&self) -> Dual<f32> {
        Dual::constant(*self as f32)
    }
    #[inline]
    fn from_superset_unchecked(element: &Dual<f32>) -> Self {
        element.re as f64
    }
    #[inline]
    fn is_in_subset(element: &Dual<f32>) -> bool {
        element.eps == 0.0
    }
}

// f32 ⊂ Dual<f64>  (lossless: f32 → f64 → constant dual)
impl SubsetOf<Dual<f64>> for f32 {
    #[inline]
    fn to_superset(&self) -> Dual<f64> {
        Dual::constant(*self as f64)
    }
    #[inline]
    fn from_superset_unchecked(element: &Dual<f64>) -> Self {
        element.re as f32
    }
    #[inline]
    fn is_in_subset(element: &Dual<f64>) -> bool {
        element.eps == 0.0
    }
}

// Identity: Reverse<F> ⊂ Reverse<F>
impl<F: Float + TapeThreadLocal> SubsetOf<Reverse<F>> for Reverse<F> {
    #[inline]
    fn to_superset(&self) -> Reverse<F> {
        *self
    }
    #[inline]
    fn from_superset_unchecked(element: &Reverse<F>) -> Self {
        *element
    }
    #[inline]
    fn is_in_subset(_: &Reverse<F>) -> bool {
        true
    }
}

// f64 ⊂ Reverse<f64>
impl SubsetOf<Reverse<f64>> for f64 {
    #[inline]
    fn to_superset(&self) -> Reverse<f64> {
        Reverse::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &Reverse<f64>) -> Self {
        element.value
    }
    #[inline]
    fn is_in_subset(element: &Reverse<f64>) -> bool {
        element.index == crate::tape::CONSTANT
    }
}

// f32 ⊂ Reverse<f32>
impl SubsetOf<Reverse<f32>> for f32 {
    #[inline]
    fn to_superset(&self) -> Reverse<f32> {
        Reverse::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &Reverse<f32>) -> Self {
        element.value
    }
    #[inline]
    fn is_in_subset(element: &Reverse<f32>) -> bool {
        element.index == crate::tape::CONSTANT
    }
}

// f64 ⊂ Reverse<f32>  (lossy: f64 → f32 → constant)
impl SubsetOf<Reverse<f32>> for f64 {
    #[inline]
    fn to_superset(&self) -> Reverse<f32> {
        Reverse::constant(*self as f32)
    }
    #[inline]
    fn from_superset_unchecked(element: &Reverse<f32>) -> Self {
        element.value as f64
    }
    #[inline]
    fn is_in_subset(element: &Reverse<f32>) -> bool {
        element.index == crate::tape::CONSTANT
    }
}

// f32 ⊂ Reverse<f64>
impl SubsetOf<Reverse<f64>> for f32 {
    #[inline]
    fn to_superset(&self) -> Reverse<f64> {
        Reverse::constant(*self as f64)
    }
    #[inline]
    fn from_superset_unchecked(element: &Reverse<f64>) -> Self {
        element.value as f32
    }
    #[inline]
    fn is_in_subset(element: &Reverse<f64>) -> bool {
        element.index == crate::tape::CONSTANT
    }
}

// ══════════════════════════════════════════════
//  AbsDiffEq / RelativeEq / UlpsEq
//  (required by RealField)
// ══════════════════════════════════════════════

impl<F: Float> AbsDiffEq for Dual<F>
where
    F: AbsDiffEq<Epsilon = F>,
{
    type Epsilon = Self;

    #[inline]
    fn default_epsilon() -> Self {
        Dual::constant(F::default_epsilon())
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.re)
    }
}

impl<F: Float> RelativeEq for Dual<F>
where
    F: RelativeEq<Epsilon = F>,
{
    #[inline]
    fn default_max_relative() -> Self {
        Dual::constant(F::default_max_relative())
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self, max_relative: Self) -> bool {
        self.re.relative_eq(&other.re, epsilon.re, max_relative.re)
    }
}

impl<F: Float> UlpsEq for Dual<F>
where
    F: UlpsEq<Epsilon = F>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        F::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon.re, max_ulps)
    }
}

impl<F: Float + TapeThreadLocal> AbsDiffEq for Reverse<F>
where
    F: AbsDiffEq<Epsilon = F>,
{
    type Epsilon = Self;

    #[inline]
    fn default_epsilon() -> Self {
        Reverse::constant(F::default_epsilon())
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self) -> bool {
        self.value.abs_diff_eq(&other.value, epsilon.value)
    }
}

impl<F: Float + TapeThreadLocal> RelativeEq for Reverse<F>
where
    F: RelativeEq<Epsilon = F>,
{
    #[inline]
    fn default_max_relative() -> Self {
        Reverse::constant(F::default_max_relative())
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self, max_relative: Self) -> bool {
        self.value
            .relative_eq(&other.value, epsilon.value, max_relative.value)
    }
}

impl<F: Float + TapeThreadLocal> UlpsEq for Reverse<F>
where
    F: UlpsEq<Epsilon = F>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        F::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self, max_ulps: u32) -> bool {
        self.value.ulps_eq(&other.value, epsilon.value, max_ulps)
    }
}

// ══════════════════════════════════════════════
//  ComplexField for Dual<F>
// ══════════════════════════════════════════════

// We implement ComplexField concretely for f32 and f64 to satisfy all trait
// bounds (SubsetOf conversions require concrete types). Use a macro to avoid
// duplication.

macro_rules! impl_complex_field_dual {
    ($f:ty) => {
        impl ComplexField for Dual<$f> {
            type RealField = Self;

            #[inline]
            fn from_real(re: Self::RealField) -> Self {
                re
            }
            #[inline]
            fn real(self) -> Self::RealField {
                self
            }
            #[inline]
            fn imaginary(self) -> Self::RealField {
                Self::zero()
            }
            #[inline]
            fn modulus(self) -> Self::RealField {
                Dual::abs(self)
            }
            #[inline]
            fn modulus_squared(self) -> Self::RealField {
                self * self
            }
            #[inline]
            fn argument(self) -> Self::RealField {
                if self.re >= <$f>::zero() {
                    Self::zero()
                } else {
                    Self::pi()
                }
            }
            #[inline]
            fn norm1(self) -> Self::RealField {
                Dual::abs(self)
            }
            #[inline]
            fn scale(self, factor: Self::RealField) -> Self {
                self * factor
            }
            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self {
                self / factor
            }
            #[inline]
            fn floor(self) -> Self {
                Dual::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                Dual::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                Dual::round(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                Dual::trunc(self)
            }
            #[inline]
            fn fract(self) -> Self {
                Dual::fract(self)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                Dual::mul_add(self, a, b)
            }
            #[inline]
            fn abs(self) -> Self::RealField {
                Dual::abs(self)
            }
            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                Dual::hypot(self, other)
            }
            #[inline]
            fn recip(self) -> Self {
                Dual::recip(self)
            }
            #[inline]
            fn conjugate(self) -> Self {
                self // real type
            }
            #[inline]
            fn sin(self) -> Self {
                Dual::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                Dual::cos(self)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                Dual::sin_cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                Dual::tan(self)
            }
            #[inline]
            fn asin(self) -> Self {
                Dual::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                Dual::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                Dual::atan(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                Dual::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                Dual::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                Dual::tanh(self)
            }
            #[inline]
            fn asinh(self) -> Self {
                Dual::asinh(self)
            }
            #[inline]
            fn acosh(self) -> Self {
                Dual::acosh(self)
            }
            #[inline]
            fn atanh(self) -> Self {
                Dual::atanh(self)
            }
            #[inline]
            fn log(self, base: Self::RealField) -> Self {
                Dual::log(self, base)
            }
            #[inline]
            fn log2(self) -> Self {
                Dual::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                Dual::log10(self)
            }
            #[inline]
            fn ln(self) -> Self {
                Dual::ln(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                Dual::ln_1p(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                Dual::sqrt(self)
            }
            #[inline]
            fn exp(self) -> Self {
                Dual::exp(self)
            }
            #[inline]
            fn exp2(self) -> Self {
                Dual::exp2(self)
            }
            #[inline]
            fn exp_m1(self) -> Self {
                Dual::exp_m1(self)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                Dual::powi(self, n)
            }
            #[inline]
            fn powf(self, n: Self::RealField) -> Self {
                Dual::powf(self, n)
            }
            #[inline]
            fn powc(self, n: Self) -> Self {
                Dual::powf(self, n)
            }
            #[inline]
            fn cbrt(self) -> Self {
                Dual::cbrt(self)
            }
            #[inline]
            fn is_finite(&self) -> bool {
                self.re.is_finite()
            }
            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                if self.re >= <$f>::zero() {
                    Some(Dual::sqrt(self))
                } else {
                    None
                }
            }
        }
    };
}

impl_complex_field_dual!(f32);
impl_complex_field_dual!(f64);

// ══════════════════════════════════════════════
//  RealField for Dual<F>
// ══════════════════════════════════════════════

macro_rules! impl_real_field_dual {
    ($f:ty) => {
        impl RealField for Dual<$f> {
            #[inline]
            fn is_sign_positive(&self) -> bool {
                self.re.is_sign_positive()
            }
            #[inline]
            fn is_sign_negative(&self) -> bool {
                self.re.is_sign_negative()
            }
            #[inline]
            fn copysign(self, sign: Self) -> Self {
                Dual::abs(self) * Dual::signum(sign)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                Dual::max(self, other)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                Dual::min(self, other)
            }
            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                Dual::max(Dual::min(self, max), min)
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                Dual::atan2(self, other)
            }
            #[inline]
            fn min_value() -> Option<Self> {
                Some(Dual::constant(<$f>::MIN))
            }
            #[inline]
            fn max_value() -> Option<Self> {
                Some(Dual::constant(<$f>::MAX))
            }

            // ── Constants ──
            #[inline]
            fn pi() -> Self {
                Dual::constant(<$f>::PI())
            }
            #[inline]
            fn two_pi() -> Self {
                Dual::constant(<$f>::TAU())
            }
            #[inline]
            fn frac_pi_2() -> Self {
                Dual::constant(<$f>::FRAC_PI_2())
            }
            #[inline]
            fn frac_pi_3() -> Self {
                Dual::constant(<$f>::FRAC_PI_3())
            }
            #[inline]
            fn frac_pi_4() -> Self {
                Dual::constant(<$f>::FRAC_PI_4())
            }
            #[inline]
            fn frac_pi_6() -> Self {
                Dual::constant(<$f>::FRAC_PI_6())
            }
            #[inline]
            fn frac_pi_8() -> Self {
                Dual::constant(<$f>::FRAC_PI_8())
            }
            #[inline]
            fn frac_1_pi() -> Self {
                Dual::constant(<$f>::FRAC_1_PI())
            }
            #[inline]
            fn frac_2_pi() -> Self {
                Dual::constant(<$f>::FRAC_2_PI())
            }
            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                Dual::constant(<$f>::FRAC_2_SQRT_PI())
            }
            #[inline]
            fn e() -> Self {
                Dual::constant(<$f>::E())
            }
            #[inline]
            fn log2_e() -> Self {
                Dual::constant(<$f>::LOG2_E())
            }
            #[inline]
            fn log10_e() -> Self {
                Dual::constant(<$f>::LOG10_E())
            }
            #[inline]
            fn ln_2() -> Self {
                Dual::constant(<$f>::LN_2())
            }
            #[inline]
            fn ln_10() -> Self {
                Dual::constant(<$f>::LN_10())
            }
        }
    };
}

impl_real_field_dual!(f32);
impl_real_field_dual!(f64);

// ══════════════════════════════════════════════
//  ComplexField for Reverse<F>
// ══════════════════════════════════════════════

macro_rules! impl_complex_field_reverse {
    ($f:ty) => {
        impl ComplexField for Reverse<$f> {
            type RealField = Self;

            #[inline]
            fn from_real(re: Self::RealField) -> Self {
                re
            }
            #[inline]
            fn real(self) -> Self::RealField {
                self
            }
            #[inline]
            fn imaginary(self) -> Self::RealField {
                Self::zero()
            }
            #[inline]
            fn modulus(self) -> Self::RealField {
                NumFloat::abs(self)
            }
            #[inline]
            fn modulus_squared(self) -> Self::RealField {
                self * self
            }
            #[inline]
            fn argument(self) -> Self::RealField {
                if self.value >= <$f>::zero() {
                    Self::zero()
                } else {
                    Self::pi()
                }
            }
            #[inline]
            fn norm1(self) -> Self::RealField {
                NumFloat::abs(self)
            }
            #[inline]
            fn scale(self, factor: Self::RealField) -> Self {
                self * factor
            }
            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self {
                self / factor
            }
            #[inline]
            fn floor(self) -> Self {
                NumFloat::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                NumFloat::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                NumFloat::round(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                NumFloat::trunc(self)
            }
            #[inline]
            fn fract(self) -> Self {
                NumFloat::fract(self)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                NumFloat::mul_add(self, a, b)
            }
            #[inline]
            fn abs(self) -> Self::RealField {
                NumFloat::abs(self)
            }
            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                NumFloat::hypot(self, other)
            }
            #[inline]
            fn recip(self) -> Self {
                NumFloat::recip(self)
            }
            #[inline]
            fn conjugate(self) -> Self {
                self
            }
            #[inline]
            fn sin(self) -> Self {
                NumFloat::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                NumFloat::cos(self)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                NumFloat::sin_cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                NumFloat::tan(self)
            }
            #[inline]
            fn asin(self) -> Self {
                NumFloat::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                NumFloat::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                NumFloat::atan(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                NumFloat::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                NumFloat::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                NumFloat::tanh(self)
            }
            #[inline]
            fn asinh(self) -> Self {
                NumFloat::asinh(self)
            }
            #[inline]
            fn acosh(self) -> Self {
                NumFloat::acosh(self)
            }
            #[inline]
            fn atanh(self) -> Self {
                NumFloat::atanh(self)
            }
            #[inline]
            fn log(self, base: Self::RealField) -> Self {
                NumFloat::log(self, base)
            }
            #[inline]
            fn log2(self) -> Self {
                NumFloat::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                NumFloat::log10(self)
            }
            #[inline]
            fn ln(self) -> Self {
                NumFloat::ln(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                NumFloat::ln_1p(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                NumFloat::sqrt(self)
            }
            #[inline]
            fn exp(self) -> Self {
                NumFloat::exp(self)
            }
            #[inline]
            fn exp2(self) -> Self {
                NumFloat::exp2(self)
            }
            #[inline]
            fn exp_m1(self) -> Self {
                NumFloat::exp_m1(self)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                NumFloat::powi(self, n)
            }
            #[inline]
            fn powf(self, n: Self::RealField) -> Self {
                NumFloat::powf(self, n)
            }
            #[inline]
            fn powc(self, n: Self) -> Self {
                NumFloat::powf(self, n)
            }
            #[inline]
            fn cbrt(self) -> Self {
                NumFloat::cbrt(self)
            }
            #[inline]
            fn is_finite(&self) -> bool {
                NumFloat::is_finite(*self)
            }
            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                if self.value >= <$f>::zero() {
                    Some(NumFloat::sqrt(self))
                } else {
                    None
                }
            }
        }
    };
}

impl_complex_field_reverse!(f32);
impl_complex_field_reverse!(f64);

// ══════════════════════════════════════════════
//  RealField for Reverse<F>
// ══════════════════════════════════════════════

macro_rules! impl_real_field_reverse {
    ($f:ty) => {
        impl RealField for Reverse<$f> {
            #[inline]
            fn is_sign_positive(&self) -> bool {
                self.value.is_sign_positive()
            }
            #[inline]
            fn is_sign_negative(&self) -> bool {
                self.value.is_sign_negative()
            }
            #[inline]
            fn copysign(self, sign: Self) -> Self {
                NumFloat::abs(self) * NumFloat::signum(sign)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                NumFloat::max(self, other)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                NumFloat::min(self, other)
            }
            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                NumFloat::max(NumFloat::min(self, max), min)
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                NumFloat::atan2(self, other)
            }
            #[inline]
            fn min_value() -> Option<Self> {
                Some(Reverse::constant(<$f>::MIN))
            }
            #[inline]
            fn max_value() -> Option<Self> {
                Some(Reverse::constant(<$f>::MAX))
            }

            #[inline]
            fn pi() -> Self {
                Reverse::constant(<$f>::PI())
            }
            #[inline]
            fn two_pi() -> Self {
                Reverse::constant(<$f>::TAU())
            }
            #[inline]
            fn frac_pi_2() -> Self {
                Reverse::constant(<$f>::FRAC_PI_2())
            }
            #[inline]
            fn frac_pi_3() -> Self {
                Reverse::constant(<$f>::FRAC_PI_3())
            }
            #[inline]
            fn frac_pi_4() -> Self {
                Reverse::constant(<$f>::FRAC_PI_4())
            }
            #[inline]
            fn frac_pi_6() -> Self {
                Reverse::constant(<$f>::FRAC_PI_6())
            }
            #[inline]
            fn frac_pi_8() -> Self {
                Reverse::constant(<$f>::FRAC_PI_8())
            }
            #[inline]
            fn frac_1_pi() -> Self {
                Reverse::constant(<$f>::FRAC_1_PI())
            }
            #[inline]
            fn frac_2_pi() -> Self {
                Reverse::constant(<$f>::FRAC_2_PI())
            }
            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                Reverse::constant(<$f>::FRAC_2_SQRT_PI())
            }
            #[inline]
            fn e() -> Self {
                Reverse::constant(<$f>::E())
            }
            #[inline]
            fn log2_e() -> Self {
                Reverse::constant(<$f>::LOG2_E())
            }
            #[inline]
            fn log10_e() -> Self {
                Reverse::constant(<$f>::LOG10_E())
            }
            #[inline]
            fn ln_2() -> Self {
                Reverse::constant(<$f>::LN_2())
            }
            #[inline]
            fn ln_10() -> Self {
                Reverse::constant(<$f>::LN_10())
            }
        }
    };
}

impl_real_field_reverse!(f32);
impl_real_field_reverse!(f64);

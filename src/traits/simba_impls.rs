//! simba trait implementations for `Dual<F>`, `DualVec<F, N>` and `Reverse<F>`.
//!
//! Enables AD types inside nalgebra matrices and solvers.
//!
//! # Active-tape requirement for `Reverse<F>`
//!
//! `Reverse` arithmetic records to a thread-local tape, so any nalgebra or
//! simba operation on `Reverse` values (matrix products, decompositions,
//! solver steps) must run inside an active recording — e.g. within the
//! closure passed to [`grad`](crate::grad) or another recording API.
//! Outside a recording, the first arithmetic operation panics with
//! "No active tape. Use echidna::grad() or similar API." `Dual` and
//! `DualVec` are tape-free and have no such requirement.

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::{Float as NumFloat, FloatConst, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::TapeThreadLocal;

// ══════════════════════════════════════════════
//  SimdValue — trivial scalar lane (LANES=1)
// ══════════════════════════════════════════════

// One scalar SimdValue story for all AD value types: each is a single
// SIMD lane (LANES = 1); the marker PrimitiveSimdValue rides along.
macro_rules! impl_simd_value_scalar {
    ([$($gen:tt)*] $ty:ty) => {
        impl<$($gen)*> SimdValue for $ty {
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
    // SAFETY: This is a single-lane (LANES=1) scalar type, so the lane index
    // is always 0 and the operation is trivially safe regardless of input.
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline(always)]
    // SAFETY: This is a single-lane (LANES=1) scalar type, so the lane index
    // is always 0 and the operation is trivially safe regardless of input.
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

        impl<$($gen)*> PrimitiveSimdValue for $ty {}
    };
}

impl_simd_value_scalar!([F: Float] Dual<F>);
impl_simd_value_scalar!([F: Float, const N: usize] DualVec<F, N>);
impl_simd_value_scalar!([F: Float + TapeThreadLocal] Reverse<F>);

// ══════════════════════════════════════════════
//  Field (must be explicit — no blanket impl)
// ══════════════════════════════════════════════

impl<F: Float> Field for Dual<F> {}
impl<F: Float, const N: usize> Field for DualVec<F, N> {}
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
        f64::from(element.re)
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
        Dual::constant(f64::from(*self))
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

// Identity: DualVec<F, N> ⊂ DualVec<F, N>
impl<F: Float, const N: usize> SubsetOf<DualVec<F, N>> for DualVec<F, N> {
    #[inline]
    fn to_superset(&self) -> DualVec<F, N> {
        *self
    }
    #[inline]
    fn from_superset_unchecked(element: &DualVec<F, N>) -> Self {
        *element
    }
    #[inline]
    fn is_in_subset(_: &DualVec<F, N>) -> bool {
        true
    }
}

// f64 ⊂ DualVec<f64, N>  (lossless: f64 → constant dual vector)
impl<const N: usize> SubsetOf<DualVec<f64, N>> for f64 {
    #[inline]
    fn to_superset(&self) -> DualVec<f64, N> {
        DualVec::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &DualVec<f64, N>) -> Self {
        element.re
    }
    #[inline]
    fn is_in_subset(element: &DualVec<f64, N>) -> bool {
        element.eps.into_iter().all(|e| e == 0.0)
    }
}

// f32 ⊂ DualVec<f32, N>  (lossless: f32 → constant dual vector)
impl<const N: usize> SubsetOf<DualVec<f32, N>> for f32 {
    #[inline]
    fn to_superset(&self) -> DualVec<f32, N> {
        DualVec::constant(*self)
    }
    #[inline]
    fn from_superset_unchecked(element: &DualVec<f32, N>) -> Self {
        element.re
    }
    #[inline]
    fn is_in_subset(element: &DualVec<f32, N>) -> bool {
        element.eps.into_iter().all(|e| e == 0.0)
    }
}

// f64 ⊂ DualVec<f32, N>  (lossy: f64 → f32 → constant dual vector)
// Required by ComplexField: SupersetOf<f64>
impl<const N: usize> SubsetOf<DualVec<f32, N>> for f64 {
    #[inline]
    fn to_superset(&self) -> DualVec<f32, N> {
        DualVec::constant(*self as f32)
    }
    #[inline]
    fn from_superset_unchecked(element: &DualVec<f32, N>) -> Self {
        f64::from(element.re)
    }
    #[inline]
    fn is_in_subset(element: &DualVec<f32, N>) -> bool {
        element.eps.into_iter().all(|e| e == 0.0)
    }
}

// f32 ⊂ DualVec<f64, N>  (lossless: f32 → f64 → constant dual vector)
impl<const N: usize> SubsetOf<DualVec<f64, N>> for f32 {
    #[inline]
    fn to_superset(&self) -> DualVec<f64, N> {
        DualVec::constant(f64::from(*self))
    }
    #[inline]
    fn from_superset_unchecked(element: &DualVec<f64, N>) -> Self {
        element.re as f32
    }
    #[inline]
    fn is_in_subset(element: &DualVec<f64, N>) -> bool {
        element.eps.into_iter().all(|e| e == 0.0)
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
        f64::from(element.value)
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
        Reverse::constant(f64::from(*self))
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

// approx-trait triple, shared across the AD value types; `$re` names each
// type's primal field (comparisons are primal-only by convention).
macro_rules! impl_approx_eq {
    ([$($gen:tt)*] $ty:ty, $re:ident) => {
        impl<$($gen)*> AbsDiffEq for $ty
        where
            F: AbsDiffEq<Epsilon = F>,
        {
    type Epsilon = Self;

    #[inline]
    fn default_epsilon() -> Self {
        <$ty>::constant(F::default_epsilon())
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self) -> bool {
        self.$re.abs_diff_eq(&other.$re, epsilon.$re)
    }
}

        impl<$($gen)*> RelativeEq for $ty
        where
            F: RelativeEq<Epsilon = F>,
        {
    #[inline]
    fn default_max_relative() -> Self {
        <$ty>::constant(F::default_max_relative())
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self, max_relative: Self) -> bool {
        self.$re.relative_eq(&other.$re, epsilon.$re, max_relative.$re)
    }
}

        impl<$($gen)*> UlpsEq for $ty
        where
            F: UlpsEq<Epsilon = F>,
        {
    #[inline]
    fn default_max_ulps() -> u32 {
        F::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self, max_ulps: u32) -> bool {
        self.$re.ulps_eq(&other.$re, epsilon.$re, max_ulps)
    }
}
    };
}

impl_approx_eq!([F: Float] Dual<F>, re);
impl_approx_eq!([F: Float, const N: usize] DualVec<F, N>, re);
impl_approx_eq!([F: Float + TapeThreadLocal] Reverse<F>, value);

// ══════════════════════════════════════════════
//  ComplexField for Dual<F>
// ══════════════════════════════════════════════

// We implement ComplexField concretely for f32 and f64 to satisfy all trait
// bounds (SubsetOf conversions require concrete types). Use a macro to avoid
// duplication.

macro_rules! impl_complex_field_fwd {
    // Dual: no extra generics. DualVec: const-generic lane count.
    ($f:ty, Dual) => { impl_complex_field_fwd!(@impl [] Dual<$f>, $f); };
    ($f:ty, DualVec) => { impl_complex_field_fwd!(@impl [const N: usize] DualVec<$f, N>, $f); };
    (@impl [$($gen:tt)*] $ty:ty, $f:ty) => {
        impl<$($gen)*> ComplexField for $ty {
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
                <$ty>::abs(self)
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
                <$ty>::abs(self)
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
                <$ty>::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                <$ty>::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                <$ty>::round(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                <$ty>::trunc(self)
            }
            #[inline]
            fn fract(self) -> Self {
                <$ty>::fract(self)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$ty>::mul_add(self, a, b)
            }
            #[inline]
            fn abs(self) -> Self::RealField {
                <$ty>::abs(self)
            }
            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                <$ty>::hypot(self, other)
            }
            #[inline]
            fn recip(self) -> Self {
                <$ty>::recip(self)
            }
            #[inline]
            fn conjugate(self) -> Self {
                self // real type
            }
            #[inline]
            fn sin(self) -> Self {
                <$ty>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$ty>::cos(self)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                <$ty>::sin_cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                <$ty>::tan(self)
            }
            #[inline]
            fn asin(self) -> Self {
                <$ty>::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                <$ty>::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                <$ty>::atan(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                <$ty>::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                <$ty>::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$ty>::tanh(self)
            }
            #[inline]
            fn asinh(self) -> Self {
                <$ty>::asinh(self)
            }
            #[inline]
            fn acosh(self) -> Self {
                <$ty>::acosh(self)
            }
            #[inline]
            fn atanh(self) -> Self {
                <$ty>::atanh(self)
            }
            #[inline]
            fn log(self, base: Self::RealField) -> Self {
                <$ty>::log(self, base)
            }
            #[inline]
            fn log2(self) -> Self {
                <$ty>::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                <$ty>::log10(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$ty>::ln(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                <$ty>::ln_1p(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            #[inline]
            fn exp2(self) -> Self {
                <$ty>::exp2(self)
            }
            #[inline]
            fn exp_m1(self) -> Self {
                <$ty>::exp_m1(self)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                <$ty>::powi(self, n)
            }
            #[inline]
            fn powf(self, n: Self::RealField) -> Self {
                <$ty>::powf(self, n)
            }
            #[inline]
            fn powc(self, n: Self) -> Self {
                <$ty>::powf(self, n)
            }
            #[inline]
            fn cbrt(self) -> Self {
                <$ty>::cbrt(self)
            }
            #[inline]
            fn is_finite(&self) -> bool {
                self.re.is_finite()
            }
            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                if self.re >= <$f>::zero() {
                    Some(<$ty>::sqrt(self))
                } else {
                    None
                }
            }
        }
        };
}

impl_complex_field_fwd!(f32, Dual);
impl_complex_field_fwd!(f64, Dual);

// ══════════════════════════════════════════════
//  RealField for Dual<F>
// ══════════════════════════════════════════════

macro_rules! impl_real_field_fwd {
    // Dual: no extra generics. DualVec: const-generic lane count.
    ($f:ty, Dual) => { impl_real_field_fwd!(@impl [] Dual<$f>, $f); };
    ($f:ty, DualVec) => { impl_real_field_fwd!(@impl [const N: usize] DualVec<$f, N>, $f); };
    (@impl [$($gen:tt)*] $ty:ty, $f:ty) => {
        impl<$($gen)*> RealField for $ty {
            #[inline]
            fn is_sign_positive(&self) -> bool {
                self.re.is_sign_positive()
            }
            #[inline]
            fn is_sign_negative(&self) -> bool {
                self.re.is_sign_negative()
            }
            #[inline]
            // copysign = magnitude of self, sign of `sign`, composed as
            // abs(self)·signum(sign) so the derivative follows the same
            // composition (not a bit-level sign copy). For a NaN `sign` the
            // result is NaN (signum(NaN) = NaN), unlike f64::copysign which
            // reads NaN's sign bit.
            fn copysign(self, sign: Self) -> Self {
                <$ty>::abs(self) * <$ty>::signum(sign)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                <$ty>::max(self, other)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                <$ty>::min(self, other)
            }
            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                <$ty>::max(<$ty>::min(self, max), min)
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                <$ty>::atan2(self, other)
            }
            #[inline]
            fn min_value() -> Option<Self> {
                Some(<$ty>::constant(<$f>::MIN))
            }
            #[inline]
            fn max_value() -> Option<Self> {
                Some(<$ty>::constant(<$f>::MAX))
            }

            // ── Constants ──
            #[inline]
            fn pi() -> Self {
                <$ty>::constant(<$f>::PI())
            }
            #[inline]
            fn two_pi() -> Self {
                <$ty>::constant(<$f>::TAU())
            }
            #[inline]
            fn frac_pi_2() -> Self {
                <$ty>::constant(<$f>::FRAC_PI_2())
            }
            #[inline]
            fn frac_pi_3() -> Self {
                <$ty>::constant(<$f>::FRAC_PI_3())
            }
            #[inline]
            fn frac_pi_4() -> Self {
                <$ty>::constant(<$f>::FRAC_PI_4())
            }
            #[inline]
            fn frac_pi_6() -> Self {
                <$ty>::constant(<$f>::FRAC_PI_6())
            }
            #[inline]
            fn frac_pi_8() -> Self {
                <$ty>::constant(<$f>::FRAC_PI_8())
            }
            #[inline]
            fn frac_1_pi() -> Self {
                <$ty>::constant(<$f>::FRAC_1_PI())
            }
            #[inline]
            fn frac_2_pi() -> Self {
                <$ty>::constant(<$f>::FRAC_2_PI())
            }
            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                <$ty>::constant(<$f>::FRAC_2_SQRT_PI())
            }
            #[inline]
            fn e() -> Self {
                <$ty>::constant(<$f>::E())
            }
            #[inline]
            fn log2_e() -> Self {
                <$ty>::constant(<$f>::LOG2_E())
            }
            #[inline]
            fn log10_e() -> Self {
                <$ty>::constant(<$f>::LOG10_E())
            }
            #[inline]
            fn ln_2() -> Self {
                <$ty>::constant(<$f>::LN_2())
            }
            #[inline]
            fn ln_10() -> Self {
                <$ty>::constant(<$f>::LN_10())
            }
        }
        };
}

impl_real_field_fwd!(f32, Dual);
impl_real_field_fwd!(f64, Dual);

// ══════════════════════════════════════════════
//  ComplexField for DualVec<F, N>
// ══════════════════════════════════════════════

// We implement ComplexField concretely for f32 and f64 to satisfy all trait
// bounds (SubsetOf conversions require concrete types). Use a macro to avoid
// duplication.

impl_complex_field_fwd!(f32, DualVec);
impl_complex_field_fwd!(f64, DualVec);

// ══════════════════════════════════════════════
//  RealField for DualVec<F, N>
// ══════════════════════════════════════════════

impl_real_field_fwd!(f32, DualVec);
impl_real_field_fwd!(f64, DualVec);

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
            // copysign = magnitude of self, sign of `sign`, composed as
            // abs(self)·signum(sign) so the derivative follows the same
            // composition (not a bit-level sign copy). For a NaN `sign` the
            // result is NaN (signum(NaN) = NaN), unlike f64::copysign which
            // reads NaN's sign bit.
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

//! `num_traits` implementations for [`BReverse<F>`].
//!
//! Mirrors `num_traits_impls.rs` for `Reverse<F>`, but each transcendental
//! pushes an opcode instead of a precomputed multiplier.

use std::num::FpCategory;

use num_traits::{
    Float as NumFloat, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::breverse::BReverse;
use crate::bytecode_tape::{self, BtapeThreadLocal, BytecodeTape, CONSTANT};
use crate::float::Float;
use crate::opcode::{OpCode, UNUSED};

// ── Helpers ──

/// Ensure a BReverse operand has a valid tape index.
#[inline]
fn ensure_on_tape<F: Float>(x: &BReverse<F>, tape: &mut BytecodeTape<F>) -> u32 {
    if x.index == CONSTANT {
        tape.push_const(x.value)
    } else {
        x.index
    }
}

/// Record a unary opcode, promoting constant if needed.
#[inline]
fn brev_unary<F: Float + BtapeThreadLocal>(x: BReverse<F>, op: OpCode, f_val: F) -> BReverse<F> {
    let index = bytecode_tape::with_active_btape(|t| {
        let xi = ensure_on_tape(&x, t);
        t.push_op(op, xi, UNUSED, f_val)
    });
    BReverse {
        value: f_val,
        index,
    }
}

/// Record a binary opcode, promoting constants if needed.
#[inline]
fn brev_binary<F: Float + BtapeThreadLocal>(
    x: BReverse<F>,
    y: BReverse<F>,
    op: OpCode,
    f_val: F,
) -> BReverse<F> {
    let index = bytecode_tape::with_active_btape(|t| {
        let xi = ensure_on_tape(&x, t);
        let yi = ensure_on_tape(&y, t);
        t.push_op(op, xi, yi, f_val)
    });
    BReverse {
        value: f_val,
        index,
    }
}

// ══════════════════════════════════════════════
//  Basic numeric traits
// ══════════════════════════════════════════════

impl<F: Float + BtapeThreadLocal> Zero for BReverse<F> {
    #[inline]
    fn zero() -> Self {
        BReverse::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<F: Float + BtapeThreadLocal> One for BReverse<F> {
    #[inline]
    fn one() -> Self {
        BReverse::constant(F::one())
    }
}

impl<F: Float + BtapeThreadLocal> Num for BReverse<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(BReverse::constant)
    }
}

impl<F: Float> FromPrimitive for BReverse<F> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(BReverse::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(BReverse::constant)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(BReverse::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(BReverse::constant)
    }
}

impl<F: Float> ToPrimitive for BReverse<F> {
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

impl<F: Float + BtapeThreadLocal> NumCast for BReverse<F> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(BReverse::constant)
    }
}

// ══════════════════════════════════════════════
//  Signed
// ══════════════════════════════════════════════

impl<F: Float + BtapeThreadLocal> Signed for BReverse<F> {
    #[inline]
    fn abs(&self) -> Self {
        brev_unary(*self, OpCode::Abs, self.value.abs())
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
        BReverse::constant(self.value.signum())
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

// ══════════════════════════════════════════════
//  FloatConst
// ══════════════════════════════════════════════

impl<F: Float + BtapeThreadLocal> FloatConst for BReverse<F> {
    fn E() -> Self {
        BReverse::constant(F::E())
    }
    fn FRAC_1_PI() -> Self {
        BReverse::constant(F::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        BReverse::constant(F::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        BReverse::constant(F::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        BReverse::constant(F::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        BReverse::constant(F::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        BReverse::constant(F::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        BReverse::constant(F::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        BReverse::constant(F::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        BReverse::constant(F::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        BReverse::constant(F::LN_10())
    }
    fn LN_2() -> Self {
        BReverse::constant(F::LN_2())
    }
    fn LOG10_E() -> Self {
        BReverse::constant(F::LOG10_E())
    }
    fn LOG2_E() -> Self {
        BReverse::constant(F::LOG2_E())
    }
    fn PI() -> Self {
        BReverse::constant(F::PI())
    }
    fn SQRT_2() -> Self {
        BReverse::constant(F::SQRT_2())
    }
    fn TAU() -> Self {
        BReverse::constant(F::TAU())
    }
    fn LOG10_2() -> Self {
        BReverse::constant(F::LOG10_2())
    }
    fn LOG2_10() -> Self {
        BReverse::constant(F::LOG2_10())
    }
}

// ══════════════════════════════════════════════
//  Float (num_traits::Float)
// ══════════════════════════════════════════════

impl<F: Float + BtapeThreadLocal> NumFloat for BReverse<F> {
    fn nan() -> Self {
        BReverse::constant(F::nan())
    }
    fn infinity() -> Self {
        BReverse::constant(F::infinity())
    }
    fn neg_infinity() -> Self {
        BReverse::constant(F::neg_infinity())
    }
    fn neg_zero() -> Self {
        BReverse::constant(F::neg_zero())
    }

    fn min_value() -> Self {
        BReverse::constant(F::min_value())
    }
    fn min_positive_value() -> Self {
        BReverse::constant(F::min_positive_value())
    }
    fn max_value() -> Self {
        BReverse::constant(F::max_value())
    }
    fn epsilon() -> Self {
        BReverse::constant(F::epsilon())
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

    // ── Rounding (zero derivative, but needed for re-evaluation) ──

    fn floor(self) -> Self {
        brev_unary(self, OpCode::Floor, self.value.floor())
    }
    fn ceil(self) -> Self {
        brev_unary(self, OpCode::Ceil, self.value.ceil())
    }
    fn round(self) -> Self {
        brev_unary(self, OpCode::Round, self.value.round())
    }
    fn trunc(self) -> Self {
        brev_unary(self, OpCode::Trunc, self.value.trunc())
    }
    fn fract(self) -> Self {
        brev_unary(self, OpCode::Fract, self.value.fract())
    }
    fn abs(self) -> Self {
        brev_unary(self, OpCode::Abs, self.value.abs())
    }
    fn signum(self) -> Self {
        brev_unary(self, OpCode::Signum, self.value.signum())
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // Decompose to Mul + Add (MulAdd is ternary, incompatible with binary encoding).
        self * a + b
    }

    fn recip(self) -> Self {
        brev_unary(self, OpCode::Recip, self.value.recip())
    }

    fn powi(self, n: i32) -> Self {
        let val = self.value.powi(n);
        let index = bytecode_tape::with_active_btape(|t| {
            let xi = ensure_on_tape(&self, t);
            t.push_powi(xi, n, val)
        });
        BReverse { value: val, index }
    }

    fn powf(self, n: Self) -> Self {
        brev_binary(self, n, OpCode::Powf, self.value.powf(n.value))
    }

    fn sqrt(self) -> Self {
        brev_unary(self, OpCode::Sqrt, self.value.sqrt())
    }
    fn cbrt(self) -> Self {
        brev_unary(self, OpCode::Cbrt, self.value.cbrt())
    }

    fn exp(self) -> Self {
        brev_unary(self, OpCode::Exp, self.value.exp())
    }
    fn exp2(self) -> Self {
        brev_unary(self, OpCode::Exp2, self.value.exp2())
    }
    fn exp_m1(self) -> Self {
        brev_unary(self, OpCode::ExpM1, self.value.exp_m1())
    }
    fn ln(self) -> Self {
        brev_unary(self, OpCode::Ln, self.value.ln())
    }
    fn log2(self) -> Self {
        brev_unary(self, OpCode::Log2, self.value.log2())
    }
    fn log10(self) -> Self {
        brev_unary(self, OpCode::Log10, self.value.log10())
    }
    fn ln_1p(self) -> Self {
        brev_unary(self, OpCode::Ln1p, self.value.ln_1p())
    }
    fn log(self, base: Self) -> Self {
        // Decompose: log_b(x) = ln(x) / ln(b)
        self.ln() / base.ln()
    }

    fn sin(self) -> Self {
        brev_unary(self, OpCode::Sin, self.value.sin())
    }
    fn cos(self) -> Self {
        brev_unary(self, OpCode::Cos, self.value.cos())
    }
    fn tan(self) -> Self {
        brev_unary(self, OpCode::Tan, self.value.tan())
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.value.sin_cos();
        (
            brev_unary(self, OpCode::Sin, s),
            brev_unary(self, OpCode::Cos, c),
        )
    }
    fn asin(self) -> Self {
        brev_unary(self, OpCode::Asin, self.value.asin())
    }
    fn acos(self) -> Self {
        brev_unary(self, OpCode::Acos, self.value.acos())
    }
    fn atan(self) -> Self {
        brev_unary(self, OpCode::Atan, self.value.atan())
    }
    fn atan2(self, other: Self) -> Self {
        brev_binary(self, other, OpCode::Atan2, self.value.atan2(other.value))
    }

    fn sinh(self) -> Self {
        brev_unary(self, OpCode::Sinh, self.value.sinh())
    }
    fn cosh(self) -> Self {
        brev_unary(self, OpCode::Cosh, self.value.cosh())
    }
    fn tanh(self) -> Self {
        brev_unary(self, OpCode::Tanh, self.value.tanh())
    }
    fn asinh(self) -> Self {
        brev_unary(self, OpCode::Asinh, self.value.asinh())
    }
    fn acosh(self) -> Self {
        brev_unary(self, OpCode::Acosh, self.value.acosh())
    }
    fn atanh(self) -> Self {
        brev_unary(self, OpCode::Atanh, self.value.atanh())
    }

    fn hypot(self, other: Self) -> Self {
        brev_binary(self, other, OpCode::Hypot, self.value.hypot(other.value))
    }

    fn max(self, other: Self) -> Self {
        brev_binary(
            self,
            other,
            OpCode::Max,
            if self.value >= other.value {
                self.value
            } else {
                other.value
            },
        )
    }

    fn min(self, other: Self) -> Self {
        brev_binary(
            self,
            other,
            OpCode::Min,
            if self.value <= other.value {
                self.value
            } else {
                other.value
            },
        )
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
        let val = self.value.to_degrees();
        let index = bytecode_tape::with_active_btape(|t| {
            let xi = ensure_on_tape(&self, t);
            let fi = t.push_const(factor);
            t.push_op(OpCode::Mul, xi, fi, val)
        });
        BReverse { value: val, index }
    }

    fn to_radians(self) -> Self {
        let factor = F::PI() / F::from(180.0).unwrap();
        let val = self.value.to_radians();
        let index = bytecode_tape::with_active_btape(|t| {
            let xi = ensure_on_tape(&self, t);
            let fi = t.push_const(factor);
            t.push_op(OpCode::Mul, xi, fi, val)
        });
        BReverse { value: val, index }
    }
}

use std::fmt::{Debug, Display};

use num_traits::FromPrimitive;

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::TapeThreadLocal;

/// The central trait for AD-generic numeric code.
///
/// Implement functions as `fn foo<T: Scalar>(x: T) -> T` and they work
/// with plain `f64`, `Dual<f64>`, and `Reverse<f64>`.
pub trait Scalar:
    num_traits::Float
    + num_traits::FloatConst
    + FromPrimitive
    + Copy
    + Default
    + Debug
    + Display
    + Send
    + 'static
{
    /// The underlying primitive float type.
    type Float: Float;

    /// Lift a plain float to this scalar (constant — zero derivative).
    fn from_f(val: Self::Float) -> Self;

    /// Extract the primal value.
    fn value(&self) -> Self::Float;
}

impl Scalar for f32 {
    type Float = f32;

    #[inline]
    fn from_f(val: f32) -> Self {
        val
    }

    #[inline]
    fn value(&self) -> f32 {
        *self
    }
}

impl Scalar for f64 {
    type Float = f64;

    #[inline]
    fn from_f(val: f64) -> Self {
        val
    }

    #[inline]
    fn value(&self) -> f64 {
        *self
    }
}

impl<F: Float> Scalar for Dual<F> {
    type Float = F;

    #[inline]
    fn from_f(val: F) -> Self {
        Dual::constant(val)
    }

    #[inline]
    fn value(&self) -> F {
        self.re
    }
}

impl<F: Float, const N: usize> Scalar for DualVec<F, N> {
    type Float = F;

    #[inline]
    fn from_f(val: F) -> Self {
        DualVec::constant(val)
    }

    #[inline]
    fn value(&self) -> F {
        self.re
    }
}

impl<F: Float + TapeThreadLocal> Scalar for Reverse<F> {
    type Float = F;

    #[inline]
    fn from_f(val: F) -> Self {
        Reverse::constant(val)
    }

    #[inline]
    fn value(&self) -> F {
        self.value
    }
}

#[cfg(feature = "bytecode")]
impl<F: Float + crate::bytecode_tape::BtapeThreadLocal> Scalar for crate::breverse::BReverse<F> {
    type Float = F;

    #[inline]
    fn from_f(val: F) -> Self {
        crate::breverse::BReverse::constant(val)
    }

    #[inline]
    fn value(&self) -> F {
        self.value
    }
}

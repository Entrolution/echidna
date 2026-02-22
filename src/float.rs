use std::fmt::{Debug, Display};

use num_traits::{Float as NumFloat, FloatConst, FromPrimitive};

use crate::dual::Dual;
use crate::dual_vec::DualVec;

/// Marker trait for floating-point types that can serve as the base of AD computations.
///
/// Bundles the numeric and utility traits needed throughout echidna.
/// Implemented by primitive types (`f32`, `f64`) and by `Dual<F>`, which enables
/// nested forward-mode: `Dual<Dual<f64>>` for second-order derivatives.
pub trait Float:
    NumFloat + FloatConst + FromPrimitive + Copy + Send + Sync + Default + Debug + Display + 'static
{
}

impl Float for f32 {}
impl Float for f64 {}
impl<F: Float> Float for Dual<F> {}
impl<F: Float, const N: usize> Float for DualVec<F, N> {}

/// Checks whether all components (primal + tangent) are zero.
///
/// Used by `reverse_tangent` to safely skip zero adjoints without
/// incorrectly dropping tangent (eps) contributions. `PartialEq` only
/// compares `.re`, so a value with `re==0` but `eps!=0` would be
/// incorrectly pruned without this trait.
#[cfg_attr(not(feature = "bytecode"), allow(dead_code))]
pub(crate) trait IsAllZero {
    fn is_all_zero(&self) -> bool;
}

impl IsAllZero for f32 {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        *self == 0.0
    }
}

impl IsAllZero for f64 {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        *self == 0.0
    }
}

impl<F: Float> IsAllZero for Dual<F> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.re == F::zero() && self.eps == F::zero()
    }
}

impl<F: Float, const N: usize> IsAllZero for DualVec<F, N> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.re == F::zero() && self.eps.iter().all(|&e| e == F::zero())
    }
}

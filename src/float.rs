use std::fmt::{Debug, Display};

use num_traits::{Float as NumFloat, FloatConst, FromPrimitive};

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::reverse::Reverse;
use crate::tape::TapeThreadLocal;

#[cfg(feature = "bytecode")]
use crate::breverse::BReverse;
#[cfg(feature = "bytecode")]
use crate::bytecode_tape::BtapeThreadLocal;

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

#[cfg(feature = "taylor")]
impl<F: Float, const K: usize> Float for crate::taylor::Taylor<F, K> {}
#[cfg(feature = "taylor")]
impl<F: Float + crate::taylor_dyn::TaylorArenaLocal> Float for crate::taylor_dyn::TaylorDyn<F> {}

impl<F: Float + TapeThreadLocal> Float for Reverse<F> {}

#[cfg(feature = "bytecode")]
impl<F: Float + BtapeThreadLocal> Float for BReverse<F> {}

#[cfg(feature = "laurent")]
impl<F: Float, const K: usize> Float for crate::laurent::Laurent<F, K> {}

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

#[cfg(feature = "taylor")]
impl<F: Float, const K: usize> IsAllZero for crate::taylor::Taylor<F, K> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == F::zero())
    }
}

#[cfg(feature = "taylor")]
impl<F: Float + crate::taylor_dyn::TaylorArenaLocal> IsAllZero for crate::taylor_dyn::TaylorDyn<F> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        if self.value != F::zero() {
            return false;
        }
        if self.index == crate::taylor_dyn::CONSTANT {
            return true;
        }
        crate::taylor_dyn::with_active_arena(|arena: &mut crate::taylor_dyn::TaylorArena<F>| {
            arena.coeffs(self.index).iter().all(|&c| c == F::zero())
        })
    }
}

impl<F: Float + TapeThreadLocal> IsAllZero for Reverse<F> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.value == F::zero()
    }
}

#[cfg(feature = "bytecode")]
impl<F: Float + BtapeThreadLocal> IsAllZero for BReverse<F> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.value == F::zero()
    }
}

#[cfg(feature = "laurent")]
impl<F: Float, const K: usize> IsAllZero for crate::laurent::Laurent<F, K> {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        self.is_all_zero_pub()
    }
}

use std::fmt::{Debug, Display};

use num_traits::{Float as NumFloat, FloatConst, FromPrimitive};

use crate::dual::Dual;

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

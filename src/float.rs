use std::fmt::{Debug, Display};

use num_traits::{Float as NumFloat, FloatConst, FromPrimitive};

/// Marker trait for base floating-point types (`f32`, `f64`).
///
/// Bundles the numeric and utility traits needed throughout echidna.
/// Only primitive float types implement this — AD wrapper types do not.
pub trait Float:
    NumFloat + FloatConst + FromPrimitive + Copy + Send + Sync + Default + Debug + Display + 'static
{
}

impl Float for f32 {}
impl Float for f64 {}

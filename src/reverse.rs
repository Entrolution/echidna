use std::fmt::{self, Display};

use crate::Float;
use crate::tape::CONSTANT;

/// Reverse-mode AD variable.
///
/// Just a value and a tape index — 12 bytes for `f64`. `Copy` because the
/// tape lives in a thread-local, not inside this struct.
#[derive(Clone, Copy, Debug)]
pub struct Reverse<F: Float> {
    pub(crate) value: F,
    pub(crate) index: u32,
}

impl<F: Float> Reverse<F> {
    /// Create a constant (not tracked on tape).
    #[inline]
    pub fn constant(value: F) -> Self {
        Reverse {
            value,
            index: CONSTANT,
        }
    }

    /// Create a reverse variable from a tape allocation.
    /// Typically only used internally by the API layer and tests.
    #[inline]
    pub fn from_tape(value: F, index: u32) -> Self {
        Reverse { value, index }
    }

    /// Get the tape index (for advanced usage / testing).
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl<F: Float> Display for Reverse<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<F: Float> Default for Reverse<F> {
    fn default() -> Self {
        Reverse::constant(F::zero())
    }
}

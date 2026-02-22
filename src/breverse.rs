//! Bytecode-tape reverse-mode AD variable.
//!
//! [`BReverse<F>`] is analogous to [`Reverse<F>`](crate::Reverse) but records
//! opcodes to a [`BytecodeTape`](crate::bytecode_tape::BytecodeTape) instead
//! of precomputed multipliers to an Adept-style tape. This allows the tape to
//! be re-evaluated at different inputs without re-recording.

use std::fmt::{self, Display};

use crate::bytecode_tape::{BtapeThreadLocal, CustomOpHandle, CONSTANT};
use crate::float::Float;

/// Bytecode-tape reverse-mode AD variable.
///
/// Same layout as [`Reverse<F>`](crate::Reverse) (12 bytes for `f64`, `Copy`).
/// Operations record opcodes to the thread-local [`BytecodeTape`](crate::bytecode_tape::BytecodeTape).
#[derive(Clone, Copy, Debug)]
pub struct BReverse<F: Float> {
    pub(crate) value: F,
    pub(crate) index: u32,
}

impl<F: Float> BReverse<F> {
    /// Create a constant (not tracked on tape).
    #[inline]
    pub fn constant(value: F) -> Self {
        BReverse {
            value,
            index: CONSTANT,
        }
    }

    /// Create from a tape allocation (internal use).
    #[inline]
    pub fn from_tape(value: F, index: u32) -> Self {
        BReverse { value, index }
    }

    /// Get the tape index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Apply a unary custom operation.
    pub fn custom_unary(self, handle: CustomOpHandle, value: F) -> Self
    where
        F: BtapeThreadLocal,
    {
        let index = crate::bytecode_tape::with_active_btape(|t| {
            let xi = if self.index == CONSTANT {
                t.push_const(self.value)
            } else {
                self.index
            };
            t.push_custom_unary(xi, handle, value)
        });
        BReverse { value, index }
    }

    /// Apply a binary custom operation.
    pub fn custom_binary(self, other: Self, handle: CustomOpHandle, value: F) -> Self
    where
        F: BtapeThreadLocal,
    {
        let index = crate::bytecode_tape::with_active_btape(|t| {
            let li = if self.index == CONSTANT {
                t.push_const(self.value)
            } else {
                self.index
            };
            let ri = if other.index == CONSTANT {
                t.push_const(other.value)
            } else {
                other.index
            };
            t.push_custom_binary(li, ri, handle, value)
        });
        BReverse { value, index }
    }
}

impl<F: Float> Display for BReverse<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<F: Float> Default for BReverse<F> {
    fn default() -> Self {
        BReverse::constant(F::zero())
    }
}

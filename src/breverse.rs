//! Bytecode-tape reverse-mode AD variable.
//!
//! [`BReverse<F>`] is analogous to [`Reverse<F>`](crate::Reverse) but records
//! opcodes to a [`BytecodeTape`] instead
//! of precomputed multipliers to an Adept-style tape. This allows the tape to
//! be re-evaluated at different inputs without re-recording.

use std::fmt::{self, Display};

use crate::bytecode_tape::{BtapeThreadLocal, BytecodeTape, CustomOpHandle, CONSTANT};
use crate::float::Float;

/// Debug-only sentinel for `BReverse` values without tape provenance
/// (constants, or values built via [`BReverse::from_tape`] outside a
/// recording). Untracked values are exempt from the cross-tape check.
#[cfg(debug_assertions)]
pub(crate) const TAPE_ID_UNTRACKED: u64 = 0;

/// Ensure a BReverse operand has a valid tape index. If it's a constant
/// (index == CONSTANT), promote it to a `Const` entry on the tape.
#[inline]
pub(crate) fn ensure_on_tape<F: Float>(x: &BReverse<F>, tape: &mut BytecodeTape<F>) -> u32 {
    #[cfg(debug_assertions)]
    x.debug_assert_same_tape(tape);
    if x.index == CONSTANT {
        tape.push_const(x.value)
    } else {
        x.index
    }
}

impl<F: Float> BytecodeTape<F> {
    /// Register every element of `x` as an input variable, returning the
    /// tracked [`BReverse`] values in order.
    pub(crate) fn new_inputs(&mut self, x: &[F]) -> Vec<BReverse<F>> {
        x.iter()
            .map(|&val| {
                let idx = self.new_input(val);
                BReverse::from_tape_of(self, val, idx)
            })
            .collect()
    }
}

/// Bytecode-tape reverse-mode AD variable.
///
/// Same layout as [`Reverse<F>`](crate::Reverse) (12 bytes of payload for
/// `f64` — 16 with alignment padding — and `Copy`) in release builds. Operations record opcodes to the thread-local
/// [`BytecodeTape`].
///
/// # Tape identity
///
/// A `BReverse` produced during a recording is bound to that recording's
/// tape: its index is only meaningful there. Using it while a *different*
/// tape is active — capturing an outer variable inside a nested `record`,
/// stashing values across two recordings, or moving them between recording
/// threads — silently references an unrelated tape slot (or panics
/// out-of-range). Debug builds carry a tape-identity tag and panic on such
/// cross-tape use; release builds omit the tag and the check.
#[derive(Clone, Copy, Debug)]
pub struct BReverse<F: Float> {
    pub(crate) value: F,
    pub(crate) index: u32,
    /// Debug-only identity of the tape this value was recorded on.
    #[cfg(debug_assertions)]
    pub(crate) tape_id: u64,
}

impl<F: Float> From<F> for BReverse<F> {
    #[inline]
    fn from(val: F) -> Self {
        BReverse::constant(val)
    }
}

impl<F: Float> BReverse<F> {
    /// Create a constant (not tracked on tape).
    #[inline]
    #[must_use]
    pub fn constant(value: F) -> Self {
        BReverse {
            value,
            index: CONSTANT,
            #[cfg(debug_assertions)]
            tape_id: TAPE_ID_UNTRACKED,
        }
    }

    /// Create from a tape allocation (internal use).
    ///
    /// Values built this way carry no tape provenance and are exempt from
    /// the debug cross-tape check; prefer indices produced by an active
    /// recording.
    #[inline]
    #[must_use]
    pub fn from_tape(value: F, index: u32) -> Self {
        BReverse {
            value,
            index,
            #[cfg(debug_assertions)]
            tape_id: TAPE_ID_UNTRACKED,
        }
    }

    /// Create from an allocation on a specific tape, carrying its identity
    /// in debug builds.
    #[inline]
    pub(crate) fn from_tape_of(
        tape: &crate::bytecode_tape::BytecodeTape<F>,
        value: F,
        index: u32,
    ) -> Self {
        #[cfg(not(debug_assertions))]
        let _ = tape;
        BReverse {
            value,
            index,
            #[cfg(debug_assertions)]
            tape_id: tape.tape_id,
        }
    }

    /// Create from an allocation on the currently active recording's tape
    /// (debug builds tag the value with that tape's identity).
    #[inline]
    pub(crate) fn from_active_recording(value: F, index: u32) -> Self
    where
        F: BtapeThreadLocal,
    {
        BReverse {
            value,
            index,
            #[cfg(debug_assertions)]
            tape_id: crate::bytecode_tape::active_btape_id::<F>(),
        }
    }

    /// Debug-only: panic if this value belongs to a different tape than the
    /// one it is being recorded onto. Untracked values (constants,
    /// `from_tape`) are exempt.
    #[cfg(debug_assertions)]
    #[inline]
    pub(crate) fn debug_assert_same_tape(&self, tape: &crate::bytecode_tape::BytecodeTape<F>) {
        if self.index != CONSTANT && self.tape_id != TAPE_ID_UNTRACKED {
            assert_eq!(
                self.tape_id, tape.tape_id,
                "BReverse value from another recording used on the active tape: values must \
                 not cross record() boundaries, nested recordings, or threads"
            );
        }
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
            let xi = ensure_on_tape(&self, t);
            t.push_custom_unary(xi, handle, value)
        });
        BReverse::from_active_recording(value, index)
    }

    /// Apply a binary custom operation.
    pub fn custom_binary(self, other: Self, handle: CustomOpHandle, value: F) -> Self
    where
        F: BtapeThreadLocal,
    {
        let index = crate::bytecode_tape::with_active_btape(|t| {
            let li = ensure_on_tape(&self, t);
            let ri = ensure_on_tape(&other, t);
            t.push_custom_binary(li, ri, handle, value)
        });
        BReverse::from_active_recording(value, index)
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

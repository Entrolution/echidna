//! Nonsmooth extensions: branch tracking, kink detection, and Clarke subdifferential.
//!
//! Implements Griewank & Walther, Chapter 14. Provides data structures for
//! tracking which branch of `abs`, `min`, `max` was taken during forward
//! evaluation, and for computing the Clarke generalized Jacobian via
//! enumeration of limiting Jacobians with forced branch choices.

use std::fmt;

use num_traits::Float;

use crate::opcode::OpCode;

/// A single kink encountered during forward evaluation.
///
/// Records which nonsmooth operation was executed, where in the tape it lives,
/// the switching value (distance from the kink), and which branch was taken.
#[derive(Clone, Debug)]
pub struct KinkEntry<F: Float> {
    /// Index into the tape's opcode/value arrays.
    pub tape_index: u32,
    /// The nonsmooth opcode (Abs, Min, or Max).
    pub opcode: OpCode,
    /// Distance from the kink point:
    /// - Abs: `x` (kink at `x = 0`)
    /// - Min/Max: `a - b` (kink at `a = b`)
    pub switching_value: F,
    /// Which branch was taken:
    /// - Abs: `+1` if `x >= 0`, `-1` if `x < 0`
    /// - Max: `+1` if `a >= b` (first wins), `-1` if `b > a`
    /// - Min: `+1` if `a <= b` (first wins), `-1` if `b < a`
    pub branch: i8,
}

/// Summary of all nonsmooth operations encountered during a forward sweep.
#[derive(Clone, Debug)]
pub struct NonsmoothInfo<F: Float> {
    /// All kink entries, in tape order.
    pub kinks: Vec<KinkEntry<F>>,
}

impl<F: Float> NonsmoothInfo<F> {
    /// Return kink entries whose switching value is within `tol` of zero.
    ///
    /// These are the "active" kinks — points near the boundary between branches.
    pub fn active_kinks(&self, tol: F) -> Vec<&KinkEntry<F>> {
        self.kinks
            .iter()
            .filter(|k| k.switching_value.abs() < tol)
            .collect()
    }

    /// True if no kinks are active within the given tolerance.
    pub fn is_smooth(&self, tol: F) -> bool {
        self.kinks
            .iter()
            .all(|k| k.switching_value.abs() >= tol)
    }

    /// Branch signature: `(tape_index, branch)` pairs for all kinks.
    ///
    /// Two evaluations at the same input produce the same signature.
    pub fn signature(&self) -> Vec<(u32, i8)> {
        self.kinks.iter().map(|k| (k.tape_index, k.branch)).collect()
    }
}

/// Errors that can occur during Clarke Jacobian computation.
#[derive(Clone, Debug)]
pub enum ClarkeError {
    /// Too many active kinks for enumeration.
    ///
    /// The Clarke Jacobian requires 2^k limiting Jacobians where k is the
    /// number of active kinks. This error is returned when k exceeds the limit.
    TooManyKinks {
        /// Number of active kinks found.
        count: usize,
        /// Maximum allowed.
        limit: usize,
    },
}

impl fmt::Display for ClarkeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClarkeError::TooManyKinks { count, limit } => {
                write!(
                    f,
                    "too many active kinks for Clarke enumeration: {} (limit {})",
                    count, limit
                )
            }
        }
    }
}

impl std::error::Error for ClarkeError {}

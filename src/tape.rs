//! Adept-style two-stack tape for reverse-mode AD.
//!
//! Stores precomputed partial derivatives (multipliers) and operand indices during the
//! forward pass. The reverse sweep is a single multiply-accumulate loop with zero-adjoint
//! skipping — no opcode dispatch overhead. Used internally by [`crate::Reverse`].

use std::cell::Cell;

use crate::Float;

/// Sentinel index indicating a constant (not recorded on tape).
pub const CONSTANT: u32 = u32::MAX;

/// A recorded operation: its result lives at `lhs_index`, and its operands'
/// multipliers/indices span `[prev.end_plus_one .. self.end_plus_one)`.
#[derive(Clone, Copy, Debug)]
struct Statement {
    lhs_index: u32,
    end_plus_one: u32,
}

/// Adept-style two-stack tape for reverse-mode AD.
///
/// Records precomputed partial derivatives (multipliers) and operand indices
/// during the forward sweep. The reverse sweep is a single multiply-accumulate
/// loop with zero-adjoint skipping — no opcode dispatch.
pub struct Tape<F: Float> {
    statements: Vec<Statement>,
    multipliers: Vec<F>,
    indices: Vec<u32>,
    num_variables: u32,
}

impl<F: Float> Default for Tape<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Tape<F> {
    /// Create an empty tape.
    pub fn new() -> Self {
        let mut tape = Tape {
            statements: Vec::new(),
            multipliers: Vec::new(),
            indices: Vec::new(),
            num_variables: 0,
        };
        // Sentinel statement at index 0 so that `statements[i-1].end_plus_one`
        // is always valid for i >= 1.
        tape.statements.push(Statement {
            lhs_index: 0,
            end_plus_one: 0,
        });
        tape
    }

    /// Create a tape with pre-allocated capacity.
    pub fn with_capacity(est_ops: usize) -> Self {
        let mut tape = Tape {
            statements: Vec::with_capacity(est_ops + 1),
            multipliers: Vec::with_capacity(est_ops * 2),
            indices: Vec::with_capacity(est_ops * 2),
            num_variables: 0,
        };
        tape.statements.push(Statement {
            lhs_index: 0,
            end_plus_one: 0,
        });
        tape
    }

    /// Register a new independent variable. Returns `(gradient_index, value)`.
    ///
    /// No statement is pushed for input variables — they are leaf nodes
    /// whose adjoints should not be zeroed during the reverse sweep.
    #[inline]
    pub fn new_variable(&mut self, value: F) -> (u32, F) {
        let idx = self.num_variables;
        self.num_variables += 1;
        (idx, value)
    }

    /// Record a unary operation: `result = f(operand)` with precomputed `multiplier = df/d(operand)`.
    #[inline]
    pub fn push_unary(&mut self, operand_idx: u32, multiplier: F) -> u32 {
        let result_idx = self.num_variables;
        self.num_variables += 1;

        if operand_idx != CONSTANT {
            self.multipliers.push(multiplier);
            self.indices.push(operand_idx);
        }

        self.statements.push(Statement {
            lhs_index: result_idx,
            end_plus_one: self.multipliers.len() as u32,
        });
        result_idx
    }

    /// Record a binary operation with precomputed partial derivatives.
    #[inline]
    pub fn push_binary(&mut self, lhs_idx: u32, lhs_mult: F, rhs_idx: u32, rhs_mult: F) -> u32 {
        let result_idx = self.num_variables;
        self.num_variables += 1;

        if lhs_idx != CONSTANT {
            self.multipliers.push(lhs_mult);
            self.indices.push(lhs_idx);
        }
        if rhs_idx != CONSTANT {
            self.multipliers.push(rhs_mult);
            self.indices.push(rhs_idx);
        }

        self.statements.push(Statement {
            lhs_index: result_idx,
            end_plus_one: self.multipliers.len() as u32,
        });
        result_idx
    }

    /// Run the reverse sweep, seeding the adjoint of `seed_index` with 1.
    /// Returns the full adjoint vector.
    pub fn reverse(&self, seed_index: u32) -> Vec<F> {
        let mut adjoints = vec![F::zero(); self.num_variables as usize];
        adjoints[seed_index as usize] = F::one();

        for i in (1..self.statements.len()).rev() {
            let stmt = self.statements[i];
            let a = adjoints[stmt.lhs_index as usize];
            if a != F::zero() {
                adjoints[stmt.lhs_index as usize] = F::zero();
                let start = self.statements[i - 1].end_plus_one as usize;
                let end = stmt.end_plus_one as usize;
                for j in start..end {
                    adjoints[self.indices[j] as usize] =
                        adjoints[self.indices[j] as usize] + self.multipliers[j] * a;
                }
            }
        }
        adjoints
    }

    /// Run the reverse sweep with custom adjoint seeds.
    pub fn reverse_seeded(&self, seeds: &[(u32, F)]) -> Vec<F> {
        let mut adjoints = vec![F::zero(); self.num_variables as usize];
        for &(idx, seed) in seeds {
            adjoints[idx as usize] = adjoints[idx as usize] + seed;
        }

        for i in (1..self.statements.len()).rev() {
            let stmt = self.statements[i];
            let a = adjoints[stmt.lhs_index as usize];
            if a != F::zero() {
                adjoints[stmt.lhs_index as usize] = F::zero();
                let start = self.statements[i - 1].end_plus_one as usize;
                let end = stmt.end_plus_one as usize;
                for j in start..end {
                    adjoints[self.indices[j] as usize] =
                        adjoints[self.indices[j] as usize] + self.multipliers[j] * a;
                }
            }
        }
        adjoints
    }
}

// Thread-local active tape pointer.
thread_local! {
    static TAPE_F32: Cell<*mut Tape<f32>> = const { Cell::new(std::ptr::null_mut()) };
    static TAPE_F64: Cell<*mut Tape<f64>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Trait to select the correct thread-local for a given float type.
pub trait TapeThreadLocal: Float {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>>;
}

impl TapeThreadLocal for f32 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>> {
        &TAPE_F32
    }
}

impl TapeThreadLocal for f64 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>> {
        &TAPE_F64
    }
}

/// Access the active tape for the current thread. Panics if no tape is active.
#[inline]
pub fn with_active_tape<F: TapeThreadLocal, R>(f: impl FnOnce(&mut Tape<F>) -> R) -> R {
    F::cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active tape. Use echidna::grad() or similar API."
        );
        // SAFETY: The TapeGuard guarantees the pointer is valid for the
        // duration of the closure-based API scope, and only one mutable
        // reference exists at a time (single-threaded access via thread-local).
        let tape = unsafe { &mut *ptr };
        f(tape)
    })
}

/// RAII guard that sets a tape as the thread-local active tape and restores
/// the previous one on drop.
pub struct TapeGuard<F: TapeThreadLocal> {
    prev: *mut Tape<F>,
}

impl<F: TapeThreadLocal> TapeGuard<F> {
    /// Activate `tape` as the thread-local tape. Returns a guard that restores
    /// the previous tape on drop.
    pub fn new(tape: &mut Tape<F>) -> Self {
        let prev = F::cell().with(|cell| {
            let prev = cell.get();
            cell.set(tape as *mut Tape<F>);
            prev
        });
        TapeGuard { prev }
    }
}

impl<F: TapeThreadLocal> Drop for TapeGuard<F> {
    fn drop(&mut self) {
        F::cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

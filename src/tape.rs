//! Adept-style two-stack tape for reverse-mode AD.
//!
//! Stores precomputed partial derivatives (multipliers) and operand indices during the
//! forward pass. The reverse sweep is a single multiply-accumulate loop with zero-adjoint
//! skipping — no opcode dispatch overhead. Used internally by [`crate::Reverse`].

use std::cell::Cell;
use std::marker::PhantomData;

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
    #[must_use]
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
    #[must_use]
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

    /// Clear all recorded operations, retaining allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.statements.clear();
        self.multipliers.clear();
        self.indices.clear();
        self.num_variables = 0;
        self.statements.push(Statement {
            lhs_index: 0,
            end_plus_one: 0,
        });
    }

    /// Register a new independent variable. Returns `(gradient_index, value)`.
    ///
    /// No statement is pushed for input variables — they are leaf nodes
    /// whose adjoints should not be zeroed during the reverse sweep.
    #[inline]
    pub fn new_variable(&mut self, value: F) -> (u32, F) {
        debug_assert!(
            self.num_variables < u32::MAX - 1,
            "tape variable count overflow: exceeded u32::MAX"
        );
        let idx = self.num_variables;
        self.num_variables += 1;
        (idx, value)
    }

    /// Record a unary operation: `result = f(operand)` with precomputed `multiplier = df/d(operand)`.
    #[inline]
    pub fn push_unary(&mut self, operand_idx: u32, multiplier: F) -> u32 {
        debug_assert!(
            self.num_variables < u32::MAX - 1,
            "tape variable count overflow: exceeded u32::MAX"
        );
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
        debug_assert!(
            self.num_variables < u32::MAX - 1,
            "tape variable count overflow: exceeded u32::MAX"
        );
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
    ///
    /// # Performance note on subnormal (denormal) floats
    ///
    /// This function does **not** set the x86/x64 MXCSR flush-to-zero bit.
    /// If a reverse sweep produces subnormal adjoints — common on long
    /// chains where `a != zero` lets tiny contributions accumulate — x86
    /// hardware falls into microcode-emulated arithmetic that can be
    /// 10-100× slower than the normal path. Adjoints that stay normal
    /// (≥ `f32::MIN_POSITIVE` ≈ 1.17e-38 or `f64::MIN_POSITIVE` ≈ 2.2e-308)
    /// are unaffected.
    ///
    /// Callers on x86 where subnormal adjoints are expected can opt into
    /// FTZ by setting `MXCSR` themselves (`_mm_setcsr(0x9FC0)` via
    /// `core::arch::x86_64::_mm_setcsr`) around the reverse-sweep call.
    /// Doing so in the library would change numerical semantics for
    /// callers who depend on subnormal precision, so the choice is
    /// deferred. ARM64 always flushes subnormals by default (FPCR.FZ=1
    /// in AArch32 compatibility, FPCR.FZ16 et al. on AArch64), so this
    /// warning is x86-specific.
    #[must_use]
    pub fn reverse(&self, seed_index: u32) -> Vec<F> {
        let mut adjoints = vec![F::zero(); self.num_variables as usize];
        adjoints[seed_index as usize] = F::one();

        for i in (1..self.statements.len()).rev() {
            let stmt = self.statements[i];
            let a = adjoints[stmt.lhs_index as usize];
            // Performance: skip zero-adjoint branches. Trade-off: `0 * NaN`
            // returns 0 instead of propagating NaN. This is a deliberate design choice
            // matching JAX convention. Use forward mode if NaN propagation is needed.
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

// Thread-local tape pool (one tape per type per thread).
thread_local! {
    static POOL_F32: Cell<Option<Tape<f32>>> = const { Cell::new(None) };
    static POOL_F64: Cell<Option<Tape<f64>>> = const { Cell::new(None) };
}

/// Trait to select the correct thread-local for a given float type.
pub trait TapeThreadLocal: Float {
    /// Returns the thread-local cell holding a pointer to the active tape.
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>>;
    /// Returns the thread-local cell holding the tape pool.
    fn pool_cell() -> &'static std::thread::LocalKey<Cell<Option<Tape<Self>>>>;
    /// Returns the per-type borrow flag cell.
    fn borrow_cell() -> &'static std::thread::LocalKey<Cell<bool>>;
}

impl TapeThreadLocal for f32 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>> {
        &TAPE_F32
    }
    fn pool_cell() -> &'static std::thread::LocalKey<Cell<Option<Tape<Self>>>> {
        &POOL_F32
    }
    fn borrow_cell() -> &'static std::thread::LocalKey<Cell<bool>> {
        &TAPE_BORROWED_F32
    }
}

impl TapeThreadLocal for f64 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut Tape<Self>>> {
        &TAPE_F64
    }
    fn pool_cell() -> &'static std::thread::LocalKey<Cell<Option<Tape<Self>>>> {
        &POOL_F64
    }
    fn borrow_cell() -> &'static std::thread::LocalKey<Cell<bool>> {
        &TAPE_BORROWED_F64
    }
}

impl<F: TapeThreadLocal> Tape<F> {
    /// Take a tape from the thread-local pool, clearing it for reuse.
    /// Falls back to creating a new tape if the pool is empty.
    pub(crate) fn take_pooled(capacity: usize) -> Self {
        F::pool_cell().with(|cell| match cell.take() {
            Some(mut tape) => {
                tape.clear();
                tape
            }
            None => Tape::with_capacity(capacity),
        })
    }

    /// Return a tape to the thread-local pool for future reuse.
    pub(crate) fn return_to_pool(self) {
        F::pool_cell().with(|cell| cell.set(Some(self)));
    }
}

thread_local! {
    // Per-type borrow guards (prevents false reentrance detection across different float types)
    static TAPE_BORROWED_F32: Cell<bool> = const { Cell::new(false) };
    static TAPE_BORROWED_F64: Cell<bool> = const { Cell::new(false) };
}

struct TapeBorrowGuard {
    cell: &'static std::thread::LocalKey<Cell<bool>>,
}

impl TapeBorrowGuard {
    fn new<F: TapeThreadLocal>() -> Self {
        let cell = F::borrow_cell();
        cell.with(|b| {
            assert!(
                !b.get(),
                "reentrant with_active_tape call detected — this would create aliased &mut references"
            );
            b.set(true);
        });
        TapeBorrowGuard { cell }
    }
}

impl Drop for TapeBorrowGuard {
    fn drop(&mut self) {
        self.cell.with(|b| b.set(false));
    }
}

/// Access the active tape for the current thread. Panics if no tape is active.
#[inline]
pub fn with_active_tape<F: TapeThreadLocal, R>(f: impl FnOnce(&mut Tape<F>) -> R) -> R {
    let _guard = TapeBorrowGuard::new::<F>();
    F::cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active tape. Use echidna::grad() or similar API."
        );
        // SAFETY: TapeGuard's `'a` lifetime statically ties the raw pointer's
        // validity to the live `&'a mut Tape<F>` borrow on the stack frame
        // that constructed the guard — the borrow checker rejects any program
        // in which the guard outlives its tape. Access is single-threaded via
        // thread-local, and the TapeBorrowGuard above ensures no reentrant
        // call creates aliased &mut references.
        let tape = unsafe { &mut *ptr };
        f(tape)
    })
}

/// RAII guard that sets a tape as the thread-local active tape and restores
/// the previous one on drop.
///
/// The `'a` lifetime ties the guard to the borrow of the tape it was
/// constructed from, so the borrow checker rejects any pattern where the
/// guard could outlive its tape.
///
/// ```compile_fail
/// use echidna::tape::{Tape, TapeGuard};
/// let guard: TapeGuard<f64>;
/// {
///     let mut tape: Tape<f64> = Tape::new();
///     guard = TapeGuard::new(&mut tape);
/// } // tape dropped, but guard would survive — rejected by the borrow checker.
/// drop(guard);
/// ```
pub struct TapeGuard<'a, F: TapeThreadLocal> {
    prev: *mut Tape<F>,
    _borrow: PhantomData<&'a mut Tape<F>>,
}

impl<'a, F: TapeThreadLocal> TapeGuard<'a, F> {
    /// Activate `tape` as the thread-local tape. Returns a guard that restores
    /// the previous tape on drop.
    #[must_use = "dropping the guard immediately deactivates the tape; bind it to extend the recording scope"]
    pub fn new(tape: &'a mut Tape<F>) -> Self {
        let prev = F::cell().with(|cell| {
            let prev = cell.get();
            cell.set(tape as *mut Tape<F>);
            prev
        });
        TapeGuard {
            prev,
            _borrow: PhantomData,
        }
    }
}

impl<'a, F: TapeThreadLocal> Drop for TapeGuard<'a, F> {
    fn drop(&mut self) {
        F::cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

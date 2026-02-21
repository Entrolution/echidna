//! Bytecode tape for re-evaluable reverse-mode AD.
//!
//! Unlike the Adept-style [`Tape`](crate::tape::Tape), this tape stores opcodes
//! rather than precomputed multipliers. This allows the tape to be re-evaluated
//! at different inputs without re-recording, at the cost of opcode dispatch
//! during the reverse sweep.
//!
//! # Limitations
//!
//! The tape records one execution path. If the recorded function contains
//! branches (`if x > 0 { ... } else { ... }`), re-evaluating at inputs that
//! take a different branch produces incorrect results.

use std::cell::Cell;

use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

/// Sentinel index for constant entries (not tracked).
pub const CONSTANT: u32 = u32::MAX;

/// A bytecode tape that can be re-evaluated at different inputs.
///
/// Created via [`crate::api::record`]. After recording, call [`forward`](Self::forward)
/// to re-evaluate and [`reverse`](Self::reverse) to compute adjoints.
pub struct BytecodeTape<F: Float> {
    opcodes: Vec<OpCode>,
    arg_indices: Vec<[u32; 2]>,
    values: Vec<F>,
    num_inputs: u32,
    num_variables: u32,
    output_index: u32,
}

impl<F: Float> BytecodeTape<F> {
    /// Create an empty bytecode tape.
    pub fn new() -> Self {
        BytecodeTape {
            opcodes: Vec::new(),
            arg_indices: Vec::new(),
            values: Vec::new(),
            num_inputs: 0,
            num_variables: 0,
            output_index: 0,
        }
    }

    /// Create a bytecode tape with pre-allocated capacity.
    pub fn with_capacity(est_ops: usize) -> Self {
        BytecodeTape {
            opcodes: Vec::with_capacity(est_ops),
            arg_indices: Vec::with_capacity(est_ops),
            values: Vec::with_capacity(est_ops),
            num_inputs: 0,
            num_variables: 0,
            output_index: 0,
        }
    }

    /// Register a new input variable. Returns its index.
    #[inline]
    pub fn new_input(&mut self, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.num_inputs += 1;
        self.opcodes.push(OpCode::Input);
        self.arg_indices.push([UNUSED, UNUSED]);
        self.values.push(value);
        idx
    }

    /// Register a scalar constant. Returns its index.
    #[inline]
    pub fn push_const(&mut self, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Const);
        self.arg_indices.push([UNUSED, UNUSED]);
        self.values.push(value);
        idx
    }

    /// Record an operation. Returns the result index.
    #[inline]
    pub fn push_op(&mut self, op: OpCode, arg0: u32, arg1: u32, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(op);
        self.arg_indices.push([arg0, arg1]);
        self.values.push(value);
        idx
    }

    /// Record a powi operation. The `i32` exponent is stored in `arg_indices[1]`.
    #[inline]
    pub fn push_powi(&mut self, arg0: u32, exp: i32, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Powi);
        self.arg_indices.push([arg0, opcode::powi_exp_encode(exp)]);
        self.values.push(value);
        idx
    }

    /// Mark the output variable.
    #[inline]
    pub fn set_output(&mut self, index: u32) {
        self.output_index = index;
    }

    /// Get the output value (available after `forward()` or initial recording).
    #[inline]
    pub fn output_value(&self) -> F {
        self.values[self.output_index as usize]
    }

    /// Number of input variables.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Number of operations (including inputs and constants).
    #[inline]
    pub fn num_ops(&self) -> usize {
        self.opcodes.len()
    }

    /// Re-evaluate the tape at new inputs (forward sweep).
    ///
    /// Overwrites `values` in-place — no allocation.
    pub fn forward(&mut self, inputs: &[F]) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        // Overwrite input values.
        for (i, &v) in inputs.iter().enumerate() {
            self.values[i] = v;
        }

        // Re-evaluate all non-Input, non-Const ops.
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        self.values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        // For powi, b holds the encoded exponent.
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    self.values[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep: compute adjoints seeded at the output.
    ///
    /// Returns the full adjoint vector (length = `num_variables`).
    pub fn reverse(&self, seed_index: u32) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();

        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoints[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                op => {
                    adjoints[i] = F::zero();
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        self.values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    let r = self.values[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        adjoints[b_idx as usize] = adjoints[b_idx as usize] + db * adj;
                    }
                }
            }
        }
        adjoints
    }

    /// Forward + reverse: compute the gradient at new inputs.
    ///
    /// Returns only the input adjoints (indices `0..num_inputs`).
    pub fn gradient(&mut self, inputs: &[F]) -> Vec<F> {
        self.forward(inputs);
        let adjoints = self.reverse(self.output_index);
        adjoints[..self.num_inputs as usize].to_vec()
    }

    /// Like [`gradient`](Self::gradient) but reuses a caller-provided buffer
    /// for the adjoint vector, avoiding allocation on repeated calls.
    pub fn gradient_with_buf(&mut self, inputs: &[F], adjoint_buf: &mut Vec<F>) -> Vec<F> {
        self.forward(inputs);

        let n = self.num_variables as usize;
        adjoint_buf.clear();
        adjoint_buf.resize(n, F::zero());
        adjoint_buf[self.output_index as usize] = F::one();

        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoint_buf[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                op => {
                    adjoint_buf[i] = F::zero();
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        self.values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    let r = self.values[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    adjoint_buf[a_idx as usize] = adjoint_buf[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        adjoint_buf[b_idx as usize] = adjoint_buf[b_idx as usize] + db * adj;
                    }
                }
            }
        }
        adjoint_buf[..self.num_inputs as usize].to_vec()
    }
}

impl<F: Float> Default for BytecodeTape<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════
//  Thread-local active bytecode tape
// ══════════════════════════════════════════════

thread_local! {
    static BTAPE_F32: Cell<*mut BytecodeTape<f32>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_F64: Cell<*mut BytecodeTape<f64>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Trait to select the correct thread-local for a given float type.
pub trait BtapeThreadLocal: Float {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>>;
}

impl BtapeThreadLocal for f32 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F32
    }
}

impl BtapeThreadLocal for f64 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F64
    }
}

/// Access the active bytecode tape for the current thread.
/// Panics if no tape is active.
#[inline]
pub fn with_active_btape<F: BtapeThreadLocal, R>(f: impl FnOnce(&mut BytecodeTape<F>) -> R) -> R {
    F::btape_cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active bytecode tape. Use echidna::record() to record a function."
        );
        // SAFETY: BtapeGuard guarantees validity for the duration of the
        // recording scope, single-threaded via thread-local.
        let tape = unsafe { &mut *ptr };
        f(tape)
    })
}

/// RAII guard that sets a bytecode tape as the thread-local active tape.
pub struct BtapeGuard<F: BtapeThreadLocal> {
    prev: *mut BytecodeTape<F>,
}

impl<F: BtapeThreadLocal> BtapeGuard<F> {
    /// Activate `tape` as the thread-local bytecode tape.
    pub fn new(tape: &mut BytecodeTape<F>) -> Self {
        let prev = F::btape_cell().with(|cell| {
            let prev = cell.get();
            cell.set(tape as *mut BytecodeTape<F>);
            prev
        });
        BtapeGuard { prev }
    }
}

impl<F: BtapeThreadLocal> Drop for BtapeGuard<F> {
    fn drop(&mut self) {
        F::btape_cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

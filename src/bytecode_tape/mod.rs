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

use std::collections::HashMap;
use std::sync::Arc;

use crate::dual::Dual;
use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

// Submodules — each adds impl blocks to BytecodeTape<F>
mod forward;
mod jacobian;
mod optimize;
mod reverse;
mod sparse;
mod tangent;

#[cfg(feature = "taylor")]
mod taylor;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "serde")]
mod serde_support;

mod thread_local;
pub use self::thread_local::{with_active_btape, BtapeGuard, BtapeThreadLocal};

/// Sentinel index for constant entries (not tracked).
pub const CONSTANT: u32 = u32::MAX;

/// Trait for user-registered custom operations on the bytecode tape.
///
/// Operations are defined on `F` (the base float type). The tape automatically
/// handles forward-mode tangent propagation and reverse-mode adjoint accumulation
/// via chain rule using the partials you provide.
///
/// For correct second-order derivatives (Hessian, HVP) through custom ops,
/// override [`eval_dual`](CustomOp::eval_dual) and
/// [`partials_dual`](CustomOp::partials_dual). The defaults use first-order
/// chain rule, which treats partials as constant.
///
/// # Example
///
/// ```ignore
/// struct Softplus;
///
/// impl CustomOp<f64> for Softplus {
///     fn eval(&self, a: f64, _b: f64) -> f64 {
///         (1.0 + a.exp()).ln()
///     }
///     fn partials(&self, a: f64, _b: f64, _r: f64) -> (f64, f64) {
///         let sig = 1.0 / (1.0 + (-a).exp());
///         (sig, 0.0)
///     }
/// }
/// ```
pub trait CustomOp<F: Float>: Send + Sync {
    /// Forward evaluation on the base float type.
    fn eval(&self, a: F, b: F) -> F;
    /// Reverse partials `(∂result/∂a, ∂result/∂b)`.
    fn partials(&self, a: F, b: F, result: F) -> (F, F);

    /// Evaluate in dual-number context for second-order derivatives (HVP, Hessian).
    ///
    /// Override this to propagate tangent information through the custom op.
    /// The default uses first-order chain rule: correct for gradients, but
    /// treats partials as constant for second-order derivatives.
    ///
    /// Primals are embedded in the dual numbers: `a.re` is the primal input.
    fn eval_dual(&self, a: Dual<F>, b: Dual<F>) -> Dual<F> {
        let result = self.eval(a.re, b.re);
        let (da, db) = self.partials(a.re, b.re, result);
        Dual::new(result, da * a.eps + db * b.eps)
    }

    /// Partials in dual-number context for second-order derivatives.
    ///
    /// Override this to return partials whose tangent components carry the
    /// derivative of the partial itself. The default returns constant partials.
    fn partials_dual(&self, a: Dual<F>, b: Dual<F>, result: Dual<F>) -> (Dual<F>, Dual<F>) {
        let (da, db) = self.partials(a.re, b.re, result.re);
        (Dual::constant(da), Dual::constant(db))
    }
}

/// Handle returned by [`BytecodeTape::register_custom`], used to invoke custom ops.
#[derive(Clone, Copy, Debug)]
pub struct CustomOpHandle(pub(crate) u16);

/// A bytecode tape that can be re-evaluated at different inputs.
///
/// Created via [`crate::api::record`]. After recording, call [`forward`](Self::forward)
/// to re-evaluate and [`reverse`](Self::reverse) to compute adjoints.
pub struct BytecodeTape<F: Float> {
    pub(crate) opcodes: Vec<OpCode>,
    pub(crate) arg_indices: Vec<[u32; 2]>,
    pub(crate) values: Vec<F>,
    pub(crate) num_inputs: u32,
    pub(crate) num_variables: u32,
    pub(crate) output_index: u32,
    /// Indices of multiple output variables (empty = single-output mode).
    pub(crate) output_indices: Vec<u32>,
    /// Registered custom operations (callback table).
    pub(crate) custom_ops: Vec<Arc<dyn CustomOp<F>>>,
    /// Second operand index for binary custom ops (sparse side table).
    /// Maps tape index → second operand index.
    pub(crate) custom_second_args: HashMap<u32, u32>,
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
            output_indices: Vec::new(),
            custom_ops: Vec::new(),
            custom_second_args: HashMap::new(),
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
            output_indices: Vec::new(),
            custom_ops: Vec::new(),
            custom_second_args: HashMap::new(),
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
    ///
    /// **Constant folding**: if all operands point to `Const` entries (not `Input`),
    /// the operation is replaced by a single `Const` with the already-computed value.
    ///
    /// **Algebraic simplification**: identity patterns (`x + 0 → x`, `x * 1 → x`,
    /// etc.) and absorbing patterns (`x * 0 → 0`, `x - x → 0`, `x / x → 1`) are
    /// detected and short-circuited. Absorbing patterns are guarded by a value check
    /// to handle NaN/Inf edge cases correctly.
    #[inline]
    pub fn push_op(&mut self, op: OpCode, arg0: u32, arg1: u32, value: F) -> u32 {
        // Constant folding: if both args (when present) are Const, emit Const instead.
        let arg0_const = self.opcodes[arg0 as usize] == OpCode::Const;
        let arg1_const = arg1 == UNUSED || self.opcodes[arg1 as usize] == OpCode::Const;
        if arg0_const && arg1_const {
            return self.push_const(value);
        }

        // Algebraic simplification: single-arg-const patterns (binary ops only).
        if (arg0_const || arg1_const) && arg1 != UNUSED {
            if let Some(idx) =
                self.try_algebraic_simplify(op, arg0, arg1, arg0_const, arg1_const, value)
            {
                return idx;
            }
        }

        // Same-index simplification: x - x → 0, x / x → 1.
        if arg0 == arg1 && arg1 != UNUSED {
            if let Some(idx) = self.try_same_index_simplify(op, value) {
                return idx;
            }
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(op);
        self.arg_indices.push([arg0, arg1]);
        self.values.push(value);
        idx
    }

    /// Try to simplify a binary op where exactly one argument is a known constant.
    ///
    /// Identity patterns (`x + 0`, `x * 1`, etc.) are always safe — they return
    /// the original index whose value is correct. Absorbing patterns (`x * 0`)
    /// use `push_const(value)` (not `push_const(F::zero())`) to preserve IEEE 754
    /// signed zero semantics, and are guarded by `value == expected` to handle
    /// NaN/Inf correctly (e.g., `NaN * 0 = NaN`, not `0`).
    #[inline(never)]
    fn try_algebraic_simplify(
        &mut self,
        op: OpCode,
        arg0: u32,
        arg1: u32,
        arg0_const: bool,
        arg1_const: bool,
        value: F,
    ) -> Option<u32> {
        let zero = F::zero();
        let one = F::one();
        match op {
            OpCode::Add => {
                if arg1_const && self.values[arg1 as usize] == zero {
                    return Some(arg0);
                }
                if arg0_const && self.values[arg0 as usize] == zero {
                    return Some(arg1);
                }
            }
            OpCode::Sub => {
                if arg1_const && self.values[arg1 as usize] == zero {
                    return Some(arg0);
                }
            }
            OpCode::Mul => {
                // Identity: x * 1 → x, 1 * x → x
                if arg1_const && self.values[arg1 as usize] == one {
                    return Some(arg0);
                }
                if arg0_const && self.values[arg0 as usize] == one {
                    return Some(arg1);
                }
                // Absorbing: x * 0 → const (guarded: NaN * 0 = NaN, not 0)
                if arg1_const && self.values[arg1 as usize] == zero && value == zero {
                    return Some(self.push_const(value));
                }
                if arg0_const && self.values[arg0 as usize] == zero && value == zero {
                    return Some(self.push_const(value));
                }
            }
            OpCode::Div => {
                if arg1_const && self.values[arg1 as usize] == one {
                    return Some(arg0);
                }
            }
            _ => {}
        }
        None
    }

    /// Try to simplify a binary op where both arguments are the same index.
    ///
    /// `x - x → 0` is guarded (Inf - Inf = NaN, not 0).
    /// `x / x → 1` is guarded (0/0 = NaN, not 1).
    #[inline(never)]
    fn try_same_index_simplify(&mut self, op: OpCode, value: F) -> Option<u32> {
        match op {
            OpCode::Sub if value == F::zero() => Some(self.push_const(value)),
            OpCode::Div if value == F::one() => Some(self.push_const(value)),
            _ => None,
        }
    }

    /// Record a powi operation. The `i32` exponent is stored in `arg_indices[1]`.
    ///
    /// **Constant folding**: if the operand is a `Const`, emit `Const` instead.
    ///
    /// **Algebraic simplification**: `x^0 → 1` (guarded), `x^1 → x`,
    /// `x^(-1) → Recip(x)` (cheaper unary dispatch).
    #[inline]
    pub fn push_powi(&mut self, arg0: u32, exp: i32, value: F) -> u32 {
        if self.opcodes[arg0 as usize] == OpCode::Const {
            return self.push_const(value);
        }

        // x^0 → 1 (guarded: 0^0 edge case — only fold when value is actually 1)
        if exp == 0 && value == F::one() {
            return self.push_const(F::one());
        }
        // x^1 → x
        if exp == 1 {
            return arg0;
        }
        // x^(-1) → Recip(x) (cheaper unary opcode dispatch)
        if exp == -1 {
            return self.push_op(OpCode::Recip, arg0, UNUSED, value);
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Powi);
        self.arg_indices.push([arg0, opcode::powi_exp_encode(exp)]);
        self.values.push(value);
        idx
    }

    /// Register a custom operation. Returns a handle for use with
    /// [`crate::BReverse::custom_unary`] and [`crate::BReverse::custom_binary`].
    pub fn register_custom(&mut self, op: Arc<dyn CustomOp<F>>) -> CustomOpHandle {
        let idx = self.custom_ops.len();
        assert!(idx <= u16::MAX as usize, "too many custom ops");
        self.custom_ops.push(op);
        CustomOpHandle(idx as u16)
    }

    /// Record a unary custom op. `arg_indices = [arg0, callback_idx]`.
    #[inline]
    pub fn push_custom_unary(&mut self, arg0: u32, handle: CustomOpHandle, value: F) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Custom);
        self.arg_indices.push([arg0, handle.0 as u32]);
        self.values.push(value);
        idx
    }

    /// Record a binary custom op. `arg_indices = [arg0, callback_idx]`,
    /// second operand stored in `custom_second_args`.
    #[inline]
    pub fn push_custom_binary(
        &mut self,
        arg0: u32,
        arg1: u32,
        handle: CustomOpHandle,
        value: F,
    ) -> u32 {
        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Custom);
        self.arg_indices.push([arg0, handle.0 as u32]);
        self.custom_second_args.insert(idx, arg1);
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

    /// Index of the (single) output variable.
    ///
    /// Use this with the buffer produced by [`forward_tangent`](Self::forward_tangent)
    /// to read the output: `buf[tape.output_index()]`.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index as usize
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

    // ── Multi-output support ──

    /// Mark multiple output variables.
    ///
    /// When set, [`num_outputs`](Self::num_outputs), [`output_values`](Self::output_values),
    /// [`jacobian`](Self::jacobian), and [`vjp_multi`](Self::vjp_multi) become available.
    /// Single-output methods (`output_index`, `gradient`, etc.) continue to work using
    /// the first output.
    pub fn set_outputs(&mut self, indices: &[u32]) {
        self.output_indices = indices.to_vec();
        if let Some(&first) = indices.first() {
            self.output_index = first;
        }
    }

    /// Number of output variables. Returns 1 in single-output mode.
    pub fn num_outputs(&self) -> usize {
        if self.output_indices.is_empty() {
            1
        } else {
            self.output_indices.len()
        }
    }

    /// Get all output values (available after `forward()` or initial recording).
    ///
    /// In single-output mode, returns a single-element vector.
    pub fn output_values(&self) -> Vec<F> {
        if self.output_indices.is_empty() {
            vec![self.values[self.output_index as usize]]
        } else {
            self.output_indices
                .iter()
                .map(|&idx| self.values[idx as usize])
                .collect()
        }
    }

    /// Indices of all output entries in the tape buffer.
    ///
    /// For multi-output tapes, returns all registered output indices.
    /// For single-output tapes, returns a single-element slice.
    pub fn all_output_indices(&self) -> &[u32] {
        if self.output_indices.is_empty() {
            std::slice::from_ref(&self.output_index)
        } else {
            &self.output_indices
        }
    }

    // ── GPU accessor methods ──

    /// Slice view of all opcodes in the tape.
    #[inline]
    pub fn opcodes_slice(&self) -> &[OpCode] {
        &self.opcodes
    }

    /// Slice view of all argument index pairs `[arg0, arg1]`.
    #[inline]
    pub fn arg_indices_slice(&self) -> &[[u32; 2]] {
        &self.arg_indices
    }

    /// Slice view of all primal values in the tape.
    #[inline]
    pub fn values_slice(&self) -> &[F] {
        &self.values
    }

    /// Total number of tape entries (inputs + constants + operations).
    #[inline]
    pub fn num_variables_count(&self) -> usize {
        self.num_variables as usize
    }

    /// Returns `true` if the tape contains any custom operations.
    #[inline]
    pub fn has_custom_ops(&self) -> bool {
        !self.custom_ops.is_empty()
    }
}

impl<F: Float> Default for BytecodeTape<F> {
    fn default() -> Self {
        Self::new()
    }
}

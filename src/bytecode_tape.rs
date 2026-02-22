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
use std::collections::HashMap;
use std::sync::Arc;

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

use crate::float::IsAllZero;
use num_traits::Float as NumFloat;

/// Sentinel index for constant entries (not tracked).
pub const CONSTANT: u32 = u32::MAX;

/// Trait for user-registered custom operations on the bytecode tape.
///
/// Operations are defined on `F` (the base float type). The tape automatically
/// handles forward-mode tangent propagation and reverse-mode adjoint accumulation
/// via chain rule using the partials you provide.
///
/// For second-order derivatives (Hessian, HVP), implement [`CustomOpSecondOrder`]
/// additionally.
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
}

/// Handle returned by [`BytecodeTape::register_custom`], used to invoke custom ops.
#[derive(Clone, Copy, Debug)]
pub struct CustomOpHandle(pub(crate) u16);

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
    /// Indices of multiple output variables (empty = single-output mode).
    output_indices: Vec<u32>,
    /// Registered custom operations (callback table).
    custom_ops: Vec<Arc<dyn CustomOp<F>>>,
    /// Second operand index for binary custom ops (sparse side table).
    /// Maps tape index → second operand index.
    custom_second_args: HashMap<u32, u32>,
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
    #[inline]
    pub fn push_op(&mut self, op: OpCode, arg0: u32, arg1: u32, value: F) -> u32 {
        // Constant folding: if both args (when present) are Const, emit Const instead.
        let arg0_const =
            self.opcodes[arg0 as usize] == OpCode::Const;
        let arg1_const = arg1 == UNUSED || self.opcodes[arg1 as usize] == OpCode::Const;
        if arg0_const && arg1_const {
            return self.push_const(value);
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(op);
        self.arg_indices.push([arg0, arg1]);
        self.values.push(value);
        idx
    }

    /// Record a powi operation. The `i32` exponent is stored in `arg_indices[1]`.
    ///
    /// **Constant folding**: if the operand is a `Const`, emit `Const` instead.
    #[inline]
    pub fn push_powi(&mut self, arg0: u32, exp: i32, value: F) -> u32 {
        if self.opcodes[arg0 as usize] == OpCode::Const {
            return self.push_const(value);
        }

        let idx = self.num_variables;
        self.num_variables += 1;
        self.opcodes.push(OpCode::Powi);
        self.arg_indices.push([arg0, opcode::powi_exp_encode(exp)]);
        self.values.push(value);
        idx
    }

    /// Register a custom operation. Returns a handle for use with
    /// [`BReverse::custom_unary`] and [`BReverse::custom_binary`].
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

    /// Reverse sweep seeded at a specific index with a given weight.
    ///
    /// Like [`reverse`](Self::reverse) but with an arbitrary seed value instead of 1.
    fn reverse_weighted(&self, seed_index: u32, seed_value: F) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = seed_value;

        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoints[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    adjoints[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let r = self.values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoints[bi as usize] = adjoints[bi as usize] + db * adj;
                    }
                }
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

    /// Reverse sweep with weighted seeds for multiple outputs.
    ///
    /// Computes `∑_i weights[i] * ∂output_i/∂x` — a vector-Jacobian product.
    ///
    /// Returns the gradient with respect to all inputs (length [`num_inputs`](Self::num_inputs)).
    pub fn reverse_seeded(&self, seeds: &[F]) -> Vec<F> {
        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        assert_eq!(
            seeds.len(),
            out_indices.len(),
            "seeds length must match number of outputs"
        );

        let n = self.num_variables as usize;
        let ni = self.num_inputs as usize;
        let mut total_adjoints = vec![F::zero(); n];

        for (k, (&out_idx, &weight)) in out_indices.iter().zip(seeds.iter()).enumerate() {
            if weight == F::zero() {
                continue;
            }
            let adjoints = self.reverse_weighted(out_idx, weight);
            for j in 0..n {
                total_adjoints[j] = total_adjoints[j] + adjoints[j];
            }
            let _ = k;
        }

        total_adjoints[..ni].to_vec()
    }

    /// Compute the full Jacobian of a multi-output tape via reverse mode.
    ///
    /// Performs `m` reverse sweeps (one per output). Returns `J[i][j] = ∂f_i/∂x_j`.
    pub fn jacobian(&mut self, inputs: &[F]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        let ni = self.num_inputs as usize;
        let mut jac = Vec::with_capacity(out_indices.len());

        for &out_idx in &out_indices {
            let adjoints = self.reverse(out_idx);
            jac.push(adjoints[..ni].to_vec());
        }

        jac
    }

    /// Vector-Jacobian product for a multi-output tape.
    ///
    /// Computes `wᵀ · J` where `J` is the Jacobian. More efficient than
    /// computing the full Jacobian when only the weighted combination is needed.
    pub fn vjp_multi(&mut self, inputs: &[F], weights: &[F]) -> Vec<F> {
        self.forward(inputs);
        self.reverse_seeded(weights)
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
                OpCode::Custom => {
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = self
                        .custom_second_args
                        .get(&(i as u32))
                        .map(|&bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    self.values[i] = self.custom_ops[cb_idx as usize].eval(a, b);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        self.values[b_idx as usize]
                    } else if op == OpCode::Powi {
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
                OpCode::Custom => {
                    adjoints[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let r = self.values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoints[bi as usize] = adjoints[bi as usize] + db * adj;
                    }
                }
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

    /// Forward evaluation into an external buffer.
    ///
    /// Reads opcodes, constants, and argument indices from `self`, but writes
    /// computed values into `values_buf` instead of `self.values`. This allows
    /// parallel evaluation of the same tape at different inputs without cloning.
    pub fn forward_into(&self, inputs: &[F], values_buf: &mut Vec<F>) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        let n = self.num_variables as usize;
        values_buf.clear();
        values_buf.resize(n, F::zero());

        // Copy constant values from the tape, then overwrite inputs.
        values_buf.copy_from_slice(&self.values[..n]);
        for (i, &v) in inputs.iter().enumerate() {
            values_buf[i] = v;
        }

        // Re-evaluate all non-Input, non-Const ops.
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = values_buf[a_idx as usize];
                    let b = self
                        .custom_second_args
                        .get(&(i as u32))
                        .map(|&bi| values_buf[bi as usize])
                        .unwrap_or(F::zero());
                    values_buf[i] = self.custom_ops[cb_idx as usize].eval(a, b);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = values_buf[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        values_buf[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    values_buf[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep reading from an external values buffer.
    ///
    /// Like [`reverse`](Self::reverse) but reads primal values from `values`
    /// instead of `self.values`. Pair with [`forward_into`](Self::forward_into)
    /// for parallel evaluation.
    pub fn reverse_from(&self, values: &[F], seed_index: u32) -> Vec<F> {
        let n = self.num_variables as usize;
        assert_eq!(values.len(), n, "values buffer has wrong length");
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();

        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoints[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    adjoints[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt
                        .map(|bi| values[bi as usize])
                        .unwrap_or(F::zero());
                    let r = values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoints[bi as usize] = adjoints[bi as usize] + db * adj;
                    }
                }
                op => {
                    adjoints[i] = F::zero();
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or_else(|| F::zero())
                    } else {
                        F::zero()
                    };
                    let r = values[i];
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
                OpCode::Custom => {
                    adjoint_buf[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let r = self.values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoint_buf[a_idx as usize] = adjoint_buf[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoint_buf[bi as usize] = adjoint_buf[bi as usize] + db * adj;
                    }
                }
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

    // ── Forward-over-reverse (second-order) ──

    /// Forward sweep with tangent-carrying numbers. Reads opcodes and constants
    /// from `self`, writing results into `buf`. Does not mutate the tape.
    ///
    /// Generic over `T: NumFloat` so it works with both `Dual<F>` and
    /// `DualVec<F, N>`.
    fn forward_tangent<T: NumFloat>(&self, inputs: &[T], buf: &mut Vec<T>) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());

        let mut input_idx = 0usize;
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input => {
                    buf[i] = inputs[input_idx];
                    input_idx += 1;
                }
                OpCode::Const => {
                    buf[i] = T::from(self.values[i]).unwrap();
                }
                OpCode::Custom => {
                    // For custom ops in tangent mode: evaluate on primal values,
                    // then propagate tangent via chain rule using partials.
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a_primal = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b_primal = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let result = self.custom_ops[cb_idx as usize].eval(a_primal, b_primal);
                    let (da, db) =
                        self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, result);
                    // Apply chain rule: tangent = da * a_tangent + db * b_tangent
                    let a_t = buf[a_idx as usize];
                    let b_t = b_idx_opt
                        .map(|bi| buf[bi as usize])
                        .unwrap_or(T::zero());
                    let result_t = T::from(result).unwrap();
                    let da_t = T::from(da).unwrap();
                    let db_t = T::from(db).unwrap();
                    // Result tangent = da * a.eps + db * b.eps (for Dual<F>)
                    // But we need to set both primal and tangent. With Dual numbers,
                    // the result = f(a.re, b.re) + (da * a.eps + db * b.eps) * eps
                    // We construct this by: result_primal + tangent_sum, where for generic
                    // T we use: result_t + (da_t * (a_t - a_re_t) + db_t * (b_t - b_re_t))
                    // But actually simpler: set buf[i] via arithmetic on T directly:
                    // buf[i].re = result, buf[i].eps = da * a.eps + db * b.eps
                    // Since T is opaque, we do: result_t + da_t * (a_t - T::from(a_primal)) + db_t * (b_t - T::from(b_primal))
                    let a_re_t = T::from(a_primal).unwrap();
                    let b_re_t = T::from(b_primal).unwrap();
                    buf[i] = result_t + da_t * (a_t - a_re_t) + db_t * (b_t - b_re_t);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = buf[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    buf[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep with tangent-carrying adjoints. Uses values from
    /// [`forward_tangent`](Self::forward_tangent). Uses [`IsAllZero`] to
    /// safely skip zero adjoints without dropping tangent contributions.
    fn reverse_tangent<T: NumFloat + IsAllZero>(&self, tangent_vals: &[T], buf: &mut Vec<T>) {
        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());
        buf[self.output_index as usize] = T::one();

        for i in (0..self.opcodes.len()).rev() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a_primal = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b_primal = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let r_primal = self.values[i];
                    let (da, db) =
                        self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, r_primal);
                    let da_t = T::from(da).unwrap();
                    let db_t = T::from(db).unwrap();
                    buf[a_idx as usize] = buf[a_idx as usize] + da_t * adj;
                    if let Some(bi) = b_idx_opt {
                        buf[bi as usize] = buf[bi as usize] + db_t * adj;
                    }
                }
                op => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = tangent_vals[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        tangent_vals[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    let r = tangent_vals[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    buf[a_idx as usize] = buf[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize] = buf[b_idx as usize] + db * adj;
                    }
                }
            }
        }
    }

    /// Hessian-vector product via forward-over-reverse.
    ///
    /// Returns `(gradient, H·v)` where both are `Vec<F>` of length
    /// [`num_inputs`](Self::num_inputs). The tape is not mutated.
    pub fn hvp(&self, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
        let mut dual_vals = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.hvp_with_buf(x, v, &mut dual_vals, &mut adjoint_buf)
    }

    /// Like [`hvp`](Self::hvp) but reuses caller-provided buffers to avoid
    /// allocation on repeated calls (e.g. inside [`hessian`](Self::hessian)).
    pub fn hvp_with_buf(
        &self,
        x: &[F],
        v: &[F],
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) -> (Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v.len(), n, "wrong number of directions");

        let dual_inputs: Vec<Dual<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| Dual::new(xi, vi))
            .collect();

        self.forward_tangent(&dual_inputs, dual_vals_buf);
        self.reverse_tangent(dual_vals_buf, adjoint_buf);

        let gradient: Vec<F> = (0..n).map(|i| adjoint_buf[i].re).collect();
        let hvp: Vec<F> = (0..n).map(|i| adjoint_buf[i].eps).collect();
        (gradient, hvp)
    }

    /// Like [`hvp_with_buf`](Self::hvp_with_buf) but also reuses a caller-provided
    /// input buffer, eliminating all allocations on repeated calls.
    fn hvp_with_all_bufs(
        &self,
        x: &[F],
        v: &[F],
        dual_input_buf: &mut Vec<Dual<F>>,
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) {
        let n = self.num_inputs as usize;

        // Reuse the input buffer instead of allocating
        dual_input_buf.clear();
        dual_input_buf.extend(x.iter().zip(v.iter()).map(|(&xi, &vi)| Dual::new(xi, vi)));

        self.forward_tangent(dual_input_buf, dual_vals_buf);
        self.reverse_tangent(dual_vals_buf, adjoint_buf);
        let _ = n; // suppress unused warning
    }

    /// Full Hessian matrix via `n` Hessian-vector products.
    ///
    /// Returns `(value, gradient, hessian)` where `hessian[i][j] = ∂²f/∂x_i∂x_j`.
    /// The tape is not mutated.
    pub fn hessian(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        for j in 0..n {
            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf
                .extend((0..n).map(|i| Dual::new(x[i], if i == j { F::one() } else { F::zero() })));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if j == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for (row, adj) in hessian.iter_mut().zip(adjoint_buf.iter()) {
                row[j] = adj.eps;
            }
        }

        (value, gradient, hessian)
    }

    /// Full Hessian matrix via batched forward-over-reverse.
    ///
    /// Processes `ceil(n/N)` batches instead of `n` individual HVPs,
    /// computing N Hessian columns simultaneously.
    pub fn hessian_vec<const N: usize>(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let num_batches = n.div_ceil(N);
        for batch in 0..num_batches {
            let base = batch * N;

            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let col = base + lane;
                    if col < n && i == col {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if batch == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for lane in 0..N {
                let col = base + lane;
                if col >= n {
                    break;
                }
                for i in 0..n {
                    hessian[i][col] = adjoint_buf[i].eps[lane];
                }
            }
        }

        (value, gradient, hessian)
    }

    /// Detect the structural sparsity pattern of the Hessian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets.
    /// At nonlinear operations, marks cross-pairs as potential Hessian interactions.
    pub fn detect_sparsity(&self) -> crate::sparse::SparsityPattern {
        crate::sparse::detect_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            self.num_inputs as usize,
            self.num_variables as usize,
        )
    }

    /// Detect the structural sparsity pattern of the Jacobian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets (first-order).
    /// For each output, determines which inputs it depends on.
    pub fn detect_jacobian_sparsity(&self) -> crate::sparse::JacobianSparsityPattern {
        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };
        crate::sparse::detect_jacobian_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            self.num_inputs as usize,
            self.num_variables as usize,
            &out_indices,
        )
    }

    /// Compute a sparse Hessian using structural sparsity detection and graph coloring.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)` where
    /// `hessian_values[k]` corresponds to `(pattern.rows[k], pattern.cols[k])`.
    ///
    /// For problems with sparse Hessians, this requires only `chromatic_number`
    /// HVP calls instead of `n`, which can be dramatically fewer for banded
    /// or sparse interaction structures.
    pub fn sparse_hessian(&self, x: &[F]) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut v = vec![F::zero(); n];

        for color in 0..num_colors {
            // Form direction vector: v[i] = 1 if colors[i] == color, else 0
            for i in 0..n {
                v[i] = if colors[i] == color {
                    F::one()
                } else {
                    F::zero()
                };
            }

            self.hvp_with_all_bufs(
                x,
                &v,
                &mut dual_input_buf,
                &mut dual_vals_buf,
                &mut adjoint_buf,
            );

            if color == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            // Extract Hessian entries for this color.
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    hessian_values[k] = adjoint_buf[row as usize].eps;
                }
            }
        }

        (value, gradient, pattern, hessian_values)
    }

    /// Batched sparse Hessian: packs N colors per sweep using DualVec.
    ///
    /// Reduces the number of forward+reverse sweeps from `num_colors` to
    /// `ceil(num_colors / N)`. Each sweep processes N colors simultaneously.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)`.
    pub fn sparse_hessian_vec<const N: usize>(
        &self,
        x: &[F],
    ) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            // Build DualVec inputs: lane k has v[i]=1 if colors[i] == base_color+k
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let target_color = base_color + lane as u32;
                    if target_color < num_colors && colors[i] == target_color {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if batch == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            // Extract Hessian entries: for entry (row, col) with colors[col] == base_color+lane,
            // read adjoint_buf[row].eps[lane]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                let col_color = colors[col as usize];
                if col_color >= base_color && col_color < base_color + N as u32 {
                    let lane = (col_color - base_color) as usize;
                    hessian_values[k] = adjoint_buf[row as usize].eps[lane];
                }
            }
        }

        (value, gradient, pattern, hessian_values)
    }

    // ── Batch evaluation ──

    /// Evaluate the gradient at multiple input points.
    ///
    /// Returns one gradient vector per input point.
    pub fn gradient_batch(&mut self, inputs: &[&[F]]) -> Vec<Vec<F>> {
        inputs.iter().map(|x| self.gradient(x)).collect()
    }

    // ── Higher-order derivatives ──

    /// Third-order directional derivative: `∑_{jk} (∂³f/∂x_i∂x_j∂x_k) v1_j v2_k`.
    ///
    /// Given directions `v1` and `v2`, computes:
    /// - `gradient`: `∇f(x)`
    /// - `hvp`: `H(x) · v1` (Hessian-vector product)
    /// - `third`: `(∂/∂v2)(H · v1)` (third-order tensor contracted with v1 and v2)
    ///
    /// Uses `Dual<Dual<F>>` (nested dual numbers): inner tangent for `v1`,
    /// outer tangent for `v2`.
    pub fn third_order_hvvp(
        &self,
        x: &[F],
        v1: &[F],
        v2: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v1.len(), n, "wrong v1 length");
        assert_eq!(v2.len(), n, "wrong v2 length");

        // Build Dual<Dual<F>> inputs:
        //   inner.re = x[i], inner.eps = v1[i]  (for HVP direction)
        //   outer.re = inner, outer.eps = v2[i]  (for third-order direction)
        //
        // outer = Dual { re: Dual(x[i], v1[i]), eps: Dual(v2[i], 0) }
        let dd_inputs: Vec<Dual<Dual<F>>> = (0..n)
            .map(|i| Dual {
                re: Dual::new(x[i], v1[i]),
                eps: Dual::new(v2[i], F::zero()),
            })
            .collect();

        let mut dd_vals: Vec<Dual<Dual<F>>> = Vec::new();
        let mut dd_adj: Vec<Dual<Dual<F>>> = Vec::new();

        self.forward_tangent(&dd_inputs, &mut dd_vals);
        self.reverse_tangent(&dd_vals, &mut dd_adj);

        let gradient: Vec<F> = (0..n).map(|i| dd_adj[i].re.re).collect();
        let hvp: Vec<F> = (0..n).map(|i| dd_adj[i].re.eps).collect();
        let third: Vec<F> = (0..n).map(|i| dd_adj[i].eps.eps).collect();

        (gradient, hvp, third)
    }

    // ── Tape optimizations ──

    /// Eliminate dead (unreachable) entries from the tape.
    ///
    /// Walks backward from all outputs, marks reachable entries, then compacts
    /// the tape in-place with an index remap. Inputs are never removed.
    pub fn dead_code_elimination(&mut self) {
        let n = self.opcodes.len();
        let mut reachable = vec![false; n];

        // Mark all inputs as reachable.
        for flag in reachable.iter_mut().take(self.num_inputs as usize) {
            *flag = true;
        }

        // Seed from outputs.
        let mut stack: Vec<u32> = Vec::new();
        stack.push(self.output_index);
        for &oi in &self.output_indices {
            stack.push(oi);
        }

        while let Some(idx) = stack.pop() {
            let i = idx as usize;
            if reachable[i] {
                continue;
            }
            reachable[i] = true;
            let [a, b] = self.arg_indices[i];
            if a != UNUSED {
                stack.push(a);
            }
            if b != UNUSED && self.opcodes[i] != OpCode::Powi {
                stack.push(b);
            }
        }

        // Build remap: old index -> new index.
        let mut remap = vec![0u32; n];
        let mut new_idx = 0u32;
        for i in 0..n {
            if reachable[i] {
                remap[i] = new_idx;
                new_idx += 1;
            }
        }
        let new_len = new_idx as usize;

        // Compact in-place.
        let mut write = 0;
        for (read, &is_reachable) in reachable.iter().enumerate().take(n) {
            if is_reachable {
                self.opcodes[write] = self.opcodes[read];
                self.values[write] = self.values[read];
                let [a, b] = self.arg_indices[read];
                let ra = if a != UNUSED { remap[a as usize] } else { UNUSED };
                let rb = if b != UNUSED && self.opcodes[read] != OpCode::Powi {
                    remap[b as usize]
                } else {
                    b
                };
                self.arg_indices[write] = [ra, rb];
                write += 1;
            }
        }

        self.opcodes.truncate(new_len);
        self.arg_indices.truncate(new_len);
        self.values.truncate(new_len);
        self.num_variables = new_len as u32;
        self.output_index = remap[self.output_index as usize];
        for oi in &mut self.output_indices {
            *oi = remap[*oi as usize];
        }
    }

    /// Common subexpression elimination.
    ///
    /// Deduplicates identical `(OpCode, arg0, arg1)` triples, normalising
    /// argument order for commutative ops. Finishes with a DCE pass to
    /// remove the now-dead duplicates.
    pub fn cse(&mut self) {
        use std::collections::HashMap;

        let n = self.opcodes.len();
        // Maps canonical (op, arg0, arg1) -> first index that computed it.
        let mut seen: HashMap<(OpCode, u32, u32), u32> = HashMap::new();
        // remap[i] = canonical index for entry i (identity by default).
        let mut remap: Vec<u32> = (0..n as u32).collect();

        let is_commutative = |op: OpCode| -> bool {
            matches!(
                op,
                OpCode::Add | OpCode::Mul | OpCode::Max | OpCode::Min | OpCode::Hypot
            )
        };

        for i in 0..n {
            let op = self.opcodes[i];
            match op {
                OpCode::Input | OpCode::Const => continue,
                _ => {}
            }

            let [mut a, mut b] = self.arg_indices[i];
            // Apply remap to args (except Powi exponent in b).
            a = remap[a as usize];
            if b != UNUSED && op != OpCode::Powi {
                b = remap[b as usize];
            }
            // Update arg_indices with remapped values.
            self.arg_indices[i] = [a, b];

            // Build the canonical key.
            let key = if b == UNUSED {
                // Unary: hash (op, arg0) only; use UNUSED as placeholder.
                (op, a, UNUSED)
            } else if is_commutative(op) {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                (op, lo, hi)
            } else {
                (op, a, b)
            };

            if let Some(&canonical) = seen.get(&key) {
                remap[i] = canonical;
            } else {
                seen.insert(key, i as u32);
            }
        }

        // Apply remap to all arg_indices (for entries that reference CSE'd nodes).
        for i in 0..n {
            let op = self.opcodes[i];
            if matches!(op, OpCode::Input | OpCode::Const) {
                continue;
            }
            let [a, b] = self.arg_indices[i];
            let ra = remap[a as usize];
            let rb = if b != UNUSED && op != OpCode::Powi {
                remap[b as usize]
            } else {
                b
            };
            self.arg_indices[i] = [ra, rb];
        }

        // Update output indices.
        self.output_index = remap[self.output_index as usize];
        for oi in &mut self.output_indices {
            *oi = remap[*oi as usize];
        }

        // DCE removes the now-unreachable duplicate entries.
        self.dead_code_elimination();
    }

    // ── Sparse Jacobian ──

    /// Compute a sparse Jacobian using structural sparsity detection and graph coloring.
    ///
    /// Auto-selects forward-mode (column compression) or reverse-mode (row compression)
    /// based on which requires fewer sweeps.
    ///
    /// Returns `(output_values, pattern, jacobian_values)`.
    pub fn sparse_jacobian(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (col_colors, num_col_colors) = crate::sparse::column_coloring(&pattern);
        let (row_colors, num_row_colors) = crate::sparse::row_coloring(&pattern);

        if num_col_colors <= num_row_colors {
            let jac_values =
                self.sparse_jacobian_forward_impl(x, &pattern, &col_colors, num_col_colors);
            let outputs = self.output_values();
            (outputs, pattern, jac_values)
        } else {
            let jac_values =
                self.sparse_jacobian_reverse_impl(x, &pattern, &row_colors, num_row_colors);
            let outputs = self.output_values();
            (outputs, pattern, jac_values)
        }
    }

    /// Sparse Jacobian via forward-mode (column compression).
    pub fn sparse_jacobian_forward(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);
        let jac_values = self.sparse_jacobian_forward_impl(x, &pattern, &colors, num_colors);
        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Jacobian via reverse-mode (row compression).
    pub fn sparse_jacobian_reverse(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::row_coloring(&pattern);
        let jac_values = self.sparse_jacobian_reverse_impl(x, &pattern, &colors, num_colors);
        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Jacobian with a precomputed sparsity pattern and coloring.
    ///
    /// Skips re-detection of sparsity on repeated calls. Use `column_coloring` colors
    /// for forward mode or `row_coloring` colors for reverse mode. The `forward` flag
    /// selects the mode.
    pub fn sparse_jacobian_with_pattern(
        &mut self,
        x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
        forward_mode: bool,
    ) -> (Vec<F>, Vec<F>) {
        self.forward(x);
        let jac_values = if forward_mode {
            self.sparse_jacobian_forward_impl(x, pattern, colors, num_colors)
        } else {
            self.sparse_jacobian_reverse_impl(x, pattern, colors, num_colors)
        };
        let outputs = self.output_values();
        (outputs, jac_values)
    }

    /// Forward-mode sparse Jacobian implementation (column compression).
    ///
    /// Each color group seeds a forward pass with tangent 1.0 in the columns
    /// sharing that color. The resulting tangent at each output gives J[row][col].
    fn sparse_jacobian_forward_impl(
        &self,
        x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<Dual<F>> = Vec::new();

        for color in 0..num_colors {
            // Build Dual inputs: tangent = 1 for inputs with this color
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                Dual::new(
                    x[i],
                    if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    },
                )
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            // Extract Jacobian entries: for entry (row, col) with colors[col] == color,
            // the tangent at output_indices[row] gives J[row][col]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    jac_values[k] = dual_vals_buf[out_indices[row as usize] as usize].eps;
                }
            }
        }

        jac_values
    }

    /// Reverse-mode sparse Jacobian implementation (row compression).
    ///
    /// Each color group seeds a reverse pass with adjoint 1.0 at the outputs
    /// sharing that color.
    fn sparse_jacobian_reverse_impl(
        &self,
        _x: &[F],
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let m = self.num_outputs();
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        for color in 0..num_colors {
            // Build seeds: weight = 1 for outputs with this color
            let seeds: Vec<F> = (0..m)
                .map(|i| {
                    if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();

            let adjoints = self.reverse_seeded_full(&seeds, &out_indices);

            // Extract Jacobian entries: for entry (row, col) with colors[row] == color,
            // adjoint[col] gives J[row][col]
            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[row as usize] == color {
                    jac_values[k] = adjoints[col as usize];
                }
            }
        }

        jac_values
    }

    /// Reverse sweep with weighted seeds, returning full adjoint vector.
    fn reverse_seeded_full(&self, seeds: &[F], out_indices: &[u32]) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];

        for (&out_idx, &weight) in out_indices.iter().zip(seeds.iter()) {
            if weight == F::zero() {
                continue;
            }
            adjoints[out_idx as usize] = adjoints[out_idx as usize] + weight;
        }

        for i in (0..self.opcodes.len()).rev() {
            let adj = adjoints[i];
            if adj == F::zero() {
                continue;
            }

            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    adjoints[i] = F::zero();
                    let [a_idx, cb_idx] = self.arg_indices[i];
                    let a = self.values[a_idx as usize];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let b = b_idx_opt
                        .map(|bi| self.values[bi as usize])
                        .unwrap_or(F::zero());
                    let r = self.values[i];
                    let (da, db) = self.custom_ops[cb_idx as usize].partials(a, b, r);
                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if let Some(bi) = b_idx_opt {
                        adjoints[bi as usize] = adjoints[bi as usize] + db * adj;
                    }
                }
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

    /// Batched sparse Jacobian: packs N colors per forward sweep using DualVec.
    ///
    /// Reduces the number of forward sweeps from `num_colors` to
    /// `ceil(num_colors / N)`.
    pub fn sparse_jacobian_vec<const N: usize>(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);

        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let target_color = base_color + lane as u32;
                    if target_color < num_colors && colors[i] == target_color {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                let col_color = colors[col as usize];
                if col_color >= base_color && col_color < base_color + N as u32 {
                    let lane = (col_color - base_color) as usize;
                    jac_values[k] =
                        dual_vals_buf[out_indices[row as usize] as usize].eps[lane];
                }
            }
        }

        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Dense Jacobian via forward mode (one forward-tangent pass per input).
    ///
    /// More efficient than reverse mode when `num_inputs < num_outputs`.
    pub fn jacobian_forward(&self, x: &[F]) -> Vec<Vec<F>> {
        let n = self.num_inputs as usize;

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };
        let m = out_indices.len();

        let mut jac = vec![vec![F::zero(); n]; m];
        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<Dual<F>> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for col in 0..n {
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                Dual::new(x[i], if i == col { F::one() } else { F::zero() })
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (row_idx, &out_idx) in out_indices.iter().enumerate() {
                jac[row_idx][col] = dual_vals_buf[out_idx as usize].eps;
            }
        }

        jac
    }

    /// Sparse Hessian with a precomputed sparsity pattern and coloring.
    ///
    /// Skips re-detection on repeated calls (e.g. in solver loops).
    pub fn sparse_hessian_with_pattern(
        &self,
        x: &[F],
        pattern: &crate::sparse::SparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> (F, Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut v = vec![F::zero(); n];

        for color in 0..num_colors {
            for i in 0..n {
                v[i] = if colors[i] == color {
                    F::one()
                } else {
                    F::zero()
                };
            }

            self.hvp_with_all_bufs(
                x,
                &v,
                &mut dual_input_buf,
                &mut dual_vals_buf,
                &mut adjoint_buf,
            );

            if color == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                if colors[col as usize] == color {
                    hessian_values[k] = adjoint_buf[row as usize].eps;
                }
            }
        }

        (value, gradient, hessian_values)
    }

    /// Run all tape optimizations: CSE followed by DCE.
    ///
    /// In debug builds, validates internal consistency after optimization.
    pub fn optimize(&mut self) {
        // In debug builds, clone the tape before optimization for validation.
        #[cfg(debug_assertions)]
        let pre_opt_len = self.opcodes.len();

        self.cse();
        self.dead_code_elimination();

        // Validate internal consistency in debug builds.
        debug_assert!({
            let n = self.opcodes.len();
            // All arg_indices must point to valid entries.
            for i in 0..n {
                let [a, b] = self.arg_indices[i];
                match self.opcodes[i] {
                    OpCode::Input | OpCode::Const => {
                        assert_eq!(a, UNUSED, "Input/Const should have UNUSED args");
                        assert_eq!(b, UNUSED, "Input/Const should have UNUSED args");
                    }
                    OpCode::Powi => {
                        assert!(
                            (a as usize) < n,
                            "Powi arg0 {} out of bounds (tape len {})",
                            a,
                            n
                        );
                    }
                    _ => {
                        assert!(
                            (a as usize) < i,
                            "arg0 {} not before op {} (tape len {})",
                            a,
                            i,
                            n
                        );
                        if b != UNUSED {
                            assert!(
                                (b as usize) < i,
                                "arg1 {} not before op {} (tape len {})",
                                b,
                                i,
                                n
                            );
                        }
                    }
                }
            }
            // output_index must be valid.
            assert!(
                (self.output_index as usize) < n,
                "output_index {} out of bounds (tape len {})",
                self.output_index,
                n
            );
            for &oi in &self.output_indices {
                assert!(
                    (oi as usize) < n,
                    "output_indices entry {} out of bounds (tape len {})",
                    oi,
                    n
                );
            }
            // num_inputs must be preserved.
            let input_count = self.opcodes.iter().filter(|&&op| op == OpCode::Input).count();
            assert_eq!(
                input_count,
                self.num_inputs as usize,
                "num_inputs mismatch after optimization"
            );
            let _ = pre_opt_len;
            true
        });
    }
}

impl<F: Float> Default for BytecodeTape<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════
//  Parallel methods (rayon)
// ══════════════════════════════════════════════

#[cfg(feature = "parallel")]
impl<F: Float> BytecodeTape<F> {
    /// Parallel gradient: forward + reverse using external buffers.
    ///
    /// Takes `&self` instead of `&mut self`, enabling shared access across threads.
    pub fn gradient_par(&self, inputs: &[F]) -> Vec<F> {
        let mut values_buf = Vec::new();
        self.forward_into(inputs, &mut values_buf);
        let adjoints = self.reverse_from(&values_buf, self.output_index);
        adjoints[..self.num_inputs as usize].to_vec()
    }

    /// Parallel Jacobian: one reverse sweep per output, parallelized.
    ///
    /// Returns `J[i][j] = ∂f_i/∂x_j`.
    pub fn jacobian_par(&self, inputs: &[F]) -> Vec<Vec<F>> {
        use rayon::prelude::*;

        let mut values_buf = Vec::new();
        self.forward_into(inputs, &mut values_buf);

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };

        let ni = self.num_inputs as usize;
        out_indices
            .par_iter()
            .map(|&out_idx| {
                let adjoints = self.reverse_from(&values_buf, out_idx);
                adjoints[..ni].to_vec()
            })
            .collect()
    }

    /// Parallel Hessian: one HVP per column, parallelized over columns.
    ///
    /// Returns `(value, gradient, hessian)`.
    pub fn hessian_par(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        // Compute gradient and value from column 0 (serial).
        let dual_input_buf: Vec<Dual<F>> = (0..n)
            .map(|i| Dual::new(x[i], if i == 0 { F::one() } else { F::zero() }))
            .collect();
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
        self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

        let value = dual_vals_buf[self.output_index as usize].re;
        let gradient: Vec<F> = (0..n).map(|i| adjoint_buf[i].re).collect();
        let col0: Vec<F> = (0..n).map(|i| adjoint_buf[i].eps).collect();

        // Parallelize remaining columns.
        let other_cols: Vec<Vec<F>> = (1..n)
            .into_par_iter()
            .map(|j| {
                let inputs: Vec<Dual<F>> = (0..n)
                    .map(|i| Dual::new(x[i], if i == j { F::one() } else { F::zero() }))
                    .collect();
                let mut dv = Vec::new();
                let mut ab = Vec::new();
                self.forward_tangent(&inputs, &mut dv);
                self.reverse_tangent(&dv, &mut ab);
                (0..n).map(|i| ab[i].eps).collect()
            })
            .collect();

        let mut hessian = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            hessian[i][0] = col0[i];
        }
        for (j_minus_1, col) in other_cols.iter().enumerate() {
            let j = j_minus_1 + 1;
            for i in 0..n {
                hessian[i][j] = col[i];
            }
        }

        (value, gradient, hessian)
    }

    /// Parallel sparse Hessian: parallelized over colors.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)`.
    pub fn sparse_hessian_par(
        &self,
        x: &[F],
    ) -> (F, Vec<F>, crate::sparse::SparsityPattern, Vec<F>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        // Compute value/gradient from color 0 (serial).
        let mut v0 = vec![F::zero(); n];
        for i in 0..n {
            v0[i] = if colors[i] == 0 {
                F::one()
            } else {
                F::zero()
            };
        }
        let di: Vec<Dual<F>> = (0..n)
            .map(|i| Dual::new(x[i], v0[i]))
            .collect();
        let mut dv = Vec::new();
        let mut ab = Vec::new();
        self.forward_tangent(&di, &mut dv);
        self.reverse_tangent(&dv, &mut ab);
        let value = dv[self.output_index as usize].re;
        let gradient: Vec<F> = (0..n).map(|i| ab[i].re).collect();

        // Collect all color results in parallel.
        let color_results: Vec<Vec<Dual<F>>> = (0..num_colors)
            .into_par_iter()
            .map(|color| {
                let mut v = vec![F::zero(); n];
                for i in 0..n {
                    v[i] = if colors[i] == color {
                        F::one()
                    } else {
                        F::zero()
                    };
                }
                let inputs: Vec<Dual<F>> = (0..n)
                    .map(|i| Dual::new(x[i], v[i]))
                    .collect();
                let mut dv_local = Vec::new();
                let mut ab_local = Vec::new();
                self.forward_tangent(&inputs, &mut dv_local);
                self.reverse_tangent(&dv_local, &mut ab_local);
                ab_local
            })
            .collect();

        let mut hessian_values = vec![F::zero(); pattern.nnz()];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let color = colors[col as usize] as usize;
            hessian_values[k] = color_results[color][row as usize].eps;
        }

        (value, gradient, pattern, hessian_values)
    }

    /// Parallel sparse Jacobian: parallelized over colors.
    ///
    /// Auto-selects forward (column compression) or reverse (row compression)
    /// based on `num_outputs` vs `num_inputs`.
    pub fn sparse_jacobian_par(
        &self,
        x: &[F],
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        use rayon::prelude::*;

        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut values_buf = Vec::new();
        self.forward_into(x, &mut values_buf);

        let out_indices = if self.output_indices.is_empty() {
            vec![self.output_index]
        } else {
            self.output_indices.clone()
        };
        let m = out_indices.len();
        let outputs: Vec<F> = out_indices.iter().map(|&oi| values_buf[oi as usize]).collect();

        let jac_pattern = self.detect_jacobian_sparsity();
        let ni = self.num_inputs as usize;

        if m <= n {
            // Row compression (reverse mode)
            let (row_colors, num_colors) =
                crate::sparse::row_coloring(&jac_pattern);

            let color_results: Vec<Vec<F>> = (0..num_colors)
                .into_par_iter()
                .map(|color| {
                    let n_vars = self.num_variables as usize;
                    let mut adjoints = vec![F::zero(); n_vars];
                    for (i, &oi) in out_indices.iter().enumerate() {
                        if row_colors[i] == color {
                            adjoints[oi as usize] = F::one();
                        }
                    }

                    for idx in (0..self.opcodes.len()).rev() {
                        let adj = adjoints[idx];
                        if adj == F::zero() {
                            continue;
                        }
                        match self.opcodes[idx] {
                            OpCode::Input | OpCode::Const => continue,
                            OpCode::Custom => {
                                adjoints[idx] = F::zero();
                                let [a_idx, cb_idx] = self.arg_indices[idx];
                                let a = values_buf[a_idx as usize];
                                let b_idx_opt =
                                    self.custom_second_args.get(&(idx as u32)).copied();
                                let b = b_idx_opt
                                    .map(|bi| values_buf[bi as usize])
                                    .unwrap_or(F::zero());
                                let r = values_buf[idx];
                                let (da, db) =
                                    self.custom_ops[cb_idx as usize].partials(a, b, r);
                                adjoints[a_idx as usize] =
                                    adjoints[a_idx as usize] + da * adj;
                                if let Some(bi) = b_idx_opt {
                                    adjoints[bi as usize] =
                                        adjoints[bi as usize] + db * adj;
                                }
                            }
                            op => {
                                adjoints[idx] = F::zero();
                                let [a_idx, b_idx] = self.arg_indices[idx];
                                let a = values_buf[a_idx as usize];
                                let b = if b_idx != UNUSED && op != OpCode::Powi {
                                    values_buf[b_idx as usize]
                                } else if op == OpCode::Powi {
                                    F::from(b_idx).unwrap_or_else(|| F::zero())
                                } else {
                                    F::zero()
                                };
                                let r = values_buf[idx];
                                let (da, db) = opcode::reverse_partials(op, a, b, r);
                                adjoints[a_idx as usize] =
                                    adjoints[a_idx as usize] + da * adj;
                                if b_idx != UNUSED && op != OpCode::Powi {
                                    adjoints[b_idx as usize] =
                                        adjoints[b_idx as usize] + db * adj;
                                }
                            }
                        }
                    }
                    adjoints[..ni].to_vec()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in
                jac_pattern.rows.iter().zip(jac_pattern.cols.iter()).enumerate()
            {
                let color = row_colors[row as usize] as usize;
                jac_values[k] = color_results[color][col as usize];
            }

            (outputs, jac_pattern, jac_values)
        } else {
            // Column compression (forward mode) — parallelize forward tangent sweeps
            let (col_colors, num_colors) =
                crate::sparse::column_coloring(&jac_pattern);

            let color_results: Vec<Vec<F>> = (0..num_colors)
                .into_par_iter()
                .map(|color| {
                    let dir: Vec<F> = (0..n)
                        .map(|i| {
                            if col_colors[i] == color {
                                F::one()
                            } else {
                                F::zero()
                            }
                        })
                        .collect();
                    // Forward tangent with F (not Dual) direction
                    // We need to do forward_into first, then forward tangent
                    // But forward_tangent uses self.values, so we need a workaround.
                    // Use Dual<F> with tangent = direction.
                    let inputs: Vec<Dual<F>> = (0..n)
                        .map(|i| Dual::new(x[i], dir[i]))
                        .collect();
                    let mut dv = Vec::new();
                    self.forward_tangent(&inputs, &mut dv);
                    out_indices.iter().map(|&oi| dv[oi as usize].eps).collect()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in
                jac_pattern.rows.iter().zip(jac_pattern.cols.iter()).enumerate()
            {
                let color = col_colors[col as usize] as usize;
                jac_values[k] = color_results[color][row as usize];
            }

            (outputs, jac_pattern, jac_values)
        }
    }

    /// Evaluate the gradient at multiple input points in parallel.
    ///
    /// Uses `forward_into` + `reverse_from` with per-thread buffers.
    pub fn gradient_batch_par(&self, inputs: &[&[F]]) -> Vec<Vec<F>> {
        use rayon::prelude::*;

        let ni = self.num_inputs as usize;
        let out_idx = self.output_index;

        inputs
            .par_iter()
            .map(|x| {
                let mut values_buf = Vec::new();
                self.forward_into(x, &mut values_buf);
                let adjoints = self.reverse_from(&values_buf, out_idx);
                adjoints[..ni].to_vec()
            })
            .collect()
    }

    /// Compute Hessian at multiple input points in parallel.
    ///
    /// Returns `(value, gradient, hessian)` for each input point.
    pub fn hessian_batch_par(&self, inputs: &[&[F]]) -> Vec<(F, Vec<F>, Vec<Vec<F>>)> {
        use rayon::prelude::*;

        inputs
            .par_iter()
            .map(|x| self.hessian_par(x))
            .collect()
    }
}

// ══════════════════════════════════════════════
//  Thread-local active bytecode tape
// ══════════════════════════════════════════════
//  Serde support
// ══════════════════════════════════════════════

#[cfg(feature = "serde")]
mod serde_support {
    use super::*;
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<F: Float + Serialize> Serialize for BytecodeTape<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if !self.custom_ops.is_empty() {
                return Err(serde::ser::Error::custom(
                    "cannot serialize a BytecodeTape containing custom ops; \
                     custom ops must be re-registered after deserialization",
                ));
            }
            let mut s = serializer.serialize_struct("BytecodeTape", 7)?;
            s.serialize_field("opcodes", &self.opcodes)?;
            s.serialize_field("arg_indices", &self.arg_indices)?;
            s.serialize_field("values", &self.values)?;
            s.serialize_field("num_inputs", &self.num_inputs)?;
            s.serialize_field("num_variables", &self.num_variables)?;
            s.serialize_field("output_index", &self.output_index)?;
            s.serialize_field("output_indices", &self.output_indices)?;
            s.serialize_field("custom_second_args", &self.custom_second_args)?;
            s.end()
        }
    }

    impl<'de, F: Float + Deserialize<'de>> Deserialize<'de> for BytecodeTape<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            struct TapeData<F> {
                opcodes: Vec<OpCode>,
                arg_indices: Vec<[u32; 2]>,
                values: Vec<F>,
                num_inputs: u32,
                num_variables: u32,
                output_index: u32,
                #[serde(default)]
                output_indices: Vec<u32>,
                #[serde(default)]
                custom_second_args: HashMap<u32, u32>,
            }

            let data = TapeData::<F>::deserialize(deserializer)?;
            Ok(BytecodeTape {
                opcodes: data.opcodes,
                arg_indices: data.arg_indices,
                values: data.values,
                num_inputs: data.num_inputs,
                num_variables: data.num_variables,
                output_index: data.output_index,
                output_indices: data.output_indices,
                custom_ops: Vec::new(),
                custom_second_args: data.custom_second_args,
            })
        }
    }
}

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

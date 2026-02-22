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

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

use crate::float::IsAllZero;
use num_traits::Float as NumFloat;

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
    /// Indices of multiple output variables (empty = single-output mode).
    output_indices: Vec<u32>,
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
        dual_input_buf.extend(
            x.iter()
                .zip(v.iter())
                .map(|(&xi, &vi)| Dual::new(xi, vi)),
        );

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
            dual_input_buf.extend((0..n).map(|i| {
                Dual::new(x[i], if i == j { F::one() } else { F::zero() })
            }));

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

            self.hvp_with_all_bufs(x, &v, &mut dual_input_buf, &mut dual_vals_buf, &mut adjoint_buf);

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

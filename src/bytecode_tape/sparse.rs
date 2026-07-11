//! Sparse Hessian and Jacobian evaluation over recorded tapes:
//! dependency-bitset sparsity detection plus colored, lane-packed
//! compressed sweeps.

use std::collections::HashMap;

use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;
use crate::opcode::{OpCode, UNUSED};
use crate::sparse::{
    column_coloring, greedy_coloring, row_coloring, JacobianSparsityPattern, SparsityPattern,
};

/// Lane-packed coloring seed: lane `k` is hot iff `colors[i] == base_color + k`
/// and that color exists. Shared by `sparse_hessian_vec` and
/// `sparse_jacobian_vec`; the dense `hessian_vec` seeds on `i == col` instead
/// and deliberately does not use this.
#[inline]
fn color_seed_eps<F: Float, const N: usize>(
    i: usize,
    colors: &[u32],
    base_color: u32,
    num_colors: u32,
) -> [F; N] {
    std::array::from_fn(|lane| {
        let target_color = base_color + lane as u32;
        if target_color < num_colors && colors[i] == target_color {
            F::one()
        } else {
            F::zero()
        }
    })
}

impl<F: Float> super::BytecodeTape<F> {
    /// Detect the structural sparsity pattern of the Hessian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets.
    /// At nonlinear operations, marks cross-pairs as potential Hessian interactions.
    #[must_use]
    pub fn detect_sparsity(&self) -> SparsityPattern {
        detect_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            &self.custom_second_args,
            self.num_inputs as usize,
            self.num_variables_count(),
        )
    }

    /// Detect the structural sparsity pattern of the Jacobian.
    ///
    /// Walks the tape forward propagating input-dependency bitsets (first-order).
    /// For each output, determines which inputs it depends on.
    #[must_use]
    pub fn detect_jacobian_sparsity(&self) -> JacobianSparsityPattern {
        let out_indices = self.all_output_indices();
        detect_jacobian_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            &self.custom_second_args,
            self.num_inputs as usize,
            self.num_variables_count(),
            out_indices,
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
    pub fn sparse_hessian(&self, x: &[F]) -> (F, Vec<F>, SparsityPattern, Vec<F>) {
        self.assert_scalar_output("sparse_hessian");
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = greedy_coloring(&pattern);
        let (value, gradient, hessian_values) =
            self.sparse_hessian_with_pattern(x, &pattern, &colors, num_colors);
        (value, gradient, pattern, hessian_values)
    }

    /// Batched sparse Hessian: packs N colors per sweep using DualVec.
    ///
    /// Reduces the number of forward+reverse sweeps from `num_colors` to
    /// `ceil(num_colors / N)`. Each sweep processes N colors simultaneously.
    ///
    /// **Custom ops limitation:** For tapes containing custom ops, this method
    /// uses first-order chain rule (linearized partials). For exact second-order
    /// derivatives through custom ops, use `sparse_hessian` instead, which calls
    /// `CustomOp::eval_dual` / `CustomOp::partials_dual`.
    ///
    /// Returns `(value, gradient, pattern, hessian_values)`.
    pub fn sparse_hessian_vec<const N: usize>(
        &self,
        x: &[F],
    ) -> (F, Vec<F>, SparsityPattern, Vec<F>) {
        self.assert_scalar_output("sparse_hessian_vec");
        assert!(
            self.custom_ops.is_empty(),
            "sparse_hessian_vec: custom ops produce approximate (first-order) second derivatives; \
             use eval_forward with Dual<Dual<F>> for exact Hessians through custom ops"
        );
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = greedy_coloring(&pattern);

        let hessian_values = vec![F::zero(); pattern.nnz()];
        let gradient = vec![F::zero(); n];
        let mut value = F::zero();

        // Constant-output tape (n == 0): no batches to sweep.
        if n == 0 {
            return (
                self.constant_output_value(),
                gradient,
                pattern,
                hessian_values,
            );
        }

        let mut hessian_values = hessian_values;
        let mut gradient = gradient;

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            // Build DualVec inputs: lane k has v[i]=1 if colors[i] == base_color+k
            dual_input_buf.clear();
            dual_input_buf.extend(
                (0..n).map(|i| {
                    DualVec::new(x[i], color_seed_eps(i, &colors, base_color, num_colors))
                }),
            );

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

    // ── Sparse Jacobian ──

    /// Compute a sparse Jacobian using structural sparsity detection and graph coloring.
    ///
    /// Auto-selects forward-mode (column compression) or reverse-mode (row compression)
    /// based on which requires fewer sweeps.
    ///
    /// Returns `(output_values, pattern, jacobian_values)`.
    pub fn sparse_jacobian(&mut self, x: &[F]) -> (Vec<F>, JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (col_colors, num_col_colors) = column_coloring(&pattern);
        let (row_colors, num_row_colors) = row_coloring(&pattern);

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
    ) -> (Vec<F>, JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = column_coloring(&pattern);
        let jac_values = self.sparse_jacobian_forward_impl(x, &pattern, &colors, num_colors);
        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Jacobian via reverse-mode (row compression).
    pub fn sparse_jacobian_reverse(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = row_coloring(&pattern);
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
        pattern: &JacobianSparsityPattern,
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
        pattern: &JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

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
        pattern: &JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let m = self.num_outputs();
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

        let mut seeds: Vec<F> = Vec::with_capacity(m);
        for color in 0..num_colors {
            // Build seeds: weight = 1 for outputs with this color
            seeds.clear();
            seeds.extend((0..m).map(|i| {
                if colors[i] == color {
                    F::one()
                } else {
                    F::zero()
                }
            }));

            let adjoints = self.reverse_seeded_full(&seeds, out_indices);

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

    /// Batched sparse Jacobian: packs N colors per forward sweep using DualVec.
    ///
    /// Reduces the number of forward sweeps from `num_colors` to
    /// `ceil(num_colors / N)`.
    pub fn sparse_jacobian_vec<const N: usize>(
        &mut self,
        x: &[F],
    ) -> (Vec<F>, JacobianSparsityPattern, Vec<F>) {
        assert!(
            self.custom_ops.is_empty(),
            "sparse_jacobian_vec: custom ops produce approximate (first-order) derivatives; \
             use eval_forward with Dual<F> for exact Jacobians through custom ops"
        );
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = column_coloring(&pattern);

        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();

        let num_batches = (num_colors as usize).div_ceil(N);
        for batch in 0..num_batches {
            let base_color = (batch * N) as u32;

            dual_input_buf.clear();
            dual_input_buf.extend(
                (0..n).map(|i| {
                    DualVec::new(x[i], color_seed_eps(i, &colors, base_color, num_colors))
                }),
            );

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
                let col_color = colors[col as usize];
                if col_color >= base_color && col_color < base_color + N as u32 {
                    let lane = (col_color - base_color) as usize;
                    jac_values[k] = dual_vals_buf[out_indices[row as usize] as usize].eps[lane];
                }
            }
        }

        let outputs = self.output_values();
        (outputs, pattern, jac_values)
    }

    /// Sparse Hessian with a precomputed sparsity pattern and coloring.
    ///
    /// Skips re-detection on repeated calls (e.g. in solver loops).
    pub fn sparse_hessian_with_pattern(
        &self,
        x: &[F],
        pattern: &SparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> (F, Vec<F>, Vec<F>) {
        self.assert_scalar_output("sparse_hessian_with_pattern");
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let hessian_values = vec![F::zero(); pattern.nnz()];
        let gradient = vec![F::zero(); n];
        let mut value = F::zero();

        // Constant-output tape (n == 0): no colors to sweep.
        if n == 0 {
            return (self.constant_output_value(), gradient, hessian_values);
        }

        let mut hessian_values = hessian_values;
        let mut gradient = gradient;

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
}

// ══════════════════════════════════════════════
//  Sparsity detection (tape walkers)
// ══════════════════════════════════════════════

/// Internal sparsity detection implementation.
///
/// Walks the tape forward propagating input-dependency bitsets.
/// At nonlinear operations, marks cross-pairs as potential Hessian interactions.
fn detect_sparsity_impl(
    opcodes: &[OpCode],
    arg_indices: &[[u32; 2]],
    custom_second_args: &HashMap<u32, u32>,
    num_inputs: usize,
    num_vars: usize,
) -> SparsityPattern {
    let num_words = num_inputs.div_ceil(64);

    // Dependency bitsets: deps[node] = set of input variables this node depends on
    let mut deps: Vec<Vec<u64>> = vec![vec![0u64; num_words]; num_vars];

    // Interaction pairs as Vec (more cache-friendly than HashSet for typical problems).
    // Deduplicated via sort + dedup at the end.
    let mut interactions: Vec<(u32, u32)> = Vec::new();

    // Push every unordered pair within `bits` (including self-pairs),
    // normalized to lower-triangle (row >= col) order.
    fn push_upper_pairs(out: &mut Vec<(u32, u32)>, bits: &[u32]) {
        for ii in 0..bits.len() {
            for jj in 0..=ii {
                let (r, c) = if bits[ii] >= bits[jj] {
                    (bits[ii], bits[jj])
                } else {
                    (bits[jj], bits[ii])
                };
                out.push((r, c));
            }
        }
    }

    // Push every cross pair between two dependency sets, normalized the
    // same way.
    fn push_cross_pairs(out: &mut Vec<(u32, u32)>, bits_a: &[u32], bits_b: &[u32]) {
        for &va in bits_a {
            for &vb in bits_b {
                let (r, c) = if va >= vb { (va, vb) } else { (vb, va) };
                out.push((r, c));
            }
        }
    }

    let mut input_idx = 0u32;
    for i in 0..opcodes.len() {
        match opcodes[i] {
            OpCode::Input => {
                // Set bit for this input
                let word = input_idx as usize / 64;
                let bit = input_idx as usize % 64;
                deps[i][word] |= 1u64 << bit;
                input_idx += 1;
            }
            OpCode::Const => {
                // No dependencies
            }
            op => {
                let [a_idx, b_idx] = arg_indices[i];
                let a = a_idx as usize;

                match classify_op(op) {
                    OpClass::Linear => {
                        // Union dependencies, no new interactions
                        union_into(&mut deps, i, a);
                        if is_binary_op(op) && b_idx != UNUSED {
                            union_into(&mut deps, i, b_idx as usize);
                        }
                    }
                    OpClass::UnaryNonlinear => {
                        // Union dependencies + mark all cross-pairs within dep set
                        union_into(&mut deps, i, a);
                        // extract_bits copies the set-bit positions into an
                        // owned Vec, so the pair loop below borrows nothing
                        // from deps.
                        let bits = extract_bits(&deps[i], num_inputs);
                        push_upper_pairs(&mut interactions, &bits);
                    }
                    OpClass::BinaryNonlinear => {
                        // For Custom ops, arg_indices[i][1] is the callback index,
                        // NOT a tape index. The real second operand (if any) is in
                        // custom_second_args. Unary custom ops have no second operand.
                        let real_b = if op == OpCode::Custom {
                            custom_second_args.get(&(i as u32)).map(|&v| v as usize)
                        } else {
                            Some(b_idx as usize)
                        };

                        if let Some(b) = real_b {
                            // Binary: full cross-pair + within-operand analysis
                            let bits_a = extract_bits(&deps[a], num_inputs);
                            let bits_b = extract_bits(&deps[b], num_inputs);
                            union_into(&mut deps, i, a);
                            union_into(&mut deps, i, b);
                            // Cross-pairs between operand dependency sets
                            push_cross_pairs(&mut interactions, &bits_a, &bits_b);
                            // Within-operand second derivatives (non-Mul ops)
                            if op != OpCode::Mul {
                                push_upper_pairs(&mut interactions, &bits_a);
                                push_upper_pairs(&mut interactions, &bits_b);
                            }
                        } else {
                            // Unary custom op: treat as UnaryNonlinear
                            union_into(&mut deps, i, a);
                            let bits = extract_bits(&deps[i], num_inputs);
                            push_upper_pairs(&mut interactions, &bits);
                        }
                    }
                    OpClass::ZeroDerivative => {
                        // Propagate dependencies for downstream ops
                        union_into(&mut deps, i, a);
                        if is_binary_op(op) && b_idx != UNUSED {
                            union_into(&mut deps, i, b_idx as usize);
                        }
                    }
                }
            }
        }
    }

    // Convert interactions to sorted, deduplicated COO format
    interactions.sort_unstable();
    interactions.dedup();
    let entries = interactions;

    let (rows, cols): (Vec<u32>, Vec<u32>) = entries.iter().copied().unzip();

    SparsityPattern {
        dim: num_inputs,
        rows,
        cols,
    }
}

#[derive(Debug, Clone, Copy)]
enum OpClass {
    Linear,
    UnaryNonlinear,
    BinaryNonlinear,
    ZeroDerivative,
}

fn classify_op(op: OpCode) -> OpClass {
    match op {
        // Linear: derivative is constant w.r.t. inputs
        OpCode::Add | OpCode::Sub | OpCode::Neg | OpCode::Fract => OpClass::Linear,

        // Unary nonlinear: second derivatives exist
        OpCode::Recip
        | OpCode::Sqrt
        | OpCode::Cbrt
        | OpCode::Powi
        | OpCode::Exp
        | OpCode::Exp2
        | OpCode::ExpM1
        | OpCode::Ln
        | OpCode::Log2
        | OpCode::Log10
        | OpCode::Ln1p
        | OpCode::Sin
        | OpCode::Cos
        | OpCode::Tan
        | OpCode::Asin
        | OpCode::Acos
        | OpCode::Atan
        | OpCode::Sinh
        | OpCode::Cosh
        | OpCode::Tanh
        | OpCode::Asinh
        | OpCode::Acosh
        | OpCode::Atanh => OpClass::UnaryNonlinear,

        // Binary nonlinear: cross-derivatives between operands
        OpCode::Mul | OpCode::Div | OpCode::Powf | OpCode::Atan2 | OpCode::Hypot => {
            OpClass::BinaryNonlinear
        }

        // Zero derivative (piecewise constant or discontinuous).
        // Abs is included here: d²|x|/dx² = 0 a.e. (the kink at zero has measure
        // zero and does not contribute structural Hessian entries).
        OpCode::Abs
        | OpCode::Signum
        | OpCode::Floor
        | OpCode::Ceil
        | OpCode::Round
        | OpCode::Trunc
        | OpCode::Max
        | OpCode::Min
        | OpCode::Rem => OpClass::ZeroDerivative,

        // Custom ops are conservatively treated as binary nonlinear
        OpCode::Custom => OpClass::BinaryNonlinear,

        OpCode::Input | OpCode::Const => unreachable!(),
    }
}

fn is_binary_op(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Rem
            | OpCode::Powf
            | OpCode::Atan2
            | OpCode::Hypot
            | OpCode::Max
            | OpCode::Min
            | OpCode::Custom
    )
}

/// Union deps[src] into deps[dst] without cloning, using `split_at_mut`.
fn union_into(deps: &mut [Vec<u64>], dst: usize, src: usize) {
    if dst == src {
        return;
    }
    let (a, b) = if dst < src {
        let (left, right) = deps.split_at_mut(src);
        (&mut left[dst], &right[0] as &[u64])
    } else {
        let (left, right) = deps.split_at_mut(dst);
        (&mut right[0], &left[src] as &[u64])
    };
    for w in 0..a.len() {
        a[w] |= b[w];
    }
}

/// Extract set bit positions from a bitset.
fn extract_bits(bitset: &[u64], max_bits: usize) -> Vec<u32> {
    let mut result = Vec::new();
    for (word_idx, &word) in bitset.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros();
            let pos = word_idx * 64 + bit as usize;
            if pos < max_bits {
                result.push(pos as u32);
            }
            w &= w - 1; // Clear lowest set bit
        }
    }
    result
}

/// Detect Jacobian sparsity by forward-propagating input-dependency bitsets.
///
/// For each output, determines which inputs it depends on (first-order only).
/// All ops propagate dependencies — unlike Hessian sparsity, linear ops matter here.
fn detect_jacobian_sparsity_impl(
    opcodes: &[OpCode],
    arg_indices: &[[u32; 2]],
    custom_second_args: &HashMap<u32, u32>,
    num_inputs: usize,
    num_vars: usize,
    output_indices: &[u32],
) -> JacobianSparsityPattern {
    let num_words = num_inputs.div_ceil(64);

    // Dependency bitsets: deps[node] = set of input variables this node depends on
    let mut deps: Vec<Vec<u64>> = vec![vec![0u64; num_words]; num_vars];

    let mut input_idx = 0u32;
    for i in 0..opcodes.len() {
        match opcodes[i] {
            OpCode::Input => {
                let word = input_idx as usize / 64;
                let bit = input_idx as usize % 64;
                deps[i][word] |= 1u64 << bit;
                input_idx += 1;
            }
            OpCode::Const => {
                // No dependencies
            }
            op => {
                let [a_idx, b_idx] = arg_indices[i];
                let a = a_idx as usize;
                // Union dependencies from all operands (first-order: all ops propagate)
                union_into(&mut deps, i, a);
                // For Custom ops, the real second operand is in custom_second_args,
                // not in arg_indices[i][1] (which is the callback index).
                if op == OpCode::Custom {
                    if let Some(&real_b) = custom_second_args.get(&(i as u32)) {
                        union_into(&mut deps, i, real_b as usize);
                    }
                } else if is_binary_op(op) && b_idx != UNUSED {
                    union_into(&mut deps, i, b_idx as usize);
                }
            }
        }
    }

    // Extract COO entries from output dependency sets
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for (out_row, &out_idx) in output_indices.iter().enumerate() {
        let bits = extract_bits(&deps[out_idx as usize], num_inputs);
        for input_col in bits {
            rows.push(out_row as u32);
            cols.push(input_col);
        }
    }

    JacobianSparsityPattern {
        num_outputs: output_indices.len(),
        num_inputs,
        rows,
        cols,
    }
}

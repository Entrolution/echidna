use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::Float;

impl<F: Float> super::BytecodeTape<F> {
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
        let out_indices = self.all_output_indices();
        crate::sparse::detect_jacobian_sparsity_impl(
            &self.opcodes,
            &self.arg_indices,
            self.num_inputs as usize,
            self.num_variables as usize,
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
        pattern: &crate::sparse::JacobianSparsityPattern,
        colors: &[u32],
        num_colors: u32,
    ) -> Vec<F> {
        let m = self.num_outputs();
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

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
    ) -> (Vec<F>, crate::sparse::JacobianSparsityPattern, Vec<F>) {
        self.forward(x);
        let pattern = self.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);

        let n = self.num_inputs as usize;
        let mut jac_values = vec![F::zero(); pattern.nnz()];

        let out_indices = self.all_output_indices();

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
}

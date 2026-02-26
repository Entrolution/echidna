use rayon::prelude::*;

use crate::dual::Dual;
use crate::float::Float;

impl<F: Float> super::BytecodeTape<F> {
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
        let mut values_buf = Vec::new();
        self.forward_into(inputs, &mut values_buf);

        let out_indices = self.all_output_indices();

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
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        // Compute gradient and value from column 0 (serial).
        let dual_input_buf: Vec<Dual<F>> = (0..n)
            .map(|i| Dual::new(x[i], if i == 0 { F::one() } else { F::zero() }))
            .collect();
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.forward_tangent_dual(&dual_input_buf, &mut dual_vals_buf);
        self.reverse_tangent_dual(&dual_vals_buf, &mut adjoint_buf);

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
                self.forward_tangent_dual(&inputs, &mut dv);
                self.reverse_tangent_dual(&dv, &mut ab);
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
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let pattern = self.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        // Compute value/gradient from color 0 (serial).
        let mut v0 = vec![F::zero(); n];
        for i in 0..n {
            v0[i] = if colors[i] == 0 { F::one() } else { F::zero() };
        }
        let di: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], v0[i])).collect();
        let mut dv = Vec::new();
        let mut ab = Vec::new();
        self.forward_tangent_dual(&di, &mut dv);
        self.reverse_tangent_dual(&dv, &mut ab);
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
                let inputs: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], v[i])).collect();
                let mut dv_local = Vec::new();
                let mut ab_local = Vec::new();
                self.forward_tangent_dual(&inputs, &mut dv_local);
                self.reverse_tangent_dual(&dv_local, &mut ab_local);
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
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut values_buf = Vec::new();
        self.forward_into(x, &mut values_buf);

        let out_indices = self.all_output_indices();
        let m = out_indices.len();
        let outputs: Vec<F> = out_indices
            .iter()
            .map(|&oi| values_buf[oi as usize])
            .collect();

        let jac_pattern = self.detect_jacobian_sparsity();
        let ni = self.num_inputs as usize;

        if m <= n {
            // Row compression (reverse mode)
            let (row_colors, num_colors) = crate::sparse::row_coloring(&jac_pattern);

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

                    self.reverse_sweep_core(&mut adjoints, &values_buf, None);
                    adjoints[..ni].to_vec()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in jac_pattern
                .rows
                .iter()
                .zip(jac_pattern.cols.iter())
                .enumerate()
            {
                let color = row_colors[row as usize] as usize;
                jac_values[k] = color_results[color][col as usize];
            }

            (outputs, jac_pattern, jac_values)
        } else {
            // Column compression (forward mode) — parallelize forward tangent sweeps
            let (col_colors, num_colors) = crate::sparse::column_coloring(&jac_pattern);

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
                    let inputs: Vec<Dual<F>> = (0..n).map(|i| Dual::new(x[i], dir[i])).collect();
                    let mut dv = Vec::new();
                    self.forward_tangent(&inputs, &mut dv);
                    out_indices.iter().map(|&oi| dv[oi as usize].eps).collect()
                })
                .collect();

            let mut jac_values = vec![F::zero(); jac_pattern.nnz()];
            for (k, (&row, &col)) in jac_pattern
                .rows
                .iter()
                .zip(jac_pattern.cols.iter())
                .enumerate()
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
        inputs.par_iter().map(|x| self.hessian_par(x)).collect()
    }
}

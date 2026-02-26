use std::collections::HashMap;

use crate::dual::Dual;
use crate::float::Float;
use crate::opcode;

impl<F: Float> super::BytecodeTape<F> {
    /// Compute the full Jacobian of a multi-output tape via reverse mode.
    ///
    /// Performs `m` reverse sweeps (one per output). Returns `J[i][j] = ∂f_i/∂x_j`.
    pub fn jacobian(&mut self, inputs: &[F]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let out_indices = self.all_output_indices();

        let ni = self.num_inputs as usize;
        let mut jac = Vec::with_capacity(out_indices.len());

        for &out_idx in out_indices {
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

    /// Compute a Jacobian with forced branch choices at specified tape indices.
    ///
    /// For each `(tape_index, sign)` in `forced_signs`, the reverse sweep uses
    /// [`forced_reverse_partials`](opcode::forced_reverse_partials) instead of the
    /// standard partials at that index.
    ///
    /// This is the building block for Clarke subdifferential enumeration.
    pub fn jacobian_limiting(&mut self, inputs: &[F], forced_signs: &[(u32, i8)]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let sign_map: HashMap<u32, i8> = forced_signs.iter().copied().collect();
        let out_indices = self.all_output_indices();

        let ni = self.num_inputs as usize;
        let mut jac = Vec::with_capacity(out_indices.len());

        for &out_idx in out_indices {
            let adjoints = self.reverse_with_forced_signs(out_idx, &sign_map);
            jac.push(adjoints[..ni].to_vec());
        }

        jac
    }

    /// Compute the Clarke generalized Jacobian via limiting Jacobian enumeration.
    ///
    /// 1. Runs `forward_nonsmooth` to detect all kink operations and their branches.
    /// 2. Identifies "active" kinks (|switching_value| < `tol`).
    /// 3. Enumerates all 2^k sign combinations for the k active kinks.
    /// 4. For each combination, computes a limiting Jacobian via forced reverse sweeps.
    ///
    /// Returns the nonsmooth info and a vector of limiting Jacobians.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ClarkeError::TooManyKinks`] if the number of active kinks exceeds
    /// the limit (default 20, overridden by `max_active_kinks`).
    pub fn clarke_jacobian(
        &mut self,
        inputs: &[F],
        tol: F,
        max_active_kinks: Option<usize>,
    ) -> Result<(crate::nonsmooth::NonsmoothInfo<F>, Vec<Vec<Vec<F>>>), crate::nonsmooth::ClarkeError>
    {
        #![allow(clippy::type_complexity)]
        let info = self.forward_nonsmooth(inputs);
        let active: Vec<&crate::nonsmooth::KinkEntry<F>> = info
            .active_kinks(tol)
            .into_iter()
            .filter(|k| opcode::has_nontrivial_subdifferential(k.opcode))
            .collect();
        let k = active.len();
        let limit = max_active_kinks.unwrap_or(20);

        if k > limit {
            return Err(crate::nonsmooth::ClarkeError::TooManyKinks { count: k, limit });
        }

        let active_indices: Vec<u32> = active.iter().map(|e| e.tape_index).collect();

        // Build sign_map from all (non-active) kinks using their natural branches,
        // then override active kinks per combination.
        let base_signs: HashMap<u32, i8> = info
            .kinks
            .iter()
            .map(|e| (e.tape_index, e.branch))
            .collect();

        let out_indices = self.all_output_indices();
        let ni = self.num_inputs as usize;

        let num_combos = 1usize << k;
        let mut jacobians = Vec::with_capacity(num_combos);

        for combo in 0..num_combos {
            let mut sign_map = base_signs.clone();
            for (bit, &idx) in active_indices.iter().enumerate() {
                let sign: i8 = if (combo >> bit) & 1 == 0 { 1 } else { -1 };
                sign_map.insert(idx, sign);
            }

            let mut jac = Vec::with_capacity(out_indices.len());
            for &out_idx in out_indices {
                let adjoints = self.reverse_with_forced_signs(out_idx, &sign_map);
                jac.push(adjoints[..ni].to_vec());
            }
            jacobians.push(jac);
        }

        Ok((info, jacobians))
    }

    /// Dense Jacobian via forward mode (one forward-tangent pass per input).
    ///
    /// More efficient than reverse mode when `num_inputs < num_outputs`.
    pub fn jacobian_forward(&self, x: &[F]) -> Vec<Vec<F>> {
        let n = self.num_inputs as usize;

        let out_indices = self.all_output_indices();
        let m = out_indices.len();

        let mut jac = vec![vec![F::zero(); n]; m];
        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<Dual<F>> = Vec::new();

        // Indexing by `col` is clearer than enumerate here: col seeds the tangent direction
        #[allow(clippy::needless_range_loop)]
        for col in 0..n {
            dual_input_buf.clear();
            dual_input_buf.extend(
                (0..n).map(|i| Dual::new(x[i], if i == col { F::one() } else { F::zero() })),
            );

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);

            for (row_idx, &out_idx) in out_indices.iter().enumerate() {
                jac[row_idx][col] = dual_vals_buf[out_idx as usize].eps;
            }
        }

        jac
    }

    /// Dense Jacobian via cross-country (vertex) elimination.
    ///
    /// Builds a linearized DAG from the tape, then eliminates intermediate
    /// vertices in Markowitz order. For functions where `m ≈ n` and the
    /// graph has moderate connectivity, this can require fewer operations
    /// than either pure forward mode (`n` passes) or reverse mode (`m` passes).
    pub fn jacobian_cross_country(&mut self, inputs: &[F]) -> Vec<Vec<F>> {
        self.forward(inputs);

        let out_indices = self.all_output_indices();

        let mut graph = crate::cross_country::LinearizedGraph::from_tape(
            &self.opcodes,
            &self.arg_indices,
            &self.values,
            self.num_inputs as usize,
            out_indices,
            &self.custom_ops,
            &self.custom_second_args,
        );

        graph.eliminate_all();
        graph.extract_jacobian()
    }
}

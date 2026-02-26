use std::collections::HashMap;

use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

impl<F: Float> super::BytecodeTape<F> {
    /// Reverse sweep with weighted seeds for multiple outputs.
    ///
    /// Computes `∑_i weights[i] * ∂output_i/∂x` — a vector-Jacobian product.
    ///
    /// Returns the gradient with respect to all inputs (length [`num_inputs`](Self::num_inputs)).
    pub fn reverse_seeded(&self, seeds: &[F]) -> Vec<F> {
        let out_indices = self.all_output_indices();

        assert_eq!(
            seeds.len(),
            out_indices.len(),
            "seeds length must match number of outputs"
        );

        let ni = self.num_inputs as usize;
        let adjoints = self.reverse_seeded_full(seeds, out_indices);
        adjoints[..ni].to_vec()
    }

    /// Core reverse sweep loop shared by all scalar reverse sweep variants.
    ///
    /// Expects `adjoints` to be pre-seeded by the caller (length = `num_variables`).
    /// Reads primal values from `values` (either `self.values` or an external buffer).
    /// When `forced_signs` is `Some`, uses forced partials at matching tape indices.
    pub(super) fn reverse_sweep_core(
        &self,
        adjoints: &mut [F],
        values: &[F],
        forced_signs: Option<&HashMap<u32, i8>>,
    ) {
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
                    let b = b_idx_opt.map(|bi| values[bi as usize]).unwrap_or(F::zero());
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

                    let (da, db) = match forced_signs.and_then(|fs| fs.get(&(i as u32))) {
                        Some(&sign) => opcode::forced_reverse_partials(op, a, b, r, sign),
                        None => opcode::reverse_partials(op, a, b, r),
                    };

                    adjoints[a_idx as usize] = adjoints[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        adjoints[b_idx as usize] = adjoints[b_idx as usize] + db * adj;
                    }
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
        self.reverse_sweep_core(&mut adjoints, &self.values, None);
        adjoints
    }

    /// Reverse sweep with forced branch choices at specified tape indices.
    pub(super) fn reverse_with_forced_signs(
        &self,
        seed_index: u32,
        forced_signs: &HashMap<u32, i8>,
    ) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];
        adjoints[seed_index as usize] = F::one();
        self.reverse_sweep_core(&mut adjoints, &self.values, Some(forced_signs));
        adjoints
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
        self.reverse_sweep_core(&mut adjoints, values, None);
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

        self.reverse_sweep_core(adjoint_buf, &self.values, None);
        adjoint_buf[..self.num_inputs as usize].to_vec()
    }

    /// Reverse sweep with weighted seeds, returning full adjoint vector.
    pub(super) fn reverse_seeded_full(&self, seeds: &[F], out_indices: &[u32]) -> Vec<F> {
        let n = self.num_variables as usize;
        let mut adjoints = vec![F::zero(); n];

        for (&out_idx, &weight) in out_indices.iter().zip(seeds.iter()) {
            if weight == F::zero() {
                continue;
            }
            adjoints[out_idx as usize] = adjoints[out_idx as usize] + weight;
        }

        self.reverse_sweep_core(&mut adjoints, &self.values, None);
        adjoints
    }

    // ── Batch evaluation ──

    /// Evaluate the gradient at multiple input points.
    ///
    /// Returns one gradient vector per input point.
    pub fn gradient_batch(&mut self, inputs: &[&[F]]) -> Vec<Vec<F>> {
        inputs.iter().map(|x| self.gradient(x)).collect()
    }
}

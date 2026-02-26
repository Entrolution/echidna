use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

impl<F: Float> super::BytecodeTape<F> {
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

    /// Forward sweep with nonsmooth branch tracking.
    ///
    /// Calls [`forward`](Self::forward) to evaluate the tape, then scans for
    /// nonsmooth operations and records which branch was taken at each one.
    ///
    /// Tracked operations:
    /// - `Abs`, `Min`, `Max` — kinks with nontrivial subdifferentials
    /// - `Signum`, `Floor`, `Ceil`, `Round`, `Trunc` — step-function
    ///   discontinuities (zero derivative on both sides, tracked for proximity
    ///   detection only)
    ///
    /// Returns [`crate::NonsmoothInfo`] containing all kink entries in tape order.
    pub fn forward_nonsmooth(&mut self, inputs: &[F]) -> crate::nonsmooth::NonsmoothInfo<F> {
        self.forward(inputs);

        let mut kinks = Vec::new();
        for i in 0..self.opcodes.len() {
            let op = self.opcodes[i];
            if !opcode::is_nonsmooth(op) {
                continue;
            }

            let [a_idx, b_idx] = self.arg_indices[i];
            let a = self.values[a_idx as usize];

            match op {
                OpCode::Abs => {
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a,
                        branch: if a >= F::zero() { 1 } else { -1 },
                    });
                }
                OpCode::Max => {
                    let b = self.values[b_idx as usize];
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - b,
                        branch: if a >= b { 1 } else { -1 },
                    });
                }
                OpCode::Min => {
                    let b = self.values[b_idx as usize];
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - b,
                        branch: if a <= b { 1 } else { -1 },
                    });
                }
                OpCode::Signum => {
                    // Kink at x = 0 (same as Abs).
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a,
                        branch: if a >= F::zero() { 1 } else { -1 },
                    });
                }
                OpCode::Floor | OpCode::Ceil | OpCode::Round | OpCode::Trunc => {
                    // Kink at integer values. switching_value = distance to
                    // nearest integer: zero exactly at kink points, works
                    // symmetrically for both approach directions.
                    kinks.push(crate::nonsmooth::KinkEntry {
                        tape_index: i as u32,
                        opcode: op,
                        switching_value: a - a.round(),
                        branch: if a - a.floor() < F::from(0.5).unwrap() {
                            1
                        } else {
                            -1
                        },
                    });
                }
                _ => unreachable!(),
            }
        }

        crate::nonsmooth::NonsmoothInfo { kinks }
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
}

use std::collections::HashMap;

use crate::float::Float;
use crate::opcode::{OpCode, UNUSED};

impl<F: Float> super::BytecodeTape<F> {
    /// Eliminate dead (unreachable) entries from the tape.
    ///
    /// Walks backward from all outputs, marks reachable entries, then compacts
    /// the tape in-place with an index remap. Inputs are never removed.
    /// Core DCE: reachability walk from `seeds`, compact tape, return index remap.
    ///
    /// Shared by [`dead_code_elimination`](Self::dead_code_elimination) and
    /// [`dead_code_elimination_for_outputs`](Self::dead_code_elimination_for_outputs).
    /// Callers handle output index updates after compaction.
    fn dce_compact(&mut self, seeds: &[u32]) -> Vec<u32> {
        let n = self.opcodes.len();
        let mut reachable = vec![false; n];

        // Mark all inputs as reachable.
        for flag in reachable.iter_mut().take(self.num_inputs as usize) {
            *flag = true;
        }

        let mut stack: Vec<u32> = seeds.to_vec();

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
                let ra = if a != UNUSED {
                    remap[a as usize]
                } else {
                    UNUSED
                };
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

        remap
    }

    pub fn dead_code_elimination(&mut self) {
        let mut seeds = vec![self.output_index];
        seeds.extend_from_slice(&self.output_indices);
        let remap = self.dce_compact(&seeds);
        self.output_index = remap[self.output_index as usize];
        for oi in &mut self.output_indices {
            *oi = remap[*oi as usize];
        }
    }

    /// Eliminate dead code, keeping only the specified outputs alive.
    ///
    /// Like [`dead_code_elimination`](Self::dead_code_elimination) but seeds
    /// reachability only from `active_outputs`. After compaction,
    /// `output_indices` contains only the active outputs (remapped), and
    /// `output_index` is set to the first active output.
    ///
    /// **Note**: Like `dead_code_elimination`, this does not remap
    /// `custom_second_args` (pre-existing limitation).
    ///
    /// # Panics
    /// Panics if `active_outputs` is empty.
    pub fn dead_code_elimination_for_outputs(&mut self, active_outputs: &[u32]) {
        assert!(
            !active_outputs.is_empty(),
            "active_outputs must not be empty"
        );
        let remap = self.dce_compact(active_outputs);
        self.output_indices = active_outputs
            .iter()
            .map(|&oi| remap[oi as usize])
            .collect();
        self.output_index = self.output_indices[0];
    }

    /// Common subexpression elimination.
    ///
    /// Deduplicates identical `(OpCode, arg0, arg1)` triples, normalising
    /// argument order for commutative ops. Finishes with a DCE pass to
    /// remove the now-dead duplicates.
    pub fn cse(&mut self) {
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

    /// Run all tape optimizations: CSE followed by DCE.
    ///
    /// In debug builds, validates internal consistency after optimization.
    pub fn optimize(&mut self) {
        self.cse();
        self.dead_code_elimination();

        // Validate internal consistency in debug builds.
        #[cfg(debug_assertions)]
        {
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
            let input_count = self
                .opcodes
                .iter()
                .filter(|&&op| op == OpCode::Input)
                .count();
            assert_eq!(
                input_count, self.num_inputs as usize,
                "num_inputs mismatch after optimization"
            );
        }
    }
}

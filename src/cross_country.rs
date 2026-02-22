//! Cross-country elimination for Jacobian computation.
//!
//! Implements vertex elimination on the linearized computational graph
//! (Griewank & Walther, Chapter 10). Intermediate vertices are removed
//! in Markowitz order, producing fill-in edges that accumulate partial
//! derivatives. After all intermediates are eliminated, remaining edges
//! connect inputs to outputs directly, yielding the full Jacobian.

use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use crate::bytecode_tape::CustomOp;
use crate::float::Float;
use crate::opcode::{self, OpCode, UNUSED};

/// Linearized DAG for cross-country elimination.
///
/// Each node corresponds to a tape entry. Edges carry local partial
/// derivative weights. Intermediate nodes are eliminated in Markowitz
/// order to compute the Jacobian.
pub(crate) struct LinearizedGraph<F: Float> {
    num_inputs: usize,
    output_indices: Vec<u32>,
    /// preds[v] = [(predecessor_index, edge_weight), ...]
    preds: Vec<Vec<(u32, F)>>,
    /// succs[v] = [(successor_index, edge_weight), ...]
    succs: Vec<Vec<(u32, F)>>,
    /// true if this node is eligible for elimination
    is_intermediate: Vec<bool>,
}

impl<F: Float> LinearizedGraph<F> {
    /// Build the linearized graph from tape data.
    ///
    /// Walks the tape in topological order, computing local partial
    /// derivatives via `reverse_partials` and constructing weighted edges.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_tape(
        opcodes: &[OpCode],
        arg_indices: &[[u32; 2]],
        values: &[F],
        num_inputs: usize,
        output_indices: &[u32],
        custom_ops: &[Arc<dyn CustomOp<F>>],
        custom_second_args: &HashMap<u32, u32>,
    ) -> Self {
        let n = opcodes.len();
        let mut preds: Vec<Vec<(u32, F)>> = vec![Vec::new(); n];
        let mut succs: Vec<Vec<(u32, F)>> = vec![Vec::new(); n];

        let zero = F::zero();

        for i in 0..n {
            match opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let [a_idx, cb_idx] = arg_indices[i];
                    let a = values[a_idx as usize];
                    let b_idx_opt = custom_second_args.get(&(i as u32));
                    let b = b_idx_opt
                        .map(|&bi| values[bi as usize])
                        .unwrap_or(zero);
                    let r = values[i];
                    let (da, db) = custom_ops[cb_idx as usize].partials(a, b, r);

                    if opcodes[a_idx as usize] != OpCode::Const && da != zero {
                        preds[i].push((a_idx, da));
                        succs[a_idx as usize].push((i as u32, da));
                    }
                    if let Some(&bi) = b_idx_opt {
                        if opcodes[bi as usize] != OpCode::Const && db != zero {
                            preds[i].push((bi, db));
                            succs[bi as usize].push((i as u32, db));
                        }
                    }
                }
                op => {
                    let [a_idx, b_idx] = arg_indices[i];
                    let a = values[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        values[b_idx as usize]
                    } else if op == OpCode::Powi {
                        F::from(b_idx).unwrap_or(zero)
                    } else {
                        zero
                    };
                    let r = values[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    // Edge from first argument
                    if opcodes[a_idx as usize] != OpCode::Const && da != zero {
                        preds[i].push((a_idx, da));
                        succs[a_idx as usize].push((i as u32, da));
                    }

                    // Edge from second argument (binary ops only, not Powi)
                    if b_idx != UNUSED
                        && op != OpCode::Powi
                        && opcodes[b_idx as usize] != OpCode::Const
                        && db != zero
                    {
                        preds[i].push((b_idx, db));
                        succs[b_idx as usize].push((i as u32, db));
                    }
                }
            }
        }

        // Classify nodes: intermediate iff not input, not const, not output
        let mut is_intermediate = vec![false; n];
        for i in 0..n {
            is_intermediate[i] = i >= num_inputs
                && opcodes[i] != OpCode::Const
                && !output_indices.contains(&(i as u32));
        }

        LinearizedGraph {
            num_inputs,
            output_indices: output_indices.to_vec(),
            preds,
            succs,
            is_intermediate,
        }
    }

    /// Accumulate an edge weight into an adjacency list.
    ///
    /// If an entry for `target` already exists, adds `weight` to it.
    /// Otherwise pushes a new entry.
    fn accumulate_edge(adj: &mut Vec<(u32, F)>, target: u32, weight: F) {
        for entry in adj.iter_mut() {
            if entry.0 == target {
                entry.1 = entry.1 + weight;
                return;
            }
        }
        adj.push((target, weight));
    }

    /// Eliminate a single intermediate vertex, creating fill-in edges
    /// between all predecessor–successor pairs.
    fn eliminate_vertex(&mut self, v: usize) {
        let preds_v = mem::take(&mut self.preds[v]);
        let succs_v = mem::take(&mut self.succs[v]);

        let v_u32 = v as u32;

        // Create fill-in edges for each (predecessor, successor) pair
        for &(u, w_uv) in &preds_v {
            for &(w, w_vw) in &succs_v {
                let fill = w_uv * w_vw;
                if fill != F::zero() {
                    Self::accumulate_edge(&mut self.succs[u as usize], w, fill);
                    Self::accumulate_edge(&mut self.preds[w as usize], u, fill);
                }
            }
        }

        // Remove v from predecessors' successor lists
        for &(u, _) in &preds_v {
            self.succs[u as usize].retain(|&(t, _)| t != v_u32);
        }

        // Remove v from successors' predecessor lists
        for &(w, _) in &succs_v {
            self.preds[w as usize].retain(|&(s, _)| s != v_u32);
        }

        self.is_intermediate[v] = false;
    }

    /// Find the intermediate vertex with the smallest Markowitz cost
    /// (|predecessors| × |successors|). Ties broken by smallest index.
    fn find_min_markowitz(&self) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None; // (index, cost)

        for (v, &is_inter) in self.is_intermediate.iter().enumerate() {
            if !is_inter {
                continue;
            }
            let cost = self.preds[v].len() * self.succs[v].len();
            match best {
                None => best = Some((v, cost)),
                Some((_, best_cost)) if cost < best_cost => {
                    best = Some((v, cost));
                }
                _ => {}
            }
        }

        best.map(|(v, _)| v)
    }

    /// Eliminate all intermediate vertices in Markowitz order.
    pub(crate) fn eliminate_all(&mut self) {
        while let Some(v) = self.find_min_markowitz() {
            self.eliminate_vertex(v);
        }
    }

    /// Extract the m×n Jacobian matrix after all intermediates are eliminated.
    ///
    /// Remaining edges connect inputs to outputs directly.
    pub(crate) fn extract_jacobian(&self) -> Vec<Vec<F>> {
        let m = self.output_indices.len();
        let n = self.num_inputs;
        let mut jac = vec![vec![F::zero(); n]; m];

        for (row, &out_idx) in self.output_indices.iter().enumerate() {
            let out = out_idx as usize;

            // If the output IS an input, add the identity contribution
            if out < n {
                jac[row][out] = jac[row][out] + F::one();
            }

            // Accumulate remaining edges from input predecessors
            for &(pred_idx, weight) in &self.preds[out] {
                let p = pred_idx as usize;
                debug_assert!(
                    p < n,
                    "non-input predecessor {} remains after elimination",
                    p
                );
                if p < n {
                    jac[row][p] = jac[row][p] + weight;
                }
            }
        }

        jac
    }
}

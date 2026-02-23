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
                    let b = b_idx_opt.map(|&bi| values[bi as usize]).unwrap_or(zero);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small graph manually for testing.
    ///
    /// Layout: 2 inputs (nodes 0, 1), 1 intermediate (node 2), 1 output (node 3).
    ///
    /// ```text
    ///  0 ──(2.0)──▶ 2 ──(3.0)──▶ 3 (output)
    ///  1 ──(5.0)──▶ 2
    /// ```
    ///
    /// So df3/dx0 = 2*3 = 6, df3/dx1 = 5*3 = 15.
    fn diamond_graph() -> LinearizedGraph<f64> {
        // preds[v] = [(predecessor, weight)]
        let preds = vec![
            vec![],                   // node 0: input
            vec![],                   // node 1: input
            vec![(0, 2.0), (1, 5.0)], // node 2: intermediate
            vec![(2, 3.0)],           // node 3: output
        ];
        let succs = vec![
            vec![(2, 2.0)], // node 0 → 2
            vec![(2, 5.0)], // node 1 → 2
            vec![(3, 3.0)], // node 2 → 3
            vec![],         // node 3: output (no successors)
        ];
        LinearizedGraph {
            num_inputs: 2,
            output_indices: vec![3],
            preds,
            succs,
            is_intermediate: vec![false, false, true, false],
        }
    }

    /// Chain graph: input → A → B → output, each with weight.
    ///
    /// ```text
    ///  0 ──(2.0)──▶ 1 ──(3.0)──▶ 2 ──(4.0)──▶ 3 (output)
    /// ```
    ///
    /// df/dx = 2*3*4 = 24.
    fn chain_graph() -> LinearizedGraph<f64> {
        let preds = vec![vec![], vec![(0, 2.0)], vec![(1, 3.0)], vec![(2, 4.0)]];
        let succs = vec![vec![(1, 2.0)], vec![(2, 3.0)], vec![(3, 4.0)], vec![]];
        LinearizedGraph {
            num_inputs: 1,
            output_indices: vec![3],
            preds,
            succs,
            is_intermediate: vec![false, true, true, false],
        }
    }

    #[test]
    fn accumulate_edge_merges_existing() {
        let mut adj: Vec<(u32, f64)> = vec![(1, 2.0), (3, 4.0)];
        LinearizedGraph::accumulate_edge(&mut adj, 1, 5.0);
        // Should merge: (1, 2.0+5.0=7.0)
        assert_eq!(adj.len(), 2);
        assert_eq!(adj[0], (1, 7.0));
        assert_eq!(adj[1], (3, 4.0));
    }

    #[test]
    fn accumulate_edge_creates_new() {
        let mut adj: Vec<(u32, f64)> = vec![(1, 2.0)];
        LinearizedGraph::accumulate_edge(&mut adj, 5, 3.0);
        assert_eq!(adj.len(), 2);
        assert_eq!(adj[1], (5, 3.0));
    }

    #[test]
    fn find_min_markowitz_picks_smallest_cost() {
        let g = diamond_graph();
        // Node 2: |preds|=2, |succs|=1, cost=2
        // Only one intermediate, so it must be picked
        let v = g.find_min_markowitz();
        assert_eq!(v, Some(2));
    }

    #[test]
    fn find_min_markowitz_chain_prefers_lower_cost() {
        let g = chain_graph();
        // Node 1: |preds|=1, |succs|=1, cost=1
        // Node 2: |preds|=1, |succs|=1, cost=1
        // Ties broken by smallest index
        let v = g.find_min_markowitz();
        assert_eq!(v, Some(1));
    }

    #[test]
    fn find_min_markowitz_none_when_no_intermediates() {
        let mut g = diamond_graph();
        g.is_intermediate = vec![false; 4];
        assert_eq!(g.find_min_markowitz(), None);
    }

    #[test]
    fn eliminate_vertex_creates_fill_in() {
        let mut g = diamond_graph();
        // Before: 0→2 (2.0), 1→2 (5.0), 2→3 (3.0)
        // Eliminating node 2 creates:
        //   0→3 with weight 2.0*3.0 = 6.0
        //   1→3 with weight 5.0*3.0 = 15.0
        g.eliminate_vertex(2);

        assert!(!g.is_intermediate[2]);
        assert!(g.preds[2].is_empty());
        assert!(g.succs[2].is_empty());

        // Check fill-in edges: preds of node 3 should now be [(0, 6.0), (1, 15.0)]
        let preds3 = &g.preds[3];
        assert_eq!(preds3.len(), 2);
        let find = |target: u32| preds3.iter().find(|(t, _)| *t == target).unwrap().1;
        assert_eq!(find(0), 6.0);
        assert_eq!(find(1), 15.0);

        // Check succs of inputs point to node 3
        assert_eq!(g.succs[0].len(), 1);
        assert_eq!(g.succs[0][0], (3, 6.0));
        assert_eq!(g.succs[1].len(), 1);
        assert_eq!(g.succs[1][0], (3, 15.0));
    }

    #[test]
    fn eliminate_chain_accumulates_correctly() {
        let mut g = chain_graph();
        // Eliminate node 1 first (cost=1): creates fill-in 0→2 with 2*3=6
        g.eliminate_vertex(1);
        assert_eq!(g.preds[2].len(), 1);
        assert_eq!(g.preds[2][0], (0, 6.0));

        // Eliminate node 2 (cost=1): creates fill-in 0→3 with 6*4=24
        g.eliminate_vertex(2);
        assert_eq!(g.preds[3].len(), 1);
        assert_eq!(g.preds[3][0], (0, 24.0));
    }

    #[test]
    fn eliminate_all_then_extract_jacobian_diamond() {
        let mut g = diamond_graph();
        g.eliminate_all();
        let jac = g.extract_jacobian();
        // 1 output, 2 inputs: df/dx0 = 6.0, df/dx1 = 15.0
        assert_eq!(jac.len(), 1);
        assert_eq!(jac[0].len(), 2);
        assert_eq!(jac[0][0], 6.0);
        assert_eq!(jac[0][1], 15.0);
    }

    #[test]
    fn eliminate_all_then_extract_jacobian_chain() {
        let mut g = chain_graph();
        g.eliminate_all();
        let jac = g.extract_jacobian();
        // df/dx = 2*3*4 = 24
        assert_eq!(jac.len(), 1);
        assert_eq!(jac[0].len(), 1);
        assert_eq!(jac[0][0], 24.0);
    }

    #[test]
    fn fill_in_merges_parallel_paths() {
        // Two parallel paths from input 0 to output 2 through node 1.
        // This tests that accumulate_edge merges duplicate fill-in edges.
        //
        //   0 ──(2.0)──▶ 1 ──(3.0)──▶ 2 (output)
        //   0 ──(4.0)──▶ 1   (second edge)
        //
        let preds = vec![
            vec![],
            vec![(0, 2.0), (0, 4.0)], // two edges from 0 to 1
            vec![(1, 3.0)],
        ];
        let succs = vec![vec![(1, 2.0), (1, 4.0)], vec![(2, 3.0)], vec![]];
        let mut g = LinearizedGraph {
            num_inputs: 1,
            output_indices: vec![2],
            preds,
            succs,
            is_intermediate: vec![false, true, false],
        };

        g.eliminate_vertex(1);
        // Fill-in: 0→2 with 2*3=6, then 0→2 with 4*3=12, merged = 18
        let preds2 = &g.preds[2];
        assert_eq!(preds2.len(), 1);
        assert_eq!(preds2[0], (0, 18.0));
    }

    #[test]
    fn markowitz_selects_cheaper_vertex() {
        // 3 inputs, 1 output, 2 intermediates with different fan-in/fan-out.
        //
        // Node 3 (intermediate): preds from 0,1,2 (fan-in=3), succ to 5 (fan-out=1) → cost=3
        // Node 4 (intermediate): pred from 0 (fan-in=1), succ to 5 (fan-out=1) → cost=1
        //
        // Markowitz should pick node 4 first.
        let preds = vec![
            vec![],                             // 0: input
            vec![],                             // 1: input
            vec![],                             // 2: input
            vec![(0, 1.0), (1, 1.0), (2, 1.0)], // 3: intermediate, fan-in=3
            vec![(0, 1.0)],                     // 4: intermediate, fan-in=1
            vec![(3, 1.0), (4, 1.0)],           // 5: output
        ];
        let succs = vec![
            vec![(3, 1.0), (4, 1.0)],
            vec![(3, 1.0)],
            vec![(3, 1.0)],
            vec![(5, 1.0)],
            vec![(5, 1.0)],
            vec![],
        ];
        let g = LinearizedGraph {
            num_inputs: 3,
            output_indices: vec![5],
            preds,
            succs,
            is_intermediate: vec![false, false, false, true, true, false],
        };

        assert_eq!(g.find_min_markowitz(), Some(4));
    }
}

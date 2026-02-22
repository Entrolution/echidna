//! Sparse Hessian computation via structural sparsity detection and graph coloring.

use crate::opcode::{OpCode, UNUSED};

/// Symmetric sparsity pattern in COO format (lower triangle + diagonal).
///
/// Entries are sorted by (row, col) and represent positions where the Hessian
/// may have non-zero values.
pub struct SparsityPattern {
    /// Dimension of the (square) Hessian matrix.
    pub dim: usize,
    /// Row indices (0-based).
    pub rows: Vec<u32>,
    /// Column indices (0-based), where `cols[k] <= rows[k]` (lower triangle).
    pub cols: Vec<u32>,
}

impl SparsityPattern {
    /// Number of non-zero entries in the pattern.
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Whether the pattern is empty (all zeros).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Check if position (i, j) is in the pattern (checks both (i,j) and (j,i)).
    pub fn contains(&self, i: usize, j: usize) -> bool {
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        self.rows
            .iter()
            .zip(self.cols.iter())
            .any(|(&row, &col)| row as usize == r && col as usize == c)
    }
}

/// Internal sparsity detection implementation.
///
/// Walks the tape forward propagating input-dependency bitsets.
/// At nonlinear operations, marks cross-pairs as potential Hessian interactions.
pub(crate) fn detect_sparsity_impl(
    opcodes: &[OpCode],
    arg_indices: &[[u32; 2]],
    num_inputs: usize,
    num_vars: usize,
) -> SparsityPattern {
    let num_words = num_inputs.div_ceil(64);

    // Dependency bitsets: deps[node] = set of input variables this node depends on
    let mut deps: Vec<Vec<u64>> = vec![vec![0u64; num_words]; num_vars];

    // Interaction pairs: HashSet<(u32, u32)> where row >= col
    let mut interactions: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

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
                        mark_all_pairs(&deps[i].clone(), num_inputs, &mut interactions);
                    }
                    OpClass::BinaryNonlinear => {
                        let b = b_idx as usize;
                        // Clone before mutation to avoid borrow issues
                        let deps_a = deps[a].clone();
                        let deps_b = deps[b].clone();
                        // Union dependencies
                        union_into(&mut deps, i, a);
                        union_into(&mut deps, i, b);
                        // Mark cross-pairs between operand dependency sets
                        mark_cross_pairs(&deps_a, &deps_b, num_inputs, &mut interactions);
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

    // Convert interactions to sorted COO format
    let mut entries: Vec<(u32, u32)> = interactions.into_iter().collect();
    entries.sort();

    let rows: Vec<u32> = entries.iter().map(|&(r, _)| r).collect();
    let cols: Vec<u32> = entries.iter().map(|&(_, c)| c).collect();

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
        | OpCode::Atanh
        | OpCode::Abs => OpClass::UnaryNonlinear,

        // Binary nonlinear: cross-derivatives between operands
        OpCode::Mul | OpCode::Div | OpCode::Powf | OpCode::Atan2 | OpCode::Hypot => {
            OpClass::BinaryNonlinear
        }

        // Zero derivative (piecewise constant or discontinuous)
        OpCode::Signum
        | OpCode::Floor
        | OpCode::Ceil
        | OpCode::Round
        | OpCode::Trunc
        | OpCode::Max
        | OpCode::Min
        | OpCode::Rem => OpClass::ZeroDerivative,

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
    )
}

/// Union deps[src] into deps[dst].
fn union_into(deps: &mut [Vec<u64>], dst: usize, src: usize) {
    if dst == src {
        return;
    }
    let num_words = deps[dst].len();
    // Clone src to avoid simultaneous mutable+immutable borrow
    let src_deps: Vec<u64> = deps[src].clone();
    for w in 0..num_words {
        deps[dst][w] |= src_deps[w];
    }
}

/// Mark all pairs (i, j) where both i and j are in the dependency set.
fn mark_all_pairs(
    dep_set: &[u64],
    num_inputs: usize,
    interactions: &mut std::collections::HashSet<(u32, u32)>,
) {
    let bits = extract_bits(dep_set, num_inputs);
    for i in 0..bits.len() {
        for j in 0..=i {
            let (r, c) = if bits[i] >= bits[j] {
                (bits[i], bits[j])
            } else {
                (bits[j], bits[i])
            };
            interactions.insert((r, c));
        }
    }
}

/// Mark cross-pairs between two dependency sets.
fn mark_cross_pairs(
    deps_a: &[u64],
    deps_b: &[u64],
    num_inputs: usize,
    interactions: &mut std::collections::HashSet<(u32, u32)>,
) {
    let bits_a = extract_bits(deps_a, num_inputs);
    let bits_b = extract_bits(deps_b, num_inputs);
    for &a in &bits_a {
        for &b in &bits_b {
            let (r, c) = if a >= b { (a, b) } else { (b, a) };
            interactions.insert((r, c));
        }
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

/// Greedy graph coloring for symmetric sparse Hessian recovery.
///
/// Colors the squared graph G^2 (vertices within distance 2 in the
/// interaction graph are adjacent) so that for each row of the Hessian,
/// at most one column in each color group has a non-zero entry. This
/// enables direct recovery of Hessian entries from compressed HVPs.
///
/// Returns `(colors, num_colors)` where `colors[i]` is the color assigned to input `i`.
/// Vertices are visited in decreasing-degree order for better results.
pub fn greedy_coloring(pattern: &SparsityPattern) -> (Vec<u32>, u32) {
    let n = pattern.dim;
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Build adjacency lists for G
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (&r, &c) in pattern.rows.iter().zip(pattern.cols.iter()) {
        let r = r as usize;
        let c = c as usize;
        if r != c {
            adj[r].push(c as u32);
            adj[c].push(r as u32);
        }
    }

    // Build G^2: two vertices are adjacent if they share a common neighbor
    // or are directly adjacent. For symmetric Hessian recovery, this ensures
    // that no two columns in the same color group have non-zero entries in the
    // same row.
    let mut adj2: Vec<std::collections::HashSet<u32>> = vec![std::collections::HashSet::new(); n];
    for v in 0..n {
        // Distance-1 neighbors
        for &u in &adj[v] {
            adj2[v].insert(u);
        }
        // Distance-2 neighbors: for each neighbor u of v, add u's neighbors
        for &u in &adj[v] {
            for &w in &adj[u as usize] {
                if w as usize != v {
                    adj2[v].insert(w);
                }
            }
        }
    }

    // Sort vertices by decreasing degree in G^2
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| adj2[b].len().cmp(&adj2[a].len()));

    let mut colors = vec![u32::MAX; n];
    let mut num_colors = 0u32;

    for &v in &order {
        // Find colors used by G^2-neighbors
        let mut used = std::collections::HashSet::new();
        for &neighbor in &adj2[v] {
            if colors[neighbor as usize] != u32::MAX {
                used.insert(colors[neighbor as usize]);
            }
        }

        // Assign smallest available color
        let mut color = 0u32;
        while used.contains(&color) {
            color += 1;
        }
        colors[v] = color;
        if color + 1 > num_colors {
            num_colors = color + 1;
        }
    }

    (colors, num_colors)
}

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

    // Interaction pairs as Vec (more cache-friendly than HashSet for typical problems).
    // Deduplicated via sort + dedup at the end.
    let mut interactions: Vec<(u32, u32)> = Vec::new();

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
                        // Safe: we read deps[i] (already updated) — no clone needed
                        // since mark_all_pairs only reads.
                        let bits = extract_bits(&deps[i], num_inputs);
                        for ii in 0..bits.len() {
                            for jj in 0..=ii {
                                let (r, c) = if bits[ii] >= bits[jj] {
                                    (bits[ii], bits[jj])
                                } else {
                                    (bits[jj], bits[ii])
                                };
                                interactions.push((r, c));
                            }
                        }
                    }
                    OpClass::BinaryNonlinear => {
                        let b = b_idx as usize;
                        // Extract bits before mutation — a and b differ from i,
                        // so we can read deps[a] and deps[b] directly.
                        let bits_a = extract_bits(&deps[a], num_inputs);
                        let bits_b = extract_bits(&deps[b], num_inputs);
                        // Union dependencies
                        union_into(&mut deps, i, a);
                        union_into(&mut deps, i, b);
                        // Mark cross-pairs between operand dependency sets
                        for &va in &bits_a {
                            for &vb in &bits_b {
                                let (r, c) = if va >= vb { (va, vb) } else { (vb, va) };
                                interactions.push((r, c));
                            }
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

    // Build G^2 using sorted Vec instead of HashSet (more cache-friendly).
    // Two vertices are adjacent if they share a common neighbor or are directly
    // adjacent. For symmetric Hessian recovery, this ensures that no two columns
    // in the same color group have non-zero entries in the same row.
    let mut adj2: Vec<Vec<u32>> = vec![Vec::new(); n];
    for v in 0..n {
        // Distance-1 neighbors
        for &u in &adj[v] {
            adj2[v].push(u);
        }
        // Distance-2 neighbors: for each neighbor u of v, add u's neighbors
        for &u in &adj[v] {
            for &w in &adj[u as usize] {
                if w as usize != v {
                    adj2[v].push(w);
                }
            }
        }
        // Deduplicate
        adj2[v].sort_unstable();
        adj2[v].dedup();
    }

    // Sort vertices by decreasing degree in G^2
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| adj2[b].len().cmp(&adj2[a].len()));

    let mut colors = vec![u32::MAX; n];
    let mut num_colors = 0u32;

    for &v in &order {
        // Find smallest available color using u64 bitset (supports up to 64 colors).
        // For virtually all practical sparse Hessians, 64 colors is sufficient.
        // Falls back to linear scan if > 64 colors are needed.
        let mut used_bits: u64 = 0;
        let mut needs_fallback = false;
        for &neighbor in &adj2[v] {
            let c = colors[neighbor as usize];
            if c != u32::MAX {
                if c < 64 {
                    used_bits |= 1u64 << c;
                } else {
                    needs_fallback = true;
                }
            }
        }

        let color = if !needs_fallback {
            (!used_bits).trailing_zeros()
        } else {
            // Fallback: collect all used colors and find first gap
            let mut used_vec: Vec<u32> = adj2[v]
                .iter()
                .filter_map(|&neighbor| {
                    let c = colors[neighbor as usize];
                    if c != u32::MAX { Some(c) } else { None }
                })
                .collect();
            used_vec.sort_unstable();
            used_vec.dedup();
            let mut c = 0u32;
            for &u in &used_vec {
                if u != c {
                    break;
                }
                c += 1;
            }
            c
        };

        colors[v] = color;
        if color + 1 > num_colors {
            num_colors = color + 1;
        }
    }

    (colors, num_colors)
}

/// Compressed Sparse Row (CSR) format for sparse matrices.
pub struct CsrPattern {
    /// Dimension of the square matrix.
    pub dim: usize,
    /// Row pointers (length `dim + 1`). Row `i` spans `col_ind[row_ptr[i]..row_ptr[i+1]]`.
    pub row_ptr: Vec<u32>,
    /// Column indices, sorted within each row.
    pub col_ind: Vec<u32>,
}

impl CsrPattern {
    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.col_ind.len()
    }

    /// Reorder values from COO order (matching `SparsityPattern`) to CSR order.
    ///
    /// Given Hessian values in COO order from `sparse_hessian`, returns values
    /// aligned with this CSR pattern's `col_ind` ordering.
    pub fn reorder_values<F: Copy>(&self, coo: &SparsityPattern, coo_vals: &[F]) -> Vec<F> {
        assert_eq!(coo_vals.len(), coo.nnz());
        assert_eq!(self.nnz(), coo.nnz());

        // Build a map from (row, col) -> value from COO
        let mut result = Vec::with_capacity(self.nnz());

        for row in 0..self.dim {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            for csr_idx in start..end {
                let col = self.col_ind[csr_idx];
                // Find this (row, col) in the COO pattern
                let coo_idx = coo
                    .rows
                    .iter()
                    .zip(coo.cols.iter())
                    .position(|(&r, &c)| r == row as u32 && c == col)
                    .expect("CSR entry not found in COO pattern");
                result.push(coo_vals[coo_idx]);
            }
        }
        result
    }
}

impl SparsityPattern {
    /// Convert to CSR format preserving the lower triangle (matching COO storage).
    pub fn to_csr_lower(&self) -> CsrPattern {
        let n = self.dim;
        let mut row_ptr = vec![0u32; n + 1];

        // Count entries per row
        for &r in &self.rows {
            row_ptr[r as usize + 1] += 1;
        }
        // Prefix sum
        for i in 1..=n {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Fill col_ind in row order (COO is already sorted by (row, col))
        let nnz = self.nnz();
        let mut col_ind = vec![0u32; nnz];
        let mut pos = vec![0u32; n]; // current position within each row
        for k in 0..nnz {
            let r = self.rows[k] as usize;
            let offset = row_ptr[r] + pos[r];
            col_ind[offset as usize] = self.cols[k];
            pos[r] += 1;
        }

        CsrPattern {
            dim: n,
            row_ptr,
            col_ind,
        }
    }

    /// Convert to symmetric CSR format (both lower and upper triangle).
    ///
    /// For every off-diagonal entry (r, c) in the lower triangle, also stores
    /// the mirrored entry (c, r). Diagonal entries appear once.
    pub fn to_csr(&self) -> CsrPattern {
        let n = self.dim;
        let mut row_ptr = vec![0u32; n + 1];

        // Count entries per row — each off-diagonal entry contributes to two rows
        for (&r, &c) in self.rows.iter().zip(self.cols.iter()) {
            row_ptr[r as usize + 1] += 1;
            if r != c {
                row_ptr[c as usize + 1] += 1;
            }
        }
        // Prefix sum
        for i in 1..=n {
            row_ptr[i] += row_ptr[i - 1];
        }

        let nnz = row_ptr[n] as usize;
        let mut col_ind = vec![0u32; nnz];
        let mut pos = vec![0u32; n];

        for (&r, &c) in self.rows.iter().zip(self.cols.iter()) {
            let ri = r as usize;
            let offset = row_ptr[ri] + pos[ri];
            col_ind[offset as usize] = c;
            pos[ri] += 1;

            if r != c {
                let ci = c as usize;
                let offset = row_ptr[ci] + pos[ci];
                col_ind[offset as usize] = r;
                pos[ci] += 1;
            }
        }

        // Sort col_ind within each row
        for i in 0..n {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;
            col_ind[start..end].sort_unstable();
        }

        CsrPattern {
            dim: n,
            row_ptr,
            col_ind,
        }
    }
}

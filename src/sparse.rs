//! Sparsity patterns and graph coloring for compressed derivative evaluation.
//!
//! Holds the pattern types ([`SparsityPattern`], [`CsrPattern`],
//! [`JacobianSparsityPattern`]) and the coloring algorithms that drive
//! compressed Hessian and Jacobian recovery. Pattern detection walks
//! recorded tapes and lives in the bytecode tape layer; this module has no
//! tape or opcode knowledge.

/// Symmetric sparsity pattern in COO format (lower triangle + diagonal).
///
/// Entries are sorted by (row, col) and represent positions where the Hessian
/// may have non-zero values.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Whether the pattern is empty (all zeros).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Check if position (i, j) is in the pattern (checks both (i,j) and (j,i)).
    #[must_use]
    pub fn contains(&self, i: usize, j: usize) -> bool {
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        self.rows
            .iter()
            .zip(self.cols.iter())
            .any(|(&row, &col)| row as usize == r && col as usize == c)
    }
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
#[must_use]
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

    greedy_distance1_coloring(&adj2, n)
}

/// Compressed Sparse Row (CSR) format for sparse matrices.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
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

        // For each CSR entry, linear-search the COO triples for its value
        // (O(nnz) per entry; fine at conversion-time scale).
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
    #[must_use]
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
    #[must_use]
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

// ══════════════════════════════════════════════
//  Jacobian sparsity pattern
// ══════════════════════════════════════════════

/// Sparsity pattern for a Jacobian matrix (non-symmetric, general m x n).
///
/// Stored in COO format: `(rows[k], cols[k])` indicates that output `rows[k]`
/// depends on input `cols[k]`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JacobianSparsityPattern {
    /// Number of outputs (rows of the Jacobian).
    pub num_outputs: usize,
    /// Number of inputs (columns of the Jacobian).
    pub num_inputs: usize,
    /// Output indices (row indices in the Jacobian).
    pub rows: Vec<u32>,
    /// Input indices (column indices in the Jacobian).
    pub cols: Vec<u32>,
}

impl JacobianSparsityPattern {
    /// Number of structural non-zeros in the pattern.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Whether the pattern is empty (all zeros).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Check if position (output_idx, input_idx) is in the pattern.
    #[must_use]
    pub fn contains(&self, output_idx: usize, input_idx: usize) -> bool {
        self.rows
            .iter()
            .zip(self.cols.iter())
            .any(|(&r, &c)| r as usize == output_idx && c as usize == input_idx)
    }
}

// ══════════════════════════════════════════════
//  Jacobian graph coloring
// ══════════════════════════════════════════════

/// Column coloring for forward-mode Jacobian compression.
///
/// Two columns (inputs) j1, j2 need different colors if any row (output) has
/// non-zeros in both columns. This builds the column intersection graph and
/// applies greedy distance-1 coloring with degree ordering.
///
/// Returns `(colors, num_colors)` where `colors[j]` is the color of column j.
#[must_use]
pub fn column_coloring(pattern: &JacobianSparsityPattern) -> (Vec<u32>, u32) {
    intersection_graph_coloring(
        &pattern.rows,
        &pattern.cols,
        pattern.num_outputs,
        pattern.num_inputs,
    )
}

/// Row coloring for reverse-mode Jacobian compression.
///
/// Two rows (outputs) i1, i2 need different colors if any column (input) has
/// non-zeros in both rows. This builds the row intersection graph and applies
/// greedy distance-1 coloring with degree ordering.
///
/// Returns `(colors, num_colors)` where `colors[i]` is the color of row i.
#[must_use]
pub fn row_coloring(pattern: &JacobianSparsityPattern) -> (Vec<u32>, u32) {
    intersection_graph_coloring(
        &pattern.cols,
        &pattern.rows,
        pattern.num_inputs,
        pattern.num_outputs,
    )
}

/// Build an intersection graph and color it greedily.
///
/// Groups entries by `group_keys` (dimension `group_dim`), then colors `color_keys`
/// (dimension `color_dim`). Two color-keys that appear in the same group need
/// different colors.
fn intersection_graph_coloring(
    group_keys: &[u32],
    color_keys: &[u32],
    group_dim: usize,
    color_dim: usize,
) -> (Vec<u32>, u32) {
    if color_dim == 0 {
        return (Vec::new(), 0);
    }

    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); group_dim];
    for (&g, &c) in group_keys.iter().zip(color_keys.iter()) {
        groups[g as usize].push(c);
    }

    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); color_dim];
    for members in &groups {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a = members[i] as usize;
                let b = members[j] as usize;
                adj[a].push(b as u32);
                adj[b].push(a as u32);
            }
        }
    }
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    greedy_distance1_coloring(&adj, color_dim)
}

/// Greedy distance-1 coloring with degree ordering and u64 bitset fast path.
fn greedy_distance1_coloring(adj: &[Vec<u32>], n: usize) -> (Vec<u32>, u32) {
    // Sort vertices by decreasing degree
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| adj[b].len().cmp(&adj[a].len()));

    let mut colors = vec![u32::MAX; n];
    let mut num_colors = 0u32;

    for &v in &order {
        let mut used_bits: u64 = 0;
        let mut needs_fallback = false;
        for &neighbor in &adj[v] {
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
            let mut used_vec: Vec<u32> = adj[v]
                .iter()
                .filter_map(|&neighbor| {
                    let c = colors[neighbor as usize];
                    if c != u32::MAX {
                        Some(c)
                    } else {
                        None
                    }
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

#![cfg(feature = "bytecode")]

use echidna::record_multi;
use echidna::sparse::{
    column_coloring, greedy_coloring, row_coloring, JacobianSparsityPattern, SparsityPattern,
};

// ══════════════════════════════════════════════
//  greedy_coloring (Hessian sparsity)
// ══════════════════════════════════════════════

#[test]
fn greedy_coloring_empty_pattern() {
    let pattern = SparsityPattern {
        dim: 0,
        rows: vec![],
        cols: vec![],
    };
    let (colors, num_colors) = greedy_coloring(&pattern);
    assert!(colors.is_empty());
    assert_eq!(num_colors, 0);
}

#[test]
fn greedy_coloring_diagonal_only() {
    // All diagonal: no off-diagonal interactions → 1 color
    let pattern = SparsityPattern {
        dim: 4,
        rows: vec![0, 1, 2, 3],
        cols: vec![0, 1, 2, 3],
    };
    let (colors, num_colors) = greedy_coloring(&pattern);
    assert_eq!(colors.len(), 4);
    assert_eq!(num_colors, 1);
}

#[test]
fn greedy_coloring_full_dense() {
    // Full lower-triangle 3×3 → all inputs interact → need 3 colors
    let pattern = SparsityPattern {
        dim: 3,
        rows: vec![0, 1, 1, 2, 2, 2],
        cols: vec![0, 0, 1, 0, 1, 2],
    };
    let (colors, num_colors) = greedy_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 3);
    // Verify all colors are distinct
    let mut unique = colors.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 3);
}

#[test]
fn greedy_coloring_banded() {
    // 4×4 tridiagonal: interactions only between adjacent variables
    // (0,0), (1,0), (1,1), (2,1), (2,2), (3,2), (3,3)
    let pattern = SparsityPattern {
        dim: 4,
        rows: vec![0, 1, 1, 2, 2, 3, 3],
        cols: vec![0, 0, 1, 1, 2, 2, 3],
    };
    let (colors, num_colors) = greedy_coloring(&pattern);
    assert_eq!(colors.len(), 4);
    // Tridiagonal Hessian needs at most 3 colors (distance-2 graph is bandwidth-4)
    assert!(num_colors <= 4, "got {} colors for tridiagonal", num_colors);
    // No adjacent variables should share the same color
    // Variables 0 and 1 interact → different colors
    assert_ne!(colors[0], colors[1]);
    // Variables 1 and 2 interact → different colors
    assert_ne!(colors[1], colors[2]);
}

#[test]
fn greedy_coloring_valid() {
    // Verify the coloring is valid: no two distance-2 neighbors share a color.
    // Use a small pattern and verify by construction.
    // 3×3 with interactions (0,0), (1,0), (1,1), (2,1), (2,2)
    // Adj in G²: 0-1 (direct), 0-2 (through 1), 1-2 (direct) → complete K3
    let pattern = SparsityPattern {
        dim: 3,
        rows: vec![0, 1, 1, 2, 2],
        cols: vec![0, 0, 1, 1, 2],
    };
    let (colors, num_colors) = greedy_coloring(&pattern);
    assert_eq!(num_colors, 3);
    assert_ne!(colors[0], colors[1]);
    assert_ne!(colors[0], colors[2]);
    assert_ne!(colors[1], colors[2]);
}

// ══════════════════════════════════════════════
//  column_coloring (Jacobian sparsity)
// ══════════════════════════════════════════════

#[test]
fn column_coloring_empty() {
    let pattern = JacobianSparsityPattern {
        num_outputs: 0,
        num_inputs: 0,
        rows: vec![],
        cols: vec![],
    };
    let (colors, num_colors) = column_coloring(&pattern);
    assert!(colors.is_empty());
    assert_eq!(num_colors, 0);
}

#[test]
fn column_coloring_independent_columns() {
    // Each output depends on exactly one input → all columns independent → 1 color
    // f0 = f(x0), f1 = f(x1), f2 = f(x2)
    let pattern = JacobianSparsityPattern {
        num_outputs: 3,
        num_inputs: 3,
        rows: vec![0, 1, 2],
        cols: vec![0, 1, 2],
    };
    let (colors, num_colors) = column_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 1);
}

#[test]
fn column_coloring_fully_dense() {
    // All outputs depend on all inputs → all columns share rows → n colors
    let pattern = JacobianSparsityPattern {
        num_outputs: 2,
        num_inputs: 3,
        rows: vec![0, 0, 0, 1, 1, 1],
        cols: vec![0, 1, 2, 0, 1, 2],
    };
    let (colors, num_colors) = column_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 3);
}

#[test]
fn column_coloring_partial_overlap() {
    // f0 depends on x0, x1; f1 depends on x1, x2
    // Columns 0 and 1 share row 0 → different colors
    // Columns 1 and 2 share row 1 → different colors
    // Columns 0 and 2 share no row → can share color
    let pattern = JacobianSparsityPattern {
        num_outputs: 2,
        num_inputs: 3,
        rows: vec![0, 0, 1, 1],
        cols: vec![0, 1, 1, 2],
    };
    let (colors, num_colors) = column_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 2);
    assert_ne!(colors[0], colors[1]); // share row 0
    assert_ne!(colors[1], colors[2]); // share row 1
    assert_eq!(colors[0], colors[2]); // no shared row
}

// ══════════════════════════════════════════════
//  row_coloring (Jacobian sparsity)
// ══════════════════════════════════════════════

#[test]
fn row_coloring_empty() {
    let pattern = JacobianSparsityPattern {
        num_outputs: 0,
        num_inputs: 0,
        rows: vec![],
        cols: vec![],
    };
    let (colors, num_colors) = row_coloring(&pattern);
    assert!(colors.is_empty());
    assert_eq!(num_colors, 0);
}

#[test]
fn row_coloring_independent_rows() {
    // Each input feeds exactly one output → all rows independent → 1 color
    let pattern = JacobianSparsityPattern {
        num_outputs: 3,
        num_inputs: 3,
        rows: vec![0, 1, 2],
        cols: vec![0, 1, 2],
    };
    let (colors, num_colors) = row_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 1);
}

#[test]
fn row_coloring_fully_dense() {
    // All outputs depend on all inputs → all rows share columns → m colors
    let pattern = JacobianSparsityPattern {
        num_outputs: 3,
        num_inputs: 2,
        rows: vec![0, 0, 1, 1, 2, 2],
        cols: vec![0, 1, 0, 1, 0, 1],
    };
    let (colors, num_colors) = row_coloring(&pattern);
    assert_eq!(colors.len(), 3);
    assert_eq!(num_colors, 3);
}

// ══════════════════════════════════════════════
//  Integration: sparsity detection + coloring
// ══════════════════════════════════════════════

#[test]
fn detected_sparsity_produces_valid_coloring() {
    // f(x0, x1, x2) = [x0 * x1, x2]
    // J sparsity: row 0 cols {0,1}, row 1 col {2}
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record_multi(|v| vec![v[0] * v[1], v[2]], &x);
    let pattern = tape.detect_jacobian_sparsity();

    let (col_colors, col_n) = column_coloring(&pattern);
    let (row_colors, row_n) = row_coloring(&pattern);

    // Columns 0,1 share row 0 → need 2 colors. Column 2 is independent.
    assert_eq!(col_n, 2);
    assert_ne!(col_colors[0], col_colors[1]);

    // Rows 0 and 1 share no column → 1 color
    assert_eq!(row_n, 1);
    assert_eq!(row_colors[0], row_colors[1]);
}

#[test]
fn sparse_jacobian_matches_dense() {
    // f(x0, x1, x2) = [x0 * x1, x1 + x2, x0 * x2]
    let x = [2.0_f64, 3.0, 4.0];
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[1], v[1] + v[2], v[0] * v[2]], &x);

    let dense_jac = tape.jacobian(&x);
    let (_, _, sparse_vals) = tape.sparse_jacobian(&x);

    let pattern = tape.detect_jacobian_sparsity();
    // Verify that sparse values match dense at structural positions
    for k in 0..pattern.nnz() {
        let row = pattern.rows[k] as usize;
        let col = pattern.cols[k] as usize;
        let dense_val = dense_jac[row][col];
        let sparse_val = sparse_vals[k];
        assert!(
            (dense_val - sparse_val).abs() < 1e-10,
            "mismatch at ({}, {}): dense={}, sparse={}",
            row,
            col,
            dense_val,
            sparse_val
        );
    }
}

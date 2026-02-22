#![cfg(feature = "bytecode")]

use echidna::{record, Scalar};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

#[test]
fn diagonal_pattern() {
    // f(x,y) = x^2 + y^2 -> diagonal Hessian, no cross terms
    let x = [3.0_f64, 4.0];
    let (tape, _) = record(|v| v[0] * v[0] + v[1] * v[1], &x);
    let pattern = tape.detect_sparsity();

    // Should have diagonal entries (0,0) and (1,1)
    assert!(pattern.contains(0, 0));
    assert!(pattern.contains(1, 1));
    // Should NOT have off-diagonal entries
    assert!(!pattern.contains(0, 1));
}

#[test]
fn full_pattern() {
    // f(x,y) = x*y -> full Hessian (cross term)
    let x = [2.0_f64, 3.0];
    let (tape, _) = record(|v| v[0] * v[1], &x);
    let pattern = tape.detect_sparsity();

    assert!(pattern.contains(0, 1));
    assert!(pattern.contains(1, 0));
}

#[test]
fn mixed_pattern() {
    // f(x,y,z) = x^2 + y*z -> {(0,0), (1,2)/(2,1)}
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record(|v| v[0] * v[0] + v[1] * v[2], &x);
    let pattern = tape.detect_sparsity();

    assert!(pattern.contains(0, 0));
    assert!(pattern.contains(1, 2));
    // x is independent of y,z
    assert!(!pattern.contains(0, 1));
    assert!(!pattern.contains(0, 2));
}

#[test]
fn tridiagonal() {
    // f(x) = sum x[i]*x[i+1] -> banded pattern, chromatic number = 2
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (tape, _) = record(
        |v| {
            let mut sum = v[0] - v[0]; // zero
            for i in 0..v.len() - 1 {
                sum = sum + v[i] * v[i + 1];
            }
            sum
        },
        &x,
    );

    let pattern = tape.detect_sparsity();

    // Check banded structure: only (i, i+1) pairs
    for i in 0..n - 1 {
        assert!(pattern.contains(i, i + 1), "missing ({}, {})", i, i + 1);
    }
    // No far-off-diagonal entries
    for i in 0..n {
        for j in 0..n {
            if (i as isize - j as isize).unsigned_abs() > 1 {
                assert!(!pattern.contains(i, j), "unexpected ({}, {})", i, j);
            }
        }
    }

    // Chromatic number of G^2 for a path graph is 3
    let (_, num_colors) = echidna::sparse::greedy_coloring(&pattern);
    assert_eq!(num_colors, 3);
}

#[test]
fn sparse_vs_dense_match() {
    // Verify sparse_hessian values match hessian at all pattern entries
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

    let (tape, _) = record(|v| rosenbrock(v), &x);
    let (val1, grad1, hess_dense) = tape.hessian(&x);
    let (val2, grad2, pattern, hess_values) = tape.sparse_hessian(&x);

    assert!((val1 - val2).abs() < 1e-10);
    for i in 0..n {
        assert!((grad1[i] - grad2[i]).abs() < 1e-10);
    }

    // Check every sparse entry matches the dense Hessian
    for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
        let r = row as usize;
        let c = col as usize;
        assert!(
            (hess_values[k] - hess_dense[r][c]).abs() < 1e-8,
            "mismatch at ({}, {}): sparse={}, dense={}",
            r,
            c,
            hess_values[k],
            hess_dense[r][c]
        );
    }
}

#[test]
fn fully_dense() {
    // f(x) = (sum x[i])^2 -> all pairs interact
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();

    let (tape, _) = record(
        |v| {
            let mut s = v[0] - v[0]; // zero
            for i in 0..v.len() {
                s = s + v[i];
            }
            s * s
        },
        &x,
    );

    let (_, _, pattern, hess_values) = tape.sparse_hessian(&x);
    let (_, _, hess_dense) = tape.hessian(&x);

    // Pattern should contain all lower-triangle entries
    for i in 0..n {
        for j in 0..=i {
            assert!(pattern.contains(i, j), "missing ({}, {})", i, j);
        }
    }

    // Values should match
    for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
        assert!(
            (hess_values[k] - hess_dense[row as usize][col as usize]).abs() < 1e-8,
            "mismatch at ({}, {})",
            row,
            col
        );
    }
}

#[test]
fn api_sparse_hessian() {
    let x = vec![1.5_f64, 2.0];
    let (val, grad, pattern, values) = echidna::sparse_hessian(|v| rosenbrock(v), &x);

    // Basic sanity: value and gradient should be correct
    let (val2, grad2, _) = echidna::hessian(|v| rosenbrock(v), &x);
    assert!((val - val2).abs() < 1e-10);
    for i in 0..2 {
        assert!((grad[i] - grad2[i]).abs() < 1e-10);
    }
    assert!(!pattern.is_empty());
    assert!(!values.is_empty());
}

// ══════════════════════════════════════════════
//  sparse_hessian_vec tests
// ══════════════════════════════════════════════

#[test]
fn sparse_hessian_vec_matches_sparse_tridiag() {
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (tape, _) = record(
        |v| {
            let mut sum = v[0] - v[0];
            for i in 0..v.len() - 1 {
                sum = sum + v[i] * v[i + 1];
            }
            sum
        },
        &x,
    );

    let (val1, grad1, pat1, vals1) = tape.sparse_hessian(&x);
    let (val2, grad2, pat2, vals2) = tape.sparse_hessian_vec::<4>(&x);

    assert!((val1 - val2).abs() < 1e-10);
    assert_eq!(pat1.nnz(), pat2.nnz());

    for i in 0..n {
        assert!((grad1[i] - grad2[i]).abs() < 1e-10);
    }

    for k in 0..vals1.len() {
        assert!(
            (vals1[k] - vals2[k]).abs() < 1e-8,
            "tridiag mismatch at k={}: scalar={}, vec={}",
            k,
            vals1[k],
            vals2[k]
        );
    }
}

#[test]
fn sparse_hessian_vec_matches_sparse_rosenbrock() {
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let (_, _, _, vals1) = tape.sparse_hessian(&x);
    let (_, _, _, vals2) = tape.sparse_hessian_vec::<4>(&x);

    for k in 0..vals1.len() {
        assert!(
            (vals1[k] - vals2[k]).abs() < 1e-8,
            "rosenbrock mismatch at k={}: scalar={}, vec={}",
            k,
            vals1[k],
            vals2[k]
        );
    }
}

#[test]
fn sparse_hessian_vec_padding() {
    // N=4 with 2 colors — tests lane padding
    let x = [3.0_f64, 4.0];
    let (tape, _) = record(|v| v[0] * v[0] + v[1] * v[1], &x);

    let (_, _, _, vals_scalar) = tape.sparse_hessian(&x);
    let (_, _, _, vals_vec) = tape.sparse_hessian_vec::<4>(&x);

    for k in 0..vals_scalar.len() {
        assert!(
            (vals_scalar[k] - vals_vec[k]).abs() < 1e-10,
            "padding mismatch at k={}",
            k
        );
    }
}

#[test]
fn sparse_hessian_vec_fully_dense() {
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let (tape, _) = record(
        |v| {
            let mut s = v[0] - v[0];
            for i in 0..v.len() {
                s = s + v[i];
            }
            s * s
        },
        &x,
    );

    let (_, _, _, vals_scalar) = tape.sparse_hessian(&x);
    let (_, _, _, vals_vec) = tape.sparse_hessian_vec::<8>(&x);

    for k in 0..vals_scalar.len() {
        assert!(
            (vals_scalar[k] - vals_vec[k]).abs() < 1e-8,
            "dense mismatch at k={}",
            k
        );
    }
}

#[test]
fn api_sparse_hessian_vec() {
    let x = vec![1.5_f64, 2.0];
    let (val1, grad1, _, vals1) = echidna::sparse_hessian(|v| rosenbrock(v), &x);
    let (val2, grad2, _, vals2) = echidna::sparse_hessian_vec::<f64, 4>(|v| rosenbrock(v), &x);

    assert!((val1 - val2).abs() < 1e-10);
    for i in 0..2 {
        assert!((grad1[i] - grad2[i]).abs() < 1e-10);
    }
    for k in 0..vals1.len() {
        assert!((vals1[k] - vals2[k]).abs() < 1e-8);
    }
}

// ══════════════════════════════════════════════
//  CSR format tests
// ══════════════════════════════════════════════

#[test]
fn csr_lower_roundtrip() {
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (tape, _) = record(|v| rosenbrock(v), &x);
    let (_, _, pattern, _) = tape.sparse_hessian(&x);

    let csr = pattern.to_csr_lower();
    assert_eq!(csr.dim, n);
    assert_eq!(csr.nnz(), pattern.nnz());
    assert_eq!(csr.row_ptr.len(), n + 1);

    // Verify row_ptr and col_ind are consistent
    for row in 0..n {
        let start = csr.row_ptr[row] as usize;
        let end = csr.row_ptr[row + 1] as usize;
        // col_ind should be sorted within each row
        for i in start + 1..end {
            assert!(csr.col_ind[i] > csr.col_ind[i - 1]);
        }
        // All col_ind should be <= row (lower triangle)
        for i in start..end {
            assert!(csr.col_ind[i] <= row as u32);
        }
    }
}

#[test]
fn csr_symmetric() {
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record(|v| v[0] * v[1] + v[1] * v[2], &x);
    let (_, _, pattern, _) = tape.sparse_hessian(&x);

    let csr = pattern.to_csr();
    // Every off-diagonal (r,c) should have both (r,c) and (c,r)
    for row in 0..csr.dim {
        let start = csr.row_ptr[row] as usize;
        let end = csr.row_ptr[row + 1] as usize;
        for idx in start..end {
            let col = csr.col_ind[idx] as usize;
            if row != col {
                // Check that (col, row) also exists
                let col_start = csr.row_ptr[col] as usize;
                let col_end = csr.row_ptr[col + 1] as usize;
                assert!(
                    csr.col_ind[col_start..col_end].contains(&(row as u32)),
                    "missing mirror ({}, {}) for ({}, {})",
                    col,
                    row,
                    row,
                    col
                );
            }
        }
    }
}

#[test]
fn csr_reorder_values() {
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (tape, _) = record(|v| rosenbrock(v), &x);
    let (_, _, pattern, hess_values) = tape.sparse_hessian(&x);

    let csr = pattern.to_csr_lower();
    let reordered = csr.reorder_values(&pattern, &hess_values);
    assert_eq!(reordered.len(), hess_values.len());

    // Verify each reordered value matches the COO value
    for row in 0..csr.dim {
        let start = csr.row_ptr[row] as usize;
        let end = csr.row_ptr[row + 1] as usize;
        for idx in start..end {
            let col = csr.col_ind[idx];
            // Find this entry in COO
            let coo_idx = pattern
                .rows
                .iter()
                .zip(pattern.cols.iter())
                .position(|(&r, &c)| r == row as u32 && c == col)
                .unwrap();
            assert!(
                (reordered[idx] - hess_values[coo_idx]).abs() < 1e-15,
                "reorder mismatch at CSR idx {}",
                idx
            );
        }
    }
}

#[test]
fn csr_empty_pattern() {
    // f(x,y) = x + y => linear, no Hessian entries
    let (tape, _) = record(|v| v[0] + v[1], &[1.0_f64, 2.0]);
    let pattern = tape.detect_sparsity();
    assert!(pattern.is_empty());

    let csr_lower = pattern.to_csr_lower();
    assert_eq!(csr_lower.nnz(), 0);
    assert_eq!(csr_lower.row_ptr, vec![0, 0, 0]);

    let csr = pattern.to_csr();
    assert_eq!(csr.nnz(), 0);
}

#[test]
fn csr_single_diagonal() {
    // f(x) = x^2 => single diagonal entry
    let (tape, _) = record(|v| v[0] * v[0], &[2.0_f64]);
    let pattern = tape.detect_sparsity();

    let csr = pattern.to_csr_lower();
    assert_eq!(csr.nnz(), 1);
    assert_eq!(csr.row_ptr, vec![0, 1]);
    assert_eq!(csr.col_ind, vec![0]);
}

// ══════════════════════════════════════════════
//  Zero-adjoint pruning test
// ══════════════════════════════════════════════

#[test]
fn pruning_preserves_eps_contributions() {
    // Ensure that zero-adjoint pruning with IsAllZero doesn't incorrectly
    // drop tangent (eps) contributions. We test this indirectly by verifying
    // Hessian correctness on a function with sparse adjoint patterns.
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

    let (tape, _) = record(
        |v| {
            // Function with many zero intermediate adjoints
            let a = v[0] * v[0]; // only depends on x0
            let b = v[3] * v[4]; // only depends on x3, x4
            a + b
        },
        &x,
    );

    // Dense Hessian should be correct even with pruning
    let (_, _, hess) = tape.hessian(&x);

    // H[0][0] = 2, H[3][4] = H[4][3] = 1, all others = 0
    assert!((hess[0][0] - 2.0).abs() < 1e-10);
    assert!((hess[3][4] - 1.0).abs() < 1e-10);
    assert!((hess[4][3] - 1.0).abs() < 1e-10);
    // Off-entries should be zero
    assert!(hess[0][1].abs() < 1e-10);
    assert!(hess[1][1].abs() < 1e-10);
    assert!(hess[2][2].abs() < 1e-10);
}

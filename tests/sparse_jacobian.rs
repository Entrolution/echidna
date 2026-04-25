#![cfg(feature = "bytecode")]

use echidna::{record_multi, Scalar};

fn linear_map<T: Scalar>(x: &[T]) -> Vec<T> {
    // f(x) = [2*x0 + x1, x1 + 3*x2]
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    let three = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(3.0).unwrap());
    vec![two * x[0] + x[1], x[1] + three * x[2]]
}

fn nonlinear_map<T: Scalar>(x: &[T]) -> Vec<T> {
    // f(x) = [x0*x1, x1*x2, x0^2]
    vec![x[0] * x[1], x[1] * x[2], x[0] * x[0]]
}

#[test]
fn jacobian_sparsity_linear() {
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record_multi(linear_map, &x);
    let pattern = tape.detect_jacobian_sparsity();

    // f0 depends on x0, x1
    assert!(pattern.contains(0, 0));
    assert!(pattern.contains(0, 1));
    assert!(!pattern.contains(0, 2));

    // f1 depends on x1, x2
    assert!(!pattern.contains(1, 0));
    assert!(pattern.contains(1, 1));
    assert!(pattern.contains(1, 2));
}

#[test]
fn jacobian_sparsity_nonlinear() {
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record_multi(nonlinear_map, &x);
    let pattern = tape.detect_jacobian_sparsity();

    // f0 = x0*x1 -> depends on x0, x1
    assert!(pattern.contains(0, 0));
    assert!(pattern.contains(0, 1));
    assert!(!pattern.contains(0, 2));

    // f1 = x1*x2 -> depends on x1, x2
    assert!(!pattern.contains(1, 0));
    assert!(pattern.contains(1, 1));
    assert!(pattern.contains(1, 2));

    // f2 = x0^2 -> depends on x0 only
    assert!(pattern.contains(2, 0));
    assert!(!pattern.contains(2, 1));
    assert!(!pattern.contains(2, 2));
}

#[test]
fn column_coloring_diagonal() {
    let x = [1.0_f64, 2.0, 3.0];
    // f(x) = [x0, x1, x2] — diagonal Jacobian, all columns independent
    let (tape, _) = record_multi(|v| vec![v[0], v[1], v[2]], &x);
    let pattern = tape.detect_jacobian_sparsity();
    let (_, num_colors) = echidna::sparse::column_coloring(&pattern);
    // Diagonal pattern: 1 color suffices
    assert_eq!(num_colors, 1);
}

#[test]
fn sparse_jacobian_vs_dense() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(nonlinear_map, &x);

    let dense_jac = tape.jacobian(&x);
    let (_, pattern, sparse_vals) = tape.sparse_jacobian(&x);

    for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
        let r = row as usize;
        let c = col as usize;
        assert!(
            (sparse_vals[k] - dense_jac[r][c]).abs() < 1e-10,
            "mismatch at ({}, {}): sparse={}, dense={}",
            r,
            c,
            sparse_vals[k],
            dense_jac[r][c]
        );
    }
}

#[test]
fn sparse_jacobian_forward_vs_reverse() {
    let x = [1.5_f64, 2.5, 0.5];
    let (mut tape, _) = record_multi(nonlinear_map, &x);

    let (vals_fwd, pat_fwd, jac_fwd) = tape.sparse_jacobian_forward(&x);
    let (vals_rev, pat_rev, jac_rev) = tape.sparse_jacobian_reverse(&x);

    assert_eq!(vals_fwd.len(), vals_rev.len());
    for i in 0..vals_fwd.len() {
        assert!((vals_fwd[i] - vals_rev[i]).abs() < 1e-10);
    }

    assert_eq!(pat_fwd.nnz(), pat_rev.nnz());
    for k in 0..jac_fwd.len() {
        assert!(
            (jac_fwd[k] - jac_rev[k]).abs() < 1e-10,
            "mismatch at k={}: fwd={}, rev={}",
            k,
            jac_fwd[k],
            jac_rev[k]
        );
    }
}

#[test]
fn sparse_jacobian_vec_matches() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(nonlinear_map, &x);

    let (_, _, jac_scalar) = tape.sparse_jacobian_forward(&x);
    let (_, _, jac_vec) = tape.sparse_jacobian_vec::<4>(&x);

    for k in 0..jac_scalar.len() {
        assert!(
            (jac_scalar[k] - jac_vec[k]).abs() < 1e-10,
            "vec mismatch at k={}",
            k
        );
    }
}

#[test]
fn sparse_jacobian_with_pattern_precomputed() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(nonlinear_map, &x);

    // First call: detect pattern and colors
    let pattern = tape.detect_jacobian_sparsity();
    let (colors, num_colors) = echidna::sparse::column_coloring(&pattern);

    // Second call with different inputs, reusing pattern
    let x2 = [3.0, 1.0, 2.0];
    let (vals, jac_precomputed) =
        tape.sparse_jacobian_with_pattern(&x2, &pattern, &colors, num_colors, true);
    let (vals2, _, jac_fresh) = tape.sparse_jacobian_forward(&x2);

    for i in 0..vals.len() {
        assert!((vals[i] - vals2[i]).abs() < 1e-10);
    }
    for k in 0..jac_precomputed.len() {
        assert!(
            (jac_precomputed[k] - jac_fresh[k]).abs() < 1e-10,
            "precomputed mismatch at k={}",
            k
        );
    }
}

#[test]
fn jacobian_forward_vs_reverse() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(nonlinear_map, &x);

    let jac_rev = tape.jacobian(&x);
    let jac_fwd = tape.jacobian_forward(&x);

    assert_eq!(jac_rev.len(), jac_fwd.len());
    for i in 0..jac_rev.len() {
        for j in 0..jac_rev[i].len() {
            assert!(
                (jac_rev[i][j] - jac_fwd[i][j]).abs() < 1e-10,
                "mismatch at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn sparse_hessian_with_pattern_precomputed() {
    let x = vec![1.5_f64, 2.0, 0.5];
    let (tape, _) = echidna::record(|v| v[0] * v[1] + v[1] * v[2] + v[0] * v[0], &x);

    let pattern = tape.detect_sparsity();
    let (colors, num_colors) = echidna::sparse::greedy_coloring(&pattern);

    let (val1, grad1, _, hess1) = tape.sparse_hessian(&x);
    let (val2, grad2, hess2) = tape.sparse_hessian_with_pattern(&x, &pattern, &colors, num_colors);

    assert!((val1 - val2).abs() < 1e-10);
    for i in 0..grad1.len() {
        assert!((grad1[i] - grad2[i]).abs() < 1e-10);
    }
    for k in 0..hess1.len() {
        assert!((hess1[k] - hess2[k]).abs() < 1e-10);
    }
}

#[test]
fn sparse_jacobian_auto_selects_reverse() {
    // 5 inputs -> 2 outputs, both outputs depend on all inputs.
    // Column coloring is expensive (all columns share both rows),
    // row coloring is cheap (only 2 rows) -> auto-selection picks reverse.
    fn wide_input_map<T: Scalar>(x: &[T]) -> Vec<T> {
        vec![
            x[0] + x[1] + x[2] + x[3] + x[4],
            x[0] * x[1] * x[2] * x[3] * x[4],
        ]
    }

    let x = [1.0_f64, 2.0, 0.5, 1.5, 3.0];
    let (mut tape, _) = record_multi(wide_input_map, &x);

    // Verify the coloring structure actually favors reverse mode.
    tape.forward(&x);
    let pattern = tape.detect_jacobian_sparsity();
    let (_, num_col_colors) = echidna::sparse::column_coloring(&pattern);
    let (_, num_row_colors) = echidna::sparse::row_coloring(&pattern);
    assert!(
        num_row_colors < num_col_colors,
        "test precondition: need row_colors ({}) < col_colors ({}) to force reverse",
        num_row_colors,
        num_col_colors
    );

    // Auto-selection should pick reverse mode for this wide-input function.
    let (_, pat_auto, jac_auto) = tape.sparse_jacobian(&x);

    // Explicit reverse should give the same result.
    let (_, pat_rev, jac_rev) = tape.sparse_jacobian_reverse(&x);

    assert_eq!(pat_auto.nnz(), pat_rev.nnz());
    for k in 0..jac_auto.len() {
        assert!(
            (jac_auto[k] - jac_rev[k]).abs() < 1e-10,
            "auto vs reverse mismatch at k={}: auto={}, rev={}",
            k,
            jac_auto[k],
            jac_rev[k]
        );
    }

    // Also verify against dense Jacobian for correctness.
    let dense_jac = tape.jacobian(&x);
    for (k, (&row, &col)) in pat_auto.rows.iter().zip(pat_auto.cols.iter()).enumerate() {
        let r = row as usize;
        let c = col as usize;
        assert!(
            (jac_auto[k] - dense_jac[r][c]).abs() < 1e-10,
            "auto vs dense mismatch at ({}, {}): auto={}, dense={}",
            r,
            c,
            jac_auto[k],
            dense_jac[r][c]
        );
    }
}

#[test]
fn api_sparse_jacobian() {
    let x = vec![1.0_f64, 2.0, 3.0];
    let (vals, pattern, jac_vals) = echidna::sparse_jacobian(nonlinear_map, &x);

    assert_eq!(vals.len(), 3);
    assert!(!pattern.is_empty());
    assert!(!jac_vals.is_empty());

    // Verify against dense
    let (mut tape, _) = record_multi(nonlinear_map, &x);
    let dense_jac = tape.jacobian(&x);

    for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
        let r = row as usize;
        let c = col as usize;
        assert!(
            (jac_vals[k] - dense_jac[r][c]).abs() < 1e-10,
            "api mismatch at ({}, {})",
            r,
            c
        );
    }
}

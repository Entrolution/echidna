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
        assert!(
            pattern.contains(i, i + 1),
            "missing ({}, {})",
            i,
            i + 1
        );
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

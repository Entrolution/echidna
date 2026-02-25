use num_traits::Float;

/// Result of LU factorization with partial pivoting.
///
/// Stores the combined L/U factors in a single matrix (L below diagonal,
/// U on and above diagonal) plus the row permutation.
pub struct LuFactors<F> {
    /// Combined L/U matrix: L is below the diagonal (unit diagonal implicit),
    /// U is on and above the diagonal.
    lu: Vec<Vec<F>>,
    /// Row permutation: `perm[i]` is the original row index for factored row `i`.
    perm: Vec<usize>,
    n: usize,
}

/// Factorize an `n x n` matrix via LU decomposition with partial pivoting.
///
/// Returns `None` if the matrix is singular (zero or near-zero pivot).
// Explicit indexing is clearer for pivoted LU: row/col indices drive pivot search and elimination
#[allow(clippy::needless_range_loop)]
pub fn lu_factor<F: Float>(a: &[Vec<F>]) -> Option<LuFactors<F>> {
    let n = a.len();
    debug_assert!(a.iter().all(|row| row.len() == n));

    let mut lu: Vec<Vec<F>> = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    let eps = F::from(1e-12).unwrap_or_else(|| F::epsilon());

    for col in 0..n {
        // Find pivot
        let mut max_val = lu[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = lu[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < eps {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            lu.swap(col, max_row);
            perm.swap(col, max_row);
        }

        let pivot = lu[col][col];

        // Eliminate below, storing L factors in-place
        for row in (col + 1)..n {
            let factor = lu[row][col] / pivot;
            lu[row][col] = factor; // Store L factor
            for j in (col + 1)..n {
                let val = lu[col][j];
                lu[row][j] = lu[row][j] - factor * val;
            }
        }
    }

    Some(LuFactors { lu, perm, n })
}

/// Solve `A * x = b` using a pre-computed LU factorization.
///
/// This avoids re-factorizing when solving multiple right-hand sides
/// against the same matrix.
// Explicit indexing is clearer for forward/back substitution with permuted indices
#[allow(clippy::needless_range_loop)]
pub fn lu_back_solve<F: Float>(factors: &LuFactors<F>, b: &[F]) -> Vec<F> {
    let n = factors.n;
    debug_assert_eq!(b.len(), n);

    // Apply permutation to b
    let mut y = vec![F::zero(); n];
    for i in 0..n {
        y[i] = b[factors.perm[i]];
    }

    // Forward substitution (L * y' = permuted_b), L has unit diagonal
    for i in 1..n {
        for j in 0..i {
            let l_ij = factors.lu[i][j];
            let y_j = y[j];
            y[i] = y[i] - l_ij * y_j;
        }
    }

    // Back substitution (U * x = y')
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum = sum - factors.lu[i][j] * x[j];
        }
        x[i] = sum / factors.lu[i][i];
    }

    x
}

/// Solve `A * x = b` via LU factorization with partial pivoting.
///
/// `a` is an `n x n` matrix stored as `a[row][col]`.
/// Returns `None` if the matrix is singular (zero or near-zero pivot).
pub fn lu_solve<F: Float>(a: &[Vec<F>], b: &[F]) -> Option<Vec<F>> {
    let factors = lu_factor(a)?;
    Some(lu_back_solve(&factors, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let x = lu_solve(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn lu_solve_2x2() {
        // [2 1] [x0]   [5]
        // [1 3] [x1] = [7]
        // Solution: x0 = 8/5, x1 = 9/5
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = lu_solve(&a, &b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-12);
        assert!((x[1] - 1.8).abs() < 1e-12);
    }

    #[test]
    fn lu_solve_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![3.0, 6.0];
        assert!(lu_solve(&a, &b).is_none());
    }

    #[test]
    fn lu_solve_needs_pivoting() {
        // First pivot is zero — requires row swap
        let a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let b = vec![3.0, 7.0];
        let x = lu_solve(&a, &b).unwrap();
        assert!((x[0] - 7.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lu_factor_then_back_solve_matches_lu_solve() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b1 = vec![5.0, 7.0];
        let b2 = vec![1.0, 0.0];

        // Factorize once
        let factors = lu_factor(&a).unwrap();

        // Solve two different RHS
        let x1 = lu_back_solve(&factors, &b1);
        let x2 = lu_back_solve(&factors, &b2);

        // Compare with lu_solve
        let x1_ref = lu_solve(&a, &b1).unwrap();
        let x2_ref = lu_solve(&a, &b2).unwrap();

        for i in 0..2 {
            assert!((x1[i] - x1_ref[i]).abs() < 1e-12);
            assert!((x2[i] - x2_ref[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn lu_factor_then_back_solve_3x3() {
        // [1 2 3] [x]   [14]
        // [4 5 6] [y] = [32]
        // [7 8 0] [z]   [23]
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let b = vec![14.0, 32.0, 23.0];
        let factors = lu_factor(&a).unwrap();
        let x = lu_back_solve(&factors, &b);
        let x_ref = lu_solve(&a, &b).unwrap();
        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-10,
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn lu_factor_singular_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(lu_factor(&a).is_none());
    }
}

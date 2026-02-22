use num_traits::Float;

/// Solve `A * x = b` via LU factorization with partial pivoting.
///
/// `a` is an `n x n` matrix stored as `a[row][col]`.
/// Returns `None` if the matrix is singular (zero or near-zero pivot).
#[allow(clippy::needless_range_loop)]
pub fn lu_solve<F: Float>(a: &[Vec<F>], b: &[F]) -> Option<Vec<F>> {
    let n = b.len();
    debug_assert!(a.len() == n && a.iter().all(|row| row.len() == n));

    // Working copy of augmented system [A | b]
    let mut m: Vec<Vec<F>> = a.to_vec();
    let mut rhs = b.to_vec();

    let eps = F::from(1e-12).unwrap_or_else(|| F::epsilon());

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = m[row][col].abs();
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
            m.swap(col, max_row);
            rhs.swap(col, max_row);
        }

        let pivot = m[col][col];

        // Eliminate below
        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            m[row][col] = F::zero();
            for j in (col + 1)..n {
                let val = m[col][j];
                m[row][j] = m[row][j] - factor * val;
            }
            let rhs_col = rhs[col];
            rhs[row] = rhs[row] - factor * rhs_col;
        }
    }

    // Back substitution
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum = sum - m[i][j] * x[j];
        }
        if m[i][i].abs() < eps {
            return None;
        }
        x[i] = sum / m[i][i];
    }

    Some(x)
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
}

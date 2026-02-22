use echidna::{BytecodeTape, Float};

use crate::linalg::{lu_back_solve, lu_factor, lu_solve};

/// Partition a full Jacobian `J_F` (m × (m+n)) into `F_z` (m × m) and `F_x` (m × n).
///
/// `num_states` is `m`, the number of state variables (first `m` columns → `F_z`).
fn partition_jacobian<F: Float>(
    jac: &[Vec<F>],
    num_states: usize,
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let m = num_states;
    let mut f_z = Vec::with_capacity(m);
    let mut f_x = Vec::with_capacity(m);
    for row in jac {
        f_z.push(row[..m].to_vec());
        f_x.push(row[m..].to_vec());
    }
    (f_z, f_x)
}

/// Transpose an m × n matrix stored as `Vec<Vec<F>>`.
fn transpose<F: Float>(mat: &[Vec<F>]) -> Vec<Vec<F>> {
    if mat.is_empty() {
        return vec![];
    }
    let rows = mat.len();
    let cols = mat[0].len();
    let mut result = vec![vec![F::zero(); rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = mat[i][j];
        }
    }
    result
}

/// Validate inputs shared by all implicit differentiation functions.
fn validate_inputs<F: Float>(
    tape: &BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    num_states: usize,
) {
    assert_eq!(
        z_star.len(),
        num_states,
        "z_star length ({}) must equal num_states ({})",
        z_star.len(),
        num_states
    );
    assert_eq!(
        tape.num_inputs(),
        num_states + x.len(),
        "tape.num_inputs() ({}) must equal num_states + x.len() ({})",
        tape.num_inputs(),
        num_states + x.len()
    );
    assert_eq!(
        tape.num_outputs(),
        num_states,
        "tape.num_outputs() ({}) must equal num_states ({}) — IFT requires F: R^(m+n) → R^m to be square in the state block",
        tape.num_outputs(),
        num_states
    );
}

/// Build concatenated input `[z_star..., x...]` and compute the full Jacobian,
/// partitioned into `(F_z, F_x)`.
fn compute_partitioned_jacobian<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    num_states: usize,
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let mut inputs = Vec::with_capacity(z_star.len() + x.len());
    inputs.extend_from_slice(z_star);
    inputs.extend_from_slice(x);

    // Debug check: warn if residual is not near zero
    #[cfg(debug_assertions)]
    {
        tape.forward(&inputs);
        let residual = tape.output_values();
        let norm_sq: F = residual.iter().fold(F::zero(), |acc, &v| acc + v * v);
        let norm = norm_sq.sqrt();
        let threshold = F::from(1e-6).unwrap_or_else(|| F::epsilon());
        if norm > threshold {
            eprintln!(
                "WARNING: implicit differentiation called with ||F(z*, x)|| = {:?} > 1e-6. \
                 Derivatives may be meaningless if z* is not a root.",
                norm.to_f64()
            );
        }
    }

    let jac = tape.jacobian(&inputs);
    partition_jacobian(&jac, num_states)
}

/// Compute the full implicit Jacobian `dz*/dx` (m × n matrix).
///
/// Given a multi-output residual tape `F: R^(m+n) → R^m` with `F(z*, x) = 0`,
/// computes `dz*/dx = -F_z^{-1} · F_x` via the Implicit Function Theorem.
///
/// The first `num_states` tape inputs are state variables `z`, the remaining are
/// parameters `x`.
///
/// Returns `None` if `F_z` is singular.
pub fn implicit_jacobian<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    num_states: usize,
) -> Option<Vec<Vec<F>>> {
    validate_inputs(tape, z_star, x, num_states);
    let (f_z, f_x) = compute_partitioned_jacobian(tape, z_star, x, num_states);

    let m = num_states;
    let n = x.len();

    // LU-factorize F_z once, then solve for each column of -F_x
    let factors = lu_factor(&f_z)?;

    // Build result column by column: solve F_z · col_j = -F_x[:, j]
    let mut result = vec![vec![F::zero(); n]; m];
    for j in 0..n {
        let neg_col: Vec<F> = (0..m).map(|i| F::zero() - f_x[i][j]).collect();
        let col = lu_back_solve(&factors, &neg_col);
        for i in 0..m {
            result[i][j] = col[i];
        }
    }

    Some(result)
}

/// Compute the implicit tangent `dz*/dx · x_dot` (m-vector).
///
/// Given a multi-output residual tape `F: R^(m+n) → R^m` with `F(z*, x) = 0`,
/// computes the directional derivative `dz*/dx · x_dot = -F_z^{-1} · (F_x · x_dot)`.
///
/// This solves a single linear system rather than computing the full Jacobian,
/// which is more efficient when only one direction is needed.
///
/// Returns `None` if `F_z` is singular.
pub fn implicit_tangent<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    x_dot: &[F],
    num_states: usize,
) -> Option<Vec<F>> {
    assert_eq!(
        x_dot.len(),
        x.len(),
        "x_dot length ({}) must equal x length ({})",
        x_dot.len(),
        x.len()
    );
    validate_inputs(tape, z_star, x, num_states);
    let (f_z, f_x) = compute_partitioned_jacobian(tape, z_star, x, num_states);

    let m = num_states;
    let n = x.len();

    // Compute F_x · x_dot (matrix-vector product)
    let mut fx_xdot = vec![F::zero(); m];
    for i in 0..m {
        for j in 0..n {
            fx_xdot[i] = fx_xdot[i] + f_x[i][j] * x_dot[j];
        }
    }

    // Negate: rhs = -(F_x · x_dot)
    let neg_fx_xdot: Vec<F> = fx_xdot.iter().map(|&v| F::zero() - v).collect();

    // Solve F_z · z_dot = -(F_x · x_dot)
    lu_solve(&f_z, &neg_fx_xdot)
}

/// Compute the implicit adjoint `(dz*/dx)^T · z_bar` (n-vector).
///
/// Given a multi-output residual tape `F: R^(m+n) → R^m` with `F(z*, x) = 0`,
/// computes `x_bar = -F_x^T · (F_z^{-T} · z_bar)`.
///
/// This is the reverse-mode (adjoint) form, useful when `n > m` or when
/// propagating gradients backward through an implicit layer.
///
/// Returns `None` if `F_z` is singular.
pub fn implicit_adjoint<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    z_bar: &[F],
    num_states: usize,
) -> Option<Vec<F>> {
    assert_eq!(
        z_bar.len(),
        num_states,
        "z_bar length ({}) must equal num_states ({})",
        z_bar.len(),
        num_states
    );
    validate_inputs(tape, z_star, x, num_states);
    let (f_z, f_x) = compute_partitioned_jacobian(tape, z_star, x, num_states);

    let m = num_states;
    let n = x.len();

    // Solve F_z^T · lambda = z_bar
    let f_z_t = transpose(&f_z);
    let lambda = lu_solve(&f_z_t, z_bar)?;

    // Compute x_bar = -F_x^T · lambda
    let f_x_t = transpose(&f_x);
    let mut x_bar = vec![F::zero(); n];
    for j in 0..n {
        for i in 0..m {
            x_bar[j] = x_bar[j] - f_x_t[j][i] * lambda[i];
        }
    }

    Some(x_bar)
}

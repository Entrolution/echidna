use echidna::{BytecodeTape, Dual, Float};

use crate::linalg::{lu_back_solve, lu_factor, lu_solve};

/// Partition a full Jacobian `J_F` (m × (m+n)) into `F_z` (m × m) and `F_x` (m × n).
///
/// `num_states` is `m`, the number of state variables (first `m` columns → `F_z`).
fn partition_jacobian<F: Float>(jac: &[Vec<F>], num_states: usize) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
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
fn validate_inputs<F: Float>(tape: &BytecodeTape<F>, z_star: &[F], x: &[F], num_states: usize) {
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

/// Compute the implicit Hessian-vector-vector product `d²z*/dx² · v · w` (m-vector).
///
/// Given a residual tape `F: R^(m+n) → R^m` with `F(z*, x) = 0`, computes the
/// second-order sensitivity by differentiating the IFT identity twice:
///
///   `F_z · h + [ṗ^T · Hess(F_i) · ẇ]_i = 0`
///
/// where `ṗ = [dz*/dx · v; v]`, `ẇ = [dz*/dx · w; w]`, and `h = d²z*/dx² · v · w`.
///
/// Uses nested `Dual<Dual<F>>` forward passes to compute the second-order correction
/// in a single O(tape_length) pass per direction pair.
///
/// Returns `None` if `F_z` is singular.
pub fn implicit_hvp<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    v: &[F],
    w: &[F],
    num_states: usize,
) -> Option<Vec<F>> {
    let n = x.len();
    let m = num_states;
    assert_eq!(
        v.len(),
        n,
        "v length ({}) must equal x length ({})",
        v.len(),
        n
    );
    assert_eq!(
        w.len(),
        n,
        "w length ({}) must equal x length ({})",
        w.len(),
        n
    );
    validate_inputs(tape, z_star, x, num_states);

    let (f_z, f_x) = compute_partitioned_jacobian(tape, z_star, x, num_states);
    let factors = lu_factor(&f_z)?;

    // First-order sensitivities: ż_v = -F_z^{-1} · (F_x · v)
    let mut fx_v = vec![F::zero(); m];
    let mut fx_w = vec![F::zero(); m];
    for i in 0..m {
        for j in 0..n {
            fx_v[i] = fx_v[i] + f_x[i][j] * v[j];
            fx_w[i] = fx_w[i] + f_x[i][j] * w[j];
        }
    }
    let neg_fx_v: Vec<F> = fx_v.iter().map(|&val| F::zero() - val).collect();
    let neg_fx_w: Vec<F> = fx_w.iter().map(|&val| F::zero() - val).collect();
    let z_dot_v = lu_back_solve(&factors, &neg_fx_v);
    let z_dot_w = lu_back_solve(&factors, &neg_fx_w);

    // Build Dual<Dual<F>> inputs for nested forward pass
    // ṗ = [ż_v; v], ẇ = [ż_w; w]
    // Input j: Dual { re: Dual(u_j, ṗ_j), eps: Dual(ẇ_j, 0) }
    let mut dd_inputs: Vec<Dual<Dual<F>>> = Vec::with_capacity(m + n);
    for i in 0..m {
        dd_inputs.push(Dual::new(
            Dual::new(z_star[i], z_dot_v[i]),
            Dual::new(z_dot_w[i], F::zero()),
        ));
    }
    for j in 0..n {
        dd_inputs.push(Dual::new(Dual::new(x[j], v[j]), Dual::new(w[j], F::zero())));
    }

    let mut buf = Vec::new();
    tape.forward_tangent(&dd_inputs, &mut buf);

    // Extract second-order correction: buf[out_idx].eps.eps for each output
    let out_indices = tape.all_output_indices();
    let mut rhs = Vec::with_capacity(m);
    for &idx in out_indices {
        rhs.push(buf[idx as usize].eps.eps);
    }

    // Solve F_z · h = -rhs
    let neg_rhs: Vec<F> = rhs.iter().map(|&val| F::zero() - val).collect();
    let h = lu_back_solve(&factors, &neg_rhs);

    Some(h)
}

/// Compute the full implicit Hessian tensor `d²z*/dx²` (m × n × n).
///
/// Returns `result[i][j][k]` = ∂²z*_i / (∂x_j ∂x_k). The tensor is symmetric
/// in the last two indices (j, k).
///
/// Cost: `n(n+1)/2` nested `Dual<Dual<F>>` forward passes plus `n(n+1)/2` back-solves,
/// all sharing a single LU factorization of `F_z`.
///
/// Returns `None` if `F_z` is singular.
pub fn implicit_hessian<F: Float>(
    tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    num_states: usize,
) -> Option<Vec<Vec<Vec<F>>>> {
    let n = x.len();
    let m = num_states;
    validate_inputs(tape, z_star, x, num_states);

    let (f_z, f_x) = compute_partitioned_jacobian(tape, z_star, x, num_states);
    let factors = lu_factor(&f_z)?;

    // First-order sensitivity columns: S[:,j] = -F_z^{-1} · F_x[:,j]
    let mut sens_cols: Vec<Vec<F>> = Vec::with_capacity(n);
    for j in 0..n {
        let neg_col: Vec<F> = f_x.iter().map(|row| F::zero() - row[j]).collect();
        sens_cols.push(lu_back_solve(&factors, &neg_col));
    }

    let out_indices = tape.all_output_indices();
    let mut result = vec![vec![vec![F::zero(); n]; n]; m];
    let mut buf: Vec<Dual<Dual<F>>> = Vec::new();

    for j in 0..n {
        for k in j..n {
            // ṗ = [S[:,j]; e_j], ẇ = [S[:,k]; e_k]
            let mut dd_inputs: Vec<Dual<Dual<F>>> = Vec::with_capacity(m + n);
            for i in 0..m {
                dd_inputs.push(Dual::new(
                    Dual::new(z_star[i], sens_cols[j][i]),
                    Dual::new(sens_cols[k][i], F::zero()),
                ));
            }
            for (l, &x_l) in x.iter().enumerate() {
                let p_l = if l == j { F::one() } else { F::zero() };
                let w_l = if l == k { F::one() } else { F::zero() };
                dd_inputs.push(Dual::new(Dual::new(x_l, p_l), Dual::new(w_l, F::zero())));
            }

            tape.forward_tangent(&dd_inputs, &mut buf);

            // Extract RHS and solve
            let mut rhs = Vec::with_capacity(m);
            for &idx in out_indices {
                rhs.push(buf[idx as usize].eps.eps);
            }
            let neg_rhs: Vec<F> = rhs.iter().map(|&val| F::zero() - val).collect();
            let h = lu_back_solve(&factors, &neg_rhs);

            for i in 0..m {
                result[i][j][k] = h[i];
                result[i][k][j] = h[i]; // Symmetric
            }
        }
    }

    Some(result)
}

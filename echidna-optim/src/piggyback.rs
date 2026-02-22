use echidna::{BytecodeTape, Dual, Float};

/// Validate that a step tape G: R^(m+n) -> R^m has the expected shape.
fn validate_step_tape<F: Float>(
    tape: &BytecodeTape<F>,
    z: &[F],
    x: &[F],
    num_states: usize,
) {
    assert_eq!(z.len(), num_states);
    assert_eq!(tape.num_inputs(), num_states + x.len());
    assert_eq!(
        tape.num_outputs(),
        num_states,
        "step tape must have num_outputs == num_states (G: R^(m+n) -> R^m)"
    );
}

/// One tangent piggyback step through a fixed-point map G.
///
/// Given the iteration `z_{k+1} = G(z_k, x)`, computes both the primal step
/// and the tangent propagation `ż_{k+1} = G_z · ż_k + G_x · ẋ` in a single
/// forward pass using dual numbers.
///
/// Returns `(z_new, z_dot_new)`.
pub fn piggyback_tangent_step<F: Float>(
    step_tape: &BytecodeTape<F>,
    z: &[F],
    x: &[F],
    z_dot: &[F],
    x_dot: &[F],
    num_states: usize,
) -> (Vec<F>, Vec<F>) {
    let mut buf = Vec::new();
    piggyback_tangent_step_with_buf(step_tape, z, x, z_dot, x_dot, num_states, &mut buf)
}

/// One tangent piggyback step, reusing `buf` across calls.
///
/// Same as [`piggyback_tangent_step`] but avoids reallocating the internal
/// dual-number buffer on each call.
pub fn piggyback_tangent_step_with_buf<F: Float>(
    step_tape: &BytecodeTape<F>,
    z: &[F],
    x: &[F],
    z_dot: &[F],
    x_dot: &[F],
    num_states: usize,
    buf: &mut Vec<Dual<F>>,
) -> (Vec<F>, Vec<F>) {
    validate_step_tape(step_tape, z, x, num_states);
    let m = num_states;
    let n = x.len();
    assert_eq!(z_dot.len(), m, "z_dot length must equal num_states");
    assert_eq!(x_dot.len(), n, "x_dot length must equal x length");

    // Build dual inputs: [Dual(z_i, ż_i), ..., Dual(x_j, ẋ_j), ...]
    let mut dual_inputs = Vec::with_capacity(m + n);
    for i in 0..m {
        dual_inputs.push(Dual::new(z[i], z_dot[i]));
    }
    for j in 0..n {
        dual_inputs.push(Dual::new(x[j], x_dot[j]));
    }

    step_tape.forward_tangent(&dual_inputs, buf);

    // Extract outputs: .re -> z_new, .eps -> z_dot_new
    let out_indices = step_tape.all_output_indices();
    let mut z_new = Vec::with_capacity(m);
    let mut z_dot_new = Vec::with_capacity(m);
    for &idx in out_indices {
        let d = buf[idx as usize];
        z_new.push(d.re);
        z_dot_new.push(d.eps);
    }

    (z_new, z_dot_new)
}

/// Tangent piggyback solve: find fixed point z* = G(z*, x) and its tangent ż*.
///
/// Iterates the fixed-point map `z_{k+1} = G(z_k, x)` while simultaneously
/// propagating tangents `ż_{k+1} = G_z · ż_k + G_x · ẋ`.
///
/// Returns `Some((z_star, z_dot_star, iterations))` on convergence, `None` on
/// divergence or exceeding `max_iter`.
pub fn piggyback_tangent_solve<F: Float>(
    step_tape: &BytecodeTape<F>,
    z0: &[F],
    x: &[F],
    x_dot: &[F],
    num_states: usize,
    max_iter: usize,
    tol: F,
) -> Option<(Vec<F>, Vec<F>, usize)> {
    let m = num_states;
    let mut z = z0.to_vec();
    let mut z_dot = vec![F::zero(); m];
    let mut buf = Vec::new();

    for k in 0..max_iter {
        let (z_new, z_dot_new) =
            piggyback_tangent_step_with_buf(step_tape, &z, x, &z_dot, x_dot, num_states, &mut buf);

        // Relative convergence: ||z_new - z|| / (1 + ||z||)
        let mut delta_sq = F::zero();
        let mut z_sq = F::zero();
        for i in 0..m {
            let d = z_new[i] - z[i];
            delta_sq = delta_sq + d * d;
            z_sq = z_sq + z[i] * z[i];
        }
        let norm = delta_sq.sqrt() / (F::one() + z_sq.sqrt());
        if !norm.is_finite() {
            return None;
        }
        if norm < tol {
            return Some((z_new, z_dot_new, k + 1));
        }

        z = z_new;
        z_dot = z_dot_new;
    }

    None
}

/// Adjoint piggyback solve at a converged fixed point z* = G(z*, x).
///
/// Iterates the adjoint fixed-point equation `λ_{k+1} = G_z^T · λ_k + z̄`
/// using reverse-mode sweeps through the step tape. At convergence, returns
/// `x̄ = G_x^T · λ*`.
///
/// Requires z* to already be computed (e.g. by the primal solver).
/// The iteration converges when G is a contraction (‖G_z‖ < 1).
///
/// Returns `Some((x_bar, iterations))` on convergence, `None` on divergence
/// or exceeding `max_iter`.
pub fn piggyback_adjoint_solve<F: Float>(
    step_tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    z_bar: &[F],
    num_states: usize,
    max_iter: usize,
    tol: F,
) -> Option<(Vec<F>, usize)> {
    validate_step_tape(step_tape, z_star, x, num_states);
    let m = num_states;
    assert_eq!(z_bar.len(), m, "z_bar length must equal num_states");

    // Set primal values: forward([z*, x])
    let mut input = Vec::with_capacity(m + x.len());
    input.extend_from_slice(z_star);
    input.extend_from_slice(x);
    step_tape.forward(&input);

    let mut lambda = z_bar.to_vec();

    for k in 0..max_iter {
        // reverse_seeded(λ) returns [G_z^T · λ; G_x^T · λ] (length m+n)
        let adj = step_tape.reverse_seeded(&lambda);

        // λ_new[i] = adj[i] + z_bar[i] for i = 0..m
        let mut lambda_new = Vec::with_capacity(m);
        let mut delta_sq = F::zero();
        let mut lam_sq = F::zero();
        for i in 0..m {
            let l_new = adj[i] + z_bar[i];
            let d = l_new - lambda[i];
            delta_sq = delta_sq + d * d;
            lam_sq = lam_sq + lambda[i] * lambda[i];
            lambda_new.push(l_new);
        }

        let norm = delta_sq.sqrt() / (F::one() + lam_sq.sqrt());
        if !norm.is_finite() {
            return None;
        }
        if norm < tol {
            // adj[m..] = G_x^T · λ* (already computed with the converged λ)
            return Some((adj[m..].to_vec(), k + 1));
        }

        lambda = lambda_new;
    }

    None
}

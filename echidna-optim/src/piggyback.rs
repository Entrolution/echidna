use std::fmt;

use echidna::{BytecodeTape, Dual, Float};

/// Reason a piggyback solve failed to converge.
///
/// Marked `#[non_exhaustive]` so future variants can be added without
/// breaking exhaustive `match`es. `last_norm` / `last_z_norm` /
/// `last_lam_norm` fields use `f64` (cast via `Float::to_f64`) for
/// uniform diagnostic output regardless of the solver's `F` type
/// — matches the precedent in `implicit.rs`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PiggybackError {
    /// The primal `z_{k+1} = G(z_k, x)` produced a non-finite norm
    /// (relative-norm `||z_new - z||/(1 + ||z||)` is NaN/Inf), or
    /// the primal vector itself contained non-finite components in
    /// the forward-adjoint loop.
    PrimalDivergence { iteration: usize },
    /// Primal stayed finite but the tangent
    /// `ż_{k+1} = G_z · ż_k + G_x · ẋ` produced non-finite values.
    /// Catches the ratio-converging case where the primal norm
    /// remains bounded while individual tangent components overflow.
    TangentDivergence { iteration: usize },
    /// Adjoint `λ_{k+1} = G_z^T · λ_k + z̄` produced non-finite
    /// values (norm or individual components).
    AdjointDivergence { iteration: usize },
    /// Reached `max_iter` without satisfying the convergence
    /// criterion. Both norms are *relative* (`||δ|| / (1 + ||state||)`),
    /// matching the scale of the `tol` argument — so a value just over
    /// `tol` signals proximity to convergence and a value many orders
    /// of magnitude over `tol` signals stagnation. `z_norm` is
    /// `Some(...)` for solvers that track a primal/tangent norm
    /// (`piggyback_tangent_solve`, `piggyback_forward_adjoint_solve`);
    /// `lam_norm` is `Some(...)` for solvers that track an adjoint
    /// norm (`piggyback_adjoint_solve`,
    /// `piggyback_forward_adjoint_solve`). At least one of the two is
    /// always `Some`. The reported value is the final iteration's
    /// residual, not a min/avg across iterations.
    MaxIterations {
        z_norm: Option<f64>,
        lam_norm: Option<f64>,
    },
}

impl fmt::Display for PiggybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PiggybackError::PrimalDivergence { iteration } => {
                write!(f, "piggyback: primal diverged at iteration {iteration}")
            }
            PiggybackError::TangentDivergence { iteration } => {
                write!(f, "piggyback: tangent diverged at iteration {iteration}")
            }
            PiggybackError::AdjointDivergence { iteration } => {
                write!(f, "piggyback: adjoint diverged at iteration {iteration}")
            }
            PiggybackError::MaxIterations { z_norm, lam_norm } => {
                write!(
                    f,
                    "piggyback: max_iter exceeded (z_norm = {z_norm:?}, lam_norm = {lam_norm:?})"
                )
            }
        }
    }
}

impl std::error::Error for PiggybackError {}

// Compile-time check that `PiggybackError` stays `Send + Sync`. Future
// variants carrying non-`Send`/`Sync` payloads will trigger a build
// failure here rather than at the (often distant) call site.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PiggybackError>();
};

/// Validate that a step tape G: R^(m+n) -> R^m has the expected shape.
///
/// Uses `assert_eq!` (panic) rather than `Result` because shape
/// mismatches are programmer errors — calling `piggyback_*_solve` with
/// an inconsistent tape is a contract violation, not a runtime
/// numerical failure that callers should recover from.
fn validate_step_tape<F: Float>(tape: &BytecodeTape<F>, z: &[F], x: &[F], num_states: usize) {
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
/// Returns `Ok((z_star, z_dot_star, iterations))` on convergence. Returns
/// `Err(PiggybackError::PrimalDivergence)` when the primal norm becomes
/// non-finite, `Err(PiggybackError::TangentDivergence)` when the primal
/// stays finite but the tangent overflows (ratio-converging case), or
/// `Err(PiggybackError::MaxIterations { z_norm, lam_norm: None })` when
/// `max_iter` is reached without satisfying `tol`.
pub fn piggyback_tangent_solve<F: Float>(
    step_tape: &BytecodeTape<F>,
    z0: &[F],
    x: &[F],
    x_dot: &[F],
    num_states: usize,
    max_iter: usize,
    tol: F,
) -> Result<(Vec<F>, Vec<F>, usize), PiggybackError> {
    let m = num_states;
    let mut z = z0.to_vec();
    let mut z_dot = vec![F::zero(); m];
    let mut buf = Vec::new();
    let mut last_norm: f64 = f64::NAN;

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
        // Variant-mapping order: norm-check first → PrimalDivergence;
        // tangent-finite check second → TangentDivergence. A non-finite
        // primal naturally produces a non-finite norm, so it falls into
        // PrimalDivergence by detection priority.
        if !norm.is_finite() {
            return Err(PiggybackError::PrimalDivergence { iteration: k });
        }
        // Detect tangent divergence even when the primal `z_new` itself is
        // finite: the JVP iteration `z_dot_{k+1} = G_z·z_dot_k + G_x·x_dot`
        // can produce Inf/NaN tangents that a primal-only norm check misses.
        if !z_dot_new.iter().all(|v| v.is_finite()) {
            return Err(PiggybackError::TangentDivergence { iteration: k });
        }
        last_norm = norm.to_f64().unwrap_or(f64::NAN);
        if norm < tol {
            return Ok((z_new, z_dot_new, k + 1));
        }

        z = z_new;
        z_dot = z_dot_new;
    }

    Err(PiggybackError::MaxIterations {
        z_norm: Some(last_norm),
        lam_norm: None,
    })
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
/// Returns `Ok((x_bar, iterations))` on convergence. Returns
/// `Err(PiggybackError::AdjointDivergence)` when the adjoint norm is
/// non-finite or `lambda_new` overflows (ratio-converging case), or
/// `Err(PiggybackError::MaxIterations { z_norm: None, lam_norm })`
/// when `max_iter` is reached without satisfying `tol`.
pub fn piggyback_adjoint_solve<F: Float>(
    step_tape: &mut BytecodeTape<F>,
    z_star: &[F],
    x: &[F],
    z_bar: &[F],
    num_states: usize,
    max_iter: usize,
    tol: F,
) -> Result<(Vec<F>, usize), PiggybackError> {
    validate_step_tape(step_tape, z_star, x, num_states);
    let m = num_states;
    assert_eq!(z_bar.len(), m, "z_bar length must equal num_states");

    // Set primal values: forward([z*, x])
    let mut input = Vec::with_capacity(m + x.len());
    input.extend_from_slice(z_star);
    input.extend_from_slice(x);
    step_tape.forward(&input);

    let mut lambda = z_bar.to_vec();
    let mut last_norm: f64 = f64::NAN;

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
            return Err(PiggybackError::AdjointDivergence { iteration: k });
        }
        // A ratio-converging iteration with exponentially-growing `lambda`
        // magnitudes (spectral radius of `G_z^T` ≥ 1) can produce finite
        // `norm` while `lambda_new` is Inf/NaN. Explicit finite check
        // catches the divergence regardless of ratio behaviour.
        if !lambda_new.iter().all(|v| v.is_finite()) {
            return Err(PiggybackError::AdjointDivergence { iteration: k });
        }
        last_norm = norm.to_f64().unwrap_or(f64::NAN);
        if norm < tol {
            // One extra reverse pass with converged lambda to get consistent x_bar.
            // Without this, adj[m..] uses the pre-convergence lambda, introducing
            // O(tol * ||G_x||) error.
            let adj_final = step_tape.reverse_seeded(&lambda_new);
            return Ok((adj_final[m..].to_vec(), k + 1));
        }

        lambda = lambda_new;
    }

    Err(PiggybackError::MaxIterations {
        z_norm: None,
        lam_norm: Some(last_norm),
    })
}

/// Interleaved forward-adjoint piggyback solve.
///
/// Simultaneously iterates the primal fixed-point `z_{k+1} = G(z_k, x)` and
/// the adjoint equation `λ_{k+1} = G_z^T · λ_k + z̄`. This cuts the total
/// iteration count from `K_primal + K_adjoint` to `max(K_primal, K_adjoint)`.
///
/// Returns `Ok((z_star, x_bar, iterations))` when both `z` and `λ` converge.
/// Returns `Err(PiggybackError::PrimalDivergence)` when `z_norm` becomes
/// non-finite or `z_new` itself contains non-finite components,
/// `Err(PiggybackError::AdjointDivergence)` when the adjoint norm or
/// `lambda_new` overflows, or
/// `Err(PiggybackError::MaxIterations { z_norm: Some, lam_norm: Some })`
/// when `max_iter` is reached without satisfying `tol`.
pub fn piggyback_forward_adjoint_solve<F: Float>(
    step_tape: &mut BytecodeTape<F>,
    z0: &[F],
    x: &[F],
    z_bar: &[F],
    num_states: usize,
    max_iter: usize,
    tol: F,
) -> Result<(Vec<F>, Vec<F>, usize), PiggybackError> {
    validate_step_tape(step_tape, z0, x, num_states);
    let m = num_states;
    assert_eq!(z_bar.len(), m, "z_bar length must equal num_states");

    // Pre-allocate input buffer [z, x]
    let mut input = Vec::with_capacity(m + x.len());
    input.extend_from_slice(z0);
    input.extend_from_slice(x);

    let mut lambda = z_bar.to_vec();
    let mut last_z_norm: f64 = f64::NAN;
    let mut last_lam_norm: f64 = f64::NAN;

    for k in 0..max_iter {
        // Forward pass at current z
        step_tape.forward(&input);
        let z_new = step_tape.output_values();

        // Reverse pass with current λ
        let adj = step_tape.reverse_seeded(&lambda);

        // Primal convergence: ||z_new - z|| / (1 + ||z||)
        let mut z_delta_sq = F::zero();
        let mut z_sq = F::zero();
        for i in 0..m {
            let d = z_new[i] - input[i];
            z_delta_sq = z_delta_sq + d * d;
            z_sq = z_sq + input[i] * input[i];
        }
        let z_norm = z_delta_sq.sqrt() / (F::one() + z_sq.sqrt());
        if !z_norm.is_finite() {
            return Err(PiggybackError::PrimalDivergence { iteration: k });
        }

        // Adjoint update and convergence: λ_new = G_z^T · λ + z̄
        let mut lam_delta_sq = F::zero();
        let mut lam_sq = F::zero();
        let mut lambda_new = Vec::with_capacity(m);
        for i in 0..m {
            let l_new = adj[i] + z_bar[i];
            let d = l_new - lambda[i];
            lam_delta_sq = lam_delta_sq + d * d;
            lam_sq = lam_sq + lambda[i] * lambda[i];
            lambda_new.push(l_new);
        }
        let lam_norm = lam_delta_sq.sqrt() / (F::one() + lam_sq.sqrt());
        if !lam_norm.is_finite() {
            return Err(PiggybackError::AdjointDivergence { iteration: k });
        }
        // Same divergence case as the standalone solvers: a ratio-converging
        // iteration with exponentially-growing lambda magnitudes can produce
        // finite `lam_norm` while `lambda_new` itself is Inf/NaN.
        if !lambda_new.iter().all(|v| v.is_finite()) {
            return Err(PiggybackError::AdjointDivergence { iteration: k });
        }
        if !z_new.iter().all(|v| v.is_finite()) {
            return Err(PiggybackError::PrimalDivergence { iteration: k });
        }

        last_z_norm = z_norm.to_f64().unwrap_or(f64::NAN);
        last_lam_norm = lam_norm.to_f64().unwrap_or(f64::NAN);

        if z_norm < tol && lam_norm < tol {
            // One extra reverse pass with converged lambda_new to get consistent x_bar,
            // matching the pattern in piggyback_adjoint_solve.
            input[..m].copy_from_slice(&z_new[..m]);
            step_tape.forward(&input);
            let adj_final = step_tape.reverse_seeded(&lambda_new);
            return Ok((z_new, adj_final[m..].to_vec(), k + 1));
        }

        // Update z in the input buffer
        input[..m].copy_from_slice(&z_new[..m]);
        lambda = lambda_new;
    }

    Err(PiggybackError::MaxIterations {
        z_norm: Some(last_z_norm),
        lam_norm: Some(last_lam_norm),
    })
}

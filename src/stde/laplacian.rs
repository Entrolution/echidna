use super::estimator::Laplacian;
use super::jet::{directional_derivatives, taylor_jet_2nd_with_buf};
use super::pipeline::estimate;
use super::types::{EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::taylor::Taylor;
use crate::Float;

/// Estimate the Laplacian (trace of Hessian) via Hutchinson's trace estimator.
///
/// Directions must satisfy E[vv^T] = I (e.g. Rademacher vectors with entries
/// +/-1, or standard Gaussian vectors). The estimator is:
///
///   Laplacian ~ (1/S) * sum_s 2*c2_s
///
/// where c2_s is the second Taylor coefficient for direction s.
///
/// Returns `(value, laplacian_estimate)`.
///
/// Note: coordinate basis vectors do **not** satisfy E[vv^T] = I and will
/// give tr(H)/n instead of tr(H). Use [`hessian_diagonal`] and sum for exact
/// computation via coordinate directions.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian<F: Float>(tape: &BytecodeTape<F>, x: &[F], directions: &[&[F]]) -> (F, F) {
    assert!(!directions.is_empty(), "directions must not be empty");

    let (value, _, second_order) = directional_derivatives(tape, x, directions);

    let two = F::from(2.0).unwrap();
    let s = F::from(directions.len()).unwrap();
    let sum: F = second_order
        .iter()
        .fold(F::zero(), |acc, &c2| acc + two * c2);
    let laplacian = sum / s;

    (value, laplacian)
}

/// Estimate the Laplacian with sample statistics via Hutchinson's trace estimator.
///
/// Same estimator as [`laplacian`], but additionally computes sample variance
/// and standard error using Welford's online algorithm (numerically stable,
/// single pass).
///
/// Each direction produces a sample `2 * c2_s`. The returned statistics
/// describe the distribution of these samples.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian_with_stats<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> EstimatorResult<F> {
    estimate(&Laplacian, tape, x, directions)
}

/// Estimate the Laplacian with a diagonal control variate.
///
/// Uses the exact Hessian diagonal (from [`hessian_diagonal`]) as a control
/// variate to reduce estimator variance. Each raw Hutchinson sample
/// `raw_s = v^T H v = 2 * c2_s` is adjusted:
///
/// ```text
/// adjusted_s = raw_s - sum_j(D_jj * v_j^2) + tr(D)
/// ```
///
/// where D is the diagonal of H. The adjustment subtracts the noisy diagonal
/// contribution and adds back its exact expectation `tr(D)`.
///
/// **Effect by distribution**:
/// - **Gaussian**: reduces variance from `2||H||_F^2` to `2 sum_{i≠j} H_ij^2`
///   (matching Rademacher performance).
/// - **Rademacher**: no effect, since `v_j^2 = 1` always, so the adjustment
///   is `sum_j D_jj * 1 - tr(D) = 0`.
///
/// Returns an [`EstimatorResult`] with statistics computed over the adjusted
/// samples.
///
/// # Panics
///
/// Panics if `directions` is empty, if any direction's length does not match
/// `tape.num_inputs()`, or if `control_diagonal.len() != tape.num_inputs()`.
pub fn laplacian_with_control<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
    control_diagonal: &[F],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    assert_eq!(
        control_diagonal.len(),
        n,
        "control_diagonal.len() must match tape.num_inputs()"
    );

    let two = F::from(2.0).unwrap();
    let trace_control: F = control_diagonal
        .iter()
        .copied()
        .fold(F::zero(), |a, b| a + b);

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for v in directions.iter() {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;

        let raw = two * c2;

        // Control variate: subtract v^T D v, add back E[v^T D v] = tr(D)
        let cv: F = control_diagonal
            .iter()
            .zip(v.iter())
            .fold(F::zero(), |acc, (&d, &vi)| acc + d * vi * vi);
        acc.update(raw - cv + trace_control);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: directions.len(),
    }
}

/// Exact Hessian diagonal via n coordinate-direction evaluations.
///
/// For each coordinate j, pushes basis vector e_j through the tape and
/// reads `2 * c2`, which equals `d^2 f / dx_j^2`.
///
/// Returns `(value, diag)` where `diag[j] = d^2 f / dx_j^2`.
pub fn hessian_diagonal<F: Float>(tape: &BytecodeTape<F>, x: &[F]) -> (F, Vec<F>) {
    let mut buf = Vec::new();
    hessian_diagonal_with_buf(tape, x, &mut buf)
}

/// Like [`hessian_diagonal`] but reuses a caller-provided buffer.
pub fn hessian_diagonal_with_buf<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    buf: &mut Vec<Taylor<F, 3>>,
) -> (F, Vec<F>) {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let two = F::from(2.0).unwrap();
    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    // Build basis vector once, mutate the hot coordinate
    let mut e = vec![F::zero(); n];
    for j in 0..n {
        e[j] = F::one();
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &e, buf);
        value = c0;
        diag.push(two * c2);
        e[j] = F::zero();
    }

    (value, diag)
}

/// Modified Gram-Schmidt orthonormalisation, in-place.
///
/// Orthonormalises the columns of the matrix represented as `columns: &mut Vec<Vec<F>>`.
/// Drops near-zero columns (norm < epsilon). Returns the rank (number of retained columns).
fn modified_gram_schmidt<F: Float>(columns: &mut Vec<Vec<F>>, epsilon: F) -> usize {
    let mut rank = 0;
    let mut i = 0;
    while i < columns.len() {
        // Orthogonalise against all previously accepted columns
        for j in 0..rank {
            // Split to satisfy the borrow checker: j < rank <= i
            let (left, right) = columns.split_at_mut(i);
            let qj = &left[j];
            let ci = &mut right[0];
            let dot: F = qj
                .iter()
                .zip(ci.iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
            for (c, &q) in ci.iter_mut().zip(qj.iter()) {
                *c = *c - dot * q;
            }
        }

        // Compute norm
        let norm_sq: F = columns[i].iter().fold(F::zero(), |acc, &v| acc + v * v);
        let norm = norm_sq.sqrt();

        if norm < epsilon {
            // Drop this column (near-zero after projection)
            columns.swap_remove(i);
            // Don't increment i — swapped element needs processing
        } else {
            // Normalise
            let inv_norm = F::one() / norm;
            for v in columns[i].iter_mut() {
                *v = *v * inv_norm;
            }
            // Move accepted column to rank position. Note: i == rank is a loop invariant
            // (both start at 0, both increment on accept, neither changes on reject+swap_remove),
            // so this swap is effectively a no-op. Kept as a defensive guard.
            if i != rank {
                columns.swap(i, rank);
            }
            rank += 1;
            i += 1;
        }
    }
    columns.truncate(rank);
    rank
}

/// Hutch++ trace estimator (Meyer et al. 2021) for the Laplacian.
///
/// Achieves O(1/S²) convergence for matrices with decaying eigenvalues by
/// splitting the work into:
///
/// 1. **Sketch phase**: k HVPs (via `tape.hvp_with_buf`) produce columns of H·S,
///    which are orthonormalised via Modified Gram-Schmidt to give a basis Q.
/// 2. **Exact subspace trace**: For each q_i in Q, `taylor_jet_2nd` gives
///    q_i^T H q_i. Sum = tr(Q^T H Q) — this part has zero variance.
/// 3. **Residual Hutchinson**: For each stochastic direction g_s, project out Q
///    (g' = g - Q(Q^T g)) and estimate the residual trace via `taylor_jet_2nd`.
/// 4. **Total** = exact_trace + residual_mean.
///
/// The variance and standard error in the result refer to the residual only.
///
/// # Arguments
///
/// - `sketch_directions`: k directions for the sketch phase. Typically Rademacher
///   or Gaussian. More directions capture more of the spectrum exactly.
/// - `stochastic_directions`: S directions for residual estimation. Rademacher recommended.
///
/// # Cost
///
/// k HVPs (≈2k forward passes) + k Taylor jets (exact subspace) + S Taylor jets
/// (residual) + O(k²·n) for Gram-Schmidt.
///
/// # Panics
///
/// Panics if `sketch_directions` or `stochastic_directions` is empty, or if any
/// direction's length does not match `tape.num_inputs()`.
pub fn laplacian_hutchpp<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sketch_directions: &[&[F]],
    stochastic_directions: &[&[F]],
) -> EstimatorResult<F> {
    assert!(
        !sketch_directions.is_empty(),
        "sketch_directions must not be empty"
    );
    assert!(
        !stochastic_directions.is_empty(),
        "stochastic_directions must not be empty"
    );

    let n = tape.num_inputs();
    let two = F::from(2.0).unwrap();
    let eps = F::epsilon().sqrt();

    // ── Step 1: Sketch — k HVPs to get columns of H·S ──
    let mut dual_vals_buf = Vec::new();
    let mut adjoint_buf = Vec::new();
    let mut hs_columns: Vec<Vec<F>> = Vec::with_capacity(sketch_directions.len());

    for s in sketch_directions {
        assert_eq!(
            s.len(),
            n,
            "sketch direction length must match tape.num_inputs()"
        );
        let (_grad, hvp) = tape.hvp_with_buf(x, s, &mut dual_vals_buf, &mut adjoint_buf);
        hs_columns.push(hvp);
    }

    // ── Step 2: QR via Modified Gram-Schmidt ──
    let rank = modified_gram_schmidt(&mut hs_columns, eps);
    let q = &hs_columns; // q[0..rank] are orthonormal basis vectors

    // ── Step 3: Exact subspace trace ──
    let mut taylor_buf = Vec::new();
    let mut value = F::zero();
    let mut exact_trace = F::zero();

    for qi in q.iter().take(rank) {
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, qi, &mut taylor_buf);
        value = c0;
        exact_trace = exact_trace + two * c2; // q_i^T H q_i
    }

    // ── Step 4: Residual Hutchinson ──
    // For each stochastic direction g, project out Q: g' = g - Q(Q^T g)
    let mut acc = WelfordAccumulator::new();
    let mut projected = vec![F::zero(); n];

    for g in stochastic_directions.iter() {
        assert_eq!(
            g.len(),
            n,
            "stochastic direction length must match tape.num_inputs()"
        );

        // g' = g - Q(Q^T g)
        // NOTE (verified correct): Dot products use original `g` (not progressively
        // projected vector). This is mathematically equivalent because Q is orthonormal:
        // (I - q₂q₂ᵀ)(I - q₁q₁ᵀ)g = g - q₁(q₁ᵀg) - q₂(q₂ᵀg) when q₁ᵀq₂ = 0.
        projected.copy_from_slice(g);
        for qi in q.iter().take(rank) {
            let dot: F = qi
                .iter()
                .zip(g.iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
            for (p, &qv) in projected.iter_mut().zip(qi.iter()) {
                *p = *p - dot * qv;
            }
        }

        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &projected, &mut taylor_buf);
        value = c0;
        acc.update(two * c2);
    }

    let (residual_mean, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate: exact_trace + residual_mean,
        sample_variance,
        standard_error,
        num_samples: stochastic_directions.len(),
    }
}

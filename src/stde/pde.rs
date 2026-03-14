use super::jet::taylor_jet_2nd_with_buf;
use super::types::{DivergenceResult, EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::dual::Dual;
use crate::Float;

/// Estimate the divergence (trace of Jacobian) of a vector field f: R^n → R^n.
///
/// Uses Hutchinson's trace estimator with first-order forward-mode AD (`Dual<F>`).
/// The tape must be recorded via [`record_multi`](crate::record_multi) with
/// `num_inputs == num_outputs`.
///
/// For each random direction v with `E[vv^T] = I`:
/// 1. Seed `Dual` inputs: `Dual(x_i, v_i)`
/// 2. Forward pass: `tape.forward_tangent(&dual_inputs, &mut buf)`
/// 3. Sample = Σ_i v_i * buf\[out_indices\[i\]\].eps = v^T J v
///
/// The mean of these samples converges to tr(J) = div(f).
///
/// # Panics
///
/// Panics if `directions` is empty, if `tape.num_outputs() != tape.num_inputs()`,
/// or if any direction's length does not match `tape.num_inputs()`.
pub fn divergence<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> DivergenceResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    let m = tape.num_outputs();
    assert_eq!(
        m, n,
        "divergence requires num_outputs ({}) == num_inputs ({})",
        m, n
    );
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let out_indices = tape.all_output_indices();
    let mut buf: Vec<Dual<F>> = Vec::new();
    let mut values = vec![F::zero(); m];
    let mut acc = WelfordAccumulator::new();

    for (k, v) in directions.iter().enumerate() {
        assert_eq!(v.len(), n, "direction length must match tape.num_inputs()");

        // Build Dual inputs
        let dual_inputs: Vec<Dual<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| Dual::new(xi, vi))
            .collect();

        tape.forward_tangent(&dual_inputs, &mut buf);

        // Extract function values (from first direction only is fine — all give same primal)
        if k == 0 {
            for (i, &oi) in out_indices.iter().enumerate() {
                values[i] = buf[oi as usize].re;
            }
        }

        // sample = v^T J v = Σ_i v_i * (J v)_i
        let sample: F = v
            .iter()
            .zip(out_indices.iter())
            .fold(F::zero(), |acc, (&vi, &oi)| acc + vi * buf[oi as usize].eps);

        acc.update(sample);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    DivergenceResult {
        values,
        estimate,
        sample_variance,
        standard_error,
        num_samples: directions.len(),
    }
}

/// Estimate the diffusion term `½ tr(σσ^T · Hess u)` for parabolic PDEs.
///
/// Uses the σ-transform: `½ tr(σσ^T H) = ½ Σ_i (σ·e_i)^T H (σ·e_i)`
/// where `σ·e_i` are the columns of σ (paper Eq 19, Appendix E).
///
/// Each column pushforward gives `(σ·e_i)^T H (σ·e_i)` via the second-order
/// Taylor coefficient: `2 * c2`. The diffusion term is `½ * Σ 2*c2 = Σ c2`.
///
/// # Arguments
///
/// - `sigma_columns`: the columns of σ as slices, each of length `tape.num_inputs()`.
///
/// Returns `(value, diffusion_term)`.
///
/// # Panics
///
/// Panics if `sigma_columns` is empty or any column's length does not match
/// `tape.num_inputs()`.
pub fn parabolic_diffusion<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sigma_columns: &[&[F]],
) -> (F, F) {
    assert!(!sigma_columns.is_empty(), "sigma_columns must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let half = F::from(0.5).unwrap();
    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut sum = F::zero();

    for col in sigma_columns {
        assert_eq!(
            col.len(),
            n,
            "sigma column length must match tape.num_inputs()"
        );
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, col, &mut buf);
        value = c0;
        sum = sum + two * c2; // (σ·e_i)^T H (σ·e_i)
    }

    (value, half * sum)
}

/// Stochastic version of [`parabolic_diffusion`]: subsample column indices.
///
/// Estimate = `(d / |J|) · ½ · Σ_{i∈J} (σ·e_i)^T H (σ·e_i)`.
///
/// # Panics
///
/// Panics if `sampled_indices` is empty or any index is out of bounds.
pub fn parabolic_diffusion_stochastic<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    sigma_columns: &[&[F]],
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    let n = tape.num_inputs();
    let d = sigma_columns.len();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let two = F::from(2.0).unwrap();
    let half = F::from(0.5).unwrap();
    let df = F::from(d).unwrap();

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for &i in sampled_indices {
        assert!(i < d, "sampled index {} out of bounds (d={})", i, d);
        let col = sigma_columns[i];
        assert_eq!(
            col.len(),
            n,
            "sigma column length must match tape.num_inputs()"
        );
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, col, &mut buf);
        value = c0;
        acc.update(two * c2); // (σ·e_i)^T H (σ·e_i)
    }

    let (mean, sample_variance, standard_error) = acc.finalize();

    // Unbiased estimator for ½ Σ_i (σ·e_i)^T H (σ·e_i): d * mean * ½
    EstimatorResult {
        value,
        estimate: mean * df * half,
        sample_variance,
        standard_error,
        num_samples: sampled_indices.len(),
    }
}

/// Dense STDE for a positive-definite 2nd-order operator.
///
/// Given a Cholesky factor `L` (lower-triangular) such that `C = L L^T`,
/// and standard Gaussian vectors `z_s ~ N(0, I)`, this computes
/// `v_s = L · z_s` internally and estimates
/// `tr(C · H_u(x)) = Σ_{ij} C_{ij} ∂²u/∂x_i∂x_j`.
///
/// The key insight: if `E[v v^T] = C`, then `E[v^T H v] = tr(C · H)`.
/// With `v = L · z` where `z ~ N(0, I)`, we get `E[vv^T] = L L^T = C`.
///
/// The caller provides `z_vectors` (standard Gaussian samples) and
/// `cholesky_rows` (the rows of L — only lower-triangular entries matter).
///
/// **Indefinite C deferred**: For indefinite operators, manually split into
/// `C⁺ - C⁻`, compute Cholesky factors for each, and call twice.
///
/// **No `rand` dependency**: callers provide z_vectors.
///
/// # Panics
///
/// Panics if `z_vectors` is empty, `cholesky_rows.len()` does not match
/// `tape.num_inputs()`, or any vector has the wrong length.
pub fn dense_stde_2nd<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    cholesky_rows: &[&[F]],
    z_vectors: &[&[F]],
) -> EstimatorResult<F> {
    assert!(!z_vectors.is_empty(), "z_vectors must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(
        cholesky_rows.len(),
        n,
        "cholesky_rows.len() must match tape.num_inputs()"
    );

    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut v = vec![F::zero(); n];
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for z in z_vectors.iter() {
        assert_eq!(z.len(), n, "z_vector length must match tape.num_inputs()");

        // Compute v = L · z (lower-triangular mat-vec)
        for i in 0..n {
            let row = cholesky_rows[i];
            let mut sum = F::zero();
            // Only use lower-triangular entries: j <= i
            for j in 0..=i {
                sum = sum + row[j] * z[j];
            }
            v[i] = sum;
        }

        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &v, &mut buf);
        value = c0;
        // 2 * c2 = v^T H v, and E[v^T H v] = tr(C · H) when E[vv^T] = C
        acc.update(two * c2);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: z_vectors.len(),
    }
}

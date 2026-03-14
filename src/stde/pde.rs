use super::jet::taylor_jet_2nd_with_buf;
use super::types::{DivergenceResult, EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::dual::Dual;
use crate::Float;

/// Inner loop for STDE: apply mat-vec, push through Taylor jet, Welford-accumulate.
///
/// `matvec` takes `(z, v)` and writes `v = M · z` for some matrix M.
/// Each z-vector produces a sample `2 * c2` where `c2 = v^T H v / 2`.
fn stde_2nd_inner<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    z_vectors: &[&[F]],
    matvec: impl Fn(&[F], &mut [F]),
) -> EstimatorResult<F> {
    let n = tape.num_inputs();
    let two = F::from(2.0).unwrap();
    let mut buf = Vec::new();
    let mut v = vec![F::zero(); n];
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for z in z_vectors.iter() {
        assert_eq!(z.len(), n, "z_vector length must match tape.num_inputs()");
        matvec(z, &mut v);
        let (c0, _, c2) = taylor_jet_2nd_with_buf(tape, x, &v, &mut buf);
        value = c0;
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

    // Unbiased estimator for ½ Σ_i (σ·e_i)^T H (σ·e_i): d * mean * ½.
    // Scale variance/SE to match the rescaled estimate.
    let scale = df * half;
    EstimatorResult {
        value,
        estimate: mean * scale,
        sample_variance: sample_variance * scale * scale,
        standard_error: standard_error * scale,
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

    stde_2nd_inner(tape, x, z_vectors, |z, v| {
        // Lower-triangular mat-vec: v = L · z
        for i in 0..n {
            let row = cholesky_rows[i];
            let mut sum = F::zero();
            for j in 0..=i {
                sum = sum + row[j] * z[j];
            }
            v[i] = sum;
        }
    })
}

/// Dense STDE for a possibly-indefinite 2nd-order operator.
///
/// Given a symmetric (possibly indefinite) coefficient matrix `C`, estimates
/// `tr(C · H_u(x)) = Σ_{ij} C_{ij} ∂²u/∂x_i∂x_j`.
///
/// # Algorithm
///
/// 1. Eigendecompose: `C = Q Λ Qᵀ` via `nalgebra::SymmetricEigen`.
/// 2. Clamp near-zero eigenvalues: `|λᵢ| < ε` → 0, where `ε = eps_factor * max(|λᵢ|)`.
/// 3. Split: `λ⁺ᵢ = max(λᵢ, 0)`, `λ⁻ᵢ = max(-λᵢ, 0)`.
/// 4. Form square-root factors: `L⁺ = Q · diag(√λ⁺)`, `L⁻ = Q · diag(√λ⁻)`.
/// 5. For each z-vector, compute `v⁺ = L⁺·z`, `v⁻ = L⁻·z`, push through Taylor jets,
///    and accumulate `2·c2⁺ - 2·c2⁻`.
///
/// If all eigenvalues are non-negative (or non-positive), the negative (or positive)
/// half is skipped entirely for efficiency.
///
/// # Arguments
///
/// - `c_matrix`: symmetric `n×n` coefficient matrix (row-major `DMatrix<f64>`).
/// - `z_vectors`: standard Gaussian random vectors, each of length n.
/// - `eps_factor`: relative threshold for clamping near-zero eigenvalues.
///   Typical value: `1e-12`. Eigenvalues with `|λ| < eps_factor * max(|λ|)` are
///   treated as zero to prevent sign-flipping from floating-point noise.
///
/// # Panics
///
/// Panics if `z_vectors` is empty, `c_matrix` is not square with dimension
/// matching `tape.num_inputs()`, or any z-vector has the wrong length.
#[cfg(feature = "nalgebra")]
pub fn dense_stde_2nd_indefinite(
    tape: &BytecodeTape<f64>,
    x: &[f64],
    c_matrix: &nalgebra::DMatrix<f64>,
    z_vectors: &[&[f64]],
    eps_factor: f64,
) -> EstimatorResult<f64> {
    assert!(!z_vectors.is_empty(), "z_vectors must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(c_matrix.nrows(), n, "c_matrix rows must match num_inputs");
    assert_eq!(c_matrix.ncols(), n, "c_matrix cols must match num_inputs");

    // Eigendecompose C = Q Λ Qᵀ
    let eigen = nalgebra::SymmetricEigen::new(c_matrix.clone());
    let eigenvalues = &eigen.eigenvalues;
    let q = &eigen.eigenvectors;

    // Epsilon threshold based on largest eigenvalue magnitude
    let max_abs = eigenvalues.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let eps = eps_factor * max_abs;

    // Classify eigenvalues
    let mut has_positive = false;
    let mut has_negative = false;
    let mut sqrt_pos = vec![0.0f64; n];
    let mut sqrt_neg = vec![0.0f64; n];

    for i in 0..n {
        let lam = eigenvalues[i];
        if lam > eps {
            sqrt_pos[i] = lam.sqrt();
            has_positive = true;
        } else if lam < -eps {
            sqrt_neg[i] = (-lam).sqrt();
            has_negative = true;
        }
        // else: near-zero, both stay 0
    }

    // All eigenvalues are zero → result is zero
    if !has_positive && !has_negative {
        let mut buf = Vec::new();
        let v = vec![0.0f64; n];
        let (value, _, _) = taylor_jet_2nd_with_buf(tape, x, &v, &mut buf);
        return EstimatorResult {
            value,
            estimate: 0.0,
            sample_variance: 0.0,
            standard_error: 0.0,
            num_samples: z_vectors.len(),
        };
    }

    // Build L⁺ = Q · diag(√λ⁺) and/or L⁻ = Q · diag(√λ⁻)
    // Store as column-major n×n for mat-vec
    let l_pos = if has_positive {
        let mut m = nalgebra::DMatrix::zeros(n, n);
        for j in 0..n {
            if sqrt_pos[j] > 0.0 {
                for i in 0..n {
                    m[(i, j)] = q[(i, j)] * sqrt_pos[j];
                }
            }
        }
        Some(m)
    } else {
        None
    };

    let l_neg = if has_negative {
        let mut m = nalgebra::DMatrix::zeros(n, n);
        for j in 0..n {
            if sqrt_neg[j] > 0.0 {
                for i in 0..n {
                    m[(i, j)] = q[(i, j)] * sqrt_neg[j];
                }
            }
        }
        Some(m)
    } else {
        None
    };

    // Optimization: if all eigenvalues have same sign, use single-pass
    if has_positive && !has_negative {
        let lp = l_pos.as_ref().unwrap();
        return stde_2nd_inner(tape, x, z_vectors, |z, v| {
            // Full mat-vec: v = L⁺ · z
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += lp[(i, j)] * z[j];
                }
                v[i] = sum;
            }
        });
    }
    if !has_positive && has_negative {
        let ln = l_neg.as_ref().unwrap();
        // All negative: tr(C·H) = -E[v⁻ᵀ H v⁻], so negate
        let mut result = stde_2nd_inner(tape, x, z_vectors, |z, v| {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += ln[(i, j)] * z[j];
                }
                v[i] = sum;
            }
        });
        result.estimate = -result.estimate;
        // variance and SE stay the same magnitude (negation doesn't change variance)
        return result;
    }

    // Mixed sign: need both passes per z-vector
    let lp = l_pos.as_ref().unwrap();
    let ln = l_neg.as_ref().unwrap();

    let mut buf = Vec::new();
    let mut v_pos = vec![0.0f64; n];
    let mut v_neg = vec![0.0f64; n];
    let mut value = 0.0f64;
    let mut acc = WelfordAccumulator::new();

    for z in z_vectors.iter() {
        assert_eq!(z.len(), n, "z_vector length must match tape.num_inputs()");

        // v⁺ = L⁺ · z
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += lp[(i, j)] * z[j];
            }
            v_pos[i] = sum;
        }
        // v⁻ = L⁻ · z
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += ln[(i, j)] * z[j];
            }
            v_neg[i] = sum;
        }

        let (c0, _, c2_pos) = taylor_jet_2nd_with_buf(tape, x, &v_pos, &mut buf);
        let (_, _, c2_neg) = taylor_jet_2nd_with_buf(tape, x, &v_neg, &mut buf);
        value = c0;
        // sample = 2·c2⁺ - 2·c2⁻ = v⁺ᵀHv⁺ - v⁻ᵀHv⁻
        acc.update(2.0 * c2_pos - 2.0 * c2_neg);
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

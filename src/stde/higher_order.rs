use super::types::{EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::taylor::Taylor;
use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn, TaylorDynGuard};
use crate::Float;

/// Exact k-th order diagonal: `[∂^k u/∂x_j^k for j in 0..n]`.
///
/// Pushes n basis vectors through order-(k+1) `TaylorDyn` jets. For each
/// coordinate j, the input jet has `coeffs_j = [x_j, 1, 0, ..., 0]` and all
/// other inputs `coeffs_i = [x_i, 0, ..., 0]`. The output coefficient at
/// index k stores `∂^k u/∂x_j^k / k!`, so the derivative is `k! * coeffs[k]`.
///
/// Requires `F: Float + TaylorArenaLocal` because the jet order is
/// runtime-determined (unlike [`hessian_diagonal`](super::hessian_diagonal) which uses const-generic
/// `Taylor<F, 3>`).
///
/// # Panics
///
/// Panics if `k < 2`, `k > 20` (factorial overflow guard for f64), or if
/// `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
) -> (F, Vec<F>) {
    let mut buf = Vec::new();
    diagonal_kth_order_with_buf(tape, x, k, &mut buf)
}

/// Like [`diagonal_kth_order`] but reuses a caller-provided buffer.
pub fn diagonal_kth_order_with_buf<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
    buf: &mut Vec<TaylorDyn<F>>,
) -> (F, Vec<F>) {
    assert!(k >= 2, "k must be >= 2 (use gradient for k=1)");
    assert!(k <= 20, "k must be <= 20 (k! overflows f64 for k > 20)");
    assert!(
        k < 13 || std::mem::size_of::<F>() > 4,
        "k must be < 13 for f32 (k! loses precision for k >= 13; use f64)"
    );
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = k + 1; // number of Taylor coefficients
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    for j in 0..n {
        // Build TaylorDyn inputs: coeffs_j = [x_j, 1, 0, ..., 0], others = [x_i, 0, ..., 0]
        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];
        diag.push(k_factorial * out_coeffs[k]);
    }

    (value, diag)
}

/// Exact k-th order diagonal using const-generic `Taylor<F, ORDER>`.
///
/// `ORDER = k + 1` where `k = ORDER - 1` is the derivative order. Uses
/// stack-allocated Taylor coefficients — faster than [`diagonal_kth_order`]
/// when k is known at compile time.
///
/// # Precision Note
///
/// For f32, `k! > 2^23` when k ≥ 13 (ORDER ≥ 14), causing precision loss.
/// The existing [`diagonal_kth_order`] has a runtime guard `k ≤ 20` for f64;
/// the const-generic version has no such guard but users should be aware
/// that for f32, ORDER ≤ 14 is the practical limit.
///
/// # Panics
///
/// Compile-time error if `ORDER < 3` (i.e., k < 2).
/// Runtime panic if `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order_const<F: Float, const ORDER: usize>(
    tape: &BytecodeTape<F>,
    x: &[F],
) -> (F, Vec<F>) {
    let mut buf: Vec<Taylor<F, ORDER>> = Vec::new();
    diagonal_kth_order_const_with_buf(tape, x, &mut buf)
}

/// Like [`diagonal_kth_order_const`] but reuses a caller-provided buffer.
pub fn diagonal_kth_order_const_with_buf<F: Float, const ORDER: usize>(
    tape: &BytecodeTape<F>,
    x: &[F],
    buf: &mut Vec<Taylor<F, ORDER>>,
) -> (F, Vec<F>) {
    const { assert!(ORDER >= 3, "ORDER must be >= 3 (k=ORDER-1 >= 2)") }

    let k = ORDER - 1;
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let mut diag = Vec::with_capacity(n);
    let mut value = F::zero();

    for j in 0..n {
        let inputs: Vec<Taylor<F, ORDER>> = (0..n)
            .map(|i| {
                let mut coeffs = [F::zero(); ORDER];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                Taylor::new(coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, buf);

        let out = buf[tape.output_index()];
        value = out.coeffs[0];
        diag.push(k_factorial * out.coeffs[k]);
    }

    (value, diag)
}

/// Stochastic k-th order diagonal estimate: `Σ_j ∂^k u/∂x_j^k`.
///
/// Evaluates only the coordinate indices in `sampled_indices`, then scales
/// by `n / |J|` to produce an unbiased estimate of the full sum.
///
/// # Panics
///
/// Panics if `sampled_indices` is empty, `k < 2`, `k > 20`, or if
/// `x.len()` does not match `tape.num_inputs()`.
pub fn diagonal_kth_order_stochastic<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    k: usize,
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    assert!(k >= 2, "k must be >= 2 (use gradient for k=1)");
    assert!(k <= 20, "k must be <= 20 (k! overflows f64 for k > 20)");
    assert!(
        k < 13 || std::mem::size_of::<F>() > 4,
        "k must be < 13 for f32 (k! loses precision for k >= 13; use f64)"
    );
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = k + 1;
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut k_factorial = F::one();
    for i in 2..=k {
        k_factorial = k_factorial * F::from(i).unwrap();
    }

    let nf = F::from(n).unwrap();

    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for &j in sampled_indices {
        assert!(j < n, "sampled index {} out of bounds (n={})", j, n);

        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                if i == j {
                    coeffs[1] = F::one();
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];
        // Per-sample: k! * coeffs[k] (the diagonal entry for coordinate j)
        acc.update(k_factorial * out_coeffs[k]);
    }

    let (mean, sample_variance, standard_error) = acc.finalize();

    // Unbiased estimator for Σ_j d_j: n * mean(sampled d_j's).
    // Scale variance/SE to match the rescaled estimate.
    EstimatorResult {
        value,
        estimate: mean * nf,
        sample_variance: sample_variance * nf * nf,
        standard_error: standard_error * nf,
        num_samples: sampled_indices.len(),
    }
}

/// Propagate direction `v` through tape using `TaylorDyn` with the given order.
///
/// Creates a `TaylorDynGuard` internally, builds `TaylorDyn` inputs from
/// `(x, v)` with coefficients `[x_i, v_i, 0, ..., 0]`, runs `forward_tangent`,
/// and returns the full coefficient vector of the output.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`,
/// or if `order < 2`.
pub fn taylor_jet_dyn<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    v: &[F],
    order: usize,
) -> Vec<F> {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(v.len(), n, "v.len() must match tape.num_inputs()");
    assert!(order >= 2, "order must be >= 2");

    let _guard = TaylorDynGuard::<F>::new(order);

    let inputs: Vec<TaylorDyn<F>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| {
            let mut coeffs = vec![F::zero(); order];
            coeffs[0] = xi;
            coeffs[1] = vi;
            TaylorDyn::from_coeffs(&coeffs)
        })
        .collect();

    let mut buf = Vec::new();
    tape.forward_tangent(&inputs, &mut buf);

    buf[tape.output_index()].coeffs()
}

/// Estimate the Laplacian via `TaylorDyn` (runtime-determined order).
///
/// Uses order 3 (coefficients c0, c1, c2) internally. Manages its own
/// arena guard.
///
/// Returns `(value, laplacian_estimate)`.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn laplacian_dyn<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> (F, F) {
    assert!(!directions.is_empty(), "directions must not be empty");
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let _guard = TaylorDynGuard::<F>::new(3);

    let two = F::from(2.0).unwrap();
    let s = F::from(directions.len()).unwrap();
    let mut sum = F::zero();
    let mut value = F::zero();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for v in directions {
        assert_eq!(v.len(), n, "direction length must match tape.num_inputs()");

        let inputs: Vec<TaylorDyn<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| TaylorDyn::from_coeffs(&[xi, vi, F::zero()]))
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out = buf[tape.output_index()];
        let coeffs = out.coeffs();
        value = coeffs[0];
        let c2 = coeffs[2];
        sum = sum + two * c2;
    }

    (value, sum / s)
}

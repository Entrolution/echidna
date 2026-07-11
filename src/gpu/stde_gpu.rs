//! GPU-accelerated STDE (Stochastic Taylor Derivative Estimator) functions.
//!
//! Provides GPU-accelerated versions of the CPU STDE functions from [`crate::stde`].
//! These use batched second-order Taylor forward propagation on the GPU to evaluate
//! many directions in parallel.
//!
//! All functions are generic over `B: GpuBackend`, working with any backend
//! (wgpu, CUDA, or future backends).

use super::{GpuBackend, GpuError, TaylorBatchResult};
use crate::stde::EstimatorResult;

/// Reject multi-output tapes: every estimator here indexes `result.c2s[i]`
/// as one c2 per batch element, which only holds for a single scalar output
/// (the batched layout is `[batch * num_outputs + out]`).
fn require_scalar_output<B: GpuBackend>(
    backend: &B,
    tape: &B::TapeBuffers,
    caller: &str,
) -> Result<(), GpuError> {
    let outputs = backend.num_outputs(tape);
    if outputs != 1 {
        return Err(GpuError::Other(format!(
            "{caller} requires a scalar-output tape; got {outputs} outputs"
        )));
    }
    Ok(())
}

/// Reject an empty evaluation point: a zero-input tape has no `quantity`,
/// and zero-sized GPU buffers would panic downstream.
fn require_nonempty_x(n: usize, caller: &str, quantity: &str) -> Result<(), GpuError> {
    if n == 0 {
        return Err(GpuError::Other(format!(
            "{caller} requires a non-empty x (a zero-input tape has no \
             {quantity}); zero-sized GPU buffers would panic"
        )));
    }
    Ok(())
}

/// GPU-accelerated Laplacian estimation via Hutchinson + Taylor-mode.
///
/// Estimates `tr(H_f(x))` using S random directions. Each direction is pushed
/// through the tape on the GPU in a single batched dispatch.
///
/// `directions` is `&[&[f32]]` with S direction vectors, each of length n.
/// The primal point `x` is replicated for each batch element.
///
/// Works with any backend that implements `GpuBackend`.
pub fn laplacian_gpu<B: GpuBackend>(
    backend: &B,
    tape: &B::TapeBuffers,
    x: &[f32],
    directions: &[&[f32]],
) -> Result<EstimatorResult<f32>, GpuError> {
    require_scalar_output(backend, tape, "laplacian_gpu")?;
    let n = x.len();
    require_nonempty_x(n, "laplacian_gpu", "Laplacian")?;
    let s = directions.len();
    if s == 0 {
        return Err(GpuError::Other("no directions provided".into()));
    }

    // Flatten primals and seeds
    let mut primals = Vec::with_capacity(s * n);
    let mut seeds = Vec::with_capacity(s * n);
    for dir in directions {
        assert_eq!(dir.len(), n, "direction length must match x");
        primals.extend_from_slice(x);
        seeds.extend_from_slice(dir);
    }

    // SAFETY(u32 cast): s is the number of random directions, bounded by practical GPU limits.
    assert!(
        s <= u32::MAX as usize,
        "too many directions for GPU dispatch"
    );
    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, s as u32)?;
    Ok(aggregate_laplacian(&result, s))
}

/// CPU-side Welford aggregation of c2 values into a Laplacian estimate.
fn aggregate_laplacian(result: &TaylorBatchResult<f32>, s: usize) -> EstimatorResult<f32> {
    // For Hutchinson: E[v^T H v] = tr(H), and v^T H v = 2 * c2 for unit-variance v.
    // The factor n (dimension) is NOT needed here because Hutchinson estimator
    // already gives tr(H) = E[v^T H v] when E[vv^T] = I.
    let value = result.values[0]; // All batch elements share the same primal

    // The CPU estimators' accumulator: skips non-finite samples, reports NaN
    // when nothing contributed, and clamps rounding-negative variance at 0.
    let mut acc = crate::stde::WelfordAccumulator::new();
    for i in 0..s {
        acc.update(2.0 * result.c2s[i]); // v^T H v
    }
    acc.into_result(value)
}

/// GPU-accelerated exact Hessian diagonal via n basis-vector pushforwards.
///
/// Uses one batch element per input dimension, with each direction being a
/// standard basis vector e_j. Returns `(f(x), diag(H))`.
pub fn hessian_diagonal_gpu<B: GpuBackend>(
    backend: &B,
    tape: &B::TapeBuffers,
    x: &[f32],
) -> Result<(f32, Vec<f32>), GpuError> {
    require_scalar_output(backend, tape, "hessian_diagonal_gpu")?;
    let n = x.len();
    require_nonempty_x(n, "hessian_diagonal_gpu", "Hessian diagonal")?;

    // Build n basis directions
    let mut primals = Vec::with_capacity(n * n);
    let mut seeds = vec![0.0f32; n * n];
    for j in 0..n {
        primals.extend_from_slice(x);
        seeds[j * n + j] = 1.0;
    }

    // SAFETY(u32 cast): n is the number of input dimensions, bounded by practical GPU limits.
    assert!(n <= u32::MAX as usize, "too many inputs for GPU dispatch");
    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, n as u32)?;

    let value = result.values[0];
    // diag(H)[j] = 2 * c2[j] (since c2 = f''(t₀)/2 along basis e_j)
    let diag: Vec<f32> = result.c2s.iter().map(|&c2| 2.0 * c2).collect();

    Ok((value, diag))
}

/// GPU-accelerated Laplacian with diagonal control variate.
///
/// Uses a precomputed Hessian diagonal to reduce estimator variance.
/// The control variate estimate is: `tr(H_diag) + mean(v^T H v - v^T diag(H) v)`.
pub fn laplacian_with_control_gpu<B: GpuBackend>(
    backend: &B,
    tape: &B::TapeBuffers,
    x: &[f32],
    directions: &[&[f32]],
    control_diagonal: &[f32],
) -> Result<EstimatorResult<f32>, GpuError> {
    require_scalar_output(backend, tape, "laplacian_with_control_gpu")?;
    let n = x.len();
    require_nonempty_x(n, "laplacian_with_control_gpu", "Laplacian")?;
    let s = directions.len();
    assert_eq!(
        control_diagonal.len(),
        n,
        "control diagonal length must match x"
    );
    if s == 0 {
        return Err(GpuError::Other("no directions provided".into()));
    }

    let mut primals = Vec::with_capacity(s * n);
    let mut seeds = Vec::with_capacity(s * n);
    for dir in directions {
        assert_eq!(dir.len(), n, "direction length must match x");
        primals.extend_from_slice(x);
        seeds.extend_from_slice(dir);
    }

    // SAFETY(u32 cast): s is the number of random directions, bounded by practical GPU limits.
    assert!(
        s <= u32::MAX as usize,
        "too many directions for GPU dispatch"
    );
    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, s as u32)?;

    // Control variate: tr(diag(H)) is exact, reduce variance of off-diagonal estimate
    let trace_diag: f32 = control_diagonal.iter().sum();

    let value = result.values[0];
    // Residual samples (after the control variate) through the CPU
    // estimators' accumulator, matching laplacian_with_control.
    let mut acc = crate::stde::WelfordAccumulator::new();
    for (i, dir) in directions.iter().enumerate() {
        let vhv = 2.0 * result.c2s[i]; // v^T H v
                                       // v^T diag(H) v = Σ_j diag[j] * v[j]²
        let v_diag_v: f32 = dir
            .iter()
            .zip(control_diagonal.iter())
            .map(|(&vj, &dj)| dj * vj * vj)
            .sum();
        acc.update(vhv - v_diag_v); // residual (off-diagonal contribution)
    }
    // Deliberately NOT into_result: the exact-trace offset below transforms
    // the raw mean, and that stays visible here (same exemption as the CPU
    // control-variate estimators).
    let (residual_mean, sample_variance, standard_error) = acc.finalize();
    let estimate = trace_diag + residual_mean;

    Ok(EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: acc.contributing(),
    })
}

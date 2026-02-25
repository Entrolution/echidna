//! GPU-accelerated STDE (Stochastic Taylor Derivative Estimator) functions.
//!
//! Provides GPU-accelerated versions of the CPU STDE functions from [`crate::stde`].
//! These use batched second-order Taylor forward propagation on the GPU to evaluate
//! many directions in parallel.
//!
//! # Supported backends
//!
//! - **wgpu** (`gpu-wgpu`): cross-platform, f32 only
//! - **CUDA** (`gpu-cuda`): NVIDIA, f32 (f64 via inherent methods on `CudaContext`)

use super::{GpuError, TaylorBatchResult};
use crate::stde::EstimatorResult;

/// GPU-accelerated Laplacian estimation via Hutchinson + Taylor-mode.
///
/// Estimates `tr(H_f(x))` using S random directions. Each direction is pushed
/// through the tape on the GPU in a single batched dispatch.
///
/// `directions` is `&[&[f32]]` with S direction vectors, each of length n.
/// The primal point `x` is replicated for each batch element.
///
/// Works with any backend that provides `taylor_forward_2nd_batch`.
#[cfg(feature = "gpu-wgpu")]
pub fn laplacian_gpu(
    backend: &super::WgpuContext,
    tape: &super::WgpuTapeBuffers,
    x: &[f32],
    directions: &[&[f32]],
) -> Result<EstimatorResult<f32>, GpuError> {
    let n = x.len();
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

    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, s as u32)?;
    Ok(aggregate_laplacian(&result, s))
}

/// GPU-accelerated Laplacian estimation (CUDA f32).
#[cfg(feature = "gpu-cuda")]
pub fn laplacian_gpu_cuda(
    backend: &super::CudaContext,
    tape: &super::CudaTapeBuffers,
    x: &[f32],
    directions: &[&[f32]],
) -> Result<EstimatorResult<f32>, GpuError> {
    let n = x.len();
    let s = directions.len();
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

    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, s as u32)?;
    Ok(aggregate_laplacian(&result, s))
}

/// CPU-side Welford aggregation of c2 values into a Laplacian estimate.
fn aggregate_laplacian(result: &TaylorBatchResult<f32>, s: usize) -> EstimatorResult<f32> {
    // For Hutchinson: E[v^T H v] = tr(H), and v^T H v = 2 * c2 for unit-variance v.
    // The factor n (dimension) is NOT needed here because Hutchinson estimator
    // already gives tr(H) = E[v^T H v] when E[vv^T] = I.
    let value = result.values[0]; // All batch elements share the same primal
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for i in 0..s {
        let sample = 2.0 * result.c2s[i]; // v^T H v
        let count = (i + 1) as f32;
        let delta = sample - mean;
        mean += delta / count;
        let delta2 = sample - mean;
        m2 += delta * delta2;
    }

    let sample_variance = if s > 1 { m2 / (s - 1) as f32 } else { 0.0 };
    let standard_error = if s > 1 {
        (sample_variance / s as f32).sqrt()
    } else {
        0.0
    };

    EstimatorResult {
        value,
        estimate: mean,
        sample_variance,
        standard_error,
        num_samples: s,
    }
}

/// GPU-accelerated exact Hessian diagonal via n basis-vector pushforwards.
///
/// Uses one batch element per input dimension, with each direction being a
/// standard basis vector e_j. Returns `(f(x), diag(H))`.
#[cfg(feature = "gpu-wgpu")]
pub fn hessian_diagonal_gpu(
    backend: &super::WgpuContext,
    tape: &super::WgpuTapeBuffers,
    x: &[f32],
) -> Result<(f32, Vec<f32>), GpuError> {
    let n = x.len();

    // Build n basis directions
    let mut primals = Vec::with_capacity(n * n);
    let mut seeds = vec![0.0f32; n * n];
    for j in 0..n {
        primals.extend_from_slice(x);
        seeds[j * n + j] = 1.0;
    }

    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, n as u32)?;

    let value = result.values[0];
    // diag(H)[j] = 2 * c2[j] (since c2 = f''(t₀)/2 along basis e_j)
    let diag: Vec<f32> = result.c2s.iter().map(|&c2| 2.0 * c2).collect();

    Ok((value, diag))
}

/// GPU-accelerated exact Hessian diagonal (CUDA f32).
#[cfg(feature = "gpu-cuda")]
pub fn hessian_diagonal_gpu_cuda(
    backend: &super::CudaContext,
    tape: &super::CudaTapeBuffers,
    x: &[f32],
) -> Result<(f32, Vec<f32>), GpuError> {
    let n = x.len();

    let mut primals = Vec::with_capacity(n * n);
    let mut seeds = vec![0.0f32; n * n];
    for j in 0..n {
        primals.extend_from_slice(x);
        seeds[j * n + j] = 1.0;
    }

    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, n as u32)?;

    let value = result.values[0];
    let diag: Vec<f32> = result.c2s.iter().map(|&c2| 2.0 * c2).collect();

    Ok((value, diag))
}

/// GPU-accelerated Laplacian with diagonal control variate.
///
/// Uses a precomputed Hessian diagonal to reduce estimator variance.
/// The control variate estimate is: `tr(H_diag) + mean(v^T H v - v^T diag(H) v)`.
#[cfg(feature = "gpu-wgpu")]
pub fn laplacian_with_control_gpu(
    backend: &super::WgpuContext,
    tape: &super::WgpuTapeBuffers,
    x: &[f32],
    directions: &[&[f32]],
    control_diagonal: &[f32],
) -> Result<EstimatorResult<f32>, GpuError> {
    let n = x.len();
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

    let result = backend.taylor_forward_2nd_batch(tape, &primals, &seeds, s as u32)?;

    // Control variate: tr(diag(H)) is exact, reduce variance of off-diagonal estimate
    let trace_diag: f32 = control_diagonal.iter().sum();

    let value = result.values[0];
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for (i, dir) in directions.iter().enumerate() {
        let vhv = 2.0 * result.c2s[i]; // v^T H v
                                       // v^T diag(H) v = Σ_j diag[j] * v[j]²
        let v_diag_v: f32 = dir
            .iter()
            .zip(control_diagonal.iter())
            .map(|(&vj, &dj)| dj * vj * vj)
            .sum();
        let sample = vhv - v_diag_v; // residual (off-diagonal contribution)
        let count = (i + 1) as f32;
        let delta = sample - mean;
        mean += delta / count;
        let delta2 = sample - mean;
        m2 += delta * delta2;
    }

    let estimate = trace_diag + mean;
    let sample_variance = if s > 1 { m2 / (s - 1) as f32 } else { 0.0 };
    let standard_error = if s > 1 {
        (sample_variance / s as f32).sqrt()
    } else {
        0.0
    };

    Ok(EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: s,
    })
}

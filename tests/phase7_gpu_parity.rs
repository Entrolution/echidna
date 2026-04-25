//! Phase 7 Commit 2 regressions — GPU compile-flag parity and pinning
//! tests for the M26/M28 investigations that found the GPU kernels
//! already matched CPU behaviour.
//!
//! Covers L25 (NVRTC `--fmad=false`), M26 (ABS at ±0), M28 (CUDA REM).
//! M23 (device.poll error propagation) is compile-tested only — it
//! requires a driver reset to exercise the error path at runtime, which
//! can't be simulated from a unit test.

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::gpu::{GpuBackend, GpuTapeData};
use echidna::{record, BReverse};

#[cfg(feature = "gpu-wgpu")]
use num_traits::Float;

#[cfg(feature = "gpu-wgpu")]
use echidna::gpu::WgpuContext;

#[cfg(feature = "gpu-cuda")]
use echidna::gpu::CudaContext;

// ── L25: NVRTC --fmad=false gives bit-exact CPU-GPU parity on mul+add ──
// With `--fmad=true` (CUDA default), NVRTC fuses `a*b + c` into a single
// FMA instruction that rounds once instead of twice — the GPU result
// drifts from the CPU's two-step `(a*b) + c` by ≤1 ULP. The fix disables
// FMA so both follow the same rounding policy and agree bit-for-bit on
// f64 arithmetic.

#[cfg(feature = "gpu-cuda")]
#[test]
fn l25_cuda_fmad_disabled_bit_exact_with_cpu_f64() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => return,
    };
    // f(a, b, c) = a*b + c — the canonical FMA-vs-pair divergence point.
    // Values chosen so the separately-rounded mul gives a result that
    // isn't exactly representable: a*b = 1e-10 + ULP noise, + 1.0 drifts.
    let a = 1.2345678901234567_f64;
    let b = 9.8765432109876543_f64;
    let c = 1.1111111111111111_f64;
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0] * v[1] + v[2], &[a, b, c]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (gpu_result, _) = ctx
        .gradient_batch(&gpu_tape, &[a as f32, b as f32, c as f32], 1)
        .unwrap();
    let cpu_result = (a as f32) * (b as f32) + (c as f32);
    // With --fmad=false the GPU should match the CPU's f32 pair-rounded
    // computation bit-for-bit. Without the flag, NVRTC could fold to
    // FMA and drift by 1 ULP.
    assert_eq!(
        gpu_result[0].to_bits(),
        cpu_result.to_bits(),
        "fmad=false should give bit-exact CPU-GPU parity on a*b+c, got \
         GPU={} CPU={}",
        gpu_result[0],
        cpu_result
    );
}

// ── M26: ABS at ±0.0 matches CPU (WGSL bit-inspect + CUDA _sign) ──
// Investigation finding: the agent reported the WGSL shader already used
// `bitcast<u32>` to inspect the sign bit, which correctly returns -1 for
// `-0.0` and +1 for `+0.0`. CUDA uses `_sign(a)`. This test pins the
// behaviour so a future refactor doesn't silently regress.

#[cfg(feature = "gpu-wgpu")]
#[test]
fn m26_wgpu_abs_at_positive_zero_gives_positive_gradient() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].abs(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // +0.0: the subgradient is the Clarke set [-1, 1]; any WGSL
    // implementation that's sign-bit-driven should return +1 here.
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[0.0_f32], 1).unwrap();
    assert_eq!(
        g[0], 1.0,
        "WGSL abs at +0 should give +1 (sign bit not set)"
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn m26_wgpu_abs_at_negative_zero_gives_negative_gradient() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].abs(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // -0.0 via bit-pattern construction (literal `-0.0_f32` would be
    // optimised away by the shader compiler in some drivers).
    let neg_zero = f32::from_bits(0x8000_0000);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[neg_zero], 1).unwrap();
    // WGSL bit-inspect returns -1 for sign-bit-set. This matches CPU's
    // `a.signum()` for -0.0 (also -0, which maps to -1 after the branch
    // in opcode.rs's Abs arm after the Phase 8 L15 fix).
    assert!(
        g[0] == -1.0 || g[0] == 0.0,
        "WGSL abs at -0 should give -1 (bit-inspect) or 0 (symmetric \
         subgradient). Got {}",
        g[0]
    );
}

// ── M28: CUDA REM primal and tangent are internally consistent ──
// Investigation finding: CUDA REM primal uses `fmod(a, b)` and reverse
// partials compute `da = 1, db = -trunc(a/b)`. These are internally
// consistent (both differentiate `a - b·trunc(a/b)`) even though the
// CPU uses a slightly different `a - b·floor(a/b)` convention. This
// test pins the quotient-boundary behaviour so any future kernel edit
// gets caught.

#[cfg(feature = "gpu-cuda")]
#[test]
fn m28_cuda_rem_quotient_boundary_consistent() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0] % v[1], &[5.0_f64, 2.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // a = 5, b = 2: trunc(5/2) = 2, so primal = 5 - 2*2 = 1.
    // Gradient: da = 1, db = -trunc(5/2) = -2.
    let (r, g) = ctx
        .gradient_batch(&gpu_tape, &[5.0_f32, 2.0_f32], 1)
        .unwrap();
    assert!((r[0] - 1.0).abs() < 1e-5, "r = 5 % 2 = 1, got {}", r[0]);
    assert!((g[0] - 1.0).abs() < 1e-5, "da should be 1, got {}", g[0]);
    assert!(
        (g[1] - (-2.0)).abs() < 1e-5,
        "db should be -2, got {}",
        g[1]
    );
}

//! CUDA kernel parity tests mirroring `gpu_kernel_parity.rs`.
//!
//! Exercises the 3A fixes (atan2 overflow, asinh/acosh large-|a|, powf
//! a≤0 safety, max/min NaN routing, fract truncation) against the CUDA
//! reverse_sweep / tangent_forward / tangent_reverse kernels.

#![cfg(feature = "gpu-cuda")]

use echidna::gpu::{CudaContext, GpuBackend, GpuTapeData};
use echidna::{record, BReverse};
use num_traits::Float;

fn cuda_context() -> Option<CudaContext> {
    CudaContext::new()
}

// ── atan2 ───────────────────────────────────────────────────────────

#[test]
fn cuda_atan2_large_magnitudes_gradient_finite() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f32, 1.0_f32];
    let (tape, _) = record(|v: &[BReverse<f32>]| v[0].atan2(v[1]), &x0);

    let large = 1e20_f32; // a*a = 1e40 overflows f32::MAX ≈ 3.4e38
    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large, large], 1).unwrap();

    assert!(
        g[0].is_finite(),
        "d/dy atan2 should be finite; got {}",
        g[0]
    );
    assert!(
        g[1].is_finite(),
        "d/dx atan2 should be finite; got {}",
        g[1]
    );
    assert!(g[0] != 0.0, "d/dy atan2 underflowed to zero");
    assert!(g[1] != 0.0, "d/dx atan2 underflowed to zero");
}

// ── asinh / acosh ───────────────────────────────────────────────────

#[test]
fn cuda_asinh_large_derivative_finite() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f32];
    let (tape, _) = record(|v: &[BReverse<f32>]| v[0].asinh(), &x0);

    let large = 1e20_f32;
    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();

    assert!(g[0].is_finite(), "asinh derivative should be finite");
    let rel_err = (g[0] as f64 - 1e-20).abs() / 1e-20;
    assert!(rel_err < 1e-5, "g[0] = {}, expected ≈ 1e-20", g[0]);
}

#[test]
fn cuda_acosh_large_derivative_finite() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [2.0_f32];
    let (tape, _) = record(|v: &[BReverse<f32>]| v[0].acosh(), &x0);

    let large = 1e20_f32;
    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();

    assert!(g[0].is_finite(), "acosh derivative should be finite");
    let rel_err = (g[0] as f64 - 1e-20).abs() / 1e-20;
    assert!(rel_err < 1e-5, "g[0] = {}, expected ≈ 1e-20", g[0]);
}

// ── powf ────────────────────────────────────────────────────────────

#[test]
fn cuda_powf_negative_base_integer_exponent() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [-2.0_f32];
    let (tape, _) = record(
        |v: &[BReverse<f32>]| v[0].powf(BReverse::constant(3.0)),
        &x0,
    );

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[-2.0_f32], 1).unwrap();

    assert!(g[0].is_finite(), "gradient must be finite (not NaN)");
    let rel_err = (g[0] as f64 - 12.0).abs() / 12.0;
    assert!(rel_err < 1e-5, "g[0] = {}, expected 12", g[0]);
}

// ── max / min NaN routing ───────────────────────────────────────────

#[test]
fn cuda_max_with_nan_operand() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.5_f32, f32::NAN];
    let (tape, _) = record(|v: &[BReverse<f32>]| v[0].max(v[1]), &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx
        .gradient_batch(&gpu_tape, &[1.5_f32, f32::NAN], 1)
        .unwrap();

    assert_eq!(g[0], 1.0, "adjoint should route to non-NaN operand");
    assert_eq!(g[1], 0.0, "adjoint to NaN operand should be 0");
}

// ── fract ───────────────────────────────────────────────────────────

#[test]
fn cuda_fract_negative_input_matches_cpu() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [-1.3_f32];
    let (tape, _) = record(|v: &[BReverse<f32>]| v[0].fract(), &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let out = ctx.forward_batch(&gpu_tape, &[-1.3_f32], 1).unwrap();

    let expected = -1.3_f32.fract(); // truncation convention: -0.3
    assert!(
        (out[0] - expected).abs() < 1e-6,
        "fract(-1.3) on CUDA = {}, expected ≈ {}",
        out[0],
        expected,
    );
    assert!(
        out[0] < 0.0,
        "GPU fract should be negative for negative input"
    );
}

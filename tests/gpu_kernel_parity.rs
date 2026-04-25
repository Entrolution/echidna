//! CPU ↔ WGSL parity tests for the hand-written kernels (forward, reverse,
//! tangent_forward, tangent_reverse) after the numerical-fix sweep.
//!
//! Each test compiles the same AD expression for both CPU (f64) and GPU
//! (f32 via wgpu/Metal) and asserts the GPU result matches CPU to within a
//! documented tolerance. These are the same patterns that prior to the
//! fix produced NaN, +/-Inf, or wrong-sign tangents on GPU.

#![cfg(feature = "gpu-wgpu")]

use echidna::gpu::{GpuBackend, GpuTapeData, WgpuContext};
use echidna::{record, BReverse};
use num_traits::Float;

fn gpu_context() -> Option<WgpuContext> {
    WgpuContext::new()
}

fn approx_eq_relaxed(gpu: f32, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let gpu = gpu as f64;
    let diff = (gpu - cpu).abs();
    if cpu.abs() < abs_tol {
        diff < abs_tol
    } else {
        diff / cpu.abs() < rel_tol
    }
}

// ── atan2: h*h overflow fix ─────────────────────────────────────────

#[test]
fn wgpu_atan2_large_magnitudes_gradient_finite() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 1.0_f64];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].atan2(v[1]), &x0);

    // f32 can't represent 1e200 — but 1e18 is still in-range and exercises
    // the overflow path since h*h = 2e36 > f32::MAX ≈ 3.4e38... actually
    // 2e36 < 3.4e38, still in-range. Use 1e19 where h*h = 2e38 ≈ MAX.
    // More robust: pick 1e20 where h*h = 2e40 = inf in f32 (pre-fix path).
    let large = 1e20_f32;
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
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
    // Expected: d/dy = x/(x²+y²) = 1e20 / 2e40 = 5e-21, nonzero.
    assert!(g[0] != 0.0, "d/dy atan2 underflowed to zero");
    assert!(g[1] != 0.0, "d/dx atan2 underflowed to zero");
}

// ── asinh / acosh: a*a ± 1 overflow fix ─────────────────────────────

#[test]
fn wgpu_asinh_large_derivative_finite() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].asinh(), &x0);

    let large = 1e20_f32; // a*a = 1e40 = inf in f32 pre-fix
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();

    assert!(g[0].is_finite(), "asinh derivative should be finite");
    // Expected: ≈ 1/|x| = 1e-20, nonzero (in f32 denormal range but nonzero).
    // f32 min subnormal ≈ 1.4e-45 — 1e-20 well above.
    assert!(g[0] > 0.0, "asinh derivative should be positive (monotone)");
    let rel_err = (g[0] as f64 - 1e-20).abs() / 1e-20;
    assert!(rel_err < 1e-5, "g[0] = {}, expected ≈ 1e-20", g[0]);
}

#[test]
fn wgpu_acosh_large_derivative_finite() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [2.0_f64];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].acosh(), &x0);

    let large = 1e20_f32;
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();

    assert!(g[0].is_finite(), "acosh derivative should be finite");
    assert!(g[0] > 0.0);
    let rel_err = (g[0] as f64 - 1e-20).abs() / 1e-20;
    assert!(rel_err < 1e-5, "g[0] = {}, expected ≈ 1e-20", g[0]);
}

// ── powf: a ≤ 0 safety net ──────────────────────────────────────────

#[test]
fn wgpu_powf_negative_base_integer_exponent() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // x.powf(3.0) at x = -2.
    let x0 = [-2.0_f64];
    let (tape, _) = record(
        |v: &[BReverse<f64>]| v[0].powf(BReverse::constant(3.0)),
        &x0,
    );

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[-2.0_f32], 1).unwrap();

    // d/dx x^3 |_{x=-2} = 12. Pre-fix: NaN.
    assert!(g[0].is_finite(), "gradient must be finite");
    let rel_err = (g[0] as f64 - 12.0).abs() / 12.0;
    assert!(rel_err < 1e-5, "g[0] = {}, expected 12", g[0]);
}

// ── max/min: NaN operand routing ────────────────────────────────────

#[test]
fn wgpu_max_with_nan_operand_primal_and_grad() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // max(x, NaN) = x (non-NaN side). Gradient should flow to x.
    let x0 = [1.5_f64, f64::NAN];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].max(v[1]), &x0);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx
        .gradient_batch(&gpu_tape, &[1.5_f32, f32::NAN], 1)
        .unwrap();

    // d/dx max(x, NaN) = 1; d/d(NaN) = 0.
    assert_eq!(g[0], 1.0, "adjoint should route to x (non-NaN)");
    assert_eq!(g[1], 0.0, "adjoint to NaN operand should be 0");
}

// ── abs: signed-zero convention ─────────────────────────────────────

#[test]
fn wgpu_abs_signed_zero() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // |-0.0| = 0.0; d|x|/dx at x = -0.0 is -1 per sign-bit (mirrors f32::signum).
    let x0 = [-0.0_f64];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].abs(), &x0);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[-0.0_f32], 1).unwrap();
    assert_eq!(g[0], -1.0, "d|x|/dx at -0.0 should be -1 (sign bit set)");
}

// ── fract: trunc semantics ──────────────────────────────────────────

#[test]
fn wgpu_fract_negative_input_matches_cpu() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // fract(-1.3): CPU returns -0.3 (trunc convention); WGSL built-in `fract`
    // returns 0.7 (floor convention). Kernel must match CPU.
    let x0 = [-1.3_f64];
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].fract(), &x0);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let out = ctx.forward_batch(&gpu_tape, &[-1.3_f32], 1).unwrap();

    let expected = -1.3_f64.fract(); // CPU convention: -0.3
    assert!(
        approx_eq_relaxed(out[0], expected, 1e-5, 1e-6),
        "fract(-1.3) on GPU = {}, expected ≈ {}",
        out[0],
        expected
    );
    assert!(
        out[0] < 0.0,
        "GPU fract should be negative for negative input"
    );
}

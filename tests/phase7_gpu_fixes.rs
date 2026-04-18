//! Phase 7 Commit 1 regressions — GPU correctness fixes.
//!
//! Covers M24 (CUDA upload_tape empty outputs fallback), M27 (WGSL
//! EXPM1/LN1P precision), M29 (GPU ATAN large-|a|), L22 (Taylor HYPOT
//! primal rescale), L23 (GPU POWI n=1 at a=0), L24 (GPU DIV small-|b|).

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::gpu::{GpuBackend, GpuTapeData};
use echidna::{record, BReverse};
use num_traits::Float;

#[cfg(feature = "gpu-wgpu")]
use echidna::gpu::WgpuContext;

#[cfg(feature = "gpu-cuda")]
use echidna::gpu::CudaContext;

// ── M29: GPU ATAN at large |a| produces finite, non-zero derivative ──
// Pre-fix: 1 + a² overflows to +Inf in f32, so `1/(1+a²) = 0` → derivative
// collapses. Post-fix: inv-based formula preserves the ≈1/a² value.

#[cfg(feature = "gpu-wgpu")]
#[test]
fn m29_wgpu_atan_large_abs_a_stays_finite() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].atan(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // For `|a| > 1.84e19` the pre-fix formula `a*a` overflows to +Inf,
    // making `1+a² = Inf` and `1/Inf = 0` — finite but the Inf
    // intermediate can contaminate other shader computations before
    // saturating. Post-fix inv-based form keeps every intermediate
    // finite. On Metal, f32 denormals are flushed, so the *result* for
    // these magnitudes is 0 regardless; what the fix guarantees is that
    // no Inf appears at any stage.
    let large = 1e20_f32;
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();
    assert!(
        g[0].is_finite(),
        "atan derivative must be finite for |a|=1e20, got {}",
        g[0]
    );
    // CUDA (which preserves denormals) tests the positive-result contract
    // separately; see `m29_cuda_atan_large_abs_a_finite_nonzero`.
}

// Stronger M29 wgpu discriminator: at `|a| = 3e9` in f32, `a² = 9e18`
// which is still within f32 range (≤ 3.4e38), so the pre-fix formula
// `1/(1+a²)` computes `≈ 1/9e18 = 1.1e-19` — a normal f32, not denormal.
// Metal won't flush this result to zero. The post-fix inv-based formula
// gives the same answer up to rounding. So this test pins that
// `g[0] > 0` at a magnitude where Metal denormal flush doesn't
// interfere, catching a regression that silently returned 0.
#[cfg(feature = "gpu-wgpu")]
#[test]
fn m29_wgpu_atan_moderate_large_a_positive_normal_f32() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].atan(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let a = 3e9_f32;
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[a], 1).unwrap();
    assert!(
        g[0].is_finite() && g[0] > 0.0,
        "atan'(3e9) should be ≈ 1.1e-19 (positive, normal f32), got {}",
        g[0]
    );
    let expected = 1.0_f64 / (1.0 + 9e18);
    let rel_err = ((g[0] as f64) - expected).abs() / expected;
    assert!(
        rel_err < 1e-3,
        "atan'(3e9) = {:e}, expected ≈ {:e}",
        g[0],
        expected
    );
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn m29_cuda_atan_large_abs_a_finite_nonzero() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].atan(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let large = 1e20_f32;
    let (_, g) = ctx.gradient_batch(&gpu_tape, &[large], 1).unwrap();
    assert!(
        g[0].is_finite(),
        "atan derivative must be finite for |a|=1e20, got {}",
        g[0]
    );
    assert!(g[0] > 0.0);
}

// ── L24: GPU DIV at small |b| stays finite ──
// Pre-fix: db = -a*inv² where inv = 1/b. For b = 1e-20 (f32), inv² ≈ 1e40
// overflows. Post-fix: db = -r*inv = -(a/b)/b which is a single division
// and stays in-range for moderate a.

#[cfg(feature = "gpu-wgpu")]
#[test]
fn l24_wgpu_div_small_denominator_db_finite() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    // f(a, b) = a / b. db = -a / b² in the reverse sweep.
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0] / v[1], &[1.0_f64, 1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // a = 1e-10, b = 1e-20. b² = 1e-40 which doesn't overflow f32 upward
    // but a/b² = 1e30 is huge. Pre-fix computes `-a * inv * inv` where
    // `inv² = 1e40 → Inf`; post-fix computes `-r * inv = -1e10 * 1e20 = -1e30`
    // which is finite.
    let (_, g) = ctx
        .gradient_batch(&gpu_tape, &[1e-10_f32, 1e-20_f32], 1)
        .unwrap();
    assert!(
        g[1].is_finite(),
        "db must be finite at small |b|, got {}",
        g[1]
    );
}

// ── L22: Taylor HYPOT primal survives large magnitudes ──
// Pre-fix: `a*a + b*b` overflows in jet primal computation. Post-fix:
// primal uses scaled `max(|a|,|b|) * sqrt((a/h)² + (b/h)²)` (WGSL) or
// CUDA `hypot(a, b)` (CUDA). Higher-order Taylor coefficients still
// use jet_mul / jet_add and may overflow; that's a follow-up.

// Post-Cycle-2 addition: scalar HYPOT across all three WGSL shaders
// (forward, tangent_forward, and the reverse-primal path that reads
// from forward's output) now honours IEEE `hypot(±Inf, x) = +Inf` and
// avoids `a*a + b*b` overflow via the shared `hypot_f32` helper.

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_hypot_inf_finite_returns_inf() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].hypot(v[1]), &[1.0_f64, 1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // hypot(+Inf, 1) must be +Inf per IEEE 754-2008.
    let result = ctx
        .forward_batch(&gpu_tape, &[f32::INFINITY, 1.0_f32], 1)
        .unwrap();
    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "hypot(Inf, 1) must be +Inf, got {}",
        result[0]
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_hypot_inf_inf_returns_inf() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].hypot(v[1]), &[1.0_f64, 1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // Pre-fix the rescale formula gave `Inf/Inf = NaN`; post-fix the
    // `ax == inf || ay == inf` guard short-circuits to +Inf.
    let result = ctx
        .forward_batch(&gpu_tape, &[f32::INFINITY, f32::INFINITY], 1)
        .unwrap();
    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "hypot(Inf, Inf) must be +Inf (IEEE), got {}",
        result[0]
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_tangent_forward_hypot_large_magnitude_primal_finite() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].hypot(v[1]), &[1.0_f64, 1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    // At (1e20, 1e20), naive `a*a + b*b = 2e40` overflows f32 to +Inf,
    // then `sqrt(Inf) = Inf`. The rescaled helper gives `1e20 * sqrt(2) ≈ 1.414e20`.
    // `gradient_batch` drives the tangent_forward kernel on the primal
    // path (via the JVP-on-single-output code path).
    let (r, g) = ctx
        .gradient_batch(&gpu_tape, &[1e20_f32, 1e20_f32], 1)
        .unwrap();
    assert!(
        r[0].is_finite(),
        "tangent_forward hypot primal at (1e20, 1e20) must be finite (rescaled), got {}",
        r[0]
    );
    assert!(g[0].is_finite() && g[1].is_finite());
    // Expected: sqrt(2) * 1e20 ≈ 1.414e20.
    let expected = 2.0_f32.sqrt() * 1e20_f32;
    let rel = ((r[0] - expected) / expected).abs();
    assert!(
        rel < 1e-4,
        "hypot(1e20, 1e20) = {}, expected ≈ {}",
        r[0],
        expected
    );
}

// ── M24: CUDA upload_tape with empty output_indices ──

#[cfg(feature = "gpu-cuda")]
#[test]
fn m24_cuda_upload_tape_empty_outputs_via_fallback() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => return,
    };
    // `GpuTapeData` has public fields, so a downstream crate can manually
    // clear `output_indices` while leaving `output_index` populated (e.g.
    // to re-target a different output slot before upload). Pre-fix the
    // CUDA path crashed here on `clone_htod(&[])`; post-fix the fallback
    // synthesises a one-element `vec![data.output_index]`.
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0] * v[0] + v[0], &[2.0_f64]);
    let mut gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    // Simulate the "misused GpuTapeData" scenario the fix defends against.
    gpu_data.output_indices.clear();
    assert!(gpu_data.output_indices.is_empty());

    // The following upload would panic on `clone_htod(&[])` without the fix.
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (r, _g) = ctx.gradient_batch(&gpu_tape, &[2.0_f32], 1).unwrap();
    assert_eq!(r.len(), 1);
    assert!((r[0] - 6.0).abs() < 1e-5, "f(2) = 6; got {}", r[0]);
}

// ── M27: WGSL EXPM1/LN1P precision at small |a| ──
// Pre-fix: tangent kernels computed `exp(a) - 1` / `log(1 + a)` directly,
// losing ~7 digits of precision for |a| < 1e-4. Post-fix: they use the
// expm1_f32 / ln1p_f32 helpers (Taylor shortcut for small |a|).

#[cfg(feature = "gpu-wgpu")]
#[test]
fn m27_wgsl_expm1_small_a_precision_improves() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    // f(x) = expm1(x). At x = 1e-6, expm1(x) ≈ 1.0000005e-6, but
    // `exp(1e-6) - 1` in f32 gives `1.0000001 - 1.0 = 1.1920929e-7` — off
    // by two orders of magnitude. The tangent kernel uses the primal
    // value internally when forming rt; with the helper the primal is
    // accurate to 1 ULP at small |a|.
    //
    // Exercising via jvp (tangent_forward kernel):
    let (tape, _) = record(
        |v: &[BReverse<f64>]| v[0].exp_m1(),
        &[1.0_f64], // trace at x=1; GPU tests below at a=1e-6
    );
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (r, _g) = ctx.gradient_batch(&gpu_tape, &[1e-6_f32], 1).unwrap();
    let expected = 1e-6_f64.exp_m1();
    let actual = r[0] as f64;
    let abs_err = (actual - expected).abs();
    assert!(
        abs_err < 1e-12,
        "expm1(1e-6) expected ≈ {:e}, got {:e} (err {:e})",
        expected,
        actual,
        abs_err
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn m27_wgsl_ln1p_small_a_precision_improves() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].ln_1p(), &[1.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (r, _g) = ctx.gradient_batch(&gpu_tape, &[1e-6_f32], 1).unwrap();
    let expected = 1e-6_f64.ln_1p();
    let actual = r[0] as f64;
    let abs_err = (actual - expected).abs();
    assert!(
        abs_err < 1e-12,
        "ln1p(1e-6) expected ≈ {:e}, got {:e} (err {:e})",
        expected,
        actual,
        abs_err
    );
}

// ── L23: GPU POWI n=1 at a=0 produces finite second derivative ──
// Pre-fix: tangent_reverse for POWI n=1 computed
// `da_eps = 1 * 0 * pow(0, -1) * at = 0 * Inf = NaN`.
// Post-fix: n==1 is special-cased to `da_re = 1, da_eps = 0`.

// Exercising this path needs the tangent_reverse CUDA path (HVP or
// second-order derivative). Easiest via echidna's HVP API which drives
// the tangent_reverse kernel.

#[cfg(feature = "gpu-cuda")]
#[test]
fn l23_cuda_powi_n1_at_zero_second_derivative_finite() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => return,
    };
    // f(x) = x.powi(1) = x. Hessian is 0. HVP at x=0, v=1 must be finite.
    // (On CPU: `0 * 0 * pow(0, -1) = 0 * Inf = NaN`; post-fix GPU path
    // short-circuits to 0.)
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].powi(1), &[0.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (hv_grad, hv) = ctx.hvp_batch(&gpu_tape, &[0.0_f32], &[1.0_f32], 1).unwrap();
    assert_eq!(hv_grad.len(), 1);
    assert_eq!(hv.len(), 1);
    assert!(
        hv[0].is_finite(),
        "HVP at powi(x,1), x=0 must be finite, got {}",
        hv[0]
    );
    assert_eq!(hv[0], 0.0, "Hessian of linear function is zero");
}

// Sibling of L23 on WGSL: the WGSL tangent_reverse must have the same
// n==1 short-circuit, otherwise the HVP for `x.powi(1)` at x=0 returns
// NaN and the backends disagree.
#[cfg(feature = "gpu-wgpu")]
#[test]
fn l23_wgpu_powi_n1_at_zero_second_derivative_finite() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => return,
    };
    let (tape, _) = record(|v: &[BReverse<f64>]| v[0].powi(1), &[0.0_f64]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);
    let (_hv_grad, hv) = ctx.hvp_batch(&gpu_tape, &[0.0_f32], &[1.0_f32], 1).unwrap();
    assert!(
        hv[0].is_finite(),
        "WGSL HVP at powi(x,1), x=0 must be finite, got {}",
        hv[0]
    );
    assert_eq!(hv[0], 0.0, "Hessian of linear function is zero");
}

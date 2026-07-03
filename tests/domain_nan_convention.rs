//! Out-of-domain derivative convention.
//!
//! Domain-restricted ops (`ln`, `log2`, `log10`, `ln_1p`, `atanh`) must emit a
//! NaN partial *strictly* outside their real domain — across every AD mode, not
//! just the bytecode `OpCode` reference — so a caller who supplies out-of-domain
//! inputs sees NaN instead of a finite but semantically meaningless derivative
//! (e.g. `d/dx ln(x) = 1/x = -0.5` at `x = -2`). Boundary values (`x = 0` for
//! `ln`, `x = -1` for `ln_1p`, `|x| = 1` for `atanh`) keep the IEEE `1/0 = ±Inf`
//! one-sided limit.

use echidna::{grad, jvp, Dual, DualVec, Reverse};
use num_traits::Float;

// Reverse (reverse-mode gradient), Dual (forward-mode JVP), and DualVec
// (forward-mode vector) must all agree with the `OpCode` convention: NaN
// derivative at an out-of-domain point.
macro_rules! out_of_domain_is_nan {
    ($name:ident, $method:ident, $p:expr) => {
        #[test]
        fn $name() {
            let p: f64 = $p;

            let g = grad(|x: &[Reverse<f64>]| x[0].$method(), &[p]);
            assert!(
                g[0].is_nan(),
                "{}: reverse grad must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                g[0]
            );

            let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].$method()], &[p], &[1.0]);
            assert!(
                t[0].is_nan(),
                "{}: dual tangent must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                t[0]
            );

            let dv = DualVec::<f64, 1>::new(p, [1.0]).$method();
            assert!(
                dv.eps[0].is_nan(),
                "{}: dualvec tangent must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                dv.eps[0]
            );
        }
    };
}

out_of_domain_is_nan!(ln_out_of_domain_is_nan, ln, -2.0);
out_of_domain_is_nan!(log2_out_of_domain_is_nan, log2, -2.0);
out_of_domain_is_nan!(log10_out_of_domain_is_nan, log10, -2.0);
out_of_domain_is_nan!(ln_1p_out_of_domain_is_nan, ln_1p, -2.0);
out_of_domain_is_nan!(atanh_out_of_domain_is_nan, atanh, 1.5);
out_of_domain_is_nan!(acosh_out_of_domain_is_nan, acosh, -2.0);

// Boundary + in-domain: the guard must NOT clobber the finite/±Inf values
// inside the domain (guards against over-correction that returns NaN there).
// `+0.0` is used deliberately — `-0.0` correctly yields `-Inf` via `1/-0.0` on
// every path and is not a boundary the convention special-cases.
#[test]
fn ln_boundary_and_in_domain_preserved() {
    let g = grad(|x: &[Reverse<f64>]| x[0].ln(), &[0.0]);
    assert!(
        g[0].is_infinite() && g[0] > 0.0,
        "ln reverse grad at x=+0 must be +Inf, got {}",
        g[0]
    );

    let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].ln()], &[0.0], &[1.0]);
    assert!(
        t[0].is_infinite() && t[0] > 0.0,
        "ln dual tangent at x=+0 must be +Inf, got {}",
        t[0]
    );

    let g = grad(|x: &[Reverse<f64>]| x[0].ln(), &[2.0]);
    assert!(
        (g[0] - 0.5).abs() < 1e-15,
        "ln reverse grad at x=2 must be 0.5, got {}",
        g[0]
    );
}

#[test]
fn atanh_boundary_preserves_inf() {
    let g = grad(|x: &[Reverse<f64>]| x[0].atanh(), &[1.0]);
    assert!(
        g[0].is_infinite() && g[0] > 0.0,
        "atanh reverse grad at x=1 must be +Inf, got {}",
        g[0]
    );
}

// acosh's domain is `a >= 1` (one-sided, unlike ln@0 / atanh@±1): the boundary
// a=1 keeps its +Inf one-sided derivative limit (1/sqrt(0)) and must not be
// clobbered to NaN by the `a < 1` guard.
#[test]
fn acosh_boundary_preserves_inf() {
    let g = grad(|x: &[Reverse<f64>]| x[0].acosh(), &[1.0]);
    assert!(
        g[0].is_infinite() && g[0] > 0.0,
        "acosh reverse grad at x=1 must be +Inf, got {}",
        g[0]
    );
    // In-domain sanity: d/dx acosh(2) = 1/sqrt(3).
    let g2 = grad(|x: &[Reverse<f64>]| x[0].acosh(), &[2.0]);
    assert!(
        (g2[0] - 1.0 / 3.0_f64.sqrt()).abs() < 1e-15,
        "acosh reverse grad at x=2 must be 1/sqrt(3), got {}",
        g2[0]
    );
}

// Cross-mode anchor: the bytecode forward-over-reverse path (gradient AND
// Hessian-vector product) already routes through the `OpCode` convention, so it
// is the canonical reference the scalar modes above must match. Green before and
// after the scalar-mode fix — it pins the target convention and cross-mode
// agreement in one place.
#[cfg(feature = "bytecode")]
#[test]
fn bytecode_hvp_out_of_domain_is_nan() {
    use echidna::{hvp, BReverse};

    type UnaryOp = fn(&[BReverse<f64>]) -> BReverse<f64>;
    let cases: [(&str, UnaryOp, f64); 6] = [
        ("ln", |x| x[0].ln(), -2.0),
        ("log2", |x| x[0].log2(), -2.0),
        ("log10", |x| x[0].log10(), -2.0),
        ("ln_1p", |x| x[0].ln_1p(), -2.0),
        ("atanh", |x| x[0].atanh(), 1.5),
        ("acosh", |x| x[0].acosh(), -2.0),
    ];
    for (label, f, p) in cases {
        let (g, h) = hvp(f, &[p], &[1.0]);
        assert!(
            g[0].is_nan(),
            "{label}: bytecode gradient must be NaN at out-of-domain x={p}, got {}",
            g[0]
        );
        assert!(
            h[0].is_nan(),
            "{label}: bytecode HVP must be NaN at out-of-domain x={p}, got {}",
            h[0]
        );
    }
}

// ── GPU parity ──
//
// The convention must hold on the GPU kernels too, or CPU and GPU disagree at
// out-of-domain inputs. Each public GPU entry point drives a *different* kernel,
// so all three are needed to cover every edited shader:
//   gradient_batch  → forward + reverse sweep (reverse.wgsl / reverse_sweep)
//   sparse_jacobian → forward tangent          (tangent_forward.wgsl / _forward)
//   hvp_batch       → forward-over-reverse      (tangent_reverse.wgsl, 2 phases)
// Tests are generic over the backend; each `wgpu_*`/`cuda_*` wrapper skips
// gracefully when no device is available.
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod gpu {
    use echidna::gpu::{GpuBackend, GpuTapeData};
    use echidna::{record, record_multi, BReverse, BytecodeTape};
    use num_traits::Float;

    #[cfg(feature = "gpu-cuda")]
    use echidna::gpu::CudaContext;
    #[cfg(feature = "gpu-wgpu")]
    use echidna::gpu::WgpuContext;

    // f64-recorded tapes (uploaded lossy to f32) for the reverse / HVP paths, one
    // per op at an out-of-domain input.
    fn out_of_domain_cases() -> [(&'static str, BytecodeTape<f64>, f32); 7] {
        [
            (
                "ln",
                record(|v: &[BReverse<f64>]| v[0].ln(), &[-2.0_f64]).0,
                -2.0,
            ),
            (
                "log2",
                record(|v: &[BReverse<f64>]| v[0].log2(), &[-2.0_f64]).0,
                -2.0,
            ),
            (
                "log10",
                record(|v: &[BReverse<f64>]| v[0].log10(), &[-2.0_f64]).0,
                -2.0,
            ),
            (
                "ln_1p",
                record(|v: &[BReverse<f64>]| v[0].ln_1p(), &[-2.0_f64]).0,
                -2.0,
            ),
            (
                "atanh",
                record(|v: &[BReverse<f64>]| v[0].atanh(), &[1.5_f64]).0,
                1.5,
            ),
            // acosh domain is a >= 1. `-2` hits the a <= -1 leg (factored
            // (a-1)(a+1) > 0 → finite garbage); `-1e9` hits the large-|a| overflow
            // branch — both must be caught by the `a < 1` guard placed *before* that
            // branch, so both are NaN.
            (
                "acosh",
                record(|v: &[BReverse<f64>]| v[0].acosh(), &[-2.0_f64]).0,
                -2.0,
            ),
            (
                "acosh_large",
                record(|v: &[BReverse<f64>]| v[0].acosh(), &[-2.0_f64]).0,
                -1e9,
            ),
        ]
    }

    // reverse.wgsl / reverse_sweep: reverse-mode gradient is NaN out of domain.
    fn check_domain_nan_gradient<B: GpuBackend>(ctx: &B) {
        for (label, tape, p) in out_of_domain_cases() {
            let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let (_, g) = ctx.gradient_batch(&gpu_tape, &[p], 1).unwrap();
            assert!(
                g[0].is_nan(),
                "{label}: GPU gradient must be NaN at out-of-domain x={p}, got {}",
                g[0]
            );
        }
    }

    // tangent_forward.wgsl / tangent_forward: forward-tangent Jacobian is NaN out
    // of domain (reached only via sparse_jacobian).
    fn check_domain_nan_jvp<B: GpuBackend>(ctx: &B) {
        let cases: [(&str, BytecodeTape<f32>, f32); 7] = [
            (
                "ln",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].ln()], &[-2.0_f32]).0,
                -2.0,
            ),
            (
                "log2",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].log2()], &[-2.0_f32]).0,
                -2.0,
            ),
            (
                "log10",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].log10()], &[-2.0_f32]).0,
                -2.0,
            ),
            (
                "ln_1p",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].ln_1p()], &[-2.0_f32]).0,
                -2.0,
            ),
            (
                "atanh",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].atanh()], &[1.5_f32]).0,
                1.5,
            ),
            (
                "acosh",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].acosh()], &[-2.0_f32]).0,
                -2.0,
            ),
            // -1e9 hits the large-|a| overflow branch on the tangent-forward kernel
            // too — pins the guard-before-1e8-branch ordering on this sweep.
            (
                "acosh_large",
                record_multi(|v: &[BReverse<f32>]| vec![v[0].acosh()], &[-2.0_f32]).0,
                -1e9,
            ),
        ];
        for (label, mut tape, p) in cases {
            let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let (_, _pattern, jac) = ctx.sparse_jacobian(&gpu_tape, &mut tape, &[p]).unwrap();
            assert!(
                !jac.is_empty() && jac[0].is_nan(),
                "{label}: GPU sparse-Jacobian entry must be NaN at out-of-domain x={p}, got {jac:?}"
            );
        }
    }

    // tangent_reverse.wgsl / tangent_reverse (both phases): HVP gradient AND H·v
    // are NaN out of domain.
    fn check_domain_nan_hvp<B: GpuBackend>(ctx: &B) {
        for (label, tape, p) in out_of_domain_cases() {
            let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let (g, hv) = ctx.hvp_batch(&gpu_tape, &[p], &[1.0_f32], 1).unwrap();
            assert!(
                g[0].is_nan(),
                "{label}: GPU HVP gradient must be NaN at out-of-domain x={p}, got {}",
                g[0]
            );
            assert!(
                hv[0].is_nan(),
                "{label}: GPU HVP (H·v) must be NaN at out-of-domain x={p}, got {}",
                hv[0]
            );
        }
    }

    // Boundary / in-domain must NOT be clobbered to NaN by the guard (would
    // signal a `>` instead of `>=`). Covers all five ops at their boundary on
    // the reverse path, plus `ln` through the forward-tangent (JVP) and
    // HVP paths so an over-eager guard is caught on every edited kernel, not
    // just reverse.
    fn check_domain_boundary<B: GpuBackend>(ctx: &B) {
        // Each op's boundary input → +Inf one-sided limit on the reverse path.
        let boundary: [(&str, BytecodeTape<f64>, f32); 6] = [
            (
                "ln",
                record(|v: &[BReverse<f64>]| v[0].ln(), &[0.0_f64]).0,
                0.0,
            ),
            (
                "log2",
                record(|v: &[BReverse<f64>]| v[0].log2(), &[0.0_f64]).0,
                0.0,
            ),
            (
                "log10",
                record(|v: &[BReverse<f64>]| v[0].log10(), &[0.0_f64]).0,
                0.0,
            ),
            (
                "ln_1p",
                record(|v: &[BReverse<f64>]| v[0].ln_1p(), &[-1.0_f64]).0,
                -1.0,
            ),
            (
                "atanh",
                record(|v: &[BReverse<f64>]| v[0].atanh(), &[1.0_f64]).0,
                1.0,
            ),
            // acosh's domain boundary is a=1 (one-sided) → +Inf, not NaN.
            (
                "acosh",
                record(|v: &[BReverse<f64>]| v[0].acosh(), &[1.0_f64]).0,
                1.0,
            ),
        ];
        for (label, tape, p) in boundary {
            let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let (_, g) = ctx.gradient_batch(&gpu_tape, &[p], 1).unwrap();
            assert!(
                g[0].is_infinite() && g[0] > 0.0,
                "{label}: GPU gradient at boundary x={p} must be +Inf, got {}",
                g[0]
            );
        }

        // `ln` through the tangent (sparse_jacobian) and HVP (hvp_batch) kernels
        // at in-domain x=2 → 0.5, proving those guards don't over-clobber either.
        let (mut jac_tape, _) = record_multi(|v: &[BReverse<f32>]| vec![v[0].ln()], &[2.0_f32]);
        let jac_gpu = ctx.upload_tape(&GpuTapeData::from_tape(&jac_tape).unwrap());
        let (_, _pattern, jac) = ctx
            .sparse_jacobian(&jac_gpu, &mut jac_tape, &[2.0_f32])
            .unwrap();
        assert!(
            !jac.is_empty() && (jac[0] as f64 - 0.5).abs() < 1e-5,
            "ln GPU sparse-Jacobian at in-domain x=2 must be 0.5, got {jac:?}"
        );

        let (hvp_tape, _) = record(|v: &[BReverse<f64>]| v[0].ln(), &[2.0_f64]);
        let hvp_gpu = ctx.upload_tape(&GpuTapeData::from_tape_f64_lossy(&hvp_tape).unwrap());
        let (g, hv) = ctx.hvp_batch(&hvp_gpu, &[2.0_f32], &[1.0_f32], 1).unwrap();
        assert!(
            (g[0] as f64 - 0.5).abs() < 1e-5,
            "ln GPU HVP gradient at in-domain x=2 must be 0.5, got {}",
            g[0]
        );
        // d²/dx² ln(x) = -1/x² = -0.25 at x=2.
        assert!(
            (hv[0] as f64 + 0.25).abs() < 1e-5,
            "ln GPU HVP (H·v) at in-domain x=2 must be -0.25, got {}",
            hv[0]
        );

        // acosh at its a=1 boundary through the tangent (sparse_jacobian) and HVP
        // (hvp_batch) kernels → +Inf (1/sqrt(0)), NOT NaN. Otherwise a `<` vs `<=`
        // slip in the acosh guard on those two sweeps (each has its own copy) would
        // go uncaught — the reverse path's acosh@1 boundary check above only covers
        // reverse.wgsl / reverse_sweep.
        let (mut ac_jac_tape, _) =
            record_multi(|v: &[BReverse<f32>]| vec![v[0].acosh()], &[1.0_f32]);
        let ac_jac_gpu = ctx.upload_tape(&GpuTapeData::from_tape(&ac_jac_tape).unwrap());
        let (_, _pattern, ac_jac) = ctx
            .sparse_jacobian(&ac_jac_gpu, &mut ac_jac_tape, &[1.0_f32])
            .unwrap();
        assert!(
            !ac_jac.is_empty() && ac_jac[0].is_infinite() && ac_jac[0] > 0.0,
            "acosh GPU sparse-Jacobian at boundary a=1 must be +Inf, got {ac_jac:?}"
        );

        let (ac_hvp_tape, _) = record(|v: &[BReverse<f64>]| v[0].acosh(), &[1.0_f64]);
        let ac_hvp_gpu = ctx.upload_tape(&GpuTapeData::from_tape_f64_lossy(&ac_hvp_tape).unwrap());
        let (ac_g, _ac_hv) = ctx
            .hvp_batch(&ac_hvp_gpu, &[1.0_f32], &[1.0_f32], 1)
            .unwrap();
        assert!(
            ac_g[0].is_infinite() && ac_g[0] > 0.0,
            "acosh GPU HVP gradient at boundary a=1 must be +Inf, got {}",
            ac_g[0]
        );
    }

    macro_rules! backend_tests {
        ($check:ident, $wgpu_name:ident, $cuda_name:ident) => {
            #[cfg(feature = "gpu-wgpu")]
            #[test]
            fn $wgpu_name() {
                if let Some(c) = WgpuContext::new() {
                    $check(&c);
                }
            }
            #[cfg(feature = "gpu-cuda")]
            #[test]
            fn $cuda_name() {
                if let Some(c) = CudaContext::new() {
                    $check(&c);
                }
            }
        };
    }

    backend_tests!(
        check_domain_nan_gradient,
        wgpu_domain_nan_gradient,
        cuda_domain_nan_gradient
    );
    backend_tests!(
        check_domain_nan_jvp,
        wgpu_domain_nan_jvp,
        cuda_domain_nan_jvp
    );
    backend_tests!(
        check_domain_nan_hvp,
        wgpu_domain_nan_hvp,
        cuda_domain_nan_hvp
    );
    backend_tests!(
        check_domain_boundary,
        wgpu_domain_boundary,
        cuda_domain_boundary
    );
}

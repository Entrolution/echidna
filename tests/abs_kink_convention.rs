//! `abs` kink-derivative convention: unified to the minimal-norm subgradient `0`
//! at `x = 0` across every AD mode and both GPU backends (value-based, so `+0`
//! and `-0` agree; independent of how the zero was produced). `sign(x)` off the
//! kink, `NaN` at `NaN`. Sharp/limiting subgradients still force `±1` via the
//! nonsmooth machinery (`forced_reverse_partials`), so this only fixes the smooth
//! default. Mirrors the `domain_nan_convention` cross-mode + GPU-parity structure.

use echidna::{grad, jvp, Dual, DualVec, Reverse};
use num_traits::Float;

// At the kink (`+0` and `-0`) every eager mode's abs derivative is 0.
#[test]
fn abs_deriv_at_kink_is_zero_all_eager_modes() {
    for p in [0.0_f64, -0.0_f64] {
        let g = grad(|x: &[Reverse<f64>]| x[0].abs(), &[p]);
        assert_eq!(g[0], 0.0, "reverse abs' at x={p} must be 0, got {}", g[0]);

        let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].abs()], &[p], &[1.0]);
        assert_eq!(t[0], 0.0, "dual abs' at x={p} must be 0, got {}", t[0]);

        let dv = DualVec::<f64, 1>::new(p, [1.0]).abs();
        assert_eq!(
            dv.eps[0], 0.0,
            "dualvec abs' at x={p} must be 0, got {}",
            dv.eps[0]
        );
    }
}

// Off the kink the derivative is `sign(x)` (regression anchor — no over-guarding).
#[test]
fn abs_deriv_off_kink_is_sign() {
    let gp = grad(|x: &[Reverse<f64>]| x[0].abs(), &[3.0]);
    assert_eq!(gp[0], 1.0);
    let gn = grad(|x: &[Reverse<f64>]| x[0].abs(), &[-3.0]);
    assert_eq!(gn[0], -1.0);
    let (_, tp) = jvp(|x: &[Dual<f64>]| vec![x[0].abs()], &[3.0], &[1.0]);
    assert_eq!(tp[0], 1.0);
}

// `NaN` flows through the derivative (via `signum(NaN) = NaN`).
#[test]
fn abs_deriv_at_nan_is_nan() {
    let g = grad(|x: &[Reverse<f64>]| x[0].abs(), &[f64::NAN]);
    assert!(
        g[0].is_nan(),
        "reverse abs' at NaN must be NaN, got {}",
        g[0]
    );
    let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].abs()], &[f64::NAN], &[1.0]);
    assert!(t[0].is_nan(), "dual abs' at NaN must be NaN, got {}", t[0]);
    let dv = DualVec::<f64, 1>::new(f64::NAN, [1.0]).abs();
    assert!(dv.eps[0].is_nan(), "dualvec abs' at NaN must be NaN");
}

// The bytecode reverse + HVP path already returns 0 at the kink (it was the
// deliberate site); this pins that the whole cross-mode convention agrees.
#[cfg(feature = "bytecode")]
#[test]
fn abs_deriv_kink_bytecode_agrees() {
    use echidna::{hvp, BReverse};
    let (g, h) = hvp(|x: &[BReverse<f64>]| x[0].abs(), &[0.0], &[1.0]);
    assert_eq!(g[0], 0.0, "bytecode gradient abs' at 0 must be 0");
    assert_eq!(h[0], 0.0, "bytecode HVP of abs at 0 must be 0");
}

// ── GPU parity ── the kink convention must hold on the GPU kernels too, on every
// edited sweep: gradient_batch (reverse.wgsl / reverse_sweep), sparse_jacobian
// (tangent_forward), hvp_batch (tangent_reverse, both phases). Generic over B.
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod gpu {
    use echidna::gpu::{GpuBackend, GpuTapeData};
    use echidna::{record, record_multi, BReverse};
    use num_traits::Float;

    #[cfg(feature = "gpu-cuda")]
    use echidna::gpu::CudaContext;
    #[cfg(feature = "gpu-wgpu")]
    use echidna::gpu::WgpuContext;

    // `-0.0` via bit pattern (a literal `-0.0_f32` can be folded to `+0.0` by some
    // shader compilers); both zeros must give 0.
    fn neg_zero() -> f32 {
        f32::from_bits(0x8000_0000)
    }

    fn check_abs_kink_zero<B: GpuBackend>(ctx: &B) {
        for p in [0.0_f32, neg_zero()] {
            // reverse (gradient_batch)
            let (tape, _) = record(|v: &[BReverse<f64>]| v[0].abs(), &[1.0_f64]);
            let gt = ctx.upload_tape(&GpuTapeData::from_tape_f64_lossy(&tape).unwrap());
            let (_, g) = ctx.gradient_batch(&gt, &[p], 1).unwrap();
            assert_eq!(
                g[0], 0.0,
                "GPU reverse abs' at x={p} must be 0, got {}",
                g[0]
            );

            // hvp (tangent_reverse, both phases): first-order 0, second-order finite.
            let (g2, hv) = ctx.hvp_batch(&gt, &[p], &[1.0_f32], 1).unwrap();
            assert_eq!(
                g2[0], 0.0,
                "GPU HVP abs' gradient at x={p} must be 0, got {}",
                g2[0]
            );
            assert!(
                hv[0].is_finite(),
                "GPU HVP abs H·v at x={p} must be finite, got {}",
                hv[0]
            );

            // jvp (sparse_jacobian -> tangent_forward)
            let (mut jtape, _) = record_multi(|v: &[BReverse<f32>]| vec![v[0].abs()], &[1.0_f32]);
            let jgt = ctx.upload_tape(&GpuTapeData::from_tape(&jtape).unwrap());
            let (_, _pat, jac) = ctx.sparse_jacobian(&jgt, &mut jtape, &[p]).unwrap();
            assert!(
                !jac.is_empty() && jac[0] == 0.0,
                "GPU sparse-Jacobian abs' at x={p} must be 0, got {jac:?}"
            );
        }
    }

    fn check_abs_nan<B: GpuBackend>(ctx: &B) {
        // abs' at a NaN primal is NaN on every sweep (the wgpu tangent shaders
        // previously zeroed it).
        let (tape, _) = record(|v: &[BReverse<f64>]| v[0].abs(), &[1.0_f64]);
        let gt = ctx.upload_tape(&GpuTapeData::from_tape_f64_lossy(&tape).unwrap());
        let (_, g) = ctx.gradient_batch(&gt, &[f32::NAN], 1).unwrap();
        assert!(
            g[0].is_nan(),
            "GPU reverse abs' at NaN must be NaN, got {}",
            g[0]
        );
        let (_, hv) = ctx.hvp_batch(&gt, &[f32::NAN], &[1.0_f32], 1).unwrap();
        assert!(
            hv[0].is_nan(),
            "GPU HVP abs at NaN must be NaN, got {}",
            hv[0]
        );

        let (mut jtape, _) = record_multi(|v: &[BReverse<f32>]| vec![v[0].abs()], &[1.0_f32]);
        let jgt = ctx.upload_tape(&GpuTapeData::from_tape(&jtape).unwrap());
        let (_, _pat, jac) = ctx.sparse_jacobian(&jgt, &mut jtape, &[f32::NAN]).unwrap();
        assert!(
            !jac.is_empty() && jac[0].is_nan(),
            "GPU sparse-Jacobian abs' at NaN must be NaN, got {jac:?}"
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

    backend_tests!(check_abs_kink_zero, wgpu_abs_kink_zero, cuda_abs_kink_zero);
    backend_tests!(check_abs_nan, wgpu_abs_nan, cuda_abs_nan);
}

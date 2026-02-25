#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::{record, Scalar};

#[cfg(feature = "gpu-wgpu")]
use echidna::gpu::{GpuBackend, GpuTapeData, WgpuContext};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

fn polynomial<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

fn trig_func<T: Scalar>(x: &[T]) -> T {
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    x[0].sin() * x[1].cos() + (x[0] * x[1] / two).exp()
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_context() -> Option<WgpuContext> {
    match WgpuContext::new() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("WARNING: No GPU adapter found — skipping GPU STDE test");
            None
        }
    }
}

// ══════════════════════════════════════════════
//  Section 1: Taylor forward 2nd-order basic tests
// ══════════════════════════════════════════════

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_taylor_2nd_polynomial() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(|v| polynomial(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let primals = [3.0f32, 4.0];
    let seeds = [1.0f32, 0.0];
    let result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &primals, &seeds, 1)
        .unwrap();

    assert!(
        (result.values[0] - 25.0).abs() < 1e-4,
        "value: {}",
        result.values[0]
    );
    assert!((result.c1s[0] - 6.0).abs() < 1e-4, "c1: {}", result.c1s[0]);
    assert!((result.c2s[0] - 1.0).abs() < 1e-4, "c2: {}", result.c2s[0]);
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_2nd_rosenbrock_matches_cpu() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.5_f64, 2.5];
    let (tape_f64, _) = record(|v| rosenbrock(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape_f64).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let x_f32 = [1.5f32, 2.5];
    let dir = [0.6f32, 0.8];

    let gpu_result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &x_f32, &dir, 1)
        .unwrap();

    let dir_f64 = [0.6_f64, 0.8];
    let (c0, c1, c2) = echidna::stde::taylor_jet_2nd(&tape_f64, &x, &dir_f64);

    let tol: f64 = 1e-2;
    assert!(
        (gpu_result.values[0] as f64 - c0).abs() < tol,
        "c0: gpu={} cpu={}",
        gpu_result.values[0],
        c0
    );
    assert!(
        (gpu_result.c1s[0] as f64 - c1).abs() < tol,
        "c1: gpu={} cpu={}",
        gpu_result.c1s[0],
        c1
    );
    assert!(
        (gpu_result.c2s[0] as f64 - c2).abs() < tol,
        "c2: gpu={} cpu={}",
        gpu_result.c2s[0],
        c2
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_taylor_2nd_batch_sizes() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [2.0_f64, 3.0];
    let (tape, _) = record(|v| polynomial(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    for batch_size in [1u32, 4, 16, 256, 1000] {
        let mut primals = Vec::new();
        let mut seeds = Vec::new();
        for b in 0..batch_size {
            primals.extend_from_slice(&[2.0f32, 3.0]);
            if b % 2 == 0 {
                seeds.extend_from_slice(&[1.0f32, 0.0]);
            } else {
                seeds.extend_from_slice(&[0.0f32, 1.0]);
            }
        }

        let result = ctx
            .taylor_forward_2nd_batch(&tape_buf, &primals, &seeds, batch_size)
            .unwrap();

        assert_eq!(result.values.len(), batch_size as usize);
        assert_eq!(result.c1s.len(), batch_size as usize);
        assert_eq!(result.c2s.len(), batch_size as usize);

        for b in 0..batch_size as usize {
            assert!(
                (result.values[b] - 13.0).abs() < 1e-4,
                "batch {} value: {}",
                b,
                result.values[b]
            );
            assert!(
                (result.c2s[b] - 1.0).abs() < 1e-4,
                "batch {} c2: {}",
                b,
                result.c2s[b]
            );
        }
    }
}

// ══════════════════════════════════════════════
//  Section 2: Per-opcode Taylor K=3 tests
// ══════════════════════════════════════════════

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
mod opcode_tests {
    use super::*;

    /// Helper: record a 1-input function, push direction 1.0 on GPU and CPU, compare.
    fn check_1d(
        ctx: &WgpuContext,
        f: fn(&[echidna::BReverse<f64>]) -> echidna::BReverse<f64>,
        x0: f64,
        label: &str,
    ) {
        let x = [x0];
        let (tape, _) = record(f, &x);
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
        let tape_buf = ctx.upload_tape(&gpu_data);

        let gpu_result = ctx
            .taylor_forward_2nd_batch(&tape_buf, &[x0 as f32], &[1.0f32], 1)
            .unwrap();

        let (c0, c1, c2) = echidna::stde::taylor_jet_2nd(&tape, &x, &[1.0]);

        let tol: f64 = 1e-2;
        assert!(
            (gpu_result.values[0] as f64 - c0).abs() < tol.max(c0.abs() * 1e-4),
            "{} c0: gpu={} cpu={}",
            label,
            gpu_result.values[0],
            c0
        );
        assert!(
            (gpu_result.c1s[0] as f64 - c1).abs() < tol.max(c1.abs() * 1e-4),
            "{} c1: gpu={} cpu={}",
            label,
            gpu_result.c1s[0],
            c1
        );
        assert!(
            (gpu_result.c2s[0] as f64 - c2).abs() < tol.max(c2.abs() * 1e-3),
            "{} c2: gpu={} cpu={}",
            label,
            gpu_result.c2s[0],
            c2
        );
    }

    fn f_exp<T: Scalar>(x: &[T]) -> T {
        x[0].exp()
    }
    fn f_ln<T: Scalar>(x: &[T]) -> T {
        x[0].ln()
    }
    fn f_exp2<T: Scalar>(x: &[T]) -> T {
        x[0].exp2()
    }
    fn f_log2<T: Scalar>(x: &[T]) -> T {
        x[0].log2()
    }
    fn f_log10<T: Scalar>(x: &[T]) -> T {
        x[0].log10()
    }
    fn f_ln_1p<T: Scalar>(x: &[T]) -> T {
        x[0].ln_1p()
    }
    fn f_exp_m1<T: Scalar>(x: &[T]) -> T {
        x[0].exp_m1()
    }
    fn f_sqrt<T: Scalar>(x: &[T]) -> T {
        x[0].sqrt()
    }
    fn f_cbrt<T: Scalar>(x: &[T]) -> T {
        x[0].cbrt()
    }
    fn f_sin<T: Scalar>(x: &[T]) -> T {
        x[0].sin()
    }
    fn f_cos<T: Scalar>(x: &[T]) -> T {
        x[0].cos()
    }
    fn f_tan<T: Scalar>(x: &[T]) -> T {
        x[0].tan()
    }
    fn f_sinh<T: Scalar>(x: &[T]) -> T {
        x[0].sinh()
    }
    fn f_cosh<T: Scalar>(x: &[T]) -> T {
        x[0].cosh()
    }
    fn f_tanh<T: Scalar>(x: &[T]) -> T {
        x[0].tanh()
    }
    fn f_asin<T: Scalar>(x: &[T]) -> T {
        x[0].asin()
    }
    fn f_acos<T: Scalar>(x: &[T]) -> T {
        x[0].acos()
    }
    fn f_atan<T: Scalar>(x: &[T]) -> T {
        x[0].atan()
    }
    fn f_asinh<T: Scalar>(x: &[T]) -> T {
        x[0].asinh()
    }
    fn f_acosh<T: Scalar>(x: &[T]) -> T {
        x[0].acosh()
    }
    fn f_atanh<T: Scalar>(x: &[T]) -> T {
        x[0].atanh()
    }
    fn f_abs_fn<T: Scalar>(x: &[T]) -> T {
        x[0].abs()
    }
    fn f_powi3<T: Scalar>(x: &[T]) -> T {
        x[0].powi(3)
    }
    fn f_powf25<T: Scalar>(x: &[T]) -> T {
        let exp = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.5).unwrap());
        x[0].powf(exp)
    }
    fn f_arith<T: Scalar>(x: &[T]) -> T {
        let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
        let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
        (x[0] + one) * (two - x[0]) + x[0].recip()
    }
    fn f_div<T: Scalar>(x: &[T]) -> T {
        let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
        (x[0] * x[0] + one) / (x[0] + one)
    }

    #[test]
    fn gpu_op_exp_ln() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_exp, 1.5, "exp");
        check_1d(&ctx, f_ln, 2.0, "ln");
        check_1d(&ctx, f_exp2, 1.0, "exp2");
        check_1d(&ctx, f_log2, 3.0, "log2");
        check_1d(&ctx, f_log10, 2.0, "log10");
        check_1d(&ctx, f_ln_1p, 0.5, "ln_1p");
        check_1d(&ctx, f_exp_m1, 0.3, "expm1");
    }

    #[test]
    fn gpu_op_sqrt_cbrt() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_sqrt, 4.0, "sqrt");
        check_1d(&ctx, f_cbrt, 8.0, "cbrt");
    }

    #[test]
    fn gpu_op_sin_cos() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_sin, 1.0, "sin");
        check_1d(&ctx, f_cos, 1.0, "cos");
    }

    #[test]
    fn gpu_op_tan() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_tan, 0.5, "tan");
    }

    #[test]
    fn gpu_op_hyperbolic() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_sinh, 1.0, "sinh");
        check_1d(&ctx, f_cosh, 1.0, "cosh");
        check_1d(&ctx, f_tanh, 0.5, "tanh");
    }

    #[test]
    fn gpu_op_inverse_trig() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_asin, 0.5, "asin");
        check_1d(&ctx, f_acos, 0.5, "acos");
        check_1d(&ctx, f_atan, 1.0, "atan");
    }

    #[test]
    fn gpu_op_inverse_hyp() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_asinh, 1.0, "asinh");
        check_1d(&ctx, f_acosh, 2.0, "acosh");
        check_1d(&ctx, f_atanh, 0.5, "atanh");
    }

    #[test]
    fn gpu_op_pow() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_powf25, 2.0, "powf");
        check_1d(&ctx, f_powi3, 2.0, "powi");
    }

    #[test]
    fn gpu_op_arithmetic() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_arith, 1.5, "arith");
    }

    #[test]
    fn gpu_op_div() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_div, 2.0, "div");
    }

    #[test]
    fn gpu_op_nonsmooth() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_1d(&ctx, f_abs_fn, -2.0, "abs_neg");
        check_1d(&ctx, f_abs_fn, 2.0, "abs_pos");
    }
}

// ══════════════════════════════════════════════
//  Section 3: GPU STDE high-level function tests
// ══════════════════════════════════════════════

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_laplacian_matches_cpu() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.5_f64, 2.5];
    let (tape_f64, _) = record(|v| rosenbrock(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape_f64).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let n = 2;
    let s = 10;
    let dirs: Vec<Vec<f64>> = (0..s)
        .map(|si| {
            (0..n)
                .map(|i| if (si * n + i) % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();

    let dir_refs_f64: Vec<&[f64]> = dirs.iter().map(|d| d.as_slice()).collect();
    let (_, cpu_laplacian) = echidna::stde::laplacian(&tape_f64, &x, &dir_refs_f64);

    let dirs_f32: Vec<Vec<f32>> = dirs
        .iter()
        .map(|d| d.iter().map(|&v| v as f32).collect())
        .collect();
    let dir_refs_f32: Vec<&[f32]> = dirs_f32.iter().map(|d| d.as_slice()).collect();

    let gpu_result =
        echidna::gpu::stde_gpu::laplacian_gpu(&ctx, &tape_buf, &[1.5f32, 2.5], &dir_refs_f32)
            .unwrap();

    let tol: f64 = 2.0;
    assert!(
        (gpu_result.estimate as f64 - cpu_laplacian).abs() < tol,
        "gpu={} cpu={}",
        gpu_result.estimate,
        cpu_laplacian
    );
    assert_eq!(gpu_result.num_samples, s);
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_hessian_diagonal_matches_cpu() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.5_f64, 2.5];
    let (tape_f64, _) = record(|v| rosenbrock(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape_f64).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let (cpu_val, cpu_diag) = echidna::stde::hessian_diagonal(&tape_f64, &x);

    let (gpu_val, gpu_diag) =
        echidna::gpu::stde_gpu::hessian_diagonal_gpu(&ctx, &tape_buf, &[1.5f32, 2.5]).unwrap();

    let tol: f64 = 0.5;
    assert!(
        (gpu_val as f64 - cpu_val).abs() < tol,
        "val: gpu={} cpu={}",
        gpu_val,
        cpu_val
    );
    for (j, (&g, &c)) in gpu_diag.iter().zip(cpu_diag.iter()).enumerate() {
        assert!(
            (g as f64 - c).abs() < tol.max(c.abs() * 1e-3),
            "diag[{}]: gpu={} cpu={}",
            j,
            g,
            c
        );
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_polynomial_exact_laplacian() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(|v| polynomial(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let dirs: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
        vec![-1.0, -1.0],
    ];
    let dir_refs: Vec<&[f32]> = dirs.iter().map(|d| d.as_slice()).collect();

    let result =
        echidna::gpu::stde_gpu::laplacian_gpu(&ctx, &tape_buf, &[3.0f32, 4.0], &dir_refs).unwrap();

    assert!(
        (result.estimate - 4.0).abs() < 1e-3,
        "Laplacian estimate: {} (expected 4)",
        result.estimate
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_polynomial_exact_hessian_diagonal() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(|v| polynomial(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let (val, diag) =
        echidna::gpu::stde_gpu::hessian_diagonal_gpu(&ctx, &tape_buf, &[3.0f32, 4.0]).unwrap();

    assert!((val - 25.0).abs() < 1e-3, "value: {}", val);
    assert!((diag[0] - 2.0).abs() < 1e-3, "diag[0]: {}", diag[0]);
    assert!((diag[1] - 2.0).abs() < 1e-3, "diag[1]: {}", diag[1]);
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_trig_taylor_2nd() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.0_f64, 0.5];
    let (tape, _) = record(|v| trig_func(v), &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let dir = [0.6f32, 0.8];
    let result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &[1.0f32, 0.5], &dir, 1)
        .unwrap();

    let (c0, c1, c2) = echidna::stde::taylor_jet_2nd(&tape, &x, &[0.6, 0.8]);

    let tol: f64 = 1e-2;
    assert!(
        (result.values[0] as f64 - c0).abs() < tol,
        "c0: {} vs {}",
        result.values[0],
        c0
    );
    assert!(
        (result.c1s[0] as f64 - c1).abs() < tol,
        "c1: {} vs {}",
        result.c1s[0],
        c1
    );
    assert!(
        (result.c2s[0] as f64 - c2).abs() < tol,
        "c2: {} vs {}",
        result.c2s[0],
        c2
    );
}

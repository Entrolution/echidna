#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::{record, Scalar};

#[cfg(feature = "gpu-cuda")]
use echidna::gpu::CudaContext;
#[cfg(feature = "gpu-wgpu")]
use echidna::gpu::WgpuContext;
use echidna::gpu::{GpuBackend, GpuTapeData};

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

#[cfg(feature = "gpu-cuda")]
fn cuda_context() -> Option<CudaContext> {
    match CudaContext::new() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("WARNING: No CUDA device found — skipping GPU STDE test");
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
    let (tape, _) = record(polynomial, &x);
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
    let (tape_f64, _) = record(rosenbrock, &x);
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
    let (tape, _) = record(polynomial, &x);
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

#[cfg(feature = "stde")]
fn check_1d(
    ctx: &impl GpuBackend,
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

/// Generate per-opcode Taylor tests for a specific GPU backend.
macro_rules! opcode_tests_for_backend {
    ($mod_name:ident, $feature:literal, $ctx_fn:path, $ctx_ty:ty) => {
        #[cfg(all(feature = $feature, feature = "stde"))]
        mod $mod_name {
            use super::*;

            fn get_ctx() -> Option<$ctx_ty> {
                $ctx_fn()
            }

            #[test]
            fn op_exp_ln() {
                let ctx = match get_ctx() {
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
            fn op_sqrt_cbrt() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_sqrt, 4.0, "sqrt");
                check_1d(&ctx, f_cbrt, 8.0, "cbrt");
            }
            #[test]
            fn op_sin_cos() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_sin, 1.0, "sin");
                check_1d(&ctx, f_cos, 1.0, "cos");
            }
            #[test]
            fn op_tan() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_tan, 0.5, "tan");
            }
            #[test]
            fn op_hyperbolic() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_sinh, 1.0, "sinh");
                check_1d(&ctx, f_cosh, 1.0, "cosh");
                check_1d(&ctx, f_tanh, 0.5, "tanh");
            }
            #[test]
            fn op_inverse_trig() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_asin, 0.5, "asin");
                check_1d(&ctx, f_acos, 0.5, "acos");
                check_1d(&ctx, f_atan, 1.0, "atan");
            }
            #[test]
            fn op_inverse_hyp() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_asinh, 1.0, "asinh");
                check_1d(&ctx, f_acosh, 2.0, "acosh");
                check_1d(&ctx, f_atanh, 0.5, "atanh");
            }
            #[test]
            fn op_pow() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_powf25, 2.0, "powf");
                check_1d(&ctx, f_powi3, 2.0, "powi");
            }
            #[test]
            fn op_arithmetic() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_arith, 1.5, "arith");
            }
            #[test]
            fn op_div() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_div, 2.0, "div");
            }
            #[test]
            fn op_nonsmooth() {
                let ctx = match get_ctx() {
                    Some(c) => c,
                    None => return,
                };
                check_1d(&ctx, f_abs_fn, -2.0, "abs_neg");
                check_1d(&ctx, f_abs_fn, 2.0, "abs_pos");
            }
        }
    };
}

#[cfg(feature = "gpu-wgpu")]
opcode_tests_for_backend!(wgpu_opcode_tests, "gpu-wgpu", gpu_context, WgpuContext);
#[cfg(feature = "gpu-cuda")]
opcode_tests_for_backend!(cuda_opcode_tests, "gpu-cuda", cuda_context, CudaContext);

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
    let (tape_f64, _) = record(rosenbrock, &x);
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
    let (tape_f64, _) = record(rosenbrock, &x);
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
    let (tape, _) = record(polynomial, &x);
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
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let (val, diag) =
        echidna::gpu::stde_gpu::hessian_diagonal_gpu(&ctx, &tape_buf, &[3.0f32, 4.0]).unwrap();

    assert!((val - 25.0).abs() < 1e-3, "value: {}", val);
    assert!((diag[0] - 2.0).abs() < 1e-3, "diag[0]: {}", diag[0]);
    assert!((diag[1] - 2.0).abs() < 1e-3, "diag[1]: {}", diag[1]);
}

// ══════════════════════════════════════════════
//  Section 4: Chunked GPU Taylor dispatch tests
// ══════════════════════════════════════════════

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_chunked_single_chunk() {
    // When batch fits in one chunk, results match direct call
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let batch_size = 4u32;
    let mut primals = Vec::new();
    let mut seeds = Vec::new();
    for b in 0..batch_size {
        primals.extend_from_slice(&[3.0f32, 4.0]);
        if b % 2 == 0 {
            seeds.extend_from_slice(&[1.0f32, 0.0]);
        } else {
            seeds.extend_from_slice(&[0.0f32, 1.0]);
        }
    }

    let direct = ctx
        .taylor_forward_2nd_batch(&tape_buf, &primals, &seeds, batch_size)
        .unwrap();

    // Use very large max_buffer_bytes so everything fits in one chunk
    let chunked = echidna::gpu::taylor_forward_2nd_batch_chunked(
        &ctx,
        &tape_buf,
        &primals,
        &seeds,
        batch_size,
        gpu_data.num_inputs,
        gpu_data.num_variables,
        1024 * 1024 * 1024, // 1 GiB
    )
    .unwrap();

    assert_eq!(direct.values.len(), chunked.values.len());
    for i in 0..direct.values.len() {
        assert!(
            (direct.values[i] - chunked.values[i]).abs() < 1e-6,
            "values[{}] mismatch",
            i
        );
        assert!(
            (direct.c1s[i] - chunked.c1s[i]).abs() < 1e-6,
            "c1s[{}] mismatch",
            i
        );
        assert!(
            (direct.c2s[i] - chunked.c2s[i]).abs() < 1e-6,
            "c2s[{}] mismatch",
            i
        );
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_chunked_multi_chunk() {
    // Force multi-chunk by setting a tiny max_buffer_bytes
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let batch_size = 8u32;
    let mut primals = Vec::new();
    let mut seeds = Vec::new();
    for b in 0..batch_size {
        primals.extend_from_slice(&[3.0f32, 4.0]);
        if b % 2 == 0 {
            seeds.extend_from_slice(&[1.0f32, 0.0]);
        } else {
            seeds.extend_from_slice(&[0.0f32, 1.0]);
        }
    }

    let direct = ctx
        .taylor_forward_2nd_batch(&tape_buf, &primals, &seeds, batch_size)
        .unwrap();

    // bytes_per_element = num_variables * 3 * 4
    // With a tiny limit, each chunk gets only ~2 elements
    let bytes_per_element = (gpu_data.num_variables as u64) * 3 * 4;
    let max_bytes = bytes_per_element * 2; // force chunks of 2

    let chunked = echidna::gpu::taylor_forward_2nd_batch_chunked(
        &ctx,
        &tape_buf,
        &primals,
        &seeds,
        batch_size,
        gpu_data.num_inputs,
        gpu_data.num_variables,
        max_bytes,
    )
    .unwrap();

    assert_eq!(direct.values.len(), chunked.values.len());
    for i in 0..direct.values.len() {
        assert!(
            (direct.values[i] - chunked.values[i]).abs() < 1e-5,
            "values[{}]: {} vs {}",
            i,
            direct.values[i],
            chunked.values[i]
        );
        assert!(
            (direct.c1s[i] - chunked.c1s[i]).abs() < 1e-5,
            "c1s[{}]: {} vs {}",
            i,
            direct.c1s[i],
            chunked.c1s[i]
        );
        assert!(
            (direct.c2s[i] - chunked.c2s[i]).abs() < 1e-5,
            "c2s[{}]: {} vs {}",
            i,
            direct.c2s[i],
            chunked.c2s[i]
        );
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_chunked_exact_boundary() {
    // Batch exactly fills one chunk (boundary condition)
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [2.0_f64, 3.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let batch_size = 4u32;
    let mut primals = Vec::new();
    let mut seeds = Vec::new();
    for _ in 0..batch_size {
        primals.extend_from_slice(&[2.0f32, 3.0]);
        seeds.extend_from_slice(&[1.0f32, 0.0]);
    }

    let bytes_per_element = (gpu_data.num_variables as u64) * 3 * 4;
    let max_bytes = bytes_per_element * (batch_size as u64); // exact fit

    let result = echidna::gpu::taylor_forward_2nd_batch_chunked(
        &ctx,
        &tape_buf,
        &primals,
        &seeds,
        batch_size,
        gpu_data.num_inputs,
        gpu_data.num_variables,
        max_bytes,
    )
    .unwrap();

    assert_eq!(result.values.len(), batch_size as usize);
    for v in &result.values {
        assert!((v - 13.0).abs() < 1e-4, "value: {}", v);
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_chunked_zero_batch() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.0_f64, 2.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let result = echidna::gpu::taylor_forward_2nd_batch_chunked(
        &ctx,
        &tape_buf,
        &[],
        &[],
        0,
        gpu_data.num_inputs,
        gpu_data.num_variables,
        1024,
    )
    .unwrap();

    assert!(result.values.is_empty());
    assert!(result.c1s.is_empty());
    assert!(result.c2s.is_empty());
}

// ══════════════════════════════════════════════
//  Section 5: General-K Taylor forward tests
// ══════════════════════════════════════════════

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_kth_polynomial_all_orders() {
    // f(x,y) = x² + y², at (3,4), direction (1,0)
    // c0 = 25, c1 = 6, c2 = 1, c3+ = 0 (polynomial of degree 2)
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    for order in 1..=5 {
        let result = ctx
            .taylor_forward_kth_batch(&tape_buf, &[3.0f32, 4.0], &[1.0f32, 0.0], 1, order)
            .unwrap();

        assert_eq!(result.order, order);
        assert_eq!(result.coefficients.len(), order);
        assert_eq!(result.coefficients[0].len(), 1);

        // c0 = 25
        assert!(
            (result.coefficients[0][0] - 25.0).abs() < 1e-3,
            "K={order} c0: {}",
            result.coefficients[0][0]
        );

        if order >= 2 {
            // c1 = 6
            assert!(
                (result.coefficients[1][0] - 6.0).abs() < 1e-3,
                "K={order} c1: {}",
                result.coefficients[1][0]
            );
        }
        if order >= 3 {
            // c2 = 1 (since d²/dt² (3+t)² = 2, and c2 = f''/2! = 1)
            assert!(
                (result.coefficients[2][0] - 1.0).abs() < 1e-3,
                "K={order} c2: {}",
                result.coefficients[2][0]
            );
        }
        if order >= 4 {
            // c3 = 0 (polynomial degree 2)
            assert!(
                result.coefficients[3][0].abs() < 1e-3,
                "K={order} c3: {}",
                result.coefficients[3][0]
            );
        }
        if order >= 5 {
            assert!(
                result.coefficients[4][0].abs() < 1e-3,
                "K={order} c4: {}",
                result.coefficients[4][0]
            );
        }
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_kth_k3_matches_2nd() {
    // K=3 should match taylor_forward_2nd_batch exactly
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.5_f64, 2.5];
    let (tape, _) = record(rosenbrock, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let primals = [1.5f32, 2.5];
    let seeds = [0.6f32, 0.8];

    let result_2nd = ctx
        .taylor_forward_2nd_batch(&tape_buf, &primals, &seeds, 1)
        .unwrap();

    let result_kth = ctx
        .taylor_forward_kth_batch(&tape_buf, &primals, &seeds, 1, 3)
        .unwrap();

    assert_eq!(result_kth.order, 3);
    assert!(
        (result_2nd.values[0] - result_kth.coefficients[0][0]).abs() < 1e-4,
        "c0: {} vs {}",
        result_2nd.values[0],
        result_kth.coefficients[0][0]
    );
    assert!(
        (result_2nd.c1s[0] - result_kth.coefficients[1][0]).abs() < 1e-4,
        "c1: {} vs {}",
        result_2nd.c1s[0],
        result_kth.coefficients[1][0]
    );
    assert!(
        (result_2nd.c2s[0] - result_kth.coefficients[2][0]).abs() < 1e-3,
        "c2: {} vs {}",
        result_2nd.c2s[0],
        result_kth.coefficients[2][0]
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_kth_exp_higher_order() {
    // f(x) = exp(x) at x=1, direction 1
    // c_k = exp(1) / k! for all k (since exp^(k) = exp)
    // c0 = e, c1 = e, c2 = e/2, c3 = e/6, c4 = e/24
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    fn f_exp<T: Scalar>(x: &[T]) -> T {
        x[0].exp()
    }

    let x = [1.0_f64];
    let (tape, _) = record(f_exp, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    // Also compute CPU reference
    let cpu_coeffs = echidna::stde::taylor_jet_dyn(&tape, &x, &[1.0], 5);

    let result = ctx
        .taylor_forward_kth_batch(&tape_buf, &[1.0f32], &[1.0f32], 1, 5)
        .unwrap();

    let e = std::f64::consts::E;
    let expected = [e, e, e / 2.0, e / 6.0, e / 24.0];

    for (k, exp_val) in expected.iter().enumerate() {
        let gpu_val = result.coefficients[k][0] as f64;
        let tol = 0.05 * exp_val.abs();
        assert!(
            (gpu_val - exp_val).abs() < tol.max(1e-2),
            "K=5 c{k}: gpu={gpu_val} expected={exp_val} cpu={:.6}",
            cpu_coeffs[k]
        );
    }
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_kth_unsupported_order() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.0_f64];
    let (tape, _) = record(|v: &[echidna::BReverse<f64>]| v[0] * v[0], &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let result = ctx.taylor_forward_kth_batch(&tape_buf, &[1.0f32], &[1.0f32], 1, 6);
    assert!(result.is_err());
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_taylor_kth_multi_batch() {
    // Verify deinterleaving is correct with batch_size > 1
    // f(x,y) = x² + y², directions (1,0) and (0,1)
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [3.0_f64, 4.0];
    let (tape, _) = record(polynomial, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let primals = [3.0f32, 4.0, 3.0, 4.0];
    let seeds = [1.0f32, 0.0, 0.0, 1.0];

    let result = ctx
        .taylor_forward_kth_batch(&tape_buf, &primals, &seeds, 2, 4)
        .unwrap();

    assert_eq!(result.order, 4);
    // Both batch elements: c0 = 25
    assert!((result.coefficients[0][0] - 25.0).abs() < 1e-3);
    assert!((result.coefficients[0][1] - 25.0).abs() < 1e-3);
    // Batch 0 dir (1,0): c1 = 2*3 = 6, c2 = 1
    assert!((result.coefficients[1][0] - 6.0).abs() < 1e-3);
    assert!((result.coefficients[2][0] - 1.0).abs() < 1e-3);
    // Batch 1 dir (0,1): c1 = 2*4 = 8, c2 = 1
    assert!((result.coefficients[1][1] - 8.0).abs() < 1e-3);
    assert!((result.coefficients[2][1] - 1.0).abs() < 1e-3);
    // c3 = 0 for both (polynomial degree 2)
    assert!(result.coefficients[3][0].abs() < 1e-3);
    assert!(result.coefficients[3][1].abs() < 1e-3);
}

// ══════════════════════════════════════════════
//  Section 6: WS2 — Higher-order Taylor HYPOT GPU rescale
// ══════════════════════════════════════════════
//
// Pre-WS2, GPU jet HYPOT computed `jet_mul(a,a) + jet_mul(b,b)` directly
// — fine for `a.v[0] ~ 10` but `1e20 * 1e20` overflows in f32, leaking
// Inf/NaN into v[1..K]. WS2 mirrors CPU `taylor_ops::taylor_hypot`'s
// max-rescale strategy. These tests pin both the rescale's correctness
// (baseline `hypot(3, 4)`) and the WS2 fix at extreme magnitudes
// (`hypot(1e20, 1e19)`), plus a documented-divergence ignored test
// for the singular-origin case where CPU does recursive shift-and-square
// unwinding the GPU intentionally skips.

#[cfg(feature = "stde")]
fn check_hypot_jet(ctx: &impl GpuBackend, x0: f64, y0: f64, seed: [f32; 2], label: &str) {
    use num_traits::Float as _;
    let f = |v: &[echidna::BReverse<f64>]| v[0].hypot(v[1]);
    let x = [x0, y0];
    let (tape, _) = record(f, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let gpu_result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &[x0 as f32, y0 as f32], &seed, 1)
        .unwrap();

    let seed_f64 = [seed[0] as f64, seed[1] as f64];
    let (c0, c1, c2) = echidna::stde::taylor_jet_2nd(&tape, &x, &seed_f64);

    // Finiteness is the WS2 contract: pre-WS2, extreme magnitudes
    // produced Inf/NaN in v[1..K]. Assert finiteness FIRST so a
    // regression surfaces with a clear message before tolerance noise.
    assert!(
        gpu_result.values[0].is_finite(),
        "{label} c0 not finite: {}",
        gpu_result.values[0]
    );
    assert!(
        gpu_result.c1s[0].is_finite(),
        "{label} c1 not finite: {}",
        gpu_result.c1s[0]
    );
    assert!(
        gpu_result.c2s[0].is_finite(),
        "{label} c2 not finite: {}",
        gpu_result.c2s[0]
    );

    // Parity vs CPU. Use relative tolerance per existing `check_1d`
    // pattern; ULP-absolute over-passes for low-magnitude `c2` and
    // under-passes for high-magnitude `c0` at extreme inputs.
    let tol: f64 = 1e-2;
    assert!(
        (gpu_result.values[0] as f64 - c0).abs() < tol.max(c0.abs() * 1e-4),
        "{label} c0: gpu={} cpu={}",
        gpu_result.values[0],
        c0
    );
    assert!(
        (gpu_result.c1s[0] as f64 - c1).abs() < tol.max(c1.abs() * 1e-4),
        "{label} c1: gpu={} cpu={}",
        gpu_result.c1s[0],
        c1
    );
    assert!(
        (gpu_result.c2s[0] as f64 - c2).abs() < tol.max(c2.abs() * 1e-3),
        "{label} c2: gpu={} cpu={}",
        gpu_result.c2s[0],
        c2
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws2_wgpu_hypot_baseline_normal_magnitude() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // hypot(3, 4) at direction (1, 0): c0=5, c1=0.6, c2=0.064.
    // Confirms the rescale rewrite hasn't regressed normal-magnitude
    // accuracy.
    check_hypot_jet(&ctx, 3.0, 4.0, [1.0, 0.0], "wgpu_hypot(3,4) dx");
    check_hypot_jet(&ctx, 3.0, 4.0, [0.0, 1.0], "wgpu_hypot(3,4) dy");
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws2_wgpu_hypot_extreme_magnitude_finite() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // The WS2 main case: `a.v[0] = 1e20` would overflow `a*a` in f32
    // pre-WS2, NaN-ing v[1..K]. Post-WS2 the rescale keeps everything
    // bounded.
    check_hypot_jet(&ctx, 1e20, 1e19, [1.0, 0.0], "wgpu_hypot(1e20,1e19) dx");
}

#[cfg(feature = "stde")]
fn check_hypot_jet_nonfinite_input(
    ctx: &impl GpuBackend,
    x0: f32,
    y0: f32,
    expected_primal_kind: &str, // "inf" or "nan"
    label: &str,
) {
    use num_traits::Float as _;
    // Build the tape at finite reference values; the GPU is run at
    // (x0, y0) which may be Inf/NaN. The tape itself is just `hypot(a, b)`.
    let f = |v: &[echidna::BReverse<f64>]| v[0].hypot(v[1]);
    let (tape, _) = record(f, &[1.0_f64, 1.0]);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let gpu_result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &[x0, y0], &[1.0f32, 0.0], 1)
        .unwrap();

    match expected_primal_kind {
        "inf" => assert!(
            gpu_result.values[0].is_infinite() && gpu_result.values[0] > 0.0,
            "{label} c0: expected +Inf, got {}",
            gpu_result.values[0]
        ),
        "nan" => assert!(
            gpu_result.values[0].is_nan(),
            "{label} c0: expected NaN, got {}",
            gpu_result.values[0]
        ),
        _ => panic!("unknown expected_primal_kind {expected_primal_kind}"),
    }
    // Higher-order coefficients are conventional zero on GPU at the
    // function-domain boundary (CPU diverges to NaN via `Inf*0=NaN`;
    // both are defensible at the singularity). Asserting zero pins
    // the GPU contract — anyone changing the special-case return
    // values (e.g. omitting the explicit zero loop, regressing back
    // to NaN) trips here.
    assert_eq!(
        gpu_result.c1s[0], 0.0,
        "{label} c1: expected 0.0 at boundary, got {}",
        gpu_result.c1s[0]
    );
    assert_eq!(
        gpu_result.c2s[0], 0.0,
        "{label} c2: expected 0.0 at boundary, got {}",
        gpu_result.c2s[0]
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws2_wgpu_hypot_nan_input_propagates_not_swallowed() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // Pins IEEE `hypot(NaN, x) = NaN` for the jet path. Without the
    // explicit NaN guard added in the WS2 review-fix, `fmax(NaN, 0) = 0`
    // would route this to the `h == 0` branch and silently return zero
    // — corrupting any downstream computation that depended on NaN
    // propagation.
    check_hypot_jet_nonfinite_input(&ctx, f32::NAN, 0.0, "nan", "wgpu_hypot(NaN, 0) jet");
    check_hypot_jet_nonfinite_input(&ctx, 0.0, f32::NAN, "nan", "wgpu_hypot(0, NaN) jet");
    check_hypot_jet_nonfinite_input(&ctx, f32::NAN, 1.0, "nan", "wgpu_hypot(NaN, 1) jet");
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws2_cuda_hypot_nan_input_propagates_not_swallowed() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet_nonfinite_input(&ctx, f32::NAN, 0.0, "nan", "cuda_hypot(NaN, 0) jet");
    check_hypot_jet_nonfinite_input(&ctx, 0.0, f32::NAN, "nan", "cuda_hypot(0, NaN) jet");
    check_hypot_jet_nonfinite_input(&ctx, f32::NAN, 1.0, "nan", "cuda_hypot(NaN, 1) jet");
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws9_wgpu_hypot_zero_origin_with_nonzero_seed_matches_cpu() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // Post-WS9: `hypot(0, 0)` with a non-zero seed now routes
    // through the WGSL `h == 0` shift-and-square unroll, matching
    // CPU's `|t|·hypot(a/t, b/t)` expansion instead of returning
    // the zero jet. The unroll is one level deep (the CPU recursion
    // is at most one level since the entry check guarantees
    // `scale > 0` in the recursive call).
    check_hypot_jet(&ctx, 0.0, 0.0, [1.0, 0.0], "wgpu_hypot(0,0) dx");
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws2_cuda_hypot_baseline_normal_magnitude() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet(&ctx, 3.0, 4.0, [1.0, 0.0], "cuda_hypot(3,4) dx");
    check_hypot_jet(&ctx, 3.0, 4.0, [0.0, 1.0], "cuda_hypot(3,4) dy");
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws2_cuda_hypot_extreme_magnitude_finite() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet(&ctx, 1e20, 1e19, [1.0, 0.0], "cuda_hypot(1e20,1e19) dx");
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws9_cuda_hypot_zero_origin_with_nonzero_seed_matches_cpu() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet(&ctx, 0.0, 0.0, [1.0, 0.0], "cuda_hypot(0,0) dx");
}

// ══════════════════════════════════════════════════════════════
//  WS9 — GPU parity at function-domain-boundary edge cases
// ══════════════════════════════════════════════════════════════
//
// Three cases where pre-WS9 GPU silently diverged from CPU at the
// boundary of `hypot`'s function domain — CPU producing the
// mathematically-informed "singular derivative" output (Inf or NaN
// higher-order coefficients) while GPU emitted a zero jet. The
// pre-WS9 divergences were pinned by `#[ignore]`-d tests documented
// as "ws2" divergences; WS9 closes the gap and the tests now pin
// parity under their `ws9_` rename above.
//
// The remaining two divergences — Inf-finite inputs and
// deeper-order-zero inputs — are pinned by the tests below, using
// a non-finite-tolerant parity helper since `check_hypot_jet`'s
// `(gpu - cpu).abs() < tol` fails when CPU or GPU produces Inf/NaN.

#[cfg(feature = "stde")]
#[derive(Copy, Clone)]
enum CoeffClass {
    Zero,
    Inf,
    NaN,
}

#[cfg(feature = "stde")]
#[allow(clippy::too_many_arguments)] // test helper exercising a specific GPU kernel signature
fn check_hypot_jet_non_finite_higher(
    ctx: &impl GpuBackend,
    x0: f64,
    y0: f64,
    seed: [f32; 2],
    primal_class: CoeffClass,
    c1_class: CoeffClass,
    c2_class: CoeffClass,
    label: &str,
) {
    use num_traits::Float as _;
    let f = |v: &[echidna::BReverse<f64>]| v[0].hypot(v[1]);
    let x = [x0, y0];
    let (tape, _) = record(f, &x);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let tape_buf = ctx.upload_tape(&gpu_data);

    let gpu_result = ctx
        .taylor_forward_2nd_batch(&tape_buf, &[x0 as f32, y0 as f32], &seed, 1)
        .unwrap();

    let check = |v: f32, cls: CoeffClass, slot: &str| match cls {
        CoeffClass::Zero => {
            assert!(v == 0.0, "{label} {slot}: expected 0, got {v}")
        }
        CoeffClass::Inf => {
            assert!(v.is_infinite(), "{label} {slot}: expected Inf, got {v}")
        }
        CoeffClass::NaN => {
            assert!(v.is_nan(), "{label} {slot}: expected NaN, got {v}")
        }
    };
    check(gpu_result.values[0], primal_class, "c0");
    check(gpu_result.c1s[0], c1_class, "c1");
    check(gpu_result.c2s[0], c2_class, "c2");
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws9_wgpu_hypot_inf_finite_propagates_nan() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // `hypot(Inf, 1.0)` and `hypot(1.0, Inf)` with non-zero seed.
    // Post-WS9: primal = Inf (IEEE `hypot(Inf, anything) = Inf`),
    // higher-order = NaN (CPU rescale path `Inf * 0 = NaN`).
    // Pre-WS9: GPU emitted zeros for higher-order. Both operand
    // orderings are covered to exercise the symmetric
    // `aa == inf || bb == inf` entry check — regressions that
    // collapsed it to a single-operand check would fail one.
    check_hypot_jet_non_finite_higher(
        &ctx,
        f32::INFINITY as f64,
        1.0,
        [1.0, 0.0],
        CoeffClass::Inf,
        CoeffClass::NaN,
        CoeffClass::NaN,
        "wgpu_hypot(Inf, 1)",
    );
    check_hypot_jet_non_finite_higher(
        &ctx,
        1.0,
        f32::INFINITY as f64,
        [1.0, 0.0],
        CoeffClass::Inf,
        CoeffClass::NaN,
        CoeffClass::NaN,
        "wgpu_hypot(1, Inf)",
    );
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws9_cuda_hypot_inf_finite_propagates_nan() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet_non_finite_higher(
        &ctx,
        f32::INFINITY as f64,
        1.0,
        [1.0, 0.0],
        CoeffClass::Inf,
        CoeffClass::NaN,
        CoeffClass::NaN,
        "cuda_hypot(Inf, 1)",
    );
    check_hypot_jet_non_finite_higher(
        &ctx,
        1.0,
        f32::INFINITY as f64,
        [1.0, 0.0],
        CoeffClass::Inf,
        CoeffClass::NaN,
        CoeffClass::NaN,
        "cuda_hypot(1, Inf)",
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn ws9_wgpu_hypot_deeper_order_zero_returns_inf_higher() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };
    // `hypot(0, 0)` with seed [0, 0]: inside the tape, both
    // primal and first-order tangents are zero for hypot's inputs,
    // so the GPU h==0 branch fires and the shift-and-square inner
    // check `a.v[1] != 0 || b.v[1] != 0` fails (the tangents are
    // also zero). Falls into the else-branch → 0 primal + Inf
    // higher, matching CPU's `taylor_sqrt` convention at a true
    // zero leading.
    check_hypot_jet_non_finite_higher(
        &ctx,
        0.0,
        0.0,
        [0.0, 0.0],
        CoeffClass::Zero,
        CoeffClass::Inf,
        CoeffClass::Inf,
        "wgpu_hypot_deeper_zero",
    );
}

#[cfg(all(feature = "gpu-cuda", feature = "stde"))]
#[test]
fn ws9_cuda_hypot_deeper_order_zero_returns_inf_higher() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };
    check_hypot_jet_non_finite_higher(
        &ctx,
        0.0,
        0.0,
        [0.0, 0.0],
        CoeffClass::Zero,
        CoeffClass::Inf,
        CoeffClass::Inf,
        "cuda_hypot_deeper_zero",
    );
}

#[cfg(all(feature = "gpu-wgpu", feature = "stde"))]
#[test]
fn gpu_trig_taylor_2nd() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x = [1.0_f64, 0.5];
    let (tape, _) = record(trig_func, &x);
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

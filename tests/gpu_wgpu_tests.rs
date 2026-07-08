#![cfg(feature = "gpu-wgpu")]

use echidna::gpu::{GpuBackend, GpuTapeData, WgpuContext};
use echidna::{record, Scalar};

/// Try to acquire a GPU. If none available, print a warning and return None.
fn gpu_context() -> Option<WgpuContext> {
    match WgpuContext::new() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("WARNING: No GPU adapter found — skipping GPU test");
            None
        }
    }
}

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

fn trig_func<T: Scalar>(x: &[T]) -> T {
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    // sin(x0) * cos(x1) + exp(x0 * x1 / 2)
    x[0].sin() * x[1].cos() + (x[0] * x[1] / two).exp()
}

/// Evaluate tape on CPU at f64 precision for reference.
fn cpu_forward_f64(tape: &mut echidna::BytecodeTape<f64>, points: &[Vec<f64>]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            tape.forward(p);
            tape.output_value()
        })
        .collect()
}

/// Check relative error, with absolute fallback for near-zero values.
fn approx_eq(gpu: f32, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (gpu as f64 - cpu).abs();
    if cpu.abs() < abs_tol {
        diff < abs_tol
    } else {
        diff / cpu.abs() < rel_tol
    }
}

#[test]
fn forward_batch_rosenbrock() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // Record tape at f64
    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    // Generate 100 test points
    let points: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let t = i as f64 / 99.0;
            vec![-2.0 + 4.0 * t, -1.0 + 3.0 * t]
        })
        .collect();

    // CPU reference (f64)
    let cpu_results = cpu_forward_f64(&mut tape, &points);

    // GPU (f32)
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let gpu_results = ctx.forward_batch(&gpu_tape, &flat_inputs, 100).unwrap();

    assert_eq!(gpu_results.len(), 100);
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            approx_eq(*gpu, *cpu, 1e-5, 1e-6),
            "point {}: gpu={}, cpu={}, rel_err={}",
            i,
            gpu,
            cpu,
            (*gpu as f64 - cpu).abs() / cpu.abs().max(1e-12)
        );
    }
}

#[test]
fn forward_batch_trig() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [0.5_f64, 0.7];
    let (mut tape, _) = record(trig_func, &x0);

    let points: Vec<Vec<f64>> = (0..50)
        .map(|i| {
            let t = i as f64 / 49.0;
            vec![-1.0 + 2.0 * t, -1.0 + 2.0 * t]
        })
        .collect();

    let cpu_results = cpu_forward_f64(&mut tape, &points);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let gpu_results = ctx.forward_batch(&gpu_tape, &flat_inputs, 50).unwrap();

    assert_eq!(gpu_results.len(), 50);
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            approx_eq(*gpu, *cpu, 1e-5, 1e-6),
            "point {}: gpu={}, cpu={}",
            i,
            gpu,
            cpu
        );
    }
}

#[test]
fn forward_batch_single_point() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [2.0_f64, 3.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    tape.forward(&[2.0, 3.0]);
    let cpu_val = tape.output_value();

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let inputs = [2.0_f32, 3.0];
    let gpu_results = ctx.forward_batch(&gpu_tape, &inputs, 1).unwrap();

    assert_eq!(gpu_results.len(), 1);
    assert!(
        approx_eq(gpu_results[0], cpu_val, 1e-5, 1e-6),
        "gpu={}, cpu={}",
        gpu_results[0],
        cpu_val
    );
}

#[test]
fn forward_batch_large() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 1.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let batch_size = 10_000u32;
    let points: Vec<Vec<f64>> = (0..batch_size)
        .map(|i| {
            let t = i as f64 / (batch_size - 1) as f64;
            vec![-5.0 + 10.0 * t, -5.0 + 10.0 * t]
        })
        .collect();

    let cpu_results = cpu_forward_f64(&mut tape, &points);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let gpu_results = ctx
        .forward_batch(&gpu_tape, &flat_inputs, batch_size)
        .unwrap();

    assert_eq!(gpu_results.len(), batch_size as usize);

    let mut max_rel_err = 0.0_f64;
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        let rel_err = if cpu.abs() > 1e-6 {
            (*gpu as f64 - cpu).abs() / cpu.abs()
        } else {
            (*gpu as f64 - cpu).abs()
        };
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            approx_eq(*gpu, *cpu, 1e-4, 1e-3),
            "point {}: gpu={}, cpu={}, rel_err={}",
            i,
            gpu,
            cpu,
            rel_err
        );
    }
    eprintln!(
        "forward_batch_large: max relative error = {:.2e}",
        max_rel_err
    );
}

#[test]
fn forward_batch_f32_native() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    // Record tape directly at f32
    let x0 = [1.0_f32, 2.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let points: Vec<f32> = vec![1.0, 2.0, 0.5, 0.5, 2.0, 4.0];
    let gpu_results = ctx.forward_batch(&gpu_tape, &points, 3).unwrap();

    // Compare against CPU f32
    for (i, chunk) in points.chunks(2).enumerate() {
        tape.forward(chunk);
        let cpu_val = tape.output_value();
        assert!(
            (gpu_results[i] - cpu_val).abs() < 1e-5,
            "point {}: gpu={}, cpu={}",
            i,
            gpu_results[i],
            cpu_val
        );
    }
}

// ── Gradient tests (forward + reverse) ──

#[test]
fn gradient_batch_rosenbrock() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let points: Vec<Vec<f64>> = (0..50)
        .map(|i| {
            let t = i as f64 / 49.0;
            vec![-2.0 + 4.0 * t, -1.0 + 3.0 * t]
        })
        .collect();

    // CPU reference gradients (f64)
    let cpu_grads: Vec<Vec<f64>> = points.iter().map(|p| tape.gradient(p)).collect();

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let (_, gpu_grads) = ctx.gradient_batch(&gpu_tape, &flat_inputs, 50).unwrap();

    let num_inputs = tape.num_inputs();
    assert_eq!(gpu_grads.len(), 50 * num_inputs);

    for (i, cpu_grad) in cpu_grads.iter().enumerate() {
        for (j, &cpu_g) in cpu_grad.iter().enumerate() {
            let gpu_g = gpu_grads[i * num_inputs + j];
            assert!(
                approx_eq(gpu_g, cpu_g, 1e-4, 1e-3),
                "point {}, grad[{}]: gpu={}, cpu={}, rel_err={}",
                i,
                j,
                gpu_g,
                cpu_g,
                (gpu_g as f64 - cpu_g).abs() / cpu_g.abs().max(1e-12)
            );
        }
    }
}

#[test]
fn gradient_batch_trig() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [0.5_f64, 0.7];
    let (mut tape, _) = record(trig_func, &x0);

    let points: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let t = i as f64 / 19.0;
            vec![-1.0 + 2.0 * t, -1.0 + 2.0 * t]
        })
        .collect();

    let cpu_grads: Vec<Vec<f64>> = points.iter().map(|p| tape.gradient(p)).collect();

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let (_, gpu_grads) = ctx.gradient_batch(&gpu_tape, &flat_inputs, 20).unwrap();

    let num_inputs = tape.num_inputs();
    for (i, cpu_grad) in cpu_grads.iter().enumerate() {
        for (j, &cpu_g) in cpu_grad.iter().enumerate() {
            let gpu_g = gpu_grads[i * num_inputs + j];
            assert!(
                approx_eq(gpu_g, cpu_g, 1e-4, 1e-3),
                "point {}, grad[{}]: gpu={}, cpu={}",
                i,
                j,
                gpu_g,
                cpu_g
            );
        }
    }
}

#[test]
fn gradient_batch_single_point() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [2.0_f64, 3.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let cpu_grad = tape.gradient(&[2.0, 3.0]);

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let (_, gpu_grads) = ctx.gradient_batch(&gpu_tape, &[2.0_f32, 3.0], 1).unwrap();

    assert_eq!(gpu_grads.len(), 2);
    for (j, &cpu_g) in cpu_grad.iter().enumerate() {
        assert!(
            approx_eq(gpu_grads[j], cpu_g, 1e-4, 1e-3),
            "grad[{}]: gpu={}, cpu={}",
            j,
            gpu_grads[j],
            cpu_g
        );
    }
}

// ── Sparse Jacobian tests ──

fn multi_output_func<T: Scalar>(x: &[T]) -> Vec<T> {
    // f(x0, x1) = [x0*x1 + sin(x0), x0^2 + x1^2]
    vec![x[0] * x[1] + x[0].sin(), x[0] * x[0] + x[1] * x[1]]
}

#[test]
fn sparse_jacobian_multi_output() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f32, 2.0];
    let (mut tape, _) = echidna::record_multi(multi_output_func, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let x = [1.5_f32, 0.5];
    let (vals, pattern, jac_vals) = ctx.sparse_jacobian(&gpu_tape, &mut tape, &x).unwrap();

    // CPU reference: full Jacobian via forward mode
    tape.forward(&x);
    let cpu_jac = tape.jacobian_forward(&x);

    // Check output values
    tape.forward(&x);
    let cpu_vals = tape.output_values();
    for (i, (&g, &c)) in vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!((g - c).abs() < 1e-5, "output[{}]: gpu={}, cpu={}", i, g, c);
    }

    // Check Jacobian entries against CPU reference
    for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
        let cpu_val = cpu_jac[row as usize][col as usize];
        assert!(
            (jac_vals[k] - cpu_val).abs() < 1e-4,
            "J[{},{}]: gpu={}, cpu={}",
            row,
            col,
            jac_vals[k],
            cpu_val
        );
    }
}

// ── HVP tests ──

#[test]
fn hvp_batch_rosenbrock() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f32, 2.0];
    let (tape, _) = record(rosenbrock, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let x = [1.5_f32, 0.5];

    // Test with direction v = [1, 0] → first column of Hessian
    let tangent_dirs = [1.0_f32, 0.0];
    let (grads, hvps) = ctx.hvp_batch(&gpu_tape, &x, &tangent_dirs, 1).unwrap();

    // CPU reference
    let (cpu_grad, cpu_hvp) = tape.hvp(&x, &[1.0, 0.0]);

    assert_eq!(grads.len(), 2);
    assert_eq!(hvps.len(), 2);

    for j in 0..2 {
        assert!(
            approx_eq(grads[j], cpu_grad[j] as f64, 1e-4, 1e-3),
            "grad[{}]: gpu={}, cpu={}",
            j,
            grads[j],
            cpu_grad[j]
        );
        assert!(
            approx_eq(hvps[j], cpu_hvp[j] as f64, 1e-4, 1e-3),
            "hvp[{}]: gpu={}, cpu={}",
            j,
            hvps[j],
            cpu_hvp[j]
        );
    }
}

#[test]
fn hvp_batch_trig() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [0.5_f32, 0.7];
    let (tape, _) = record(trig_func, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let x = [0.3_f32, 0.8];

    // Two directions: [1,0] and [0,1] → full Hessian
    let tangent_dirs = [1.0_f32, 0.0, 0.0, 1.0];
    let (_grads, hvps) = ctx.hvp_batch(&gpu_tape, &x, &tangent_dirs, 2).unwrap();

    let (_, cpu_hvp0) = tape.hvp(&x, &[1.0, 0.0]);
    let (_, cpu_hvp1) = tape.hvp(&x, &[0.0, 1.0]);

    // Check both HVP results
    for j in 0..2 {
        assert!(
            approx_eq(hvps[j], cpu_hvp0[j] as f64, 1e-3, 1e-3),
            "hvp0[{}]: gpu={}, cpu={}",
            j,
            hvps[j],
            cpu_hvp0[j]
        );
        assert!(
            approx_eq(hvps[2 + j], cpu_hvp1[j] as f64, 1e-3, 1e-3),
            "hvp1[{}]: gpu={}, cpu={}",
            j,
            hvps[2 + j],
            cpu_hvp1[j]
        );
    }
}

// ── Sparse Hessian tests ──

#[test]
fn sparse_hessian_rosenbrock() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f32, 2.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let x = [1.5_f32, 0.5];
    let (gpu_val, gpu_grad, gpu_pattern, gpu_hess) =
        ctx.sparse_hessian(&gpu_tape, &mut tape, &x).unwrap();

    // CPU reference
    let (cpu_val, cpu_grad, cpu_pattern, cpu_hess) = tape.sparse_hessian(&x);

    // Check value
    assert!(
        approx_eq(gpu_val, cpu_val as f64, 1e-4, 1e-3),
        "value: gpu={}, cpu={}",
        gpu_val,
        cpu_val
    );

    // Check gradient
    for j in 0..2 {
        assert!(
            approx_eq(gpu_grad[j], cpu_grad[j] as f64, 1e-4, 1e-3),
            "grad[{}]: gpu={}, cpu={}",
            j,
            gpu_grad[j],
            cpu_grad[j]
        );
    }

    // Check Hessian entries
    assert_eq!(gpu_pattern.nnz(), cpu_pattern.nnz());
    for k in 0..gpu_hess.len() {
        assert!(
            approx_eq(gpu_hess[k], cpu_hess[k] as f64, 1e-3, 1e-2),
            "hess[{}] (row={}, col={}): gpu={}, cpu={}",
            k,
            gpu_pattern.rows[k],
            gpu_pattern.cols[k],
            gpu_hess[k],
            cpu_hess[k]
        );
    }
}

#[test]
fn sparse_hessian_trig() {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [0.5_f32, 0.7];
    let (mut tape, _) = record(trig_func, &x0);

    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let x = [0.3_f32, 0.8];
    let (gpu_val, gpu_grad, gpu_pattern, gpu_hess) =
        ctx.sparse_hessian(&gpu_tape, &mut tape, &x).unwrap();

    // CPU reference
    let (cpu_val, cpu_grad, cpu_pattern, cpu_hess) = tape.sparse_hessian(&x);

    assert!(
        approx_eq(gpu_val, cpu_val as f64, 1e-4, 1e-3),
        "value: gpu={}, cpu={}",
        gpu_val,
        cpu_val
    );

    for j in 0..2 {
        assert!(
            approx_eq(gpu_grad[j], cpu_grad[j] as f64, 1e-4, 1e-3),
            "grad[{}]: gpu={}, cpu={}",
            j,
            gpu_grad[j],
            cpu_grad[j]
        );
    }

    assert_eq!(gpu_pattern.nnz(), cpu_pattern.nnz());
    for k in 0..gpu_hess.len() {
        assert!(
            approx_eq(gpu_hess[k], cpu_hess[k] as f64, 1e-3, 1e-2),
            "hess[{}] (row={}, col={}): gpu={}, cpu={}",
            k,
            gpu_pattern.rows[k],
            gpu_pattern.cols[k],
            gpu_hess[k],
            cpu_hess[k]
        );
    }
}

#[test]
fn custom_ops_rejected() {
    use echidna::bytecode_tape::CustomOp;
    use std::sync::Arc;

    struct Dummy;
    impl CustomOp<f32> for Dummy {
        fn eval(&self, a: f32, _b: f32) -> f32 {
            a
        }
        fn partials(&self, _a: f32, _b: f32, _r: f32) -> (f32, f32) {
            (1.0, 0.0)
        }
    }

    let mut tape = echidna::BytecodeTape::<f32>::new();
    let x = tape.new_input(1.0);
    let handle = tape.register_custom(Arc::new(Dummy));
    let _y = tape.push_custom_unary(x, handle, 1.0);
    tape.set_output(x);

    let result = GpuTapeData::from_tape(&tape);
    assert!(
        matches!(result, Err(echidna::gpu::GpuError::CustomOpsNotSupported)),
        "expected CustomOpsNotSupported error"
    );
}

#[test]
fn wgpu_implements_gpu_backend() {
    fn assert_backend<B: GpuBackend>() {}
    assert_backend::<WgpuContext>();
}

// ══════════════════════════════════════════════
//  Per-opcode GPU HVP parity tests
// ══════════════════════════════════════════════
//
// Tests CPU vs GPU Hessian-vector products for every opcode with a
// nontrivial second derivative. Catches CPU-GPU formula mismatches
// like the cbrt HVP bug (finding #1).

mod hvp_opcode_parity {
    use super::*;
    use echidna::gpu::GpuTapeData;
    use num_traits::Float;

    /// Test HVP parity for a single-input scalar function.
    fn check_hvp_parity(
        ctx: &WgpuContext,
        f: impl FnOnce(&[echidna::BReverse<f32>]) -> echidna::BReverse<f32>,
        x_record: f32,
        x_eval: f32,
        label: &str,
    ) {
        let (tape, _) = record(f, &[x_record]);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        let x = [x_eval];
        let v = [1.0_f32]; // direction = 1 → HVP = f''(x)

        let (gpu_grad, gpu_hvp) = ctx.hvp_batch(&gpu_tape, &x, &v, 1).unwrap();
        let (cpu_grad, cpu_hvp) = tape.hvp(&x, &[1.0]);

        assert!(
            approx_eq(gpu_grad[0], cpu_grad[0] as f64, 1e-3, 1e-3),
            "{label} grad: gpu={}, cpu={}",
            gpu_grad[0],
            cpu_grad[0]
        );
        assert!(
            approx_eq(gpu_hvp[0], cpu_hvp[0] as f64, 1e-3, 1e-3),
            "{label} hvp: gpu={}, cpu={}",
            gpu_hvp[0],
            cpu_hvp[0]
        );
    }

    #[test]
    fn hvp_parity_exp() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].exp(), 0.5, 1.0, "exp");
    }

    #[test]
    fn hvp_parity_log() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].ln(), 1.0, 2.0, "log");
    }

    #[test]
    fn hvp_parity_sqrt() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].sqrt(), 1.0, 4.0, "sqrt");
    }

    #[test]
    fn hvp_parity_cbrt() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].cbrt(), 1.0, 8.0, "cbrt");
    }

    #[test]
    fn hvp_parity_recip() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].recip(), 1.0, 2.0, "recip");
    }

    #[test]
    fn hvp_parity_sin() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].sin(), 0.5, 1.0, "sin");
    }

    #[test]
    fn hvp_parity_cos() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].cos(), 0.5, 1.0, "cos");
    }

    #[test]
    fn hvp_parity_tan() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].tan(), 0.5, 0.5, "tan");
    }

    #[test]
    fn hvp_parity_asin() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].asin(), 0.3, 0.5, "asin");
    }

    #[test]
    fn hvp_parity_acos() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].acos(), 0.3, 0.5, "acos");
    }

    #[test]
    fn hvp_parity_atan() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].atan(), 0.5, 1.0, "atan");
    }

    #[test]
    fn hvp_parity_sinh() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].sinh(), 0.5, 1.0, "sinh");
    }

    #[test]
    fn hvp_parity_cosh() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].cosh(), 0.5, 1.0, "cosh");
    }

    #[test]
    fn hvp_parity_tanh() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        check_hvp_parity(&ctx, |x| x[0].tanh(), 0.5, 1.0, "tanh");
    }

    #[test]
    fn hvp_parity_powf() {
        let ctx = match gpu_context() {
            Some(c) => c,
            None => return,
        };
        // x^2.5 — nontrivial second derivative
        let two_five = 2.5_f32;
        check_hvp_parity(
            &ctx,
            move |x| x[0].powf(echidna::BReverse::constant(two_five)),
            1.0,
            2.0,
            "powf",
        );
    }
}

#[cfg(feature = "gpu-wgpu")]
mod hvp_edge_parity {
    use super::*;
    use echidna::gpu::GpuTapeData;
    use echidna::record;

    fn approx_eq(a: f32, b: f64, rel: f64, abs: f64) -> bool {
        let a = a as f64;
        (a - b).abs() <= abs || (a - b).abs() <= rel * b.abs()
    }

    /// HYPOT at magnitudes where a naive sqrt(a*a + b*b) primal overflows
    /// f32 (a*a = 4e38 > f32::MAX) while the true hypot ≈ 2.2e19 is
    /// representable: the HVP kernel's forward phase must stay finite and
    /// its gradient must match the CPU reference.
    #[test]
    fn hvp_parity_hypot_large_magnitude() {
        let ctx = match super::gpu_context() {
            Some(c) => c,
            None => return,
        };
        use num_traits::Float as _;
        let f = |v: &[echidna::BReverse<f32>]| v[0].hypot(v[1]);
        let (tape, _) = record(f, &[3.0_f32, 4.0]);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);
        let x = [2e19_f32, 1e19];
        let v = [1.0_f32, 0.0];
        let (gpu_grad, gpu_hvp) = ctx.hvp_batch(&gpu_tape, &x, &v, 1).unwrap();
        let (cpu_grad, cpu_hvp) = tape.hvp(&x, &[1.0, 0.0]);
        for i in 0..2 {
            assert!(
                approx_eq(gpu_grad[i], cpu_grad[i] as f64, 1e-3, 1e-3),
                "hypot-large grad[{i}]: gpu={}, cpu={}",
                gpu_grad[i],
                cpu_grad[i]
            );
            assert!(
                approx_eq(gpu_hvp[i], cpu_hvp[i] as f64, 1e-3, 1e-3),
                "hypot-large hvp[{i}]: gpu={}, cpu={}",
                gpu_hvp[i],
                cpu_hvp[i]
            );
        }
    }

    /// MAX/MIN with a NaN second operand: the adjoint must route to the
    /// first operand (IEEE maxNum drops the NaN). Pins the bit-pattern NaN
    /// test in the HVP kernel's reverse phase — a bare `b != b` can fold to
    /// false under fast-math backends and route the adjoint to the wrong
    /// operand.
    #[test]
    fn hvp_max_min_nan_second_operand_routes_to_first() {
        let ctx = match super::gpu_context() {
            Some(c) => c,
            None => return,
        };
        use num_traits::Float as _;
        for (name, f) in [
            (
                "max",
                (|v: &[echidna::BReverse<f32>]| v[0].max(v[1]))
                    as fn(&[echidna::BReverse<f32>]) -> echidna::BReverse<f32>,
            ),
            ("min", |v: &[echidna::BReverse<f32>]| v[0].min(v[1])),
        ] {
            let (tape, _) = record(f, &[1.0_f32, 2.0]);
            let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let x = [1.0_f32, f32::NAN];
            let v = [1.0_f32, 0.0];
            let (gpu_grad, _) = ctx.hvp_batch(&gpu_tape, &x, &v, 1).unwrap();
            assert_eq!(
                gpu_grad[0], 1.0,
                "{name}(1, NaN): adjoint must route to the first operand"
            );
            assert_eq!(
                gpu_grad[1], 0.0,
                "{name}(1, NaN): no adjoint may reach the NaN operand"
            );
        }
    }

    /// Singular primal with a structurally-zero direction component: the
    /// zero lane's tangent and HVP contributions stay exactly zero (the
    /// CPU chain-rule convention) instead of NaN from 0·Inf / 0/0, while
    /// the live lane and the first-order multipliers are untouched.
    #[test]
    fn hvp_singular_primal_zero_direction_lane() {
        let ctx = match super::gpu_context() {
            Some(c) => c,
            None => return,
        };
        use num_traits::Float as _;
        // sqrt(x)·y + ln(y) at x = 0: the sqrt node's forward tangent is
        // structurally zero for direction (0, 1) — without the guard it is
        // 0/0 = NaN and poisons the y-lane HVP through the product rule.
        let f = |v: &[echidna::BReverse<f32>]| v[0].sqrt() * v[1] + v[1].ln();
        let (tape, _) = record(f, &[1.0_f32, 2.0]);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);
        let x = [0.0_f32, 2.0];
        let v = [0.0_f32, 1.0];
        let (gpu_grad, gpu_hvp) = ctx.hvp_batch(&gpu_tape, &x, &v, 1).unwrap();
        let (cpu_grad, cpu_hvp) = tape.hvp(&x, &[0.0, 1.0]);

        // Live lane: grad = sqrt(0) + 1/y = 0.5; hvp = -1/y² = -0.25.
        assert!(
            approx_eq(gpu_grad[1], cpu_grad[1] as f64, 1e-4, 1e-6)
                && (cpu_grad[1] - 0.5).abs() < 1e-6,
            "live-lane gradient: gpu={}, cpu={}",
            gpu_grad[1],
            cpu_grad[1]
        );
        assert!(
            approx_eq(gpu_hvp[1], cpu_hvp[1] as f64, 1e-4, 1e-6)
                && (cpu_hvp[1] - (-0.25)).abs() < 1e-6,
            "live-lane hvp must stay finite (guarded sqrt tangent): gpu={}, cpu={}",
            gpu_hvp[1],
            cpu_hvp[1]
        );
        // Singular lane: the first-order multiplier is genuinely +Inf
        // (y·(1/(2·sqrt(0)))), on both backends.
        assert!(
            gpu_grad[0].is_infinite() && cpu_grad[0].is_infinite(),
            "singular-lane gradient stays +Inf on both: gpu={}, cpu={}",
            gpu_grad[0],
            cpu_grad[0]
        );
        // The singular lane's HVP is genuinely non-finite (the second
        // derivative of sqrt at 0 is unbounded and the direction moves its
        // adjoint). The two backends currently reach different non-finite
        // kinds — GPU +Inf via its explicit second-derivative arm, CPU NaN
        // via the Dual division product rule — so pin non-finiteness, not
        // the kind. Unifying reverse-accumulation semantics at singular
        // points is a separate convention decision.
        assert!(
            !gpu_hvp[0].is_finite() && !cpu_hvp[0].is_finite(),
            "singular-lane hvp must be non-finite on both: gpu={}, cpu={}",
            gpu_hvp[0],
            cpu_hvp[0]
        );
    }

    /// Overflowed derivative with a constant seed lane through the JVP
    /// kernel: exp(100) is Inf in f32, so the exp node's tangent under the
    /// y-column seed (x component zero) is Inf·0 — the uniform unary guard
    /// keeps it exactly 0, so the ∂f/∂y Jacobian entry stays finite instead
    /// of being poisoned to NaN.
    #[test]
    fn jvp_overflowed_derivative_zero_seed_lane() {
        let ctx = match super::gpu_context() {
            Some(c) => c,
            None => return,
        };
        use num_traits::Float as _;
        let f = |v: &[echidna::BReverse<f32>]| v[0].exp() + v[1].ln();
        let (mut tape, _) = record(f, &[1.0_f32, 2.0]);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);
        let (_vals, _pat, jac) = ctx
            .sparse_jacobian(&gpu_tape, &mut tape, &[100.0, 2.0])
            .unwrap();
        assert!(
            jac.iter().any(|v| (v - 0.5).abs() < 1e-4),
            "∂f/∂y = 1/y = 0.5 must survive the overflowed exp lane: {jac:?}"
        );
        assert!(
            !jac.iter().any(|v| v.is_nan()),
            "no Jacobian entry may be NaN (∂f/∂x is a legitimate +Inf): {jac:?}"
        );
    }
}

/// Degenerate batched inputs must surface as recoverable errors, not
/// zero-sized-buffer panics deep inside the backend.
#[cfg(feature = "gpu-wgpu")]
mod degenerate_inputs {
    use super::*;
    use echidna::gpu::GpuTapeData;
    use echidna::record;

    #[test]
    fn zero_batch_and_zero_input_tapes_error_cleanly() {
        let ctx = match super::gpu_context() {
            Some(c) => c,
            None => return,
        };
        // Normal tape, zero batch size.
        let (tape, _) = record(|x: &[echidna::BReverse<f32>]| x[0] * x[0], &[2.0_f32]);
        let gd = GpuTapeData::from_tape(&tape).unwrap();
        let gt = ctx.upload_tape(&gd);
        assert!(ctx.forward_batch(&gt, &[], 0).is_err());
        assert!(ctx.gradient_batch(&gt, &[], 0).is_err());
        assert!(ctx.hvp_batch(&gt, &[2.0], &[], 0).is_err());

        // Zero-input (constant-function) tape with a non-zero batch.
        let (ctape, _) = record(
            |_: &[echidna::BReverse<f32>]| echidna::BReverse::constant(3.0),
            &[],
        );
        let cgd = GpuTapeData::from_tape(&ctape).unwrap();
        let cgt = ctx.upload_tape(&cgd);
        assert!(ctx.forward_batch(&cgt, &[], 4).is_err());
        assert!(ctx.gradient_batch(&cgt, &[], 4).is_err());

        // STDE entry point with empty x.
        #[cfg(feature = "stde")]
        assert!(echidna::gpu::stde_gpu::hessian_diagonal_gpu(&ctx, &gt, &[]).is_err());

        // Taylor batch entry points share the exposure: zero batch on a
        // normal tape, and a non-zero batch on a zero-input tape.
        #[cfg(feature = "stde")]
        {
            use echidna::gpu::GpuBackend as _;
            assert!(ctx.taylor_forward_2nd_batch(&gt, &[], &[], 0).is_err());
            assert!(ctx.taylor_forward_2nd_batch(&cgt, &[], &[], 4).is_err());
        }
    }
}

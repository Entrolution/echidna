#![cfg(feature = "gpu-wgpu")]

use echidna::gpu::{GpuTapeData, WgpuContext};
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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| trig_func(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| trig_func(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = echidna::record_multi(|v| multi_output_func(v), &x0);

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
    let (tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (tape, _) = record(|v| trig_func(v), &x0);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (mut tape, _) = record(|v| trig_func(v), &x0);

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

#![cfg(feature = "gpu-cuda")]

use echidna::gpu::{CudaContext, GpuBackend, GpuTapeData};
use echidna::{record, Scalar};

/// Try to acquire a CUDA GPU. If none available, print a warning and return None.
fn cuda_context() -> Option<CudaContext> {
    match CudaContext::new() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("WARNING: No CUDA device found — skipping CUDA test");
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
    x[0].sin() * x[1].cos() + (x[0] * x[1] / two).exp()
}

fn approx_eq_f32(gpu: f32, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (gpu as f64 - cpu).abs();
    if cpu.abs() < abs_tol {
        diff < abs_tol
    } else {
        diff / cpu.abs() < rel_tol
    }
}

fn approx_eq_f64(gpu: f64, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (gpu - cpu).abs();
    if cpu.abs() < abs_tol {
        diff < abs_tol
    } else {
        diff / cpu.abs() < rel_tol
    }
}

// ── f32 tests ──

#[test]
fn forward_batch_rosenbrock_f32() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

    let points: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let t = i as f64 / 99.0;
            vec![-2.0 + 4.0 * t, -1.0 + 3.0 * t]
        })
        .collect();

    let cpu_results: Vec<f64> = points
        .iter()
        .map(|p| {
            tape.forward(p);
            tape.output_value()
        })
        .collect();

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
            approx_eq_f32(*gpu, *cpu, 1e-5, 1e-6),
            "point {}: gpu={}, cpu={}",
            i,
            gpu,
            cpu
        );
    }
}

#[test]
fn gradient_batch_rosenbrock_f32() {
    let ctx = match cuda_context() {
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

    let cpu_grads: Vec<Vec<f64>> = points.iter().map(|p| tape.gradient(p)).collect();

    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let flat_inputs: Vec<f32> = points
        .iter()
        .flat_map(|p| p.iter().map(|&v| v as f32))
        .collect();
    let (_, gpu_grads) = ctx.gradient_batch(&gpu_tape, &flat_inputs, 50).unwrap();

    let num_inputs = tape.num_inputs();
    for (i, cpu_grad) in cpu_grads.iter().enumerate() {
        for (j, &cpu_g) in cpu_grad.iter().enumerate() {
            let gpu_g = gpu_grads[i * num_inputs + j];
            assert!(
                approx_eq_f32(gpu_g, cpu_g, 1e-4, 1e-3),
                "point {}, grad[{}]: gpu={}, cpu={}",
                i,
                j,
                gpu_g,
                cpu_g
            );
        }
    }
}

// ── f64 tests ──

#[test]
fn forward_batch_rosenbrock_f64() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

    let points: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let t = i as f64 / 99.0;
            vec![-2.0 + 4.0 * t, -1.0 + 3.0 * t]
        })
        .collect();

    let cpu_results: Vec<f64> = points
        .iter()
        .map(|p| {
            tape.forward(p);
            tape.output_value()
        })
        .collect();

    let gpu_tape = ctx.upload_tape_f64(&tape).unwrap();

    let flat_inputs: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
    let gpu_results = ctx.forward_batch_f64(&gpu_tape, &flat_inputs, 100).unwrap();

    assert_eq!(gpu_results.len(), 100);
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            approx_eq_f64(*gpu, *cpu, 1e-10, 1e-12),
            "point {}: gpu={}, cpu={}",
            i,
            gpu,
            cpu
        );
    }
}

#[test]
fn gradient_batch_rosenbrock_f64() {
    let ctx = match cuda_context() {
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

    let cpu_grads: Vec<Vec<f64>> = points.iter().map(|p| tape.gradient(p)).collect();

    let gpu_tape = ctx.upload_tape_f64(&tape).unwrap();

    let flat_inputs: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
    let (_, gpu_grads) = ctx.gradient_batch_f64(&gpu_tape, &flat_inputs, 50).unwrap();

    let num_inputs = tape.num_inputs();
    for (i, cpu_grad) in cpu_grads.iter().enumerate() {
        for (j, &cpu_g) in cpu_grad.iter().enumerate() {
            let gpu_g = gpu_grads[i * num_inputs + j];
            assert!(
                approx_eq_f64(gpu_g, cpu_g, 1e-10, 1e-12),
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
fn forward_batch_trig_f64() {
    let ctx = match cuda_context() {
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

    let cpu_results: Vec<f64> = points
        .iter()
        .map(|p| {
            tape.forward(p);
            tape.output_value()
        })
        .collect();

    let gpu_tape = ctx.upload_tape_f64(&tape).unwrap();

    let flat_inputs: Vec<f64> = points.iter().flat_map(|p| p.iter().copied()).collect();
    let gpu_results = ctx.forward_batch_f64(&gpu_tape, &flat_inputs, 50).unwrap();

    assert_eq!(gpu_results.len(), 50);
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            approx_eq_f64(*gpu, *cpu, 1e-10, 1e-12),
            "point {}: gpu={}, cpu={}",
            i,
            gpu,
            cpu
        );
    }
}

#[test]
fn sparse_hessian_rosenbrock_f64() {
    let ctx = match cuda_context() {
        Some(c) => c,
        None => return,
    };

    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

    let gpu_tape = ctx.upload_tape_f64(&tape).unwrap();

    let x = [1.5_f64, 0.5];
    let (gpu_val, gpu_grad, gpu_pattern, gpu_hess) =
        ctx.sparse_hessian_f64(&gpu_tape, &mut tape, &x).unwrap();

    let (cpu_val, cpu_grad, cpu_pattern, cpu_hess) = tape.sparse_hessian(&x);

    assert!(
        approx_eq_f64(gpu_val, cpu_val, 1e-10, 1e-12),
        "value: gpu={}, cpu={}",
        gpu_val,
        cpu_val
    );

    for j in 0..2 {
        assert!(
            approx_eq_f64(gpu_grad[j], cpu_grad[j], 1e-10, 1e-12),
            "grad[{}]: gpu={}, cpu={}",
            j,
            gpu_grad[j],
            cpu_grad[j]
        );
    }

    assert_eq!(gpu_pattern.nnz(), cpu_pattern.nnz());
    for k in 0..gpu_hess.len() {
        assert!(
            approx_eq_f64(gpu_hess[k], cpu_hess[k], 1e-10, 1e-12),
            "hess[{}]: gpu={}, cpu={}",
            k,
            gpu_hess[k],
            cpu_hess[k]
        );
    }
}

#[test]
fn cuda_implements_gpu_backend() {
    fn assert_backend<B: GpuBackend>() {}
    assert_backend::<CudaContext>();
}

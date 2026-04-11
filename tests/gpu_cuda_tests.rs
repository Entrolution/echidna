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

// ══════════════════════════════════════════════
//  K-th order Taylor forward tests
// ══════════════════════════════════════════════

#[cfg(feature = "stde")]
mod taylor_kth {
    use super::*;

    #[test]
    fn cuda_taylor_kth_polynomial_all_orders() {
        // f(x,y) = x² + y² at (3,4), direction (1,0)
        // c0=25, c1=6, c2=1, c3+=0
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        let x = [3.0_f64, 4.0];
        let (tape, _) = record(|v| polynomial(v), &x);
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
        let tape_buf = ctx.upload_tape(&gpu_data);

        for order in 1..=5 {
            let result = ctx
                .taylor_forward_kth_batch(&tape_buf, &[3.0f32, 4.0], &[1.0f32, 0.0], 1, order)
                .unwrap();

            assert_eq!(result.order, order);
            assert_eq!(result.coefficients.len(), order);
            assert!(
                (result.coefficients[0][0] - 25.0).abs() < 1e-3,
                "K={order} c0: {}",
                result.coefficients[0][0]
            );
            if order >= 2 {
                assert!(
                    (result.coefficients[1][0] - 6.0).abs() < 1e-3,
                    "K={order} c1: {}",
                    result.coefficients[1][0]
                );
            }
            if order >= 3 {
                assert!(
                    (result.coefficients[2][0] - 1.0).abs() < 1e-3,
                    "K={order} c2: {}",
                    result.coefficients[2][0]
                );
            }
            if order >= 4 {
                assert!(
                    result.coefficients[3][0].abs() < 1e-3,
                    "K={order} c3: {}",
                    result.coefficients[3][0]
                );
            }
        }
    }

    #[test]
    fn cuda_taylor_kth_k3_matches_2nd() {
        // K=3 should match taylor_forward_2nd_batch exactly
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        let x = [1.5_f64, 2.5];
        let (tape, _) = record(|v| rosenbrock(v), &x);
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

    #[test]
    fn cuda_taylor_kth_exp_higher_order() {
        // f(x) = exp(x) at x=1, direction 1
        // c_k = e / k!
        let ctx = match cuda_context() {
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

        // CPU reference
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

    #[test]
    fn cuda_taylor_kth_unsupported_order() {
        let ctx = match cuda_context() {
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

    #[test]
    fn cuda_taylor_kth_multi_batch() {
        // batch_size=2, two directions, verify deinterleaving
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        let x = [3.0_f64, 4.0];
        let (tape, _) = record(|v| polynomial(v), &x);
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
        let tape_buf = ctx.upload_tape(&gpu_data);

        let primals = [3.0f32, 4.0, 3.0, 4.0];
        let seeds = [1.0f32, 0.0, 0.0, 1.0]; // dir (1,0) and (0,1)

        let result = ctx
            .taylor_forward_kth_batch(&tape_buf, &primals, &seeds, 2, 4)
            .unwrap();

        assert_eq!(result.order, 4);
        // Both: c0 = 25
        assert!((result.coefficients[0][0] - 25.0).abs() < 1e-3);
        assert!((result.coefficients[0][1] - 25.0).abs() < 1e-3);
        // Batch 0 dir (1,0): c1=6, c2=1
        assert!((result.coefficients[1][0] - 6.0).abs() < 1e-3);
        assert!((result.coefficients[2][0] - 1.0).abs() < 1e-3);
        // Batch 1 dir (0,1): c1=8, c2=1
        assert!((result.coefficients[1][1] - 8.0).abs() < 1e-3);
        assert!((result.coefficients[2][1] - 1.0).abs() < 1e-3);
        // c3 = 0 (polynomial degree 2)
        assert!(result.coefficients[3][0].abs() < 1e-3);
        assert!(result.coefficients[3][1].abs() < 1e-3);
    }

    #[test]
    fn cuda_taylor_kth_k1_primal_only() {
        // K=1 should return only primal values
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        let x = [2.0_f64, 3.0];
        let (tape, _) = record(|v| polynomial(v), &x);
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
        let tape_buf = ctx.upload_tape(&gpu_data);

        let result = ctx
            .taylor_forward_kth_batch(&tape_buf, &[2.0f32, 3.0], &[1.0f32, 0.0], 1, 1)
            .unwrap();

        assert_eq!(result.order, 1);
        assert_eq!(result.coefficients.len(), 1);
        assert!(
            (result.coefficients[0][0] - 13.0).abs() < 1e-3,
            "c0: {}",
            result.coefficients[0][0]
        );
    }

    #[test]
    fn cuda_taylor_kth_f64() {
        // f64 variant: exp(x) at x=1, K=3
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        fn f_exp<T: Scalar>(x: &[T]) -> T {
            x[0].exp()
        }

        let x = [1.0_f64];
        let (tape, _) = record(f_exp, &x);
        let tape_buf = ctx.upload_tape_f64(&tape).unwrap();

        let cpu_coeffs = echidna::stde::taylor_jet_dyn(&tape, &x, &[1.0], 3);

        let result = ctx
            .taylor_forward_kth_batch_f64(&tape_buf, &[1.0], &[1.0], 1, 3)
            .unwrap();

        assert_eq!(result.order, 3);
        for k in 0..3 {
            assert!(
                (result.coefficients[k][0] - cpu_coeffs[k]).abs() < 1e-10,
                "f64 c{k}: gpu={} cpu={}",
                result.coefficients[k][0],
                cpu_coeffs[k]
            );
        }
    }
}

fn polynomial<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

// ── Multi-output test helper ──

#[cfg(feature = "stde")]
mod taylor_kth_multi_output {
    use super::*;

    #[test]
    fn cuda_taylor_kth_multi_output() {
        // f(x,y) = (x²+y², x*y) — two outputs
        // At (3,4) dir (1,0): out0: c0=25, c1=6, c2=1; out1: c0=12, c1=4, c2=0
        let ctx = match cuda_context() {
            Some(c) => c,
            None => return,
        };

        let x = [3.0_f64, 4.0];
        let (tape, _) = echidna::record_multi(
            |v: &[echidna::BReverse<f64>]| vec![v[0] * v[0] + v[1] * v[1], v[0] * v[1]],
            &x,
        );
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
        let tape_buf = ctx.upload_tape(&gpu_data);

        let result = ctx
            .taylor_forward_kth_batch(&tape_buf, &[3.0f32, 4.0], &[1.0f32, 0.0], 1, 3)
            .unwrap();

        assert_eq!(result.order, 3);
        // 2 outputs per batch element
        assert_eq!(result.coefficients[0].len(), 2);

        // Output 0: x²+y²
        assert!((result.coefficients[0][0] - 25.0).abs() < 1e-3, "out0 c0");
        assert!((result.coefficients[1][0] - 6.0).abs() < 1e-3, "out0 c1");
        assert!((result.coefficients[2][0] - 1.0).abs() < 1e-3, "out0 c2");

        // Output 1: x*y
        assert!((result.coefficients[0][1] - 12.0).abs() < 1e-3, "out1 c0");
        assert!((result.coefficients[1][1] - 4.0).abs() < 1e-3, "out1 c1");
        assert!(result.coefficients[2][1].abs() < 1e-3, "out1 c2");
    }
}

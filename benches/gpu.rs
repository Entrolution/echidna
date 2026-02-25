use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::gpu::{GpuBackend, GpuTapeData, WgpuContext};
use echidna::record;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn gpu_context() -> Option<WgpuContext> {
    WgpuContext::new()
}

fn bench_forward_batch(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => {
            eprintln!("WARNING: No GPU — skipping GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_forward");

    // (a) Small tape × large batch
    {
        let x0 = vec![1.0_f32; 2];
        let (tape, _) = record(
            |v| {
                let one = f32::from(1.0);
                let hundred = f32::from(100.0);
                let dx = v[0] - one;
                let t = v[1] - v[0] * v[0];
                dx * dx + hundred * t * t
            },
            &x0,
        );
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        for &batch_size in &[100u32, 1000, 10000] {
            let inputs: Vec<f32> = (0..batch_size * 2).map(|i| (i as f32) * 0.01).collect();
            group.bench_with_input(
                BenchmarkId::new("small_tape", batch_size),
                &batch_size,
                |b, &bs| {
                    b.iter(|| {
                        ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                            .unwrap()
                    })
                },
            );
        }
    }

    // (b) Large tape × small batch
    {
        let n = 50;
        let x0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let (tape, _) = record(|v| rosenbrock(v), &x0);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        for &batch_size in &[1u32, 10, 100] {
            let inputs: Vec<f32> = (0..(batch_size as usize * n))
                .map(|i| (i as f32) * 0.01)
                .collect();
            group.bench_with_input(
                BenchmarkId::new("large_tape", batch_size),
                &batch_size,
                |b, &bs| {
                    b.iter(|| {
                        ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                            .unwrap()
                    })
                },
            );
        }
    }

    // (c) Medium tape × sweep
    {
        let n = 10;
        let x0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let (tape, _) = record(|v| rosenbrock(v), &x0);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        for &batch_size in &[10u32, 100, 1000, 10000] {
            let inputs: Vec<f32> = (0..(batch_size as usize * n))
                .map(|i| (i as f32) * 0.01)
                .collect();
            group.bench_with_input(
                BenchmarkId::new("medium_tape", batch_size),
                &batch_size,
                |b, &bs| {
                    b.iter(|| {
                        ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                            .unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_gradient_batch(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_gradient");

    for &n in &[2usize, 10, 50] {
        let x0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let (tape, _) = record(|v| rosenbrock(v), &x0);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        for &batch_size in &[10u32, 100, 1000] {
            let inputs: Vec<f32> = (0..(batch_size as usize * n))
                .map(|i| (i as f32) * 0.01)
                .collect();
            group.bench_with_input(
                BenchmarkId::new(format!("n{}", n), batch_size),
                &batch_size,
                |b, &bs| {
                    b.iter(|| {
                        ctx.gradient_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                            .unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_vs_cpu");

    let n = 10;
    let x0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);
    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    for &batch_size in &[100u32, 1000, 10000] {
        let inputs: Vec<f32> = (0..(batch_size as usize * n))
            .map(|i| (i as f32) * 0.01)
            .collect();
        let points: Vec<Vec<f32>> = inputs.chunks(n).map(|c| c.to_vec()).collect();

        group.bench_with_input(
            BenchmarkId::new("gpu_forward", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                        .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu_forward", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for p in &points {
                        tape.forward(black_box(p));
                        black_box(tape.output_value());
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_gradient", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    ctx.gradient_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                        .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu_gradient", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for p in &points {
                        black_box(tape.gradient(black_box(p)));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_transfer_overhead(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_transfer");

    let n = 10;
    let x0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let (tape, _) = record(|v| rosenbrock(v), &x0);
    let gpu_data = GpuTapeData::from_tape(&tape).unwrap();

    // Measure upload cost
    group.bench_function("upload_tape", |b| {
        b.iter(|| black_box(ctx.upload_tape(black_box(&gpu_data))))
    });

    // Measure full round-trip (upload + compute + download) for different batch sizes
    let gpu_tape = ctx.upload_tape(&gpu_data);
    for &batch_size in &[1u32, 10, 100, 1000] {
        let inputs: Vec<f32> = (0..(batch_size as usize * n))
            .map(|i| (i as f32) * 0.01)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("round_trip", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "stde")]
fn bench_gpu_taylor_2nd(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_taylor_2nd");

    let n = 100;
    let x0: Vec<f32> = make_input(n).iter().map(|&v| v as f32).collect();
    let x0_f64: Vec<f64> = make_input(n);
    let (tape, _) = record(|v| rosenbrock(v), &x0_f64);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    for &batch_size in &[100u32, 1000, 10000] {
        let mut primals = Vec::with_capacity(batch_size as usize * n);
        let mut seeds = Vec::with_capacity(batch_size as usize * n);
        for b in 0..batch_size {
            primals.extend_from_slice(&x0);
            let seed: Vec<f32> = (0..n)
                .map(|i| {
                    if (b as usize * n + i) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            seeds.extend_from_slice(&seed);
        }

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    ctx.taylor_forward_2nd_batch(
                        black_box(&gpu_tape),
                        black_box(&primals),
                        black_box(&seeds),
                        bs,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "stde")]
fn bench_gpu_laplacian_vs_cpu(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_laplacian_vs_cpu");

    let n = 100;
    let x_f64: Vec<f64> = make_input(n);
    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    let (tape_f64, _) = record(|v| rosenbrock(v), &x_f64);
    let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape_f64).unwrap();
    let gpu_tape = ctx.upload_tape(&gpu_data);

    let s = 100;
    let dirs_f64: Vec<Vec<f64>> = (0..s)
        .map(|si| {
            (0..n)
                .map(|i| if (si * n + i) % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();
    let dir_refs_f64: Vec<&[f64]> = dirs_f64.iter().map(|d| d.as_slice()).collect();

    let dirs_f32: Vec<Vec<f32>> = dirs_f64
        .iter()
        .map(|d| d.iter().map(|&v| v as f32).collect())
        .collect();
    let dir_refs_f32: Vec<&[f32]> = dirs_f32.iter().map(|d| d.as_slice()).collect();

    group.bench_function(BenchmarkId::new("gpu", n), |b| {
        b.iter(|| {
            echidna::gpu::stde_gpu::laplacian_gpu(
                black_box(&ctx),
                black_box(&gpu_tape),
                black_box(&x_f32),
                black_box(&dir_refs_f32),
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("cpu", n), |b| {
        b.iter(|| {
            echidna::stde::laplacian(
                black_box(&tape_f64),
                black_box(&x_f64),
                black_box(&dir_refs_f64),
            )
        })
    });

    group.finish();
}

#[cfg(feature = "stde")]
fn bench_gpu_hessian_diag_vs_cpu(c: &mut Criterion) {
    let ctx = match gpu_context() {
        Some(c) => c,
        None => return,
    };

    let mut group = c.benchmark_group("gpu_hessian_diag_vs_cpu");

    for &n in &[100usize, 500] {
        let x_f64: Vec<f64> = make_input(n);
        let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
        let (tape_f64, _) = record(|v| rosenbrock(v), &x_f64);
        let gpu_data = GpuTapeData::from_tape_f64_lossy(&tape_f64).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        group.bench_function(BenchmarkId::new("gpu", n), |b| {
            b.iter(|| {
                echidna::gpu::stde_gpu::hessian_diagonal_gpu(
                    black_box(&ctx),
                    black_box(&gpu_tape),
                    black_box(&x_f32),
                )
                .unwrap()
            })
        });

        group.bench_function(BenchmarkId::new("cpu", n), |b| {
            b.iter(|| echidna::stde::hessian_diagonal(black_box(&tape_f64), black_box(&x_f64)))
        });
    }

    group.finish();
}

#[cfg(feature = "stde")]
criterion_group!(
    benches,
    bench_forward_batch,
    bench_gradient_batch,
    bench_gpu_vs_cpu,
    bench_transfer_overhead,
    bench_gpu_taylor_2nd,
    bench_gpu_laplacian_vs_cpu,
    bench_gpu_hessian_diag_vs_cpu,
);

#[cfg(not(feature = "stde"))]
criterion_group!(
    benches,
    bench_forward_batch,
    bench_gradient_batch,
    bench_gpu_vs_cpu,
    bench_transfer_overhead,
);

criterion_main!(benches);

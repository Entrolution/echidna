use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::gpu::{GpuTapeData, WgpuContext};
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
        let (tape, _) = record(|v| {
            let one = f32::from(1.0);
            let hundred = f32::from(100.0);
            let dx = v[0] - one;
            let t = v[1] - v[0] * v[0];
            dx * dx + hundred * t * t
        }, &x0);
        let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
        let gpu_tape = ctx.upload_tape(&gpu_data);

        for &batch_size in &[100u32, 1000, 10000] {
            let inputs: Vec<f32> = (0..batch_size * 2).map(|i| (i as f32) * 0.01).collect();
            group.bench_with_input(
                BenchmarkId::new("small_tape", batch_size),
                &batch_size,
                |b, &bs| {
                    b.iter(|| ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                    b.iter(|| ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                    b.iter(|| ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                    b.iter(|| ctx.gradient_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                b.iter(|| ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                b.iter(|| ctx.gradient_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
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
                b.iter(|| ctx.forward_batch(black_box(&gpu_tape), black_box(&inputs), bs).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_batch,
    bench_gradient_batch,
    bench_gpu_vs_cpu,
    bench_transfer_overhead,
);
criterion_main!(benches);

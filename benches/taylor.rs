use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::record;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_taylor_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("taylor_grad");
    for n in [2, 10, 100] {
        let x = make_input(n);
        let v = make_direction(n);

        // Rosenbrock
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(
            BenchmarkId::new("rosenbrock_taylor2", n),
            &x,
            |b, x| b.iter(|| black_box(tape.taylor_grad::<2>(black_box(x), black_box(&v)))),
        );

        group.bench_with_input(BenchmarkId::new("rosenbrock_hvp", n), &x, |b, x| {
            b.iter(|| black_box(tape.hvp(black_box(x), black_box(&v))))
        });

        // Rastrigin
        let (tape_r, _) = record(|v| rastrigin(v), &x);

        group.bench_with_input(
            BenchmarkId::new("rastrigin_taylor2", n),
            &x,
            |b, x| b.iter(|| black_box(tape_r.taylor_grad::<2>(black_box(x), black_box(&v)))),
        );

        group.bench_with_input(BenchmarkId::new("rastrigin_hvp", n), &x, |b, x| {
            b.iter(|| black_box(tape_r.hvp(black_box(x), black_box(&v))))
        });
    }
    group.finish();
}

fn bench_taylor_grad_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group("taylor_grad_buf");
    for n in [2, 10, 100] {
        let x = make_input(n);
        let v = make_direction(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(BenchmarkId::new("taylor_grad", n), &x, |b, x| {
            b.iter(|| black_box(tape.taylor_grad::<2>(black_box(x), black_box(&v))))
        });

        group.bench_with_input(BenchmarkId::new("taylor_grad_with_buf", n), &x, |b, x| {
            let mut fwd_buf = Vec::new();
            let mut adj_buf = Vec::new();
            b.iter(|| {
                black_box(tape.taylor_grad_with_buf::<2>(
                    black_box(x),
                    black_box(&v),
                    &mut fwd_buf,
                    &mut adj_buf,
                ))
            })
        });
    }
    group.finish();
}

fn bench_taylor_grad_higher(c: &mut Criterion) {
    let mut group = c.benchmark_group("taylor_grad_higher");
    for n in [2, 10, 100] {
        let x = make_input(n);
        let v = make_direction(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(BenchmarkId::new("order_2", n), &x, |b, x| {
            b.iter(|| black_box(tape.taylor_grad::<2>(black_box(x), black_box(&v))))
        });

        group.bench_with_input(BenchmarkId::new("order_3", n), &x, |b, x| {
            b.iter(|| black_box(tape.taylor_grad::<3>(black_box(x), black_box(&v))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_taylor_grad,
    bench_taylor_grad_buf,
    bench_taylor_grad_higher
);
criterion_main!(benches);

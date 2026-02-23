use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{record, record_multi, Scalar};

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_cross_country(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_country");
    for n in [5, 10, 20] {
        let x = make_input(n);

        let (mut tape, _) = record_multi(|v| pde_poisson_vec(v), &x);

        group.bench_with_input(BenchmarkId::new("cross_country", n), &x, |b, x| {
            b.iter(|| black_box(tape.jacobian_cross_country(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("reverse", n), &x, |b, x| {
            b.iter(|| black_box(tape.jacobian(black_box(x))))
        });
    }
    group.finish();
}

fn bench_sparse_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_jacobian");
    for n in [10, 50, 100] {
        let x = make_input(n);

        let (mut tape, _) = record_multi(|v| pde_poisson_vec(v), &x);

        group.bench_with_input(BenchmarkId::new("sparse", n), &x, |b, x| {
            b.iter(|| black_box(tape.sparse_jacobian(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("dense", n), &x, |b, x| {
            b.iter(|| black_box(tape.jacobian(black_box(x))))
        });
    }
    group.finish();
}

fn abs_sum<T: Scalar>(x: &[T]) -> T {
    let mut sum = T::zero();
    for &xi in x {
        sum = sum + xi.abs();
    }
    sum
}

fn bench_nonsmooth_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonsmooth_overhead");
    for n in [2, 10, 100] {
        let x = make_input(n);

        let (mut tape, _) = record(|v| abs_sum(v), &x);

        group.bench_with_input(BenchmarkId::new("forward", n), &x, |b, x| {
            b.iter(|| {
                tape.forward(black_box(x));
                black_box(tape.output_value())
            })
        });

        group.bench_with_input(BenchmarkId::new("forward_nonsmooth", n), &x, |b, x| {
            b.iter(|| black_box(tape.forward_nonsmooth(black_box(x))))
        });
    }
    group.finish();
}

fn bench_clarke(c: &mut Criterion) {
    let mut group = c.benchmark_group("clarke");
    // Use inputs near the origin so kinks are active
    for n in [2, 5, 10] {
        let x: Vec<f64> = (0..n).map(|i| if i < 2 { 1e-12 } else { 0.5 + 0.01 * i as f64 }).collect();

        let (mut tape, _) = record(|v| abs_sum(v), &x);

        group.bench_with_input(BenchmarkId::new("clarke_jacobian", n), &x, |b, x| {
            b.iter(|| {
                black_box(tape.clarke_jacobian(black_box(x), 0.1, Some(10)))
            })
        });

        group.bench_with_input(BenchmarkId::new("standard_gradient", n), &x, |b, x| {
            b.iter(|| black_box(tape.gradient(black_box(x))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cross_country,
    bench_sparse_jacobian,
    bench_nonsmooth_overhead,
    bench_clarke
);
criterion_main!(benches);

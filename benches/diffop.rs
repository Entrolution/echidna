use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::record;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_diffop_mixed_partial(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffop_mixed_partial");
    for n in [2, 5, 10] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        // Second-order diagonal: d²u/dx₀²
        let mut orders = vec![0u8; n];
        orders[0] = 2;

        group.bench_with_input(BenchmarkId::new("d2_dx0_sq", n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::diffop::mixed_partial(
                    &tape,
                    black_box(x),
                    black_box(&orders),
                ))
            })
        });

        // Mixed second-order: d²u/(dx₀ dx₁)
        if n >= 2 {
            let mut orders_mix = vec![0u8; n];
            orders_mix[0] = 1;
            orders_mix[1] = 1;

            group.bench_with_input(BenchmarkId::new("d2_dx0dx1", n), &x, |b, x| {
                b.iter(|| {
                    black_box(echidna::diffop::mixed_partial(
                        &tape,
                        black_box(x),
                        black_box(&orders_mix),
                    ))
                })
            });
        }
    }
    group.finish();
}

fn bench_diffop_hessian_vs_tape(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffop_hessian_vs_tape");
    for n in [2, 5] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(BenchmarkId::new("diffop_hessian", n), &x, |b, x| {
            b.iter(|| black_box(echidna::diffop::hessian(&tape, black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("tape_hessian", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian(black_box(x))))
        });
    }
    group.finish();
}

fn bench_diffop_high_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffop_high_order");

    // High-order derivatives of exp(x) (1D)
    let x = vec![1.0];
    let (tape, _) = record(|v| v[0].exp(), &x);

    for order in [3, 4, 5, 6] {
        group.bench_with_input(BenchmarkId::new("exp_deriv", order), &x, |b, x| {
            let orders = vec![order as u8];
            b.iter(|| {
                black_box(echidna::diffop::mixed_partial(
                    &tape,
                    black_box(x),
                    black_box(&orders),
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_diffop_mixed_partial,
    bench_diffop_hessian_vs_tape,
    bench_diffop_high_order,
);
criterion_main!(benches);

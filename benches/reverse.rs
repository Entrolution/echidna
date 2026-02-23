use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{grad, Scalar};

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_reverse_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_gradient");
    for n in [2, 10, 100, 1000] {
        let x = make_input(n);

        group.bench_with_input(BenchmarkId::new("f64_eval", n), &x, |b, x| {
            b.iter(|| black_box(rosenbrock_f64(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("rosenbrock_rev", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock(v), black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("rosenbrock_fd", n), &x, |b, x| {
            b.iter(|| black_box(finite_diff_gradient(rosenbrock_f64, x, 1e-7)))
        });

        group.bench_with_input(BenchmarkId::new("rastrigin_rev", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rastrigin(v), black_box(x))))
        });
    }
    group.finish();
}

fn bench_reverse_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_fwd_vs_rev");
    for n in [2, 3, 5, 10, 20] {
        let x = make_input(n);

        group.bench_with_input(BenchmarkId::new("forward_n_passes", n), &x, |b, x| {
            b.iter(|| {
                let n = x.len();
                let mut g = vec![0.0; n];
                for i in 0..n {
                    let inputs: Vec<echidna::Dual<f64>> = x
                        .iter()
                        .enumerate()
                        .map(|(k, &xi)| {
                            if k == i {
                                echidna::Dual::variable(xi)
                            } else {
                                echidna::Dual::constant(xi)
                            }
                        })
                        .collect();
                    g[i] = rosenbrock::<echidna::Dual<f64>>(&inputs).eps;
                }
                black_box(g)
            })
        });

        group.bench_with_input(BenchmarkId::new("reverse_1_pass", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock(v), black_box(x))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reverse_gradient, bench_reverse_crossover);
criterion_main!(benches);

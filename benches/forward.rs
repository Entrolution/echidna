use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{jacobian, Dual, Scalar};

#[path = "common/mod.rs"]
mod common;
use common::*;

fn forward_gradient<F: Fn(&[Dual<f64>]) -> Dual<f64>>(f: &F, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let inputs: Vec<Dual<f64>> = x
            .iter()
            .enumerate()
            .map(|(k, &xi)| {
                if k == i {
                    Dual::variable(xi)
                } else {
                    Dual::constant(xi)
                }
            })
            .collect();
        grad[i] = f(&inputs).eps;
    }
    grad
}

fn bench_forward_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_overhead");
    for n in [2, 10, 100] {
        let x = make_input(n);

        group.bench_with_input(BenchmarkId::new("f64_eval", n), &x, |b, x| {
            b.iter(|| black_box(rosenbrock_f64(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("dual_single_dir", n), &x, |b, x| {
            b.iter(|| {
                let inputs: Vec<Dual<f64>> = x.iter().map(|&xi| Dual::variable(xi)).collect();
                black_box(rosenbrock::<Dual<f64>>(black_box(&inputs)))
            })
        });
    }
    group.finish();
}

fn bench_forward_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_gradient");
    for n in [2, 10, 100, 1000] {
        let x = make_input(n);

        group.bench_with_input(BenchmarkId::new("rosenbrock_fwd", n), &x, |b, x| {
            b.iter(|| black_box(forward_gradient(&rosenbrock, black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("rosenbrock_fd", n), &x, |b, x| {
            b.iter(|| black_box(finite_diff_gradient(rosenbrock_f64, x, 1e-7)))
        });

        group.bench_with_input(BenchmarkId::new("rastrigin_fwd", n), &x, |b, x| {
            b.iter(|| black_box(forward_gradient(&rastrigin, black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("rastrigin_fd", n), &x, |b, x| {
            b.iter(|| {
                black_box(finite_diff_gradient(
                    |v| {
                        let ten = 10.0;
                        let two_pi = 2.0 * std::f64::consts::PI;
                        let mut s = ten * v.len() as f64;
                        for &xi in v {
                            s += xi * xi - ten * (two_pi * xi).cos();
                        }
                        s
                    },
                    x,
                    1e-7,
                ))
            })
        });
    }
    group.finish();
}

fn bench_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian");
    for n in [2, 10] {
        let x = make_input(n);

        group.bench_with_input(BenchmarkId::new("forward_jacobian", n), &x, |b, x| {
            b.iter(|| black_box(jacobian(|v| vec![rosenbrock(v)], black_box(x))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_overhead,
    bench_forward_gradient,
    bench_jacobian
);
criterion_main!(benches);

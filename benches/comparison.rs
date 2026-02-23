use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{grad, record};
use nalgebra::DVector;
use num_dual::DualNum;

#[path = "common/mod.rs"]
mod common;
use common::*;

// ─── num-dual implementations ──────────────────────────────────────────────
// num-dual requires functions generic over DualNum<f64>, using nalgebra vectors.

fn rosenbrock_nd<D: DualNum<f64> + Clone>(x: &[D]) -> D {
    let mut sum = D::from(0.0);
    for i in 0..x.len() - 1 {
        let t1 = D::from(1.0) - x[i].clone();
        let t2 = x[i + 1].clone() - x[i].clone() * x[i].clone();
        sum = sum + t1.clone() * t1 + D::from(100.0) * t2.clone() * t2;
    }
    sum
}

fn two_output_nd<D: DualNum<f64> + Clone>(x: &[D]) -> [D; 2] {
    let mut s1 = D::from(0.0);
    let mut s2 = D::from(0.0);
    for i in 0..x.len() {
        s1 = s1 + x[i].clone() * x[i].clone();
        s2 = s2 + x[i].clone().sin();
    }
    [s1, s2]
}

// ─── echidna 2-output function ─────────────────────────────────────────────

fn two_output_echidna<T: echidna::Scalar>(x: &[T]) -> Vec<T> {
    let mut s1 = T::zero();
    let mut s2 = T::zero();
    for &xi in x {
        s1 = s1 + xi * xi;
        s2 = s2 + xi.sin();
    }
    vec![s1, s2]
}

// ─── Gradient comparison ───────────────────────────────────────────────────

fn bench_gradient_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_comparison");
    for n in [2, 10, 100] {
        let x = make_input(n);

        // echidna reverse-mode gradient
        group.bench_with_input(BenchmarkId::new("echidna_grad", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock(v), black_box(x))))
        });

        // echidna bytecode gradient
        group.bench_with_input(BenchmarkId::new("echidna_bytecode", n), &x, |b, x| {
            b.iter(|| {
                let (mut tape, _) = record(|v| rosenbrock(v), black_box(x));
                black_box(tape.gradient(x))
            })
        });

        // num-dual gradient (dynamic)
        let x_dv = DVector::from_column_slice(&x);
        group.bench_with_input(
            BenchmarkId::new("num_dual_grad", n),
            &x_dv,
            |b, x_dv| {
                b.iter(|| {
                    let (f, g) = num_dual::gradient(
                        |v: DVector<num_dual::DualDVec64>| rosenbrock_nd(v.as_slice()),
                        black_box(x_dv.clone()),
                    );
                    black_box((f, g))
                })
            },
        );
    }
    group.finish();
}

// ─── Jacobian comparison ───────────────────────────────────────────────────

fn bench_jacobian_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian_comparison");
    for n in [2, 5, 10] {
        let x = make_input(n);

        // echidna forward-mode Jacobian
        group.bench_with_input(BenchmarkId::new("echidna_fwd", n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::jacobian(
                    |v| two_output_echidna(v),
                    black_box(x),
                ))
            })
        });

        // echidna bytecode Jacobian (reverse)
        group.bench_with_input(BenchmarkId::new("echidna_rev", n), &x, |b, x| {
            b.iter(|| {
                let (mut tape, _) = echidna::record_multi(|v| two_output_echidna(v), black_box(x));
                black_box(tape.jacobian(x))
            })
        });

        // num-dual Jacobian (dynamic)
        let x_dv = DVector::from_column_slice(&x);
        group.bench_with_input(
            BenchmarkId::new("num_dual_jac", n),
            &x_dv,
            |b, x_dv| {
                b.iter(|| {
                    let (f, jac) = num_dual::jacobian(
                        |v: DVector<num_dual::DualDVec64>| {
                            let out = two_output_nd(v.as_slice());
                            DVector::from_column_slice(&out)
                        },
                        black_box(x_dv.clone()),
                    );
                    black_box((f, jac))
                })
            },
        );
    }
    group.finish();
}

// ─── Hessian comparison ────────────────────────────────────────────────────

fn bench_hessian_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_comparison");
    for n in [2, 10] {
        let x = make_input(n);

        // echidna fwd-over-rev Hessian
        let (tape, _) = record(|v| rosenbrock(v), &x);
        group.bench_with_input(BenchmarkId::new("echidna_hessian", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian(black_box(x))))
        });

        // num-dual hyper-dual Hessian (dynamic)
        let x_dv = DVector::from_column_slice(&x);
        group.bench_with_input(
            BenchmarkId::new("num_dual_hessian", n),
            &x_dv,
            |b, x_dv| {
                b.iter(|| {
                    let (f, g, h) = num_dual::hessian(
                        |v: DVector<num_dual::Dual2DVec64>| rosenbrock_nd(v.as_slice()),
                        black_box(x_dv.clone()),
                    );
                    black_box((f, g, h))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gradient_comparison,
    bench_jacobian_comparison,
    bench_hessian_comparison
);
criterion_main!(benches);

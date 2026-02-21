use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{jacobian, Dual, Scalar};
use num_traits::Float;

fn rosenbrock_f64(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = 1.0 - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum += t1 * t1 + 100.0 * t2 * t2;
    }
    sum
}

fn rosenbrock_dual(x: &[Dual<f64>]) -> Dual<f64> {
    rosenbrock_generic(x)
}

fn rosenbrock_generic<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

fn forward_gradient(x: &[f64]) -> Vec<f64> {
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
        grad[i] = rosenbrock_dual(&inputs).eps;
    }
    grad
}

fn finite_diff_gradient(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let h = 1e-7;
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (rosenbrock_f64(&xp) - rosenbrock_f64(&xm)) / (2.0 * h);
    }
    grad
}

fn bench_forward_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_overhead");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

        group.bench_with_input(BenchmarkId::new("f64_eval", n), &x, |b, x| {
            b.iter(|| black_box(rosenbrock_f64(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("dual_single_dir", n), &x, |b, x| {
            b.iter(|| {
                let inputs: Vec<Dual<f64>> = x
                    .iter()
                    .map(|&xi| Dual::variable(xi))
                    .collect();
                black_box(rosenbrock_dual(black_box(&inputs)))
            })
        });
    }
    group.finish();
}

fn bench_forward_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_gradient");
    for n in [2, 10, 100, 1000] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

        group.bench_with_input(BenchmarkId::new("forward_mode", n), &x, |b, x| {
            b.iter(|| black_box(forward_gradient(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("finite_diff", n), &x, |b, x| {
            b.iter(|| black_box(finite_diff_gradient(black_box(x))))
        });
    }
    group.finish();
}

fn bench_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian");
    for n in [2, 10] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

        group.bench_with_input(BenchmarkId::new("forward_jacobian", n), &x, |b, x| {
            b.iter(|| {
                black_box(jacobian(
                    |v| vec![rosenbrock_generic(v)],
                    black_box(x),
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_forward_overhead, bench_forward_gradient, bench_jacobian);
criterion_main!(benches);

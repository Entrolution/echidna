use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{grad, Reverse, Scalar};

fn rosenbrock_f64(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = 1.0 - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum += t1 * t1 + 100.0 * t2 * t2;
    }
    sum
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

fn bench_reverse_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_gradient");
    for n in [2, 10, 100, 1000] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

        group.bench_with_input(BenchmarkId::new("f64_eval", n), &x, |b, x| {
            b.iter(|| black_box(rosenbrock_f64(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("reverse_mode", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock_generic(v), black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("finite_diff_2n", n), &x, |b, x| {
            b.iter(|| black_box(finite_diff_gradient(black_box(x))))
        });
    }
    group.finish();
}

fn bench_reverse_crossover(c: &mut Criterion) {
    // Compare forward (n passes) vs reverse (1 pass) to find crossover point.
    let mut group = c.benchmark_group("crossover_fwd_vs_rev");
    for n in [2, 3, 5, 10, 20] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

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
                    g[i] = rosenbrock_generic(&inputs).eps;
                }
                black_box(g)
            })
        });

        group.bench_with_input(BenchmarkId::new("reverse_1_pass", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock_generic(v), black_box(x))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reverse_gradient, bench_reverse_crossover);
criterion_main!(benches);

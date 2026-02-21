use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{grad, record, BReverse, Scalar};

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

/// Bytecode gradient() vs Adept grad() for Rosenbrock.
fn bench_bytecode_vs_adept(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytecode_vs_adept");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();

        group.bench_with_input(BenchmarkId::new("adept_grad", n), &x, |b, x| {
            b.iter(|| black_box(grad(|v| rosenbrock_generic(v), black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("bytecode_gradient", n), &x, |b, x| {
            b.iter(|| {
                let (mut tape, _) = record(|v| rosenbrock_generic(v), black_box(x));
                black_box(tape.gradient(x))
            })
        });
    }
    group.finish();
}

/// Tape reuse: record once + N gradient evaluations vs N fresh grad() calls.
fn bench_tape_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_reuse");

    for (n_vars, label) in [(2, "n2"), (100, "n100")] {
        let x: Vec<f64> = (0..n_vars).map(|i| 0.5 + 0.01 * i as f64).collect();
        let x2: Vec<f64> = (0..n_vars).map(|i| 0.6 + 0.01 * i as f64).collect();

        for n_evals in [1, 5, 10, 50, 100] {
            // N fresh Adept grad() calls.
            group.bench_with_input(
                BenchmarkId::new(format!("{}_fresh_adept", label), n_evals),
                &x,
                |b, x| {
                    b.iter(|| {
                        for _ in 0..n_evals {
                            black_box(grad(|v| rosenbrock_generic(v), black_box(&x2)));
                        }
                    })
                },
            );

            // Record once + N gradient() calls.
            group.bench_with_input(
                BenchmarkId::new(format!("{}_reuse_bytecode", label), n_evals),
                &x,
                |b, x| {
                    b.iter(|| {
                        let (mut tape, _) = record(|v| rosenbrock_generic(v), black_box(x));
                        for _ in 0..n_evals {
                            black_box(tape.gradient(&x2));
                        }
                    })
                },
            );
        }
    }
    group.finish();
}

/// gradient_with_buf vs gradient (buffer reuse benefit).
fn bench_buf_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_buf_reuse");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let (mut tape, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("gradient", n), &x, |b, x| {
            b.iter(|| black_box(tape.gradient(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("gradient_with_buf", n), &x, |b, x| {
            let mut buf = Vec::new();
            b.iter(|| black_box(tape.gradient_with_buf(black_box(x), &mut buf)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bytecode_vs_adept,
    bench_tape_reuse,
    bench_buf_reuse
);
criterion_main!(benches);

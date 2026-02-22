use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::{grad, record, BReverse, Scalar};
use num_traits::Float;

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

/// HVP via forward-over-reverse vs finite-difference HVP.
fn bench_hvp(c: &mut Criterion) {
    let mut group = c.benchmark_group("hvp");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let (tape, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("fwd_over_rev", n), &x, |b, x| {
            b.iter(|| black_box(tape.hvp(black_box(x), black_box(&v))))
        });

        // Finite-difference HVP via two gradient calls.
        let h = 1e-5;
        group.bench_with_input(BenchmarkId::new("finite_diff", n), &x, |b, x| {
            let (mut tape2, _) = record(|v| rosenbrock_generic(v), x);
            let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| xi + h * vi).collect();
            let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| xi - h * vi).collect();
            b.iter(|| {
                let gp = tape2.gradient(black_box(&xp));
                let gm = tape2.gradient(black_box(&xm));
                let hvp: Vec<f64> = gp
                    .iter()
                    .zip(gm.iter())
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect();
                black_box(hvp)
            })
        });
    }
    group.finish();
}

/// Full Hessian computation.
fn bench_hessian(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian");
    for n in [2, 10] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let (tape, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("full_hessian", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian(black_box(x))))
        });
    }
    group.finish();
}

/// hvp_with_buf vs hvp (buffer reuse benefit).
fn bench_hvp_buf_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("hvp_buf_reuse");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let (tape, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("hvp", n), &x, |b, x| {
            b.iter(|| black_box(tape.hvp(black_box(x), black_box(&v))))
        });

        group.bench_with_input(BenchmarkId::new("hvp_with_buf", n), &x, |b, x| {
            let mut dv_buf = Vec::new();
            let mut adj_buf = Vec::new();
            b.iter(|| {
                black_box(tape.hvp_with_buf(black_box(x), black_box(&v), &mut dv_buf, &mut adj_buf))
            })
        });
    }
    group.finish();
}

/// Sparse Hessian vs dense Hessian.
fn bench_sparse_hessian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_hessian");

    // Tridiagonal function: sparse should win dramatically
    for n in [10, 50, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let (tape_tri, _) = record(
            |v| {
                let mut sum = v[0] - v[0];
                for i in 0..v.len() - 1 {
                    sum = sum + v[i] * v[i + 1];
                }
                sum
            },
            &x,
        );

        group.bench_with_input(BenchmarkId::new("tridiag_dense", n), &x, |b, x| {
            b.iter(|| black_box(tape_tri.hessian(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("tridiag_sparse", n), &x, |b, x| {
            b.iter(|| black_box(tape_tri.sparse_hessian(black_box(x))))
        });
    }

    // Rosenbrock
    for n in [10] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let (tape_ros, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("rosenbrock_dense", n), &x, |b, x| {
            b.iter(|| black_box(tape_ros.hessian(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("rosenbrock_sparse", n), &x, |b, x| {
            b.iter(|| black_box(tape_ros.sparse_hessian(black_box(x))))
        });
    }

    group.finish();
}

/// Checkpointed gradient vs naive (all steps in one tape).
fn bench_checkpointing(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpointing");

    let x0 = [0.5_f64, 0.3];

    for num_steps in [10, 100] {
        group.bench_with_input(BenchmarkId::new("naive", num_steps), &x0, |b, x0| {
            b.iter(|| {
                let (mut tape, _) = record(
                    |x| {
                        let mut state = x.to_vec();
                        for _ in 0..num_steps {
                            let half = BReverse::constant(0.5_f64);
                            state = vec![
                                state[0].sin() * half + state[1] * half,
                                state[0] * half + state[1].cos() * half,
                            ];
                        }
                        state[0] + state[1]
                    },
                    black_box(x0),
                );
                black_box(tape.gradient(x0))
            })
        });

        let step = |x: &[BReverse<f64>]| {
            let half = BReverse::constant(0.5_f64);
            vec![
                x[0].sin() * half + x[1] * half,
                x[0] * half + x[1].cos() * half,
            ]
        };

        for num_ckpts in [1, 3, 10] {
            let ckpts = num_ckpts.min(num_steps);
            group.bench_with_input(
                BenchmarkId::new(format!("ckpt_{}", ckpts), num_steps),
                &x0,
                |b, x0| {
                    b.iter(|| {
                        black_box(echidna::grad_checkpointed(
                            step,
                            |x| x[0] + x[1],
                            black_box(x0),
                            num_steps,
                            ckpts,
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

/// Batched Hessian (hessian_vec) vs scalar Hessian.
fn bench_hessian_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_vec");
    for n in [2, 10, 100] {
        let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let (tape, _) = record(|v| rosenbrock_generic(v), &x);

        group.bench_with_input(BenchmarkId::new("hessian", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("hessian_vec_4", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian_vec::<4>(black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("hessian_vec_8", n), &x, |b, x| {
            b.iter(|| black_box(tape.hessian_vec::<8>(black_box(x))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bytecode_vs_adept,
    bench_tape_reuse,
    bench_buf_reuse,
    bench_hvp,
    bench_hessian,
    bench_hvp_buf_reuse,
    bench_sparse_hessian,
    bench_checkpointing,
    bench_hessian_vec
);
criterion_main!(benches);

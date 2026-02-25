use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use echidna::record;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn make_rademacher_directions(n: usize, s: usize) -> Vec<Vec<f64>> {
    (0..s)
        .map(|si| {
            (0..n)
                .map(|i| if (si * n + i) % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect()
}

fn bench_stde_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_laplacian");
    for n in [10, 100] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        for s in [5, 10, 50] {
            let dirs = make_rademacher_directions(n, s);
            let dir_refs: Vec<&[f64]> = dirs.iter().map(|d| d.as_slice()).collect();

            group.bench_with_input(BenchmarkId::new(format!("stde_S{}", s), n), &x, |b, x| {
                b.iter(|| {
                    black_box(echidna::stde::laplacian(
                        &tape,
                        black_box(x),
                        black_box(&dir_refs),
                    ))
                })
            });
        }

        // Full Hessian trace as baseline
        group.bench_with_input(BenchmarkId::new("hessian_trace", n), &x, |b, x| {
            b.iter(|| {
                let (_, _, h) = tape.hessian(black_box(x));
                let trace: f64 = (0..n).map(|i| h[i][i]).sum();
                black_box(trace)
            })
        });
    }
    group.finish();
}

fn bench_stde_hessian_diag(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_hessian_diag");
    for n in [10, 50, 100] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(BenchmarkId::new("stde_diag", n), &x, |b, x| {
            b.iter(|| black_box(echidna::stde::hessian_diagonal(&tape, black_box(x))))
        });

        group.bench_with_input(BenchmarkId::new("full_hessian_diag", n), &x, |b, x| {
            b.iter(|| {
                let (_, _, h) = tape.hessian(black_box(x));
                let diag: Vec<f64> = (0..n).map(|i| h[i][i]).collect();
                black_box(diag)
            })
        });
    }
    group.finish();
}

fn bench_stde_jet(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_jet");
    for n in [2, 10, 100] {
        let x = make_input(n);
        let v = make_direction(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        group.bench_with_input(BenchmarkId::new("taylor_jet_2nd", n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::stde::taylor_jet_2nd(
                    &tape,
                    black_box(x),
                    black_box(&v),
                ))
            })
        });

        group.bench_with_input(
            BenchmarkId::new("taylor_jet_2nd_with_buf", n),
            &x,
            |b, x| {
                let mut buf = Vec::new();
                b.iter(|| {
                    black_box(echidna::stde::taylor_jet_2nd_with_buf(
                        &tape,
                        black_box(x),
                        black_box(&v),
                        &mut buf,
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_stde_hutchpp(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_hutchpp");
    let k = 10; // sketch directions
    let s = 20; // stochastic directions

    for n in [10, 50, 100] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        let sketch = make_rademacher_directions(n, k);
        let stoch = make_rademacher_directions(n, s);
        let sketch_refs: Vec<&[f64]> = sketch.iter().map(|d| d.as_slice()).collect();
        let stoch_refs: Vec<&[f64]> = stoch.iter().map(|d| d.as_slice()).collect();

        group.bench_with_input(
            BenchmarkId::new(format!("hutchpp_k{}_S{}", k, s), n),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(echidna::stde::laplacian_hutchpp(
                        &tape,
                        black_box(x),
                        black_box(&sketch_refs),
                        black_box(&stoch_refs),
                    ))
                })
            },
        );

        // Standard Hutchinson with same total budget (k + s directions)
        let total_dirs = make_rademacher_directions(n, k + s);
        let total_refs: Vec<&[f64]> = total_dirs.iter().map(|d| d.as_slice()).collect();

        group.bench_with_input(
            BenchmarkId::new(format!("hutchinson_S{}", k + s), n),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(echidna::stde::laplacian_with_stats(
                        &tape,
                        black_box(x),
                        black_box(&total_refs),
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_stde_diagonal_kth_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_diagonal_kth_order");
    for n in [10, 100] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        for k in [2, 3, 4] {
            group.bench_with_input(BenchmarkId::new(format!("k{}", k), n), &x, |b, x| {
                b.iter(|| {
                    black_box(echidna::stde::diagonal_kth_order(
                        &tape,
                        black_box(x),
                        black_box(k),
                    ))
                })
            });
        }

        // k=2 baseline: existing hessian_diagonal
        group.bench_with_input(BenchmarkId::new("hessian_diagonal", n), &x, |b, x| {
            b.iter(|| black_box(echidna::stde::hessian_diagonal(&tape, black_box(x))))
        });
    }
    group.finish();
}

fn bench_stde_diagonal_const_vs_dyn(c: &mut Criterion) {
    let mut group = c.benchmark_group("stde_diagonal_const_vs_dyn");
    for n in [10, 100] {
        let x = make_input(n);
        let (tape, _) = record(|v| rosenbrock(v), &x);

        // Const-generic ORDER=4 (k=3)
        group.bench_with_input(BenchmarkId::new("const_k3", n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::stde::diagonal_kth_order_const::<_, 4>(
                    &tape,
                    black_box(x),
                ))
            })
        });

        // Dynamic k=3
        group.bench_with_input(BenchmarkId::new("dyn_k3", n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::stde::diagonal_kth_order(
                    &tape,
                    black_box(x),
                    black_box(3),
                ))
            })
        });

        // Const-generic with buffer reuse
        group.bench_with_input(BenchmarkId::new("const_k3_with_buf", n), &x, |b, x| {
            let mut buf = Vec::new();
            b.iter(|| {
                black_box(echidna::stde::diagonal_kth_order_const_with_buf::<_, 4>(
                    &tape,
                    black_box(x),
                    &mut buf,
                ))
            })
        });
    }
    group.finish();
}

#[cfg(feature = "diffop")]
fn bench_stde_sparse(c: &mut Criterion) {
    use echidna::diffop::DiffOp;

    let mut group = c.benchmark_group("stde_sparse");
    let n = 100;
    let x = make_input(n);
    let (tape, _) = record(|v| rosenbrock(v), &x);

    // Laplacian operator
    let lap: DiffOp<f64> = DiffOp::laplacian(n);
    let dist = lap.sparse_distribution();

    // Deterministic sample indices (round-robin over entries)
    for s in [10, 100] {
        let indices: Vec<usize> = (0..s).map(|i| i % dist.len()).collect();

        group.bench_with_input(BenchmarkId::new(format!("sparse_S{}", s), n), &x, |b, x| {
            b.iter(|| {
                black_box(echidna::stde::stde_sparse(
                    &tape,
                    black_box(x),
                    black_box(&dist),
                    black_box(&indices),
                ))
            })
        });
    }

    // Exact DiffOp::eval baseline
    group.bench_with_input(BenchmarkId::new("diffop_eval_exact", n), &x, |b, x| {
        b.iter(|| black_box(lap.eval(&tape, black_box(x))))
    });

    // Hutchinson baseline with same budget (S=100)
    let dirs = make_rademacher_directions(n, 100);
    let dir_refs: Vec<&[f64]> = dirs.iter().map(|d| d.as_slice()).collect();
    group.bench_with_input(BenchmarkId::new("hutchinson_S100", n), &x, |b, x| {
        b.iter(|| {
            black_box(echidna::stde::laplacian(
                &tape,
                black_box(x),
                black_box(&dir_refs),
            ))
        })
    });

    group.finish();
}

#[cfg(feature = "diffop")]
criterion_group!(
    benches,
    bench_stde_laplacian,
    bench_stde_hessian_diag,
    bench_stde_jet,
    bench_stde_hutchpp,
    bench_stde_diagonal_kth_order,
    bench_stde_diagonal_const_vs_dyn,
    bench_stde_sparse,
);

#[cfg(not(feature = "diffop"))]
criterion_group!(
    benches,
    bench_stde_laplacian,
    bench_stde_hessian_diag,
    bench_stde_jet,
    bench_stde_hutchpp,
    bench_stde_diagonal_kth_order,
    bench_stde_diagonal_const_vs_dyn,
);

criterion_main!(benches);

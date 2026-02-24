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

criterion_group!(
    benches,
    bench_stde_laplacian,
    bench_stde_hessian_diag,
    bench_stde_jet,
    bench_stde_hutchpp
);
criterion_main!(benches);

#![cfg(feature = "bytecode")]

use echidna::{record, Scalar};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

#[test]
fn gradient_batch_matches_individual() {
    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

    let points: Vec<[f64; 2]> = vec![[1.0, 2.0], [0.5, 0.5], [2.0, 4.0], [-1.0, 1.0]];
    let refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

    let batch_grads = tape.gradient_batch(&refs);

    for (i, p) in points.iter().enumerate() {
        let single_grad = tape.gradient(p.as_slice());
        for (j, (b, s)) in batch_grads[i].iter().zip(single_grad.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-12,
                "point {}, component {}: batch={}, single={}",
                i,
                j,
                b,
                s
            );
        }
    }
}

#[cfg(feature = "parallel")]
mod parallel_batch {
    use super::*;

    #[test]
    fn gradient_batch_par_matches_serial() {
        let x0 = [1.0_f64, 2.0];
        let (mut tape, _) = record(|v| rosenbrock(v), &x0);

        let points: Vec<[f64; 2]> = vec![
            [1.0, 2.0],
            [0.5, 0.5],
            [2.0, 4.0],
            [-1.0, 1.0],
            [3.0, 9.0],
            [0.0, 0.0],
        ];
        let refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

        let serial = tape.gradient_batch(&refs);
        let parallel = tape.gradient_batch_par(&refs);

        assert_eq!(serial.len(), parallel.len());
        for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
            for (j, (sv, pv)) in s.iter().zip(p.iter()).enumerate() {
                assert!(
                    (sv - pv).abs() < 1e-12,
                    "point {}, component {}: serial={}, parallel={}",
                    i,
                    j,
                    sv,
                    pv
                );
            }
        }
    }

    #[test]
    fn hessian_batch_par_matches_serial() {
        let x0 = [1.0_f64, 2.0];
        let (tape, _) = record(|v| rosenbrock(v), &x0);

        let points: Vec<[f64; 2]> = vec![[1.0, 2.0], [0.5, 0.5], [2.0, 4.0]];
        let refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

        let parallel_results = tape.hessian_batch_par(&refs);

        for (i, p) in points.iter().enumerate() {
            let (val_s, grad_s, hess_s) = tape.hessian(p.as_slice());
            let (val_p, grad_p, hess_p) = &parallel_results[i];

            assert!(
                (val_s - val_p).abs() < 1e-12,
                "point {}: val serial={}, parallel={}",
                i,
                val_s,
                val_p
            );
            for (s, p) in grad_s.iter().zip(grad_p.iter()) {
                assert!((s - p).abs() < 1e-12);
            }
            for (row_s, row_p) in hess_s.iter().zip(hess_p.iter()) {
                for (s, p) in row_s.iter().zip(row_p.iter()) {
                    assert!((s - p).abs() < 1e-10);
                }
            }
        }
    }
}

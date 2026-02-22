#![cfg(feature = "parallel")]

use echidna::{record, record_multi, Scalar};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

fn trig_mix<T: Scalar>(x: &[T]) -> T {
    x[0].sin() * x[1].cos() + x[2].exp()
}

fn multi_output<T: Scalar>(x: &[T]) -> Vec<T> {
    vec![x[0] * x[1], x[1] * x[2], x[0] * x[0]]
}

#[test]
fn gradient_par_matches_serial() {
    let x = [1.5_f64, 2.5];
    let (mut tape, _) = record(|v| rosenbrock(v), &x);

    let serial = tape.gradient(&x);
    let parallel = tape.gradient_par(&x);

    for (s, p) in serial.iter().zip(parallel.iter()) {
        assert!((s - p).abs() < 1e-12, "serial={}, parallel={}", s, p);
    }
}

#[test]
fn gradient_par_at_multiple_points() {
    let x0 = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record(|v| trig_mix(v), &x0);

    for &(a, b, c) in &[(0.5, 1.0, 0.1), (2.0, 3.0, -1.0), (0.0, 0.0, 0.0)] {
        let xv = [a, b, c];
        let serial = tape.gradient(&xv);
        let parallel = tape.gradient_par(&xv);
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert!(
                (s - p).abs() < 1e-10,
                "at ({},{},{}): serial={}, parallel={}",
                a,
                b,
                c,
                s,
                p
            );
        }
    }
}

#[test]
fn jacobian_par_matches_serial() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(|v| multi_output(v), &x);

    let serial = tape.jacobian(&x);
    let parallel = tape.jacobian_par(&x);

    assert_eq!(serial.len(), parallel.len());
    for (si, pi) in serial.iter().zip(parallel.iter()) {
        for (s, p) in si.iter().zip(pi.iter()) {
            assert!((s - p).abs() < 1e-12, "serial={}, parallel={}", s, p);
        }
    }
}

#[test]
fn hessian_par_matches_serial() {
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let (val_s, grad_s, hess_s) = tape.hessian(&x);
    let (val_p, grad_p, hess_p) = tape.hessian_par(&x);

    assert!((val_s - val_p).abs() < 1e-12);
    for (s, p) in grad_s.iter().zip(grad_p.iter()) {
        assert!((s - p).abs() < 1e-12, "grad: serial={}, parallel={}", s, p);
    }
    for (row_s, row_p) in hess_s.iter().zip(hess_p.iter()) {
        for (s, p) in row_s.iter().zip(row_p.iter()) {
            assert!(
                (s - p).abs() < 1e-10,
                "hessian: serial={}, parallel={}",
                s,
                p
            );
        }
    }
}

#[test]
fn sparse_hessian_par_matches_serial() {
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let (val_s, grad_s, pat_s, hval_s) = tape.sparse_hessian(&x);
    let (val_p, grad_p, pat_p, hval_p) = tape.sparse_hessian_par(&x);

    assert!((val_s - val_p).abs() < 1e-12);
    assert_eq!(pat_s.nnz(), pat_p.nnz());
    for (s, p) in grad_s.iter().zip(grad_p.iter()) {
        assert!((s - p).abs() < 1e-12);
    }
    for (s, p) in hval_s.iter().zip(hval_p.iter()) {
        assert!(
            (s - p).abs() < 1e-10,
            "sparse hessian value: serial={}, parallel={}",
            s,
            p
        );
    }
}

#[test]
fn sparse_jacobian_par_matches_serial() {
    let x = [1.0_f64, 2.0, 3.0];
    let (mut tape, _) = record_multi(|v| multi_output(v), &x);

    let (out_s, pat_s, jval_s) = tape.sparse_jacobian(&x);
    let (out_p, pat_p, jval_p) = tape.sparse_jacobian_par(&x);

    assert_eq!(out_s.len(), out_p.len());
    for (s, p) in out_s.iter().zip(out_p.iter()) {
        assert!((s - p).abs() < 1e-12);
    }
    assert_eq!(pat_s.nnz(), pat_p.nnz());
    for (s, p) in jval_s.iter().zip(jval_p.iter()) {
        assert!(
            (s - p).abs() < 1e-10,
            "sparse jacobian value: serial={}, parallel={}",
            s,
            p
        );
    }
}

#[test]
fn gradient_par_is_immutable() {
    // Verify gradient_par takes &self, not &mut self.
    // We call it twice concurrently-compatible (though sequentially here)
    // to show it doesn't mutate.
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(|v| rosenbrock(v), &x);

    let g1 = tape.gradient_par(&x);
    let g2 = tape.gradient_par(&[2.0, 3.0]);
    let g3 = tape.gradient_par(&x);

    // g1 and g3 should be identical
    for (a, b) in g1.iter().zip(g3.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    // g2 should differ
    assert!((g2[0] - g1[0]).abs() > 1e-6);
}

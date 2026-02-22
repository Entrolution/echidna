#![cfg(feature = "bytecode")]

use echidna::record_multi;
use num_traits::Float;

#[test]
fn record_multi_basic() {
    // f(x,y) = [x+y, x*y]
    let (tape, values) = record_multi(|v| vec![v[0] + v[1], v[0] * v[1]], &[2.0_f64, 3.0]);

    assert_eq!(values.len(), 2);
    assert!((values[0] - 5.0).abs() < 1e-10);
    assert!((values[1] - 6.0).abs() < 1e-10);
    assert_eq!(tape.num_outputs(), 2);
}

#[test]
fn record_multi_output_values() {
    let (tape, values) = record_multi(|v| vec![v[0] + v[1], v[0] * v[1]], &[2.0_f64, 3.0]);

    let out_vals = tape.output_values();
    assert_eq!(out_vals.len(), 2);
    assert!((out_vals[0] - values[0]).abs() < 1e-15);
    assert!((out_vals[1] - values[1]).abs() < 1e-15);
}

#[test]
fn jacobian_basic() {
    // f(x,y) = [x+y, x*y]
    // J = [[1, 1], [y, x]]
    let (mut tape, _) = record_multi(|v| vec![v[0] + v[1], v[0] * v[1]], &[2.0_f64, 3.0]);
    let jac = tape.jacobian(&[2.0, 3.0]);

    assert_eq!(jac.len(), 2);
    assert_eq!(jac[0].len(), 2);
    // Row 0: [1, 1]
    assert!((jac[0][0] - 1.0).abs() < 1e-10);
    assert!((jac[0][1] - 1.0).abs() < 1e-10);
    // Row 1: [y, x] = [3, 2]
    assert!((jac[1][0] - 3.0).abs() < 1e-10);
    assert!((jac[1][1] - 2.0).abs() < 1e-10);
}

#[test]
fn jacobian_vs_forward_mode() {
    // Cross-validate: tape.jacobian() should match echidna::jacobian()
    fn f_forward(x: &[echidna::Dual<f64>]) -> Vec<echidna::Dual<f64>> {
        vec![x[0].sin() + x[1].cos(), x[0] * x[1].exp()]
    }

    let x = [0.7_f64, 0.3];
    let (_, jac_forward) = echidna::jacobian(f_forward, &x);

    let (mut tape, _) = record_multi(|v| vec![v[0].sin() + v[1].cos(), v[0] * v[1].exp()], &x);
    let jac_tape = tape.jacobian(&x);

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (jac_forward[i][j] - jac_tape[i][j]).abs() < 1e-10,
                "Jacobian mismatch at ({}, {}): fwd={}, tape={}",
                i,
                j,
                jac_forward[i][j],
                jac_tape[i][j]
            );
        }
    }
}

#[test]
fn reverse_seeded_unit_vectors() {
    // reverse_seeded with unit vector e_i should match reverse(output_i)
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[0] + v[1], v[0] * v[1]], &[2.0_f64, 3.0]);
    tape.forward(&[2.0, 3.0]);

    let grad0 = tape.reverse_seeded(&[1.0, 0.0]);
    let grad1 = tape.reverse_seeded(&[0.0, 1.0]);

    let jac = tape.jacobian(&[2.0, 3.0]);

    for j in 0..2 {
        assert!(
            (grad0[j] - jac[0][j]).abs() < 1e-10,
            "seeded[0] mismatch at {}: {} vs {}",
            j,
            grad0[j],
            jac[0][j]
        );
        assert!(
            (grad1[j] - jac[1][j]).abs() < 1e-10,
            "seeded[1] mismatch at {}: {} vs {}",
            j,
            grad1[j],
            jac[1][j]
        );
    }
}

#[test]
fn vjp_multi_vs_vjp() {
    // Compare vjp_multi with manual weighted sum
    let x = [0.7_f64, 0.3];
    let weights = [0.5, -0.3];

    let (mut tape, _) = record_multi(|v| vec![v[0].sin() + v[1].cos(), v[0] * v[1].exp()], &x);
    let vjp_result = tape.vjp_multi(&x, &weights);

    // Compute manually: J^T * w
    let jac = tape.jacobian(&x);
    for j in 0..2 {
        let expected = weights[0] * jac[0][j] + weights[1] * jac[1][j];
        assert!(
            (vjp_result[j] - expected).abs() < 1e-10,
            "vjp_multi mismatch at {}: {} vs {}",
            j,
            vjp_result[j],
            expected
        );
    }
}

#[test]
fn single_output_backward_compat() {
    // Existing single-output methods should work unchanged
    let (mut tape, val) = echidna::record(|v| v[0] * v[0] + v[1] * v[1], &[3.0_f64, 4.0]);
    assert!((val - 25.0).abs() < 1e-10);
    assert_eq!(tape.num_outputs(), 1);

    let g = tape.gradient(&[3.0, 4.0]);
    assert!((g[0] - 6.0).abs() < 1e-10);
    assert!((g[1] - 8.0).abs() < 1e-10);
}

#[test]
fn jacobian_reeval() {
    // Jacobian at different inputs using the same tape
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[1], v[0] + v[1]], &[1.0_f64, 1.0]);

    let jac1 = tape.jacobian(&[2.0, 3.0]);
    // f1 = x*y, f2 = x+y => J = [[y, x], [1, 1]]
    assert!((jac1[0][0] - 3.0).abs() < 1e-10);
    assert!((jac1[0][1] - 2.0).abs() < 1e-10);
    assert!((jac1[1][0] - 1.0).abs() < 1e-10);
    assert!((jac1[1][1] - 1.0).abs() < 1e-10);

    let jac2 = tape.jacobian(&[5.0, 7.0]);
    assert!((jac2[0][0] - 7.0).abs() < 1e-10);
    assert!((jac2[0][1] - 5.0).abs() < 1e-10);
}

#![cfg(feature = "bytecode")]

use approx::assert_relative_eq;
use echidna::{record_multi, BReverse};
use num_traits::Float;

const TOL: f64 = 1e-10;

// ── 1. matches_reverse_linear ──────────────────────────────────────

#[test]
fn matches_reverse_linear() {
    // f(x,y) = [x+y, x*y]
    let x = [2.0_f64, 3.0];
    let (mut tape, _) = record_multi(|v| vec![v[0] + v[1], v[0] * v[1]], &x);

    let jac_rev = tape.jacobian(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_rev.len(), jac_cc.len());
    for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
        for (r, c) in row_r.iter().zip(row_c.iter()) {
            assert_relative_eq!(r, c, max_relative = TOL);
        }
    }
}

// ── 2. matches_reverse_nonlinear ───────────────────────────────────

#[test]
fn matches_reverse_nonlinear() {
    // f(x,y) = [sin(x)+cos(y), exp(x*y)]
    let x = [0.7_f64, 1.3];
    let (mut tape, _) = record_multi(|v| vec![v[0].sin() + v[1].cos(), (v[0] * v[1]).exp()], &x);

    let jac_rev = tape.jacobian(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_rev.len(), jac_cc.len());
    for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
        for (r, c) in row_r.iter().zip(row_c.iter()) {
            assert_relative_eq!(r, c, max_relative = TOL);
        }
    }
}

// ── 3. identity_passthrough ────────────────────────────────────────

#[test]
fn identity_passthrough() {
    // f(x,y) = [x, y] → J = I₂
    let x = [4.0_f64, 5.0];
    let (mut tape, _) = record_multi(|v| vec![v[0], v[1]], &x);

    let jac = tape.jacobian_cross_country(&x);

    assert_eq!(jac.len(), 2);
    assert_relative_eq!(jac[0][0], 1.0, max_relative = TOL);
    assert_relative_eq!(jac[0][1], 0.0, epsilon = TOL);
    assert_relative_eq!(jac[1][0], 0.0, epsilon = TOL);
    assert_relative_eq!(jac[1][1], 1.0, max_relative = TOL);
}

// ── 4. swap_function ───────────────────────────────────────────────

#[test]
fn swap_function() {
    // f(x,y) = [y, x] → J = [[0,1],[1,0]]
    let x = [4.0_f64, 5.0];
    let (mut tape, _) = record_multi(|v| vec![v[1], v[0]], &x);

    let jac = tape.jacobian_cross_country(&x);

    assert_eq!(jac.len(), 2);
    assert_relative_eq!(jac[0][0], 0.0, epsilon = TOL);
    assert_relative_eq!(jac[0][1], 1.0, max_relative = TOL);
    assert_relative_eq!(jac[1][0], 1.0, max_relative = TOL);
    assert_relative_eq!(jac[1][1], 0.0, epsilon = TOL);
}

// ── 5. constant_scaling ────────────────────────────────────────────

#[test]
fn constant_scaling() {
    // f(x) = 3*x + 2 → J = [[3]]
    let c = BReverse::constant;
    let x = [5.0_f64];
    let (mut tape, _) = record_multi(|v| vec![c(3.0) * v[0] + c(2.0)], &x);

    let jac = tape.jacobian_cross_country(&x);

    assert_eq!(jac.len(), 1);
    assert_eq!(jac[0].len(), 1);
    assert_relative_eq!(jac[0][0], 3.0, max_relative = TOL);
}

// ── 6. more_outputs_than_inputs ────────────────────────────────────

#[test]
fn more_outputs_than_inputs() {
    // f(x) = [x², x³, sin(x)]
    let x = [1.5_f64];
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[0], v[0] * v[0] * v[0], v[0].sin()], &x);

    let jac_rev = tape.jacobian(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_cc.len(), 3);
    for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
        for (r, c) in row_r.iter().zip(row_c.iter()) {
            assert_relative_eq!(r, c, max_relative = TOL);
        }
    }
}

// ── 7. square_jacobian_3x3 ────────────────────────────────────────

#[test]
fn square_jacobian_3x3() {
    // f(x,y,z) = [x*y-z, y*z-x, x*z-y]
    let x = [2.0_f64, 3.0, 4.0];
    let (mut tape, _) = record_multi(
        |v| vec![v[0] * v[1] - v[2], v[1] * v[2] - v[0], v[0] * v[2] - v[1]],
        &x,
    );

    let jac_rev = tape.jacobian(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_cc.len(), 3);
    for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
        for (r, c) in row_r.iter().zip(row_c.iter()) {
            assert_relative_eq!(r, c, max_relative = TOL);
        }
    }
}

// ── 8. fan_out ─────────────────────────────────────────────────────

#[test]
fn fan_out() {
    // Single input feeding all outputs: f(x) = [x+1, x*x, exp(x)]
    let c = BReverse::constant;
    let x = [2.0_f64];
    let (mut tape, _) = record_multi(|v| vec![v[0] + c(1.0), v[0] * v[0], v[0].exp()], &x);

    let jac_rev = tape.jacobian(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_cc.len(), 3);
    for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
        for (r, c) in row_r.iter().zip(row_c.iter()) {
            assert_relative_eq!(r, c, max_relative = TOL);
        }
    }
}

// ── 9. reeval_multiple_points ──────────────────────────────────────

#[test]
fn reeval_multiple_points() {
    // Same tape evaluated at 3 different input points
    let (mut tape, _) = record_multi(
        |v| vec![v[0].sin() * v[1], v[0] + v[1].cos()],
        &[1.0_f64, 2.0],
    );

    let points = [[0.5, 1.0], [2.0, -1.0], [-0.3, 0.7]];

    for pt in &points {
        let jac_rev = tape.jacobian(pt);
        let jac_cc = tape.jacobian_cross_country(pt);

        assert_eq!(jac_rev.len(), jac_cc.len());
        for (row_r, row_c) in jac_rev.iter().zip(jac_cc.iter()) {
            for (r, c) in row_r.iter().zip(row_c.iter()) {
                assert_relative_eq!(r, c, max_relative = TOL);
            }
        }
    }
}

// ── 10. matches_forward_mode ───────────────────────────────────────

#[test]
fn matches_forward_mode() {
    // Cross-validate with jacobian_forward() on a nonlinear function
    // f(x,y) = [x²+y, x*sin(y)]
    let x = [1.5_f64, 0.8];
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[0] + v[1], v[0] * v[1].sin()], &x);

    tape.forward(&x);
    let jac_fwd = tape.jacobian_forward(&x);
    let jac_cc = tape.jacobian_cross_country(&x);

    assert_eq!(jac_fwd.len(), jac_cc.len());
    for (row_f, row_c) in jac_fwd.iter().zip(jac_cc.iter()) {
        for (f, c) in row_f.iter().zip(row_c.iter()) {
            assert_relative_eq!(f, c, max_relative = TOL);
        }
    }
}

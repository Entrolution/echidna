//! Implicit differentiation through residual tapes containing custom
//! operations.
//!
//! The residual `F(z, x) = cube(z) - x` (with `cube` a registered custom op)
//! has the implicit solution `z*(x) = x^(1/3)`, giving analytic references
//! `dz/dx = 1/(3 z²)` and `d²z/dx² = -(2/9) x^(-5/3)`. Second-order results
//! require the nested pass to route custom ops through
//! `eval_dual`/`partials_dual`; a first-order linearization of the custom op
//! yields zero curvature and h = 0 instead.

use std::sync::Arc;

use echidna::bytecode_tape::{BtapeGuard, BytecodeTape, CustomOp, CustomOpHandle};
use echidna::{BReverse, Dual};
use echidna_optim::{implicit_hessian, implicit_hvp, implicit_jacobian};

/// z³ with exact dual implementations: `partials_dual` carries the
/// derivative of the partial (d(3a²) = 6a·ȧ), which is what makes exact
/// second-order implicit derivatives possible.
struct Cube;

impl CustomOp<f64> for Cube {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        a * a * a
    }
    fn partials(&self, a: f64, _b: f64, _r: f64) -> (f64, f64) {
        (3.0 * a * a, 0.0)
    }
    fn eval_dual(&self, a: Dual<f64>, _b: Dual<f64>) -> Dual<f64> {
        a * a * a
    }
    fn partials_dual(&self, a: Dual<f64>, _b: Dual<f64>, _r: Dual<f64>) -> (Dual<f64>, Dual<f64>) {
        (Dual::constant(3.0) * a * a, Dual::constant(0.0))
    }
}

/// Record a single-output residual `f(z, x)` on a tape carrying custom ops.
fn record_residual(
    inputs_primal: &[f64],
    ops: Vec<Arc<dyn CustomOp<f64>>>,
    f: impl FnOnce(&[BReverse<f64>], &[CustomOpHandle], &[f64]) -> BReverse<f64>,
) -> BytecodeTape<f64> {
    let mut tape = BytecodeTape::with_capacity(inputs_primal.len() * 10);
    let handles: Vec<CustomOpHandle> = ops.into_iter().map(|op| tape.register_custom(op)).collect();
    let inputs: Vec<BReverse<f64>> = inputs_primal
        .iter()
        .map(|&val| {
            let idx = tape.new_input(val);
            BReverse::from_tape(val, idx)
        })
        .collect();
    let output = {
        let _guard = BtapeGuard::new(&mut tape);
        f(&inputs, &handles, inputs_primal)
    };
    tape.set_output(output.index());
    tape
}

/// F(z, x) = cube(z) - x, recorded at the actual root (z, x) = (2, 8).
fn cube_residual_tape() -> BytecodeTape<f64> {
    record_residual(&[2.0, 8.0], vec![Arc::new(Cube)], |v, h, xv| {
        let c = v[0].custom_unary(h[0], xv[0] * xv[0] * xv[0]);
        c - v[1]
    })
}

#[test]
fn implicit_jacobian_through_custom_op() {
    let mut tape = cube_residual_tape();
    let jac = implicit_jacobian(&mut tape, &[2.0], &[8.0], 1).unwrap();
    // dz/dx = 1/(3 z²) = 1/12
    assert!(
        (jac[0][0] - 1.0 / 12.0).abs() < 1e-12,
        "dz/dx = {}, expected 1/12",
        jac[0][0]
    );
}

#[test]
fn implicit_hvp_through_custom_op() {
    let mut tape = cube_residual_tape();
    let h = implicit_hvp(&mut tape, &[2.0], &[8.0], &[1.0], &[1.0], 1).unwrap();
    // d²z/dx² = -(2/9)·8^(-5/3) = -1/144. A first-order custom-op
    // linearization contributes zero curvature and returns h = 0.
    assert!(
        (h[0] - (-1.0 / 144.0)).abs() < 1e-12,
        "h = {}, expected -1/144",
        h[0]
    );
}

#[test]
fn implicit_hessian_through_custom_op() {
    let mut tape = cube_residual_tape();
    let hess = implicit_hessian(&mut tape, &[2.0], &[8.0], 1).unwrap();
    assert!(
        (hess[0][0][0] - (-1.0 / 144.0)).abs() < 1e-12,
        "d²z/dx² = {}, expected -1/144",
        hess[0][0][0]
    );
}

#[test]
fn implicit_hvp_custom_op_away_from_recording_point() {
    // Evaluate at the root (z, x) = (3, 27) of a tape recorded at (2, 8):
    // the custom op's curvature must be evaluated at the CURRENT point
    // (F_zz = 6z = 18), not the recording point (12).
    let mut tape = cube_residual_tape();
    let h = implicit_hvp(&mut tape, &[3.0], &[27.0], &[1.0], &[1.0], 1).unwrap();
    // d²z/dx² = -(2/9)·27^(-5/3) = -2/2187
    assert!(
        (h[0] - (-2.0 / 2187.0)).abs() < 1e-12,
        "h = {}, expected -2/2187",
        h[0]
    );
}

#[test]
fn implicit_hvp_composed_custom_op_away_from_recording_point() {
    // F(z, x) = sin(cube(z)) - sin(x) shares the roots of cube(z) - x and
    // the same implicit solution z*(x) = x^(1/3). The custom op feeds a
    // nonlinear builtin, so a stale recording-point linearization of the
    // custom op corrupts the second-order term with nonzero garbage rather
    // than a clean zero — this pins the current-point tangent routing, not
    // just the curvature.
    let mut tape = record_residual(&[2.0, 8.0], vec![Arc::new(Cube)], |v, h, xv| {
        use num_traits::Float as _;
        let c = v[0].custom_unary(h[0], xv[0] * xv[0] * xv[0]);
        c.sin() - v[1].sin()
    });
    let h = implicit_hvp(&mut tape, &[3.0], &[27.0], &[1.0], &[1.0], 1).unwrap();
    assert!(
        (h[0] - (-2.0 / 2187.0)).abs() < 1e-12,
        "h = {}, expected -2/2187",
        h[0]
    );
}

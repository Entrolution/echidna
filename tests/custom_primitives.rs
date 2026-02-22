#![cfg(feature = "bytecode")]

use std::sync::Arc;

use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
use echidna::{BReverse, CustomOp, CustomOpHandle};

// --- Custom op definitions ---

/// Softplus: f(x) = ln(1 + e^x), f'(x) = sigmoid(x)
struct Softplus;

impl CustomOp<f64> for Softplus {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        (1.0 + a.exp()).ln()
    }
    fn partials(&self, a: f64, _b: f64, _result: f64) -> (f64, f64) {
        let sig = 1.0 / (1.0 + (-a).exp());
        (sig, 0.0)
    }
}

/// Smooth max: f(a, b) = ln(e^a + e^b), binary custom op
struct SmoothMax;

impl CustomOp<f64> for SmoothMax {
    fn eval(&self, a: f64, b: f64) -> f64 {
        let max = a.max(b);
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
    fn partials(&self, a: f64, b: f64, _result: f64) -> (f64, f64) {
        let ea = a.exp();
        let eb = b.exp();
        let s = ea + eb;
        (ea / s, eb / s)
    }
}

/// Simple scaling: f(x) = 3*x
struct TripleScale;

impl CustomOp<f64> for TripleScale {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        3.0 * a
    }
    fn partials(&self, _a: f64, _b: f64, _result: f64) -> (f64, f64) {
        (3.0, 0.0)
    }
}

// --- Helper: record with custom ops ---

/// Record a function that uses custom ops. The closure receives input BReverse
/// variables and registered custom op handles. Input values are captured at
/// recording time; the tape can be re-evaluated at different points later.
fn record_with_customs(
    x: &[f64],
    ops: Vec<Arc<dyn CustomOp<f64>>>,
    f: impl FnOnce(&[BReverse<f64>], &[CustomOpHandle], &[f64]) -> BReverse<f64>,
) -> BytecodeTape<f64> {
    let mut tape = BytecodeTape::with_capacity(x.len() * 10);

    let handles: Vec<CustomOpHandle> = ops.into_iter().map(|op| tape.register_custom(op)).collect();

    let inputs: Vec<BReverse<f64>> = x
        .iter()
        .map(|&val| {
            let idx = tape.new_input(val);
            BReverse::from_tape(val, idx)
        })
        .collect();

    let _guard = BtapeGuard::new(&mut tape);
    let output = f(&inputs, &handles, x);

    tape.set_output(output.index());
    tape
}

fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn smooth_max(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// --- Tests ---

#[test]
fn custom_unary_forward_value() {
    let x = [2.0_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        v[0].custom_unary(h[0], sp_val)
    });

    tape.forward(&x);
    let expected = softplus(2.0);
    assert!((tape.output_value() - expected).abs() < 1e-12);
}

#[test]
fn custom_unary_gradient() {
    let x = [2.0_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        v[0].custom_unary(h[0], sp_val)
    });

    let grad = tape.gradient(&x);
    let expected = sigmoid(2.0);
    assert!(
        (grad[0] - expected).abs() < 1e-12,
        "grad={}, expected={}",
        grad[0],
        expected
    );
}

#[test]
fn custom_unary_gradient_at_different_points() {
    let x0 = [2.0_f64];
    let mut tape = record_with_customs(&x0, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        v[0].custom_unary(h[0], sp_val)
    });

    for &x_val in &[-1.0, 0.0, 0.5, 3.0, 5.0] {
        let xv = [x_val];
        let grad = tape.gradient(&xv);
        let expected = sigmoid(x_val);
        assert!(
            (grad[0] - expected).abs() < 1e-10,
            "at x={}: grad={}, expected={}",
            x_val,
            grad[0],
            expected
        );
    }
}

#[test]
fn custom_binary_forward_value() {
    let x = [1.0_f64, 3.0];
    let mut tape = record_with_customs(&x, vec![Arc::new(SmoothMax)], |v, h, xv| {
        let val = smooth_max(xv[0], xv[1]);
        v[0].custom_binary(v[1], h[0], val)
    });

    tape.forward(&x);
    let expected = smooth_max(1.0, 3.0);
    assert!((tape.output_value() - expected).abs() < 1e-12);
}

#[test]
fn custom_binary_gradient() {
    let x = [1.0_f64, 3.0];
    let mut tape = record_with_customs(&x, vec![Arc::new(SmoothMax)], |v, h, xv| {
        let val = smooth_max(xv[0], xv[1]);
        v[0].custom_binary(v[1], h[0], val)
    });

    let grad = tape.gradient(&x);
    let ea = 1.0_f64.exp();
    let eb = 3.0_f64.exp();
    let s = ea + eb;
    assert!(
        (grad[0] - ea / s).abs() < 1e-12,
        "d/da: got {}, expected {}",
        grad[0],
        ea / s
    );
    assert!(
        (grad[1] - eb / s).abs() < 1e-12,
        "d/db: got {}, expected {}",
        grad[1],
        eb / s
    );
}

#[test]
fn custom_binary_gradient_at_different_points() {
    let x0 = [1.0_f64, 3.0];
    let mut tape = record_with_customs(&x0, vec![Arc::new(SmoothMax)], |v, h, xv| {
        let val = smooth_max(xv[0], xv[1]);
        v[0].custom_binary(v[1], h[0], val)
    });

    for &(a, b) in &[(0.0, 0.0), (-2.0, 1.0), (5.0, 5.0), (0.1, 0.2)] {
        let xv = [a, b];
        let grad = tape.gradient(&xv);
        let ea = a.exp();
        let eb = b.exp();
        let s = ea + eb;
        assert!(
            (grad[0] - ea / s).abs() < 1e-10,
            "at ({},{}): d/da={}, expected={}",
            a,
            b,
            grad[0],
            ea / s
        );
        assert!(
            (grad[1] - eb / s).abs() < 1e-10,
            "at ({},{}): d/db={}, expected={}",
            a,
            b,
            grad[1],
            eb / s
        );
    }
}

#[test]
fn custom_op_composed_with_builtins() {
    // f(x) = softplus(x)^2
    let x = [1.5_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        let sp = v[0].custom_unary(h[0], sp_val);
        sp * sp
    });

    let grad = tape.gradient(&x);
    // d/dx [softplus(x)^2] = 2*softplus(x)*sigmoid(x)
    let sp = softplus(1.5);
    let sig = sigmoid(1.5);
    let expected = 2.0 * sp * sig;
    assert!(
        (grad[0] - expected).abs() < 1e-10,
        "grad={}, expected={}",
        grad[0],
        expected
    );
}

#[test]
fn custom_op_with_constant_input() {
    // f(x) = smooth_max(x, 0) (ReLU-like)
    let x = [2.0_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(SmoothMax)], |v, h, xv| {
        let zero = BReverse::constant(0.0);
        let val = smooth_max(xv[0], 0.0);
        v[0].custom_binary(zero, h[0], val)
    });

    let grad = tape.gradient(&x);
    let ea = 2.0_f64.exp();
    let eb = 1.0_f64; // e^0
    let s = ea + eb;
    let expected = ea / s;
    assert!(
        (grad[0] - expected).abs() < 1e-12,
        "grad={}, expected={}",
        grad[0],
        expected
    );
}

#[test]
fn multiple_custom_ops_on_same_tape() {
    // f(x) = triple(softplus(x))
    let x = [1.0_f64];
    let mut tape = record_with_customs(
        &x,
        vec![Arc::new(Softplus), Arc::new(TripleScale)],
        |v, h, xv| {
            let sp_val = softplus(xv[0]);
            let sp = v[0].custom_unary(h[0], sp_val);
            let triple_val = 3.0 * sp_val;
            sp.custom_unary(h[1], triple_val)
        },
    );

    let grad = tape.gradient(&x);
    // d/dx [3*softplus(x)] = 3*sigmoid(x)
    let sig = sigmoid(1.0);
    let expected = 3.0 * sig;
    assert!(
        (grad[0] - expected).abs() < 1e-12,
        "grad={}, expected={}",
        grad[0],
        expected
    );
}

#[test]
fn custom_op_jvp() {
    // f(x, y) = softplus(x) + y
    // Test JVP (which exercises forward tangent) via hvp on a 2-input function
    let x = [2.0_f64, 3.0];
    let tape = record_with_customs(&x, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        let sp = v[0].custom_unary(h[0], sp_val);
        sp + v[1]
    });

    // Use jacobian_forward to test forward tangent mode
    let jac = tape.jacobian_forward(&x);
    let expected_dx = sigmoid(2.0);
    assert!(
        (jac[0][0] - expected_dx).abs() < 1e-10,
        "df/dx={}, expected={}",
        jac[0][0],
        expected_dx
    );
    assert!(
        (jac[0][1] - 1.0).abs() < 1e-10,
        "df/dy={}, expected=1.0",
        jac[0][1]
    );
}

#[test]
fn custom_op_tape_reuse() {
    let x0 = [1.0_f64];
    let mut tape = record_with_customs(&x0, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        v[0].custom_unary(h[0], sp_val)
    });

    for &x_val in &[-3.0, -1.0, 0.0, 1.0, 3.0, 10.0] {
        let xv = [x_val];
        tape.forward(&xv);
        let expected = softplus(x_val);
        assert!(
            (tape.output_value() - expected).abs() < 1e-10,
            "at x={}: value={}, expected={}",
            x_val,
            tape.output_value(),
            expected
        );
    }
}

#[test]
fn custom_op_hvp() {
    // f(x) = softplus(x)^2
    // f'(x) = 2 * softplus(x) * sigmoid(x)
    // Custom ops compute partials at F level only, so the HVP treats
    // the custom op's derivative as constant w.r.t. the tangent direction.
    let x = [1.5_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(Softplus)], |v, h, xv| {
        let sp_val = softplus(xv[0]);
        let sp = v[0].custom_unary(h[0], sp_val);
        sp * sp
    });

    tape.forward(&x);
    let (grad, hvp) = tape.hvp(&x, &[1.0]);

    let sig = sigmoid(1.5);
    let sp = softplus(1.5);

    // Gradient should be correct
    let expected_grad = 2.0 * sp * sig;
    assert!(
        (grad[0] - expected_grad).abs() < 1e-10,
        "grad={}, expected={}",
        grad[0],
        expected_grad
    );

    // HVP: the custom op's sigmoid partial is treated as constant,
    // so we get 2 * sig * sig from the chain rule of the multiplication.
    let expected_hvp = 2.0 * sig * sig;
    assert!(
        (hvp[0] - expected_hvp).abs() < 1e-8,
        "hvp={}, expected={}",
        hvp[0],
        expected_hvp
    );
}

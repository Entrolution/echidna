#![cfg(feature = "bytecode")]

use std::sync::Arc;

use echidna::bytecode_tape::{BtapeGuard, BytecodeTape};
use echidna::dual::Dual;
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

// --- Second-order custom ops (with eval_dual / partials_dual overrides) ---

/// Softplus with second-order support via eval_dual / partials_dual.
struct SoftplusSecondOrder;

impl CustomOp<f64> for SoftplusSecondOrder {
    fn eval(&self, a: f64, _b: f64) -> f64 {
        (1.0 + a.exp()).ln()
    }
    fn partials(&self, a: f64, _b: f64, _r: f64) -> (f64, f64) {
        let sig = 1.0 / (1.0 + (-a).exp());
        (sig, 0.0)
    }
    fn eval_dual(&self, a: Dual<f64>, _b: Dual<f64>) -> Dual<f64> {
        let one = Dual::constant(1.0);
        (one + a.exp()).ln()
    }
    fn partials_dual(&self, a: Dual<f64>, _b: Dual<f64>, _r: Dual<f64>) -> (Dual<f64>, Dual<f64>) {
        let one = Dual::constant(1.0);
        let sig = one / (one + (-a).exp());
        (sig, Dual::constant(0.0))
    }
}

/// SmoothMax with second-order support.
struct SmoothMaxSecondOrder;

impl CustomOp<f64> for SmoothMaxSecondOrder {
    fn eval(&self, a: f64, b: f64) -> f64 {
        let max = a.max(b);
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
    fn partials(&self, a: f64, b: f64, _r: f64) -> (f64, f64) {
        let ea = a.exp();
        let eb = b.exp();
        let s = ea + eb;
        (ea / s, eb / s)
    }
    fn eval_dual(&self, a: Dual<f64>, b: Dual<f64>) -> Dual<f64> {
        // Use the stable form: max + ln(exp(a-max) + exp(b-max))
        // For Dual numbers, max is based on primals only.
        let max_val = Dual::constant(a.re.max(b.re));
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    }
    fn partials_dual(&self, a: Dual<f64>, b: Dual<f64>, _r: Dual<f64>) -> (Dual<f64>, Dual<f64>) {
        let ea = a.exp();
        let eb = b.exp();
        let s = ea + eb;
        (ea / s, eb / s)
    }
}

#[test]
fn custom_op_hvp_second_order_unary() {
    // f(x) = softplus(x)^2 with second-order custom op
    // f'(x)  = 2 * softplus(x) * sigmoid(x)
    // f''(x) = 2 * sigmoid(x)^2 + 2 * softplus(x) * sigmoid(x) * (1 - sigmoid(x))
    let x = [1.5_f64];
    let mut tape = record_with_customs(&x, vec![Arc::new(SoftplusSecondOrder)], |v, h, xv| {
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

    // With second-order support, HVP should be the true second derivative:
    // f''(x) = 2*sig^2 + 2*sp*sig*(1-sig)
    let expected_hvp = 2.0 * sig * sig + 2.0 * sp * sig * (1.0 - sig);
    assert!(
        (hvp[0] - expected_hvp).abs() < 1e-8,
        "hvp={}, expected={}",
        hvp[0],
        expected_hvp
    );
}

#[test]
fn custom_op_hvp_second_order_binary() {
    // f(a, b) = smooth_max(a, b)^2 with second-order custom op
    // Test with tangent direction v = [1, 1]
    let x = [1.0_f64, 2.0];
    let mut tape = record_with_customs(&x, vec![Arc::new(SmoothMaxSecondOrder)], |v, h, xv| {
        let val = smooth_max(xv[0], xv[1]);
        let sm = v[0].custom_binary(v[1], h[0], val);
        sm * sm
    });

    tape.forward(&x);
    let v = [1.0, 1.0];
    let (grad, hvp) = tape.hvp(&x, &v);

    // Analytical: g(a,b) = smooth_max(a,b), f = g^2
    // dg/da = ea/(ea+eb), dg/db = eb/(ea+eb)
    let ea = 1.0_f64.exp();
    let eb = 2.0_f64.exp();
    let s = ea + eb;
    let pa = ea / s; // dg/da
    let pb = eb / s; // dg/db
    let g = smooth_max(1.0, 2.0);

    // df/da = 2*g*pa, df/db = 2*g*pb
    let expected_grad_a = 2.0 * g * pa;
    let expected_grad_b = 2.0 * g * pb;
    assert!(
        (grad[0] - expected_grad_a).abs() < 1e-10,
        "grad[0]={}, expected={}",
        grad[0],
        expected_grad_a
    );
    assert!(
        (grad[1] - expected_grad_b).abs() < 1e-10,
        "grad[1]={}, expected={}",
        grad[1],
        expected_grad_b
    );

    // HVP = H * v where v = [1, 1]
    // d²f/da² = 2*pa^2 + 2*g*d(pa)/da
    // d²f/db² = 2*pb^2 + 2*g*d(pb)/db
    // d²f/dadb = 2*pa*pb + 2*g*d(pa)/db
    // d(pa)/da = (ea*s - ea*ea) / s^2 = ea*eb / s^2
    // d(pa)/db = -ea*eb / s^2
    // d(pb)/db = ea*eb / s^2
    let d2_pa_da = ea * eb / (s * s);
    let d2_pa_db = -ea * eb / (s * s);
    let d2_pb_db = ea * eb / (s * s);

    let h_aa = 2.0 * pa * pa + 2.0 * g * d2_pa_da;
    let h_ab = 2.0 * pa * pb + 2.0 * g * d2_pa_db;
    let h_bb = 2.0 * pb * pb + 2.0 * g * d2_pb_db;

    let expected_hvp_0 = h_aa * v[0] + h_ab * v[1];
    let expected_hvp_1 = h_ab * v[0] + h_bb * v[1];

    assert!(
        (hvp[0] - expected_hvp_0).abs() < 1e-8,
        "hvp[0]={}, expected={}",
        hvp[0],
        expected_hvp_0
    );
    assert!(
        (hvp[1] - expected_hvp_1).abs() < 1e-8,
        "hvp[1]={}, expected={}",
        hvp[1],
        expected_hvp_1
    );
}

#[test]
fn custom_op_hessian_second_order() {
    // Verify the full Hessian matrix for f(a, b) = smooth_max(a, b)^2
    let x = [1.0_f64, 2.0];
    let tape = record_with_customs(&x, vec![Arc::new(SmoothMaxSecondOrder)], |v, h, xv| {
        let val = smooth_max(xv[0], xv[1]);
        let sm = v[0].custom_binary(v[1], h[0], val);
        sm * sm
    });

    let (_val, _grad, hess) = tape.hessian(&x);

    let ea = 1.0_f64.exp();
    let eb = 2.0_f64.exp();
    let s = ea + eb;
    let pa = ea / s;
    let pb = eb / s;
    let g = smooth_max(1.0, 2.0);
    let cross = ea * eb / (s * s);

    let h_aa = 2.0 * pa * pa + 2.0 * g * cross;
    let h_ab = 2.0 * pa * pb + 2.0 * g * (-cross);
    let h_bb = 2.0 * pb * pb + 2.0 * g * cross;

    assert!(
        (hess[0][0] - h_aa).abs() < 1e-8,
        "H[0][0]={}, expected={}",
        hess[0][0],
        h_aa
    );
    assert!(
        (hess[0][1] - h_ab).abs() < 1e-8,
        "H[0][1]={}, expected={}",
        hess[0][1],
        h_ab
    );
    assert!(
        (hess[1][0] - h_ab).abs() < 1e-8,
        "H[1][0]={}, expected={}",
        hess[1][0],
        h_ab
    );
    assert!(
        (hess[1][1] - h_bb).abs() < 1e-8,
        "H[1][1]={}, expected={}",
        hess[1][1],
        h_bb
    );
}

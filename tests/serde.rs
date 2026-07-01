#![cfg(all(feature = "bytecode", feature = "serde"))]

use echidna::{record, record_multi, Scalar};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let dx = x[0] - one;
    let t = x[1] - x[0] * x[0];
    dx * dx + hundred * t * t
}

#[test]
fn roundtrip_tape_json() {
    let x = [1.5_f64, 2.5];
    let (mut tape, _) = record(rosenbrock, &x);

    let json = serde_json::to_string(&tape).unwrap();
    let mut tape2: echidna::BytecodeTape<f64> = serde_json::from_str(&json).unwrap();

    // Evaluate on original tape
    let grad_orig = tape.gradient(&x);
    // Evaluate on deserialized tape
    let grad_deser = tape2.gradient(&x);

    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-12, "original={}, deserialized={}", o, d);
    }
}

#[test]
fn roundtrip_tape_at_different_point() {
    let x0 = [1.0_f64, 2.0];
    let (mut tape, _) = record(rosenbrock, &x0);

    let json = serde_json::to_string(&tape).unwrap();
    let mut tape2: echidna::BytecodeTape<f64> = serde_json::from_str(&json).unwrap();

    // Evaluate at a different point
    let x1 = [2.0, 3.0];
    let grad_orig = tape.gradient(&x1);
    let grad_deser = tape2.gradient(&x1);

    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-12, "original={}, deserialized={}", o, d);
    }
}

#[test]
fn deserialized_tape_supports_hessian() {
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(rosenbrock, &x);

    let json = serde_json::to_string(&tape).unwrap();
    let tape2: echidna::BytecodeTape<f64> = serde_json::from_str(&json).unwrap();

    let (val_orig, grad_orig, hess_orig) = tape.hessian(&x);
    let (val_deser, grad_deser, hess_deser) = tape2.hessian(&x);

    assert!((val_orig - val_deser).abs() < 1e-12);
    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-12);
    }
    for (row_o, row_d) in hess_orig.iter().zip(hess_deser.iter()) {
        for (o, d) in row_o.iter().zip(row_d.iter()) {
            assert!((o - d).abs() < 1e-10);
        }
    }
}

#[test]
fn custom_op_tape_serialization_fails() {
    use echidna::bytecode_tape::BtapeGuard;
    use echidna::{BReverse, CustomOp};
    use std::sync::Arc;

    struct Scale;
    impl CustomOp<f64> for Scale {
        fn eval(&self, a: f64, _b: f64) -> f64 {
            2.0 * a
        }
        fn partials(&self, _a: f64, _b: f64, _r: f64) -> (f64, f64) {
            (2.0, 0.0)
        }
    }

    let x = [1.0_f64];
    let mut tape = echidna::BytecodeTape::with_capacity(10);
    let handle = tape.register_custom(Arc::new(Scale));
    let idx = tape.new_input(x[0]);
    let input = BReverse::from_tape(x[0], idx);
    let output = {
        let _guard = BtapeGuard::new(&mut tape);
        input.custom_unary(handle, 2.0 * x[0])
    };
    tape.set_output(output.index());

    let result = serde_json::to_string(&tape);
    assert!(
        result.is_err(),
        "should fail to serialize tape with custom ops"
    );
}

#[test]
fn sparsity_pattern_roundtrip() {
    let x = [1.0_f64, 2.0, 3.0];
    let (tape, _) = record(rosenbrock, &x);
    let pattern = tape.detect_sparsity();

    let json = serde_json::to_string(&pattern).unwrap();
    let pattern2: echidna::SparsityPattern = serde_json::from_str(&json).unwrap();

    assert_eq!(pattern.dim, pattern2.dim);
    assert_eq!(pattern.rows, pattern2.rows);
    assert_eq!(pattern.cols, pattern2.cols);
}

#[test]
fn roundtrip_tape_f32() {
    let x = [1.5_f32, 2.5];
    let (mut tape, _) = record(rosenbrock, &x);

    let json = serde_json::to_string(&tape).unwrap();
    let mut tape2: echidna::BytecodeTape<f32> = serde_json::from_str(&json).unwrap();

    let grad_orig = tape.gradient(&x);
    let grad_deser = tape2.gradient(&x);

    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-5, "original={}, deserialized={}", o, d);
    }
}

#[test]
fn roundtrip_multi_output() {
    // f: R^3 -> R^2, f(x,y,z) = (x*y + z, x - y*z)
    let x = [2.0_f64, 3.0, 0.5];
    let (mut tape, _) = record_multi(|v| vec![v[0] * v[1] + v[2], v[0] - v[1] * v[2]], &x);

    let json = serde_json::to_string(&tape).unwrap();
    let mut tape2: echidna::BytecodeTape<f64> = serde_json::from_str(&json).unwrap();

    let jac_orig = tape.jacobian(&x);
    let jac_deser = tape2.jacobian(&x);

    assert_eq!(jac_orig.len(), jac_deser.len());
    for (row_o, row_d) in jac_orig.iter().zip(jac_deser.iter()) {
        for (o, d) in row_o.iter().zip(row_d.iter()) {
            assert!((o - d).abs() < 1e-12, "original={}, deserialized={}", o, d);
        }
    }
}

#[test]
fn roundtrip_tape_cbor() {
    let x = [1.5_f64, 2.5];
    let (mut tape, _) = record(rosenbrock, &x);

    let mut bytes = Vec::new();
    ciborium::into_writer(&tape, &mut bytes).unwrap();
    let mut tape2: echidna::BytecodeTape<f64> = ciborium::from_reader(&bytes[..]).unwrap();

    let grad_orig = tape.gradient(&x);
    let grad_deser = tape2.gradient(&x);

    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-12, "original={}, deserialized={}", o, d);
    }
}

// ── #26: Malformed tape deserialization returns error ──

#[test]
fn regression_26_malformed_tape_deserialization_returns_error() {
    // Mismatched opcodes/arg_indices lengths: 2 opcodes but only 1 arg_indices entry
    let json = r#"{"opcodes":[0,1],"arg_indices":[[0,0]],"values":[0.0,1.0],"num_inputs":1,"num_variables":2,"output_index":1,"output_indices":[1],"custom_ops":{},"custom_second_args":{}}"#;
    let result: Result<echidna::BytecodeTape<f64>, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "mismatched opcodes/arg_indices lengths should fail deserialization"
    );
}

// ── Corrupt-payload rejection ──
//
// A deserialized tape is the one place tape data crosses a trust boundary:
// the bytes may be truncated, tampered with, or produced by a buggy writer.
// Each test below serializes a real tape to a `serde_json::Value`, applies
// one targeted corruption, and asserts deserialization returns a clean
// error (never a panic, never a silently-wrong tape).

fn tape_value() -> serde_json::Value {
    let x = [1.5_f64, 2.5];
    let (tape, _) = record(rosenbrock, &x);
    serde_json::to_value(&tape).unwrap()
}

/// Like `tape_value`, but the recorded function contains a unary op (`sin`).
fn tape_value_with_unary() -> (serde_json::Value, usize) {
    use num_traits::Float;
    let x = [0.5_f64, 1.0];
    let (tape, _) = record(|v| (v[0] * v[0]).sin() + v[1], &x);
    let v = serde_json::to_value(&tape).unwrap();
    let sin_pos = v["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|op| op == "Sin")
        .expect("recorded tape should contain a Sin opcode");
    (v, sin_pos)
}

/// Index of the last non-Input/non-Const opcode (a real operation).
fn last_real_op(v: &serde_json::Value) -> usize {
    v["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .rposition(|op| op != "Input" && op != "Const")
        .expect("tape should contain at least one real op")
}

fn assert_rejected(v: serde_json::Value, what: &str) {
    let r: Result<echidna::BytecodeTape<f64>, _> = serde_json::from_value(v);
    assert!(r.is_err(), "{what}: corrupt payload deserialized Ok");
}

#[test]
fn corrupt_payload_baseline_roundtrips() {
    let v = tape_value();
    let r: Result<echidna::BytecodeTape<f64>, _> = serde_json::from_value(v);
    assert!(r.is_ok(), "unmutated payload must still deserialize");
}

#[test]
fn deserialize_rejects_input_opcode_after_prefix() {
    let mut v = tape_value();
    let last = last_real_op(&v);
    v["opcodes"][last] = serde_json::Value::from("Input");
    assert_rejected(v, "Input opcode past the num_inputs prefix");
}

#[test]
fn deserialize_rejects_unset_output_index_sentinel() {
    let mut v = tape_value();
    v["output_index"] = serde_json::Value::from(u32::MAX);
    assert_rejected(v, "output_index = u32::MAX sentinel");
}

#[test]
fn deserialize_rejects_forward_referencing_arg() {
    let mut v = tape_value();
    let last = last_real_op(&v);
    // Self-reference: still < num_variables, so only a DAG-order check
    // catches it. Evaluating such a tape reads an uncomputed slot.
    v["arg_indices"][last][0] = serde_json::Value::from(last as u32);
    assert_rejected(v, "arg0 referencing a not-yet-computed slot");
}

#[test]
fn deserialize_rejects_unary_with_arg1_index() {
    let (mut v, sin_pos) = tape_value_with_unary();
    // A unary op must carry the UNUSED sentinel in slot 1. A real index
    // here makes the reverse sweep accumulate an adjoint into an
    // unrelated slot.
    v["arg_indices"][sin_pos][1] = serde_json::Value::from(0u32);
    assert_rejected(v, "unary op with a real index in arg slot 1");
}

#[test]
fn deserialize_rejects_stale_custom_second_args() {
    let mut v = tape_value();
    v["custom_second_args"]["3"] = serde_json::Value::from(1u32);
    assert_rejected(v, "non-empty custom_second_args without custom ops");
}

#[test]
fn deserialize_rejects_mismatched_lengths_without_panic() {
    // More arg_indices entries than opcodes: must be a clean error (this
    // pins that no validation step indexes one array by another's length).
    let mut v = tape_value();
    let extra = serde_json::json!([0u32, 0u32]);
    v["arg_indices"].as_array_mut().unwrap().push(extra);
    assert_rejected(v, "arg_indices longer than opcodes");
}

#[test]
fn roundtrip_nonsmooth_info() {
    use num_traits::Float;

    // f(x, y) = |x| + max(x, y)
    let x = [0.0_f64, 0.0];
    let (mut tape, _) = record(|v| v[0].abs() + v[0].max(v[1]), &x);

    let info = tape.forward_nonsmooth(&x);
    assert!(!info.kinks.is_empty());

    let json = serde_json::to_string(&info).unwrap();
    let info2: echidna::NonsmoothInfo<f64> = serde_json::from_str(&json).unwrap();

    assert_eq!(info.kinks.len(), info2.kinks.len());
    for (k1, k2) in info.kinks.iter().zip(info2.kinks.iter()) {
        assert_eq!(k1.tape_index, k2.tape_index);
        assert_eq!(k1.opcode, k2.opcode);
        assert_eq!(k1.branch, k2.branch);
        assert!((k1.switching_value - k2.switching_value).abs() < 1e-15);
    }
    assert_eq!(info.signature(), info2.signature());
}

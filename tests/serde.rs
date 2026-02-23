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
    let (mut tape, _) = record(|v| rosenbrock(v), &x);

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
    let (mut tape, _) = record(|v| rosenbrock(v), &x0);

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
    let (tape, _) = record(|v| rosenbrock(v), &x);

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
    let _guard = BtapeGuard::new(&mut tape);
    let output = input.custom_unary(handle, 2.0 * x[0]);
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
    let (tape, _) = record(|v| rosenbrock(v), &x);
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
    let (mut tape, _) = record(|v| rosenbrock(v), &x);

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
    let (mut tape, _) = record_multi(
        |v| vec![v[0] * v[1] + v[2], v[0] - v[1] * v[2]],
        &x,
    );

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
fn roundtrip_tape_bincode() {
    let x = [1.5_f64, 2.5];
    let (mut tape, _) = record(|v| rosenbrock(v), &x);

    let bytes = bincode::serialize(&tape).unwrap();
    let mut tape2: echidna::BytecodeTape<f64> = bincode::deserialize(&bytes).unwrap();

    let grad_orig = tape.gradient(&x);
    let grad_deser = tape2.gradient(&x);

    for (o, d) in grad_orig.iter().zip(grad_deser.iter()) {
        assert!((o - d).abs() < 1e-12, "original={}, deserialized={}", o, d);
    }
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

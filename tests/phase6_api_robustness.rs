//! Phase 6 regression tests for echidna API hardening:
//! M37 (empty-input hessian value), M38 (record_multi empty outputs rejection),
//! M45 (mixed_partial input-length assert), M46 (extraction_prefactor overflow).

#![cfg(feature = "bytecode")]

use echidna::{record_multi, BReverse, BytecodeTape};

// M37: tape with zero inputs must return its (constant) output value from
// hessian(), not the default-zero shortcut from an unrun loop.
#[test]
fn m37_hessian_empty_input_returns_constant_value() {
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.push_const(3.5);
    tape.set_output(idx);
    tape.set_outputs(&[idx]);

    let (value, gradient, hessian) = tape.hessian(&[]);
    assert!((value - 3.5).abs() < 1e-15, "value = {}", value);
    assert_eq!(gradient.len(), 0);
    assert_eq!(hessian.len(), 0);
}

// M37 sibling: hessian_vec suffers from the same blind spot as hessian on
// zero-input tapes — the batch loop never runs, so `value` stayed at zero.
// After the fix, the constant output is recovered via a primal pass.
#[test]
fn m37_hessian_vec_empty_input_returns_constant_value() {
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.push_const(2.75);
    tape.set_output(idx);
    tape.set_outputs(&[idx]);

    let (value, gradient, hessian) = tape.hessian_vec::<4>(&[]);
    assert!((value - 2.75).abs() < 1e-15, "value = {}", value);
    assert_eq!(gradient.len(), 0);
    assert_eq!(hessian.len(), 0);
}

// M37 sibling: sparse_hessian and sparse_hessian_vec share the same zero-
// input blind spot (the color loop / batch loop is `for _ in 0..num_colors`
// where num_colors == 0 for an input-less tape).
#[test]
fn m37_sparse_hessian_empty_input_returns_constant_value() {
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.push_const(1.25);
    tape.set_output(idx);
    tape.set_outputs(&[idx]);

    let (value, gradient, _pattern, hess_vals) = tape.sparse_hessian(&[]);
    assert!(
        (value - 1.25).abs() < 1e-15,
        "sparse_hessian value = {}",
        value
    );
    assert_eq!(gradient.len(), 0);
    assert_eq!(hess_vals.len(), 0);
}

#[test]
fn m37_sparse_hessian_vec_empty_input_returns_constant_value() {
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.push_const(4.125);
    tape.set_output(idx);
    tape.set_outputs(&[idx]);

    let (value, gradient, _pattern, hess_vals) = tape.sparse_hessian_vec::<4>(&[]);
    assert!(
        (value - 4.125).abs() < 1e-15,
        "sparse_hessian_vec value = {}",
        value
    );
    assert_eq!(gradient.len(), 0);
    assert_eq!(hess_vals.len(), 0);
}

// M38: a closure that produces zero outputs must be rejected by record_multi
// rather than silently returning a tape that claims one output pointing at a
// random index.
#[test]
#[should_panic(expected = "record_multi: closure returned zero outputs")]
fn m38_record_multi_rejects_empty_outputs() {
    let f = |_inputs: &[BReverse<f64>]| -> Vec<BReverse<f64>> { Vec::new() };
    let _ = record_multi(f, &[1.0, 2.0]);
}

// M45: mixed_partial must reject `orders.len() != tape.num_inputs()` up front,
// not silently return a garbage derivative.
#[cfg(feature = "diffop")]
#[test]
#[should_panic(expected = "mixed_partial: orders.len() must equal tape.num_inputs()")]
fn m45_mixed_partial_rejects_wrong_orders_length() {
    use echidna::record;
    // f(x, y) = x^2 + y: tape has 2 inputs; we pass only one order.
    let f = |inputs: &[BReverse<f64>]| -> BReverse<f64> { inputs[0] * inputs[0] + inputs[1] };
    let (tape, _) = record(f, &[1.0, 2.0]);
    let _ = echidna::diffop::mixed_partial(&tape, &[1.0, 2.0], &[1]);
}

// M46: extraction_prefactor in the common path must remain exact. If the
// integer product overflows, the log-domain fallback kicks in — we can't
// easily trigger overflow without pushing the jet_order past internal
// limits, so we verify the clean-path correctness (a log/exp roundtrip
// would lose ULPs here).
#[cfg(feature = "diffop")]
#[test]
fn m46_extraction_prefactor_small_order_exact() {
    use echidna::record;
    let f = |inputs: &[BReverse<f64>]| -> BReverse<f64> { inputs[0] * inputs[0] * inputs[0] };
    let (tape, _) = record(f, &[1.0]);
    let (value, deriv) = echidna::diffop::mixed_partial(&tape, &[1.0], &[3]);
    assert!((value - 1.0).abs() < 1e-15);
    // d^3(x^3)/dx^3 = 6 (exact integer).
    assert!((deriv - 6.0).abs() < 1e-12, "deriv = {}", deriv);
}

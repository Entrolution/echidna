//! Phase 8 Commit 4 regressions — miscellaneous hardening (L13, L26, L36,
//! L37, L38, L39, L40).

#![cfg(feature = "bytecode")]

use echidna::BytecodeTape;

// L13 was dropped: the original finding recommended excluding non-finite
// switching values from `active_kinks`, but an earlier bug-hunt cycle
// (regression_24_nan_switching_value_is_not_smooth in tests/nonsmooth.rs)
// explicitly pinned the opposite contract — NaN switching values are
// treated as active because an upstream numerical blow-up means the
// branch cannot be decided, so conservative inclusion is safer than
// silent exclusion. No code change here.

// L26: set_outputs must bounds-check indices and panic with an actionable
// message instead of silently allowing out-of-range output slots.
#[test]
#[should_panic(expected = "set_outputs: indices[")]
fn l26_set_outputs_panics_on_out_of_bounds() {
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let _ = tape.new_input(1.0); // index 0, values.len() becomes 1
    tape.set_outputs(&[5]); // way out of range
}

#[test]
fn l26_set_outputs_accepts_duplicate_indices() {
    // Duplicates are legitimate — two outputs sharing a value — so the
    // bounds-check must not also reject duplicates.
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.new_input(1.0);
    tape.set_outputs(&[idx, idx]);
    assert_eq!(tape.num_outputs(), 2);
}

// L36: hessian_diagonal_with_buf on a zero-input tape must return the
// constant output value (mirrors the M37 fix applied to
// hessian / hessian_vec / sparse_hessian).
#[cfg(feature = "stde")]
#[test]
fn l36_hessian_diagonal_empty_input_returns_constant() {
    use echidna::stde::hessian_diagonal;
    let mut tape = BytecodeTape::<f64>::with_capacity(4);
    let idx = tape.push_const(7.25);
    tape.set_output(idx);
    tape.set_outputs(&[idx]);
    let (value, diag) = hessian_diagonal(&tape, &[]);
    assert!((value - 7.25).abs() < 1e-15, "value = {}", value);
    assert_eq!(diag.len(), 0);
}

// L37: ndarray wrappers copy inputs element-wise so all memory layouts are
// accepted. The wrapper's public signature is `&Array1<F>`, and safe
// constructors always produce contiguous `Array1`s — so constructing a
// genuinely non-contiguous `Array1` to drive the contiguous/non-contiguous
// discrimination would require `from_shape_vec_unchecked` (unsafe) or a
// signature widening to `ArrayView1<F>` (breaking).
//
// This regression is therefore a smoke test: it verifies the helper
// produces correct results on a well-formed input, ensuring the rewrite
// didn't regress the happy path. True non-contiguous coverage would
// require changing the wrapper's input type, which is out of scope.
#[cfg(feature = "ndarray")]
#[test]
fn l37_ndarray_wrappers_accept_owned_arrays() {
    use echidna::ndarray_support::grad_ndarray;
    use echidna::BReverse;
    use ndarray::Array1;

    let x: Array1<f64> = Array1::from_vec(vec![1.0, 5.0, 3.0]);
    let f = |v: &[BReverse<f64>]| -> BReverse<f64> { v[0] * v[0] + v[1] + v[2] * v[2] * v[2] };
    let g = grad_ndarray(f, &x);
    assert!((g[0] - 2.0).abs() < 1e-12, "∂f/∂a at a=1 → 2, got {}", g[0]);
    assert!((g[1] - 1.0).abs() < 1e-12, "∂f/∂b → 1, got {}", g[1]);
    assert!(
        (g[2] - 27.0).abs() < 1e-12,
        "∂f/∂c at c=3 → 27, got {}",
        g[2]
    );
}

// L38: eval_dyn with an empty plan returns `value = 0`, not the previous
// `Σx[i]` placeholder. Exercised indirectly via mixed_partial with a
// single-input tape and all-zero orders, which takes a non-empty plan
// path — we can't construct an empty-plan DiffOp from public API alone.
// The regression here is that normal mixed_partial still returns a
// correct value under the changed default.
#[cfg(feature = "diffop")]
#[test]
fn l38_mixed_partial_still_returns_correct_value() {
    use echidna::diffop::mixed_partial;
    use echidna::{record, BReverse};

    let f = |v: &[BReverse<f64>]| -> BReverse<f64> { v[0] * v[0] + v[0] };
    let (tape, _) = record(f, &[2.0]);
    let (value, _d) = mixed_partial(&tape, &[2.0], &[1]);
    assert!(
        (value - 6.0).abs() < 1e-12,
        "value (= f(2) = 6) should not be affected by the placeholder change, got {}",
        value
    );
}

// L39: sparse_jacobian_ndarray now returns (outputs, pattern, values).
#[cfg(feature = "ndarray")]
#[test]
fn l39_sparse_jacobian_ndarray_returns_outputs() {
    use echidna::ndarray_support::sparse_jacobian_ndarray;
    use echidna::BReverse;
    use ndarray::Array1;

    // f(x, y) = [x, y, x+y] — diagonal plus a sum row.
    let f = |v: &[BReverse<f64>]| -> Vec<BReverse<f64>> { vec![v[0], v[1], v[0] + v[1]] };
    let x = Array1::from_vec(vec![3.0, 5.0]);
    let (outputs, _pattern, _values) = sparse_jacobian_ndarray(f, &x);
    assert_eq!(outputs.len(), 3);
    assert!((outputs[0] - 3.0).abs() < 1e-12);
    assert!((outputs[1] - 5.0).abs() < 1e-12);
    assert!((outputs[2] - 8.0).abs() < 1e-12);
}

// L40: mixed_partial doc used to claim a panic on all-zero orders, but
// the code returns f(x). Align the test with the real contract.
#[cfg(feature = "diffop")]
#[test]
fn l40_mixed_partial_all_zero_orders_returns_value() {
    use echidna::diffop::mixed_partial;
    use echidna::{record, BReverse};

    let f = |v: &[BReverse<f64>]| -> BReverse<f64> { v[0] * v[0] * v[0] };
    let (tape, _) = record(f, &[2.0]);
    let (value, deriv) = mixed_partial(&tape, &[2.0], &[0]);
    assert!(
        (value - 8.0).abs() < 1e-12,
        "value = f(2) = 8, got {}",
        value
    );
    // An all-zero multi-index is the identity operator → derivative equals value.
    assert!(
        (deriv - 8.0).abs() < 1e-12,
        "all-zero orders → deriv = f(x) = 8, got {}",
        deriv
    );
}

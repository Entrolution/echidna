//! Pins `taylor_hypot`'s leading-zero handling: a zero-scale prefix peels
//! away as `|t|^m · hypot(a(t)/t^m, b(t)/t^m)`, and NaN/Inf at the first
//! signal index must surface rather than being swallowed by the peel.
//! Expected values were captured from the one-order-at-a-time recursion
//! this convention originated with; the peel must reproduce them exactly.

#![cfg(feature = "taylor")]

use echidna::Taylor;

const N: usize = 6;

fn hypot_jet(a: [f64; N], b: [f64; N]) -> [f64; N] {
    let ta = Taylor::<f64, N>::new(a);
    let tb = Taylor::<f64, N>::new(b);
    ta.hypot(tb).coeffs
}

fn assert_jet_eq(actual: [f64; N], expected: [f64; N], what: &str) {
    for k in 0..N {
        let (a, e) = (actual[k], expected[k]);
        assert!(
            (a.is_nan() && e.is_nan()) || a == e,
            "{what}: coeff {k}: expected {e}, got {a} (full: {actual:?})"
        );
    }
}

#[test]
fn peels_shared_leading_zeros() {
    assert_jet_eq(
        hypot_jet([0., 0., 1., 2., 0., 0.], [0., 0., 0., 1., 0., 0.]),
        [0., 0., 1., 2., 0.5, -1.0],
        "m = 2",
    );
}

#[test]
fn nan_at_first_signal_index_poisons_tail() {
    assert_jet_eq(
        hypot_jet([0., f64::NAN, 1., 0., 0., 0.], [0.; N]),
        [0., f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        "NaN at m = 1",
    );
}

#[test]
fn nan_paired_with_inf_at_first_signal_index_is_inf() {
    // IEEE hypot(±Inf, NaN) = +Inf: the override must survive the peel at
    // the primal slot, with the derivative tail staying NaN.
    assert_jet_eq(
        hypot_jet(
            [0., f64::NAN, 1., 0., 0., 0.],
            [0., f64::INFINITY, 0., 0., 0., 0.],
        ),
        [0., f64::INFINITY, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        "NaN paired with Inf at m = 1",
    );
}

#[test]
fn inf_at_first_signal_index_matches_general_path() {
    assert_jet_eq(
        hypot_jet(
            [0., 0., f64::INFINITY, 1., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
        ),
        [0., 0., f64::INFINITY, f64::NAN, f64::NAN, f64::NAN],
        "Inf at m = 2",
    );
}

#[test]
fn signal_only_at_last_order() {
    assert_jet_eq(
        hypot_jet([0., 0., 0., 0., 0., 3.], [0.; N]),
        [0., 0., 0., 0., 0., 3.],
        "m = N - 1",
    );
}

#[test]
fn both_all_zero_stays_zero() {
    assert_jet_eq(hypot_jet([0.; N], [0.; N]), [0.; N], "all-zero");
}

#[test]
fn nondegenerate_path_unchanged() {
    assert_jet_eq(
        hypot_jet([1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]),
        [
            7.0710678118654755,
            8.20243866176395,
            9.384721199907856,
            10.60976955617793,
            11.870557917596285,
            13.161059969448717,
        ],
        "m = 0",
    );
}

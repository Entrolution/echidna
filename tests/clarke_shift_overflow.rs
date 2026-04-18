//! `clarke_jacobian`'s `1usize << k` overflow protection.
//!
//! The combo enumeration enumerates `2^k` sign patterns for `k` active
//! kinks. Before the fix, a caller could pass `max_active_kinks = Some(64)`
//! (or `≥ usize::BITS` on 32-bit) and, if the tape actually had that many
//! kinks, the `1usize << k` shift would panic in debug and wrap to 1 in
//! release — silently enumerating only a single combo. Post-fix, the limit
//! is hard-capped at `usize::BITS - 1`, and excess kinks return a
//! `TooManyKinks` error.

#![cfg(feature = "bytecode")]

use echidna::nonsmooth::ClarkeError;
use echidna::{record_multi, BReverse};
use num_traits::Float;

/// Construct a tape with many active kinks (65 independent `abs` terms) and
/// invoke `clarke_jacobian` with `max_active_kinks = Some(64)` on 64-bit —
/// the shift would have overflowed. The fix caps the limit at 63.
#[test]
fn clarke_shift_overflow_is_rejected() {
    let n_inputs = 65;
    let inputs = vec![0.0_f64; n_inputs]; // every abs kink lives at x = 0

    let (mut tape, _) = record_multi(
        |xs: &[BReverse<f64>]| {
            // Each output is abs(x_i), all with the same kink at 0.
            (0..n_inputs).map(|i| xs[i].abs()).collect()
        },
        &inputs,
    );

    // Ask for up to 64 kinks. The actual count is 65, so we expect
    // TooManyKinks — but critically, we must not panic.
    let result = tape.clarke_jacobian(&inputs, 1e-10, Some(64));

    match result {
        Err(ClarkeError::TooManyKinks { count, limit }) => {
            assert_eq!(count, 65);
            // Limit was clamped from 64 to 63 = usize::BITS - 1 on 64-bit.
            let expected_limit = (usize::BITS as usize) - 1;
            assert_eq!(
                limit, expected_limit,
                "limit should be clamped to usize::BITS - 1"
            );
        }
        Ok(_) => panic!("should have errored with TooManyKinks"),
    }
}

/// Sanity: the normal case with small k still works.
#[test]
fn clarke_small_k_still_works() {
    let inputs = vec![0.0_f64, 0.0_f64];
    let (mut tape, _) = record_multi(
        |xs: &[BReverse<f64>]| vec![xs[0].abs() + xs[1].abs()],
        &inputs,
    );
    let result = tape.clarke_jacobian(&inputs, 1e-10, None);
    assert!(result.is_ok(), "small k should succeed");
    let (_info, jacobians) = result.unwrap();
    // 2 kinks → 4 combos
    assert_eq!(jacobians.len(), 4);
}

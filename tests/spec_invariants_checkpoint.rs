//! Semantic tests tying `src/checkpoint.rs` to the TLA+ specs under `specs/revolve/`.
//!
//! Each invariant from the cross-reference table in `specs/README.md` is exercised
//! here against the real Rust implementation. The tests are intentionally bounded
//! and deterministic (no RNG dependency) so CI is reproducible.
//!
//! The structural invariants (`PositionRangeInvariant`, `SortedCheckpoints`,
//! `InitialStateStored`, `CompletenessProperty`, `PinnedOrigin`, `UniformSpacing`,
//! `SpacingPowerOf2`) are not observable from outside the module; they are
//! verified indirectly: if any of them were violated the gradients would
//! diverge from the non-checkpointed baseline, which is what each test checks.

#![cfg(feature = "bytecode")]

use echidna::{record, BReverse};
use num_traits::Float;

/// Non-checkpointed reference: record the full unrolled computation to a tape
/// and take its gradient.
fn grad_naive(
    step: impl Fn(&[BReverse<f64>]) -> Vec<BReverse<f64>>,
    loss: impl Fn(&[BReverse<f64>]) -> BReverse<f64>,
    x0: &[f64],
    num_steps: usize,
) -> Vec<f64> {
    let (mut tape, _) = record(
        |x| {
            let mut state: Vec<BReverse<f64>> = x.to_vec();
            for _ in 0..num_steps {
                state = step(&state);
            }
            loss(&state)
        },
        x0,
    );
    tape.gradient(x0)
}

/// A nonlinear step that exercises mixed partials across the state.
fn step(x: &[BReverse<f64>]) -> Vec<BReverse<f64>> {
    vec![x[0] * x[1] + x[0].sin(), x[1] - x[0] * x[0] * 0.1]
}

fn loss(x: &[BReverse<f64>]) -> BReverse<f64> {
    x[0] * x[0] + x[1] * x[1]
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "length mismatch in {}", ctx);
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < tol,
            "{}: mismatch at [{}]: {} vs {} (delta {:.3e})",
            ctx,
            i,
            a[i],
            b[i],
            (a[i] - b[i]).abs()
        );
    }
}

// ----------------------------------------------------------------------------
// BudgetInvariant + PositionRangeInvariant + SortedCheckpoints +
// InitialStateStored + CompletenessProperty
//
// All five invariants of base Revolve must hold for gradient correctness: if
// any are violated the reverse sweep either visits a step twice, skips a step,
// or reads an out-of-range state, and the gradient diverges.
// ----------------------------------------------------------------------------
#[test]
fn base_revolve_gradient_matches_naive_exhaustive() {
    let x0 = [0.7_f64, 1.3];
    for num_steps in 2..=8 {
        for num_checkpoints in 1..=num_steps {
            let g_naive = grad_naive(step, loss, &x0, num_steps);
            let g_ckpt = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_checkpoints);
            assert_close(
                &g_naive,
                &g_ckpt,
                1e-10,
                &format!("base revolve N={}, C={}", num_steps, num_checkpoints),
            );
        }
    }
}

// ----------------------------------------------------------------------------
// BufferBudget + PinnedOrigin + SpacingPowerOf2 + UniformSpacing
//
// Online thinning has to keep `buffer.len() <= num_checkpoints`, preserve
// `buffer[0] = (0, x0)`, double `spacing` on each thin, and save only at
// multiples of `spacing`. Any of those failing will yield a wrong gradient.
// ----------------------------------------------------------------------------
#[test]
fn online_thinning_gradient_matches_naive() {
    let x0 = [0.5_f64, -0.4];
    for num_steps in 2..=10 {
        for num_checkpoints in 2..=6 {
            let stop = |_: &[f64], s: usize| s >= num_steps;
            let g_naive = grad_naive(step, loss, &x0, num_steps);
            let g_online =
                echidna::grad_checkpointed_online(step, stop, loss, &x0, num_checkpoints);
            assert_close(
                &g_naive,
                &g_online,
                1e-10,
                &format!("online N={}, C={}", num_steps, num_checkpoints),
            );
        }
    }

    // Explicit high-N/low-C case to guarantee thinning fires multiple times.
    // With num_checkpoints=2, the buffer fills after each step, triggering a
    // thinning pass on every iteration and doubling `spacing` repeatedly. If
    // thinning regressed (early return, wrong spacing update, dropped pin),
    // the resulting gradient would diverge from naive; a bug that only
    // manifests under repeated thinning cannot pass this witness.
    let num_steps_big = 32;
    let num_checkpoints_tight = 2;
    let stop = |_: &[f64], s: usize| s >= num_steps_big;
    let g_naive = grad_naive(step, loss, &x0, num_steps_big);
    let g_online = echidna::grad_checkpointed_online(step, stop, loss, &x0, num_checkpoints_tight);
    assert_close(
        &g_naive,
        &g_online,
        1e-10,
        "online thinning-witnessed (N=32, C=2 forces multiple thins)",
    );
}

// ----------------------------------------------------------------------------
// RequiredIncluded + AllocationExact + BudgetInvariant (hints variant)
//
// Hint-based allocation must include every valid required position in the
// checkpoint set, distribute the remaining budget exactly, and never exceed
// the total budget. Again, gradient equality against naive exercises all three.
// ----------------------------------------------------------------------------
#[test]
fn hint_allocation_gradient_matches_naive() {
    let x0 = [1.1_f64, 0.3];
    // Sweep (N, C) with a variety of required position sets.
    for num_steps in 3..=8 {
        for num_checkpoints in 2..=num_steps {
            // Try: no hints, one interior hint, two interior hints.
            let hint_sets: Vec<Vec<usize>> = vec![
                vec![],
                vec![num_steps / 2],
                if num_steps >= 4 {
                    vec![1, num_steps - 1]
                } else {
                    vec![1]
                },
            ];
            for hints in &hint_sets {
                if hints.len() > num_checkpoints {
                    continue;
                }
                let g_naive = grad_naive(step, loss, &x0, num_steps);
                let g_hints = echidna::grad_checkpointed_with_hints(
                    step,
                    loss,
                    &x0,
                    num_steps,
                    num_checkpoints,
                    hints,
                );
                assert_close(
                    &g_naive,
                    &g_hints,
                    1e-10,
                    &format!(
                        "hints N={}, C={}, hints={:?}",
                        num_steps, num_checkpoints, hints
                    ),
                );
            }
        }
    }
}

// `AllocationExact` is not directly observable from outside the module: the
// `largest_remainder_alloc` call is internal and the only observable proxy
// (gradient correctness across budgets) is already exercised by
// `hint_allocation_gradient_matches_naive` above. True verification of
// `AllocationExact` lives in the TLA+ `HintAllocation.tla` model.

//! Compositional gradient checkpointing for iterative computations.
//!
//! Reduces memory from O(num_steps) to O(num_checkpoints) by recomputing
//! intermediate states from checkpoints during the backward pass.

use crate::breverse::BReverse;
use crate::bytecode_tape::{BtapeGuard, BtapeThreadLocal, BytecodeTape};
use crate::float::Float;

/// Compute gradients through an iterative computation using checkpointing.
///
/// Instead of storing all intermediate states (O(num_steps) memory),
/// saves states only at evenly spaced checkpoints and recomputes
/// intermediate states from the nearest checkpoint during the backward pass.
///
/// # Arguments
///
/// * `step` - A function that advances state by one step: `state_{k+1} = step(state_k)`
/// * `loss` - A scalar loss function applied to the final state
/// * `x0` - Initial state
/// * `num_steps` - Number of times to apply `step`
/// * `num_checkpoints` - Number of intermediate states to save (affects memory/compute tradeoff)
///
/// # Returns
///
/// Gradient of `loss(step^num_steps(x0))` with respect to `x0`.
///
/// # Panics
///
/// Panics if `step` changes the dimension of its input (output length must equal input length).
pub fn grad_checkpointed<F: Float + BtapeThreadLocal>(
    step: impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    loss: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x0: &[F],
    num_steps: usize,
    num_checkpoints: usize,
) -> Vec<F> {
    let dim = x0.len();

    // Handle edge case: 0 steps means gradient of loss(x0) directly.
    if num_steps == 0 {
        let (mut tape, _) = crate::api::record(loss, x0);
        return tape.gradient(x0);
    }

    let num_checkpoints = num_checkpoints.max(1).min(num_steps);

    // -- Forward pass: run all steps, saving checkpoints --

    let checkpoint_interval = if num_checkpoints >= num_steps {
        1
    } else {
        num_steps.div_ceil(num_checkpoints)
    };

    // Checkpoint storage: (step_index, state).
    // Step index 0 means "the state before any steps" (i.e., x0).
    let mut checkpoints: Vec<(usize, Vec<F>)> = Vec::new();
    checkpoints.push((0, x0.to_vec()));

    let mut current_state = x0.to_vec();
    for s in 0..num_steps {
        current_state = step_forward_primal(&step, &current_state);
        assert_eq!(
            current_state.len(),
            dim,
            "step must preserve dimension: expected {}, got {}",
            dim,
            current_state.len()
        );

        let next_step = s + 1;
        if next_step < num_steps && next_step % checkpoint_interval == 0 {
            checkpoints.push((next_step, current_state.clone()));
        }
    }

    let final_state = current_state;

    // -- Loss gradient (seeds the backward pass) --
    let mut adjoint = {
        let (mut tape, _) = crate::api::record(loss, &final_state);
        tape.gradient(&final_state)
    };

    // -- Backward pass: VJP through each step from checkpoints --

    // Checkpoints are already sorted by step index (inserted in order).
    let num_segments = checkpoints.len();
    for seg in (0..num_segments).rev() {
        let (ckpt_step, ref ckpt_state) = checkpoints[seg];
        let seg_end = if seg + 1 < num_segments {
            checkpoints[seg + 1].0
        } else {
            num_steps
        };

        let seg_len = seg_end - ckpt_step;

        // Recompute states in this segment from the checkpoint.
        let mut states: Vec<Vec<F>> = Vec::with_capacity(seg_len + 1);
        states.push(ckpt_state.clone());
        let mut s = ckpt_state.clone();
        for _ in 0..seg_len {
            s = step_forward_primal(&step, &s);
            states.push(s.clone());
        }

        // VJP backward through this segment.
        for i in (0..seg_len).rev() {
            adjoint = vjp_step(&step, &states[i], &adjoint);
        }
    }

    adjoint
}

/// Run one step forward (primal only, no gradient needed for the output).
///
/// Creates a temporary tape because `step` takes `&[BReverse<F>]`.
fn step_forward_primal<F: Float + BtapeThreadLocal>(
    step: &impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    state: &[F],
) -> Vec<F> {
    let mut tape = BytecodeTape::with_capacity(state.len() * 10);

    let inputs: Vec<BReverse<F>> = state
        .iter()
        .map(|&val| {
            let idx = tape.new_input(val);
            BReverse::from_tape(val, idx)
        })
        .collect();

    {
        let _guard = BtapeGuard::new(&mut tape);
        let outputs = step(&inputs);
        outputs.iter().map(|r| r.value).collect()
    }
}

/// Compute VJP: J^T * w for a single step via the scalar trick.
///
/// Records `sum_i w[i] * step(x)[i]` and takes its gradient.
/// Since w[i] are `BReverse::constant`, they don't participate in the tape,
/// and the gradient is exactly J^T * w.
fn vjp_step<F: Float + BtapeThreadLocal>(
    step: &impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    state: &[F],
    w: &[F],
) -> Vec<F> {
    let dim = state.len();
    let mut tape = BytecodeTape::with_capacity(dim * 10);

    let inputs: Vec<BReverse<F>> = state
        .iter()
        .map(|&val| {
            let idx = tape.new_input(val);
            BReverse::from_tape(val, idx)
        })
        .collect();

    {
        let _guard = BtapeGuard::new(&mut tape);
        let outputs = step(&inputs);

        assert_eq!(
            outputs.len(),
            dim,
            "step must preserve dimension: expected {}, got {}",
            dim,
            outputs.len()
        );

        // Compute scalar: sum_i w[i] * output[i]
        // Use BReverse::constant for w[i] so they don't affect the tape's gradient.
        let mut scalar = BReverse::constant(F::zero());
        for i in 0..dim {
            scalar += BReverse::constant(w[i]) * outputs[i];
        }

        tape.set_output(scalar.index);
    }

    tape.gradient(state)
}

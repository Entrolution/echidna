//! Compositional gradient checkpointing for iterative computations.
//!
//! Uses the optimal binomial (Revolve) checkpointing schedule
//! (Griewank & Walther, 2000) to minimize recomputation for a given
//! number of checkpoint slots.

use crate::breverse::BReverse;
use crate::bytecode_tape::{BtapeGuard, BtapeThreadLocal, BytecodeTape};
use crate::float::Float;

/// Compute gradients through an iterative computation using checkpointing.
///
/// Uses the Revolve (optimal binomial) checkpointing schedule to minimize
/// recomputation given the available checkpoint slots.
///
/// # Arguments
///
/// * `step` - A function that advances state by one step: `state_{k+1} = step(state_k)`
/// * `loss` - A scalar loss function applied to the final state
/// * `x0` - Initial state
/// * `num_steps` - Number of times to apply `step`
/// * `num_checkpoints` - Number of checkpoint slots (affects memory/compute tradeoff)
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

    // Compute optimal checkpoint positions using Revolve schedule
    let checkpoint_positions = revolve_schedule(num_steps, num_checkpoints);

    // -- Forward pass: run all steps, saving checkpoints --
    let mut checkpoints: Vec<(usize, Vec<F>)> = Vec::with_capacity(num_checkpoints + 1);
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
        if next_step < num_steps && checkpoint_positions.contains(&next_step) {
            checkpoints.push((next_step, current_state.clone()));
        }
    }

    let final_state = current_state;

    backward_from_checkpoints(&step, loss, &final_state, &checkpoints, num_steps)
}

// ══════════════════════════════════════════════
//  Online checkpointing (R9a)
// ══════════════════════════════════════════════

/// Compute gradients through an iterative computation with unknown step count.
///
/// Uses periodic thinning: maintains a buffer of checkpoint slots, and when the
/// buffer fills, discards every other checkpoint and doubles the spacing. This
/// guarantees O(log(N)) recomputation overhead for N total steps using only
/// `num_checkpoints` memory slots.
///
/// # Arguments
///
/// * `step` - A function that advances state by one step: `state_{k+1} = step(state_k)`
/// * `stop` - Predicate `stop(state, step_index)` returning `true` to stop iteration.
///   Step 0 is the initial state; `stop` is first called after step 1 with `(state_1, 1)`.
/// * `loss` - A scalar loss function applied to the final state
/// * `x0` - Initial state
/// * `num_checkpoints` - Number of checkpoint slots (must be >= 2)
///
/// # Returns
///
/// Gradient of `loss(step^N(x0))` with respect to `x0`, where N is determined by `stop`.
///
/// # Panics
///
/// Panics if `num_checkpoints < 2` or if `step` changes the dimension of its input.
pub fn grad_checkpointed_online<F: Float + BtapeThreadLocal>(
    step: impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    stop: impl Fn(&[F], usize) -> bool,
    loss: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x0: &[F],
    num_checkpoints: usize,
) -> Vec<F> {
    assert!(
        num_checkpoints >= 2,
        "online checkpointing requires at least 2 checkpoint slots, got {}",
        num_checkpoints,
    );

    let dim = x0.len();

    // Edge case: stop at step 0 means gradient of loss(x0) directly.
    if stop(x0, 0) {
        let (mut tape, _) = crate::api::record(loss, x0);
        return tape.gradient(x0);
    }

    // Checkpoint buffer: buffer[0] is always (0, x0), pinned during thinning.
    let mut buffer: Vec<(usize, Vec<F>)> = Vec::with_capacity(num_checkpoints);
    buffer.push((0, x0.to_vec()));

    let mut spacing = 1usize;
    let mut current_state = x0.to_vec();
    let mut step_index = 0usize;

    loop {
        // Advance one step.
        current_state = step_forward_primal(&step, &current_state);
        step_index += 1;
        assert_eq!(
            current_state.len(),
            dim,
            "step must preserve dimension: expected {}, got {}",
            dim,
            current_state.len()
        );

        // Save checkpoint if on the spacing grid.
        if step_index % spacing == 0 {
            buffer.push((step_index, current_state.clone()));
        }

        // Check stop condition.
        if stop(&current_state, step_index) {
            break;
        }

        // Thin when buffer is full.
        if buffer.len() == num_checkpoints {
            // Keep buffer[0] (pinned). Among buffer[1..], keep even-indexed entries.
            let tail: Vec<(usize, Vec<F>)> = buffer[1..].iter().step_by(2).cloned().collect();
            buffer.truncate(1);
            buffer.extend(tail);
            spacing *= 2;
        }
    }

    let num_steps = step_index;
    let final_state = current_state;

    backward_from_checkpoints(&step, loss, &final_state, &buffer, num_steps)
}

// ══════════════════════════════════════════════
//  Checkpoint placement hints (R9c)
// ══════════════════════════════════════════════

/// Compute gradients with user-specified required checkpoint positions.
///
/// Distributes remaining checkpoint slots optimally (via Revolve) across the
/// sub-intervals defined by the required positions. Required positions are always
/// stored as checkpoints; the remaining slots are distributed proportionally to
/// sub-interval length.
///
/// # Arguments
///
/// * `step` - A function that advances state by one step
/// * `loss` - A scalar loss function applied to the final state
/// * `x0` - Initial state
/// * `num_steps` - Number of times to apply `step`
/// * `num_checkpoints` - Total number of checkpoint slots (must be >= required positions count)
/// * `required_positions` - Step indices that must be checkpointed. Positions outside
///   `[1, num_steps-1]` are silently ignored.
///
/// # Panics
///
/// Panics if the number of valid required positions exceeds `num_checkpoints`.
pub fn grad_checkpointed_with_hints<F: Float + BtapeThreadLocal>(
    step: impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    loss: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x0: &[F],
    num_steps: usize,
    num_checkpoints: usize,
    required_positions: &[usize],
) -> Vec<F> {
    let dim = x0.len();

    if num_steps == 0 {
        let (mut tape, _) = crate::api::record(loss, x0);
        return tape.gradient(x0);
    }

    let num_checkpoints = num_checkpoints.max(1).min(num_steps);

    // Filter, sort, dedup required positions to valid range [1, num_steps-1].
    let mut required: Vec<usize> = required_positions
        .iter()
        .copied()
        .filter(|&p| p >= 1 && p < num_steps)
        .collect();
    required.sort_unstable();
    required.dedup();

    assert!(
        required.len() <= num_checkpoints,
        "required positions ({}) exceed available checkpoint slots ({})",
        required.len(),
        num_checkpoints,
    );

    // Free slots after allocating required positions.
    let free = num_checkpoints.saturating_sub(required.len());

    // Build sub-intervals: boundaries are [0, r1, r2, ..., rk, num_steps].
    let mut boundaries = Vec::with_capacity(required.len() + 2);
    boundaries.push(0);
    boundaries.extend_from_slice(&required);
    boundaries.push(num_steps);
    boundaries.dedup(); // In case required contains 0 or num_steps somehow.

    // Compute sub-interval lengths.
    let intervals: Vec<(usize, usize)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    let interval_lengths: Vec<usize> = intervals.iter().map(|(s, e)| e - s).collect();
    let total_len: usize = interval_lengths.iter().sum();

    // Distribute free slots proportionally using largest-remainder method.
    let slot_alloc = largest_remainder_alloc(free, &interval_lengths, total_len);

    // Run Revolve on each sub-interval and merge all positions.
    let mut all_positions: Vec<usize> = required.clone();
    for (i, &(start, end)) in intervals.iter().enumerate() {
        let sub_steps = end - start;
        let sub_slots = slot_alloc[i];
        if sub_steps > 1 && sub_slots > 0 {
            let sub_positions = revolve_schedule(sub_steps, sub_slots);
            // Shift positions to global coordinates.
            all_positions.extend(sub_positions.iter().map(|&p| p + start));
        }
    }
    all_positions.sort_unstable();
    all_positions.dedup();

    // Forward pass using the merged position list.
    let mut checkpoints: Vec<(usize, Vec<F>)> = Vec::with_capacity(all_positions.len() + 1);
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
        if next_step < num_steps && all_positions.contains(&next_step) {
            checkpoints.push((next_step, current_state.clone()));
        }
    }

    let final_state = current_state;
    backward_from_checkpoints(&step, loss, &final_state, &checkpoints, num_steps)
}

/// Distribute `total` items across buckets proportionally to `weights`,
/// using the largest-remainder method for rounding.
fn largest_remainder_alloc(total: usize, weights: &[usize], weight_sum: usize) -> Vec<usize> {
    if weight_sum == 0 || weights.is_empty() {
        return vec![0; weights.len()];
    }

    // Integer quotients.
    let mut alloc: Vec<usize> = weights.iter().map(|&w| (w * total) / weight_sum).collect();
    let allocated: usize = alloc.iter().sum();
    let mut remaining = total - allocated;

    if remaining > 0 {
        // Compute remainders and sort by descending remainder.
        let mut remainders: Vec<(usize, f64)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let exact = (w as f64 * total as f64) / weight_sum as f64;
                (i, exact - alloc[i] as f64)
            })
            .collect();
        remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, _) in remainders {
            if remaining == 0 {
                break;
            }
            alloc[idx] += 1;
            remaining -= 1;
        }
    }

    alloc
}

// ══════════════════════════════════════════════
//  Disk-backed checkpointing (R9b)
// ══════════════════════════════════════════════

/// Compute gradients using disk-backed checkpointing for large state vectors.
///
/// Stores checkpoint states as raw binary files on disk instead of in memory.
/// Uses the same Revolve schedule as [`grad_checkpointed`]. Checkpoint files
/// are cleaned up on completion and on panic (via a Drop guard).
///
/// # Arguments
///
/// * `step` - A function that advances state by one step
/// * `loss` - A scalar loss function applied to the final state
/// * `x0` - Initial state
/// * `num_steps` - Number of times to apply `step`
/// * `num_checkpoints` - Number of checkpoint slots
/// * `dir` - Directory to store checkpoint files. Must exist.
///
/// # Safety considerations
///
/// Uses raw byte transmutation for serialization. This is safe for all `Float` types
/// (which are `Copy + Sized` and never contain pointers). The checkpoint files are
/// platform-specific binary and not portable.
///
/// # Panics
///
/// Panics if `dir` doesn't exist, on I/O errors, or if `step` changes dimension.
pub fn grad_checkpointed_disk<F: Float + BtapeThreadLocal>(
    step: impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    loss: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x0: &[F],
    num_steps: usize,
    num_checkpoints: usize,
    dir: &std::path::Path,
) -> Vec<F> {
    assert!(
        dir.is_dir(),
        "checkpoint directory does not exist: {}",
        dir.display()
    );

    let dim = x0.len();

    if num_steps == 0 {
        let (mut tape, _) = crate::api::record(loss, x0);
        return tape.gradient(x0);
    }

    let num_checkpoints = num_checkpoints.max(1).min(num_steps);

    // Compute optimal checkpoint positions using Revolve schedule.
    let checkpoint_positions = revolve_schedule(num_steps, num_checkpoints);

    // Drop guard ensures cleanup even on panic.
    let mut guard = DiskCheckpointGuard { files: Vec::new() };

    // Write initial state (step 0).
    let path_0 = dir.join("ckpt_0.bin");
    write_checkpoint(x0, &path_0);
    guard.files.push(path_0);

    // Forward pass: run all steps, saving checkpoints to disk.
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
        if next_step < num_steps && checkpoint_positions.contains(&next_step) {
            let path = dir.join(format!("ckpt_{}.bin", next_step));
            write_checkpoint(&current_state, &path);
            guard.files.push(path);
        }
    }

    let final_state = current_state;

    // Build checkpoint index: sorted list of (step, path) for reading back.
    // Step 0 is always first.
    let mut ckpt_steps: Vec<usize> = vec![0];
    ckpt_steps.extend(checkpoint_positions.iter().filter(|&&p| p < num_steps));
    ckpt_steps.sort_unstable();
    ckpt_steps.dedup();

    // Loss gradient (seeds the backward pass).
    let mut adjoint = {
        let (mut tape, _) = crate::api::record(loss, &final_state);
        tape.gradient(&final_state)
    };

    // Backward pass: iterate segments in reverse, reading checkpoints from disk.
    let num_segments = ckpt_steps.len();
    for seg in (0..num_segments).rev() {
        let ckpt_step = ckpt_steps[seg];
        let seg_end = if seg + 1 < num_segments {
            ckpt_steps[seg + 1]
        } else {
            num_steps
        };

        let seg_len = seg_end - ckpt_step;

        // Read checkpoint state from disk.
        let path = dir.join(format!("ckpt_{}.bin", ckpt_step));
        let ckpt_state = read_checkpoint::<F>(&path, dim);

        // Recompute states in this segment from the checkpoint.
        let mut states: Vec<Vec<F>> = Vec::with_capacity(seg_len + 1);
        states.push(ckpt_state);
        let mut s = states[0].clone();
        for _ in 0..seg_len {
            s = step_forward_primal(&step, &s);
            states.push(s.clone());
        }

        // VJP backward through this segment.
        for i in (0..seg_len).rev() {
            adjoint = vjp_step(&step, &states[i], &adjoint);
        }
    }

    // Explicit cleanup (guard.drop will also attempt it).
    guard.cleanup();

    adjoint
}

fn write_checkpoint<F: Float>(state: &[F], path: &std::path::Path) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), std::mem::size_of_val(state))
    };
    std::fs::write(path, bytes).expect("checkpoint write failed");
}

fn read_checkpoint<F: Float>(path: &std::path::Path, dim: usize) -> Vec<F> {
    let bytes = std::fs::read(path).expect("checkpoint read failed");
    assert_eq!(
        bytes.len(),
        dim * std::mem::size_of::<F>(),
        "checkpoint file size mismatch: expected {}, got {}",
        dim * std::mem::size_of::<F>(),
        bytes.len()
    );
    let mut state = vec![F::zero(); dim];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), state.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    state
}

struct DiskCheckpointGuard {
    files: Vec<std::path::PathBuf>,
}

impl DiskCheckpointGuard {
    fn cleanup(&mut self) {
        for f in self.files.drain(..) {
            let _ = std::fs::remove_file(f);
        }
    }
}

impl Drop for DiskCheckpointGuard {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ══════════════════════════════════════════════
//  Shared backward pass
// ══════════════════════════════════════════════

/// Compute gradients by seeding the loss and VJP-ing backward through checkpoint segments.
///
/// Shared by all checkpointing variants. Each variant implements its own forward pass
/// to produce `(final_state, checkpoints)`, then calls this function for the backward pass.
///
/// `checkpoints` must be sorted by step index and include step 0 (the initial state).
fn backward_from_checkpoints<F: Float + BtapeThreadLocal>(
    step: &impl Fn(&[BReverse<F>]) -> Vec<BReverse<F>>,
    loss: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    final_state: &[F],
    checkpoints: &[(usize, Vec<F>)],
    num_steps: usize,
) -> Vec<F> {
    // Loss gradient (seeds the backward pass).
    let mut adjoint = {
        let (mut tape, _) = crate::api::record(loss, final_state);
        tape.gradient(final_state)
    };

    // Backward pass: VJP through each segment from checkpoints.
    // Checkpoints are sorted by step index (inserted in order).
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
            s = step_forward_primal(step, &s);
            states.push(s.clone());
        }

        // VJP backward through this segment.
        for i in (0..seg_len).rev() {
            adjoint = vjp_step(step, &states[i], &adjoint);
        }
    }

    adjoint
}

// ══════════════════════════════════════════════
//  Revolve schedule computation
// ══════════════════════════════════════════════

/// Compute optimal checkpoint positions using the Revolve algorithm.
///
/// Given `num_steps` forward steps and `num_checkpoints` available slots,
/// returns the set of step indices where checkpoints should be placed.
/// Step 0 (initial state) is always stored implicitly.
fn revolve_schedule(num_steps: usize, num_checkpoints: usize) -> Vec<usize> {
    if num_checkpoints >= num_steps {
        // Store everything
        return (1..num_steps).collect();
    }

    let mut positions = Vec::new();
    schedule_recursive(0, num_steps, num_checkpoints, &mut positions);
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// Recursively determine checkpoint positions for the interval [start, end).
///
/// Places one checkpoint optimally (using the binomial formula), then
/// recurses on the two resulting sub-intervals.
fn schedule_recursive(start: usize, end: usize, checkpoints: usize, positions: &mut Vec<usize>) {
    let steps = end - start;
    if steps <= 1 || checkpoints == 0 {
        return;
    }

    // Find optimal split point: advance by `optimal_advance` steps
    let advance = optimal_advance(steps, checkpoints);
    let split = start + advance;

    if split < end && split > start {
        positions.push(split);

        // First sub-interval: [start, split) with (checkpoints - 1) remaining slots
        // (we used one slot for the checkpoint at `split`)
        schedule_recursive(start, split, checkpoints - 1, positions);

        // Second sub-interval: [split, end) with all checkpoint slots
        // (the checkpoint at `split` has been consumed by the backward pass
        //  before we process the first sub-interval)
        schedule_recursive(split, end, checkpoints, positions);
    }
}

/// Compute the optimal advance distance before placing a checkpoint.
///
/// For `steps` remaining forward steps and `c` checkpoint slots, finds
/// the advance distance that minimizes total recomputation using the
/// binomial formula from Griewank & Walther.
fn optimal_advance(steps: usize, c: usize) -> usize {
    if c == 0 || steps <= 1 {
        return steps;
    }

    // Find smallest t where beta(t, c) >= steps
    let mut t = 1usize;
    while beta(t, c) < steps {
        t += 1;
    }

    // Optimal advance = beta(t-1, c-1)
    if t > 0 && c > 0 {
        beta(t - 1, c - 1).max(1).min(steps - 1)
    } else {
        1
    }
}

/// Binomial coefficient function: beta(s, c) = C(s+c, c).
///
/// Represents the maximum number of steps that can be reversed with
/// `s` forward recomputations and `c` checkpoint slots.
fn beta(s: usize, c: usize) -> usize {
    if c == 0 {
        return s + 1;
    }
    if s == 0 {
        return 1;
    }
    // C(s+c, c) via multiplicative formula to avoid overflow
    let mut result = 1usize;
    for i in 0..c {
        result = result.saturating_mul(s + c - i);
        result /= i + 1;
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_base_cases() {
        // beta(s, 0) = s + 1
        assert_eq!(beta(0, 0), 1);
        assert_eq!(beta(1, 0), 2);
        assert_eq!(beta(5, 0), 6);

        // beta(0, c) = 1
        assert_eq!(beta(0, 1), 1);
        assert_eq!(beta(0, 5), 1);

        // beta(1, 1) = C(2,1) = 2
        assert_eq!(beta(1, 1), 2);

        // beta(2, 2) = C(4,2) = 6
        assert_eq!(beta(2, 2), 6);

        // beta(3, 2) = C(5,2) = 10
        assert_eq!(beta(3, 2), 10);
    }

    #[test]
    fn revolve_schedule_store_all() {
        let positions = revolve_schedule(5, 5);
        assert_eq!(positions, vec![1, 2, 3, 4]);
    }

    #[test]
    fn revolve_schedule_one_checkpoint() {
        let positions = revolve_schedule(4, 1);
        // With 1 checkpoint, should place it somewhere in [1, 3]
        assert!(!positions.is_empty());
        for &p in &positions {
            assert!(p > 0 && p < 4);
        }
    }

    #[test]
    fn revolve_schedule_two_checkpoints() {
        let positions = revolve_schedule(10, 2);
        assert!(!positions.is_empty());
        for &p in &positions {
            assert!(p > 0 && p < 10);
        }
    }
}

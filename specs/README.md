# TLA+ Formal Specifications

Formal specifications for echidna's gradient checkpointing subsystem
(`src/checkpoint.rs`), written in PlusCal and verified with the TLC
model checker.

## What These Specs Verify

These specifications model the **protocol-level correctness** of the
checkpointing algorithms: schedule computation, checkpoint placement,
buffer management, and backward-pass coverage.

They verify properties like:
- Checkpoint budget is never exceeded
- The backward pass covers every step
- The online thinning buffer stays sorted and uniformly spaced
- Hint-based allocation includes all required positions

They do **not** verify:
- Numerical correctness of gradients (covered by Rust tests against
  finite differences)
- Tape recording or VJP computation mechanics
- GPU dispatch or thread-local tape management

## Specs

| Spec | What it models | Rust function |
|------|---------------|---------------|
| `revolve/BinomialBeta.tla` | Shared operators: `Beta(s,c)`, `OptimalAdvance` | `beta()`, `optimal_advance()` |
| `revolve/Revolve.tla` | Base Revolve schedule + forward/backward | `grad_checkpointed()` |
| `revolve/RevolveOnline.tla` | Online thinning with nondeterministic stop | `grad_checkpointed_online()` |
| `revolve/HintAllocation.tla` | Hint-based slot allocation | `grad_checkpointed_with_hints()` |

**Not specified:** `grad_checkpointed_disk()` uses the identical Revolve
schedule as the base variant. Its correctness reduces to base Revolve
correctness plus I/O round-trip safety, which is a Rust type-level
property (`F: Copy + Sized`).

**Excluded edge cases:** `NumSteps=0`, `NumSteps=1`, and
`NumCheckpoints=0` hit guard clauses in the Rust code that
short-circuit before the protocol runs. The specs start at `NumSteps>=2`,
`NumCheckpoints>=1`.

## Prerequisites

- **Java 11+** (tested with Java 21)
- **tla2tools.jar** — download from
  [TLA+ releases](https://github.com/tlaplus/tlaplus/releases) (v1.8.0+)
- Optional: [VS Code TLA+ extension](https://marketplace.visualstudio.com/items?itemName=alygin.vscode-tlaplus)

Place `tla2tools.jar` in this directory (it's gitignored) or anywhere
on your system.

## Running the Model Checker

From the repository root:

```bash
# Base Revolve (fast — seconds)
java -cp specs/tla2tools.jar tlc2.TLC -config specs/revolve/Revolve.cfg specs/revolve/Revolve.tla

# Online thinning (moderate — seconds to minutes depending on bounds)
java -cp specs/tla2tools.jar tlc2.TLC -config specs/revolve/RevolveOnline.cfg specs/revolve/RevolveOnline.tla

# Hint allocation (fast — seconds)
java -cp specs/tla2tools.jar tlc2.TLC -config specs/revolve/HintAllocation.cfg specs/revolve/HintAllocation.tla
```

To override constants (e.g. for parameter sweeps), edit the `.cfg` files
or pass `-D` flags.

## Invariant Cross-Reference

| TLA+ Invariant | Rust Correspondence |
|---------------|---------------------|
| `BudgetInvariant` | `all_positions.truncate(num_checkpoints)` in `grad_checkpointed` |
| `PositionRangeInvariant` | `next_step < num_steps` guard in forward pass |
| `SortedCheckpoints` | `positions.sort_unstable(); positions.dedup()` in `revolve_schedule` |
| `InitialStateStored` | `checkpoints.push((0, x0.to_vec()))` — always first |
| `CompletenessProperty` | Backward loop `for seg in (0..num_segments).rev()` covers all segments |
| `BufferBudget` | `buffer.len() >= num_checkpoints` thinning trigger in `grad_checkpointed_online` |
| `PinnedOrigin` | `buffer[0]` is `(0, x0)`, preserved during thinning |
| `SpacingPowerOf2` | `spacing *= 2` is the only mutation of `spacing` |
| `UniformSpacing` | `step_index.is_multiple_of(spacing)` checkpoint condition |
| `RequiredIncluded` | `all_positions` starts as `required.iter().copied().collect()` in `grad_checkpointed_with_hints` |
| `AllocationExact` | `largest_remainder_alloc` returns values summing to `total` |

## Variable Mapping

| TLA+ Variable | Rust Code |
|--------------|-----------|
| `positions` | `checkpoint_positions: HashSet<usize>` in `grad_checkpointed` |
| `storedCheckpoints` | `checkpoints: Vec<(usize, Vec<F>)>` (step indices only — state vectors abstracted away) |
| `workStack` | Implicit recursion stack in `schedule_recursive` |
| `coveredSteps` | Implicit — the backward loop covers segments sequentially |
| `buffer` | `buffer: Vec<(usize, Vec<F>)>` in `grad_checkpointed_online` (step indices only) |
| `spacing` | `spacing: usize` in `grad_checkpointed_online` |

## Recommended Parameter Sweeps

**Revolve.tla:**
- `NumSteps in {2, 3, 5, 8, 10, 12, 15}`, `NumCheckpoints in {1..NumSteps}`
- All complete in seconds

**RevolveOnline.tla:**
- Quick: `MaxSteps in {5, 10}`, `NumCheckpoints in {2, 3, 4}`
- Thorough: `MaxSteps=20`, `NumCheckpoints in {2, 3, 4, 5}` — minutes each
- Deep: `MaxSteps=50` — hours, run overnight

**HintAllocation.tla:**
- `NumSteps in {4, 6, 8, 10}`, `NumCheckpoints in {2..NumSteps}`
- All complete in seconds

## Design Decisions

- **State vectors are abstracted to step indices.** The specs verify
  bookkeeping (which steps are checkpointed, which segments are covered),
  not numerical values.
- **The backward pass is modelled as coarse atomic segments.** Each
  segment marks its steps as covered in one action, rather than modelling
  the sub-segment recomputation. This is safe because recomputation uses
  only a local buffer, not additional checkpoint slots.
- **The online `stop` predicate is nondeterministic.** This verifies
  invariants hold for all possible stopping points, not just specific ones.
- **Largest-remainder allocation uses integer arithmetic in TLA+.** The
  Rust code uses `f64` for remainders. The spec uses cross-multiplication
  to compare fractions exactly. Tie-breaking may differ — the spec
  verifies structural properties (budget, inclusion) regardless of
  tie-breaking order.

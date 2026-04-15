# TLA+ Formal Specifications

Formal specifications for echidna's core subsystems, verified with the
TLC model checker.

## What These Specs Verify

These specifications model the **protocol-level correctness** of:

**Gradient checkpointing** (`src/checkpoint.rs`):
- Checkpoint budget is never exceeded
- The backward pass covers every step
- The online thinning buffer stays sorted and uniformly spaced
- Hint-based allocation includes all required positions

**Bytecode tape optimizer** (`src/bytecode_tape/optimize.rs`):
- CSE + DCE preserve all structural invariants (DAG order, input prefix,
  valid references)
- CSE remap is monotone and idempotent
- DCE preserves all inputs and the output
- No CSE duplicates remain after optimization
- Optimization is idempotent: `optimize(optimize(tape)) = optimize(tape)`

They do **not** verify:
- Numerical correctness of gradients or evaluations (covered by Rust tests)
- Tape recording or VJP computation mechanics
- GPU dispatch or thread-local tape management
- Powi/Custom opcode special cases (encoding tricks, covered by Rust assertions)

## Specs

### Gradient Checkpointing

| Spec | What it models | Rust function |
|------|---------------|---------------|
| `revolve/BinomialBeta.tla` | Shared operators: `Beta(s,c)`, `OptimalAdvance` | `beta()`, `optimal_advance()` |
| `revolve/Revolve.tla` | Base Revolve schedule + forward/backward | `grad_checkpointed()` |
| `revolve/RevolveOnline.tla` | Online thinning with nondeterministic stop | `grad_checkpointed_online()` |
| `revolve/HintAllocation.tla` | Hint-based slot allocation | `grad_checkpointed_with_hints()` |

### Bytecode Tape Optimizer

| Spec | What it models | Rust function |
|------|---------------|---------------|
| `tape_optimizer/TapeOptimizer.tla` | CSE + DCE as stepwise state machine with 9 invariants | `optimize()` → `cse()` → `dce_compact()` |
| `tape_optimizer/Idempotency.tla` | Idempotency: `optimize(optimize(t)) = optimize(t)` via functional operators | `optimize()` |

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

# Tape optimizer CSE+DCE (~20s at default bounds)
java -XX:+UseParallelGC -cp specs/tla2tools.jar tlc2.TLC -config specs/tape_optimizer/TapeOptimizer.cfg specs/tape_optimizer/TapeOptimizer.tla -workers auto

# Tape optimizer idempotency (~2s at default bounds)
java -XX:+UseParallelGC -cp specs/tla2tools.jar tlc2.TLC -config specs/tape_optimizer/Idempotency.cfg specs/tape_optimizer/Idempotency.tla -workers auto
```

To override constants, edit the `.cfg` files. The tape optimizer specs
benefit from `-workers auto` and `-XX:+UseParallelGC` for larger bounds.

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

### Tape Optimizer

| TLA+ Invariant | Rust Correspondence |
|---------------|---------------------|
| `InputPrefixInvariant` | First `num_inputs` entries are `OpCode::Input` with `UNUSED` args |
| `DAGOrderInvariant` | `arg0 < i` and `arg1 < i` — debug assertion at optimize.rs:255-269 |
| `ValidRefsInvariant` | All `arg_indices` entries `< n` — debug assertion at optimize.rs:240-269 |
| `OutputValidInvariant` | `output_index < n` — debug assertion at optimize.rs:275-280 |
| `InputsPreserved` | Input count unchanged — debug assertion at optimize.rs:290-298 |
| `CSERemapMonotone` | `remap[i] <= i` — CSE only redirects to earlier entries |
| `CSERemapIdempotent` | `remap[remap[i]] = remap[i]` — canonical entries are fixed points |
| `DCEInputsReachable` | `reachable[..num_inputs] = true` in `dce_compact()` |
| `PostOptValid` | Comprehensive structural check at optimize.rs:235-299 |
| `IdempotencyProperty` | `optimize(optimize(tape)) = optimize(tape)` for all valid tapes |

## Variable Mapping

### Gradient Checkpointing

| TLA+ Variable | Rust Code |
|--------------|-----------|
| `positions` | `checkpoint_positions: HashSet<usize>` in `grad_checkpointed` |
| `storedCheckpoints` | `checkpoints: Vec<(usize, Vec<F>)>` (step indices only — state vectors abstracted away) |
| `workStack` | Implicit recursion stack in `schedule_recursive` |
| `coveredSteps` | Implicit — the backward loop covers segments sequentially |
| `buffer` | `buffer: Vec<(usize, Vec<F>)>` in `grad_checkpointed_online` (step indices only) |
| `spacing` | `spacing: usize` in `grad_checkpointed_online` |

### Tape Optimizer

| TLA+ Variable | Rust Code |
|--------------|-----------|
| `opcodes` | `self.opcodes: Vec<OpCode>` (abstracted to 5 kinds) |
| `args` | `self.arg_indices: Vec<[u32; 2]>` |
| `numEntries` | `self.opcodes.len()` / `self.num_variables` |
| `outputIdx` | `self.output_index` |
| `remap` | `remap: Vec<u32>` in `cse()` |
| `seen` | `seen: HashMap<(OpCode, u32, u32), u32>` in `cse()` |
| `scanPos` | Loop variable `i` in `cse()` forward passes |
| `reachable` | `reachable: Vec<bool>` in `dce_compact()` |
| `dceStack` | `stack: Vec<u32>` in `dce_compact()` |
| `writePos` | `write` counter in `dce_compact()` compaction loop |
| `dceRemap` | `remap: Vec<u32>` in `dce_compact()` (distinct from CSE remap) |

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

**TapeOptimizer.tla:**
- Quick: `MaxTapeLen in {3, 4}`, `NumInputs in {1, 2}` — seconds
- Default: `MaxTapeLen=5`, `NumInputs=2` — ~20s (~960K states)
- Thorough: `MaxTapeLen=6`, `NumInputs=2` — minutes to hours
- Skip `NumInputs >= MaxTapeLen` (no operations to optimize)

**Idempotency.tla:**
- Quick: `MaxTapeLen in {3, 4}`, `NumInputs in {1, 2}` — sub-second
- Default: `MaxTapeLen=5`, `NumInputs=2` — ~2s (~55K states)
- Thorough: `MaxTapeLen=6`, `NumInputs=2` — minutes

## Design Decisions

### Gradient Checkpointing

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

### Tape Optimizer

- **5 abstract opcode kinds instead of 44 real opcodes.** `Input`,
  `Const`, `Unary`, `BinComm`, `BinNonComm`. This is the minimal
  partition that captures every structurally distinct code path: the
  optimizer is opcode-aware only for commutative normalization (captured
  by the `BinComm`/`BinNonComm` split). The abstraction is a safe
  overapproximation — the spec's CSE is more aggressive than the real
  CSE.
- **Powi and Custom opcodes are excluded.** Their special-case handling
  (exponent-as-u32, callback index, side table) is encoding detail
  well-covered by Rust debug assertions.
- **Values are abstracted away.** The spec checks structural properties
  only. Numerical correctness is the domain of Rust tests
  (`optimize_rosenbrock`, etc.).
- **Single-output only.** Multi-output adds the same reachability and
  remapping logic but with multiple seeds. Safe to defer because the
  Rust optimizer treats `output_indices` identically to `output_index`.
- **Nondeterministic tape construction via build phase.** Rather than
  pre-computing all valid tapes (combinatorial explosion), the spec
  builds tapes entry by entry. TLC explores all branches naturally.
- **UNUSED sentinel is `MaxTapeLen + 100`.** Guaranteed out of range for
  any valid tape index.
- **Idempotency uses pure functional operators.** CSE and DCE are
  expressed as recursive TLA+ functions (no variables), enabling direct
  equality comparison of `optimize(optimize(t))` vs `optimize(t)`.

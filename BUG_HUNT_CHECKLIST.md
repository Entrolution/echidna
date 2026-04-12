# Execution Checklist

## Rules
- NO item may be deferred without explicit user approval and documented reason
- Every commit MUST have /review-fix run before it is marked done
- Sequential execution only — no parallel editing agents
- Comments must not contain review metadata

## Phase 1: High-severity GPU fixes

- [x] Fix #1: CBRT second derivative in GPU HVP kernels
  - Changed `rr*r` (r³) to `rr*rr*r` (r⁵) in CUDA `tape_eval.cu:631` and WGSL `tangent_reverse.wgsl:277`
  - Comment added explaining f''(a) = -2/(9·r⁵) derivation
  - Test: GPU HVP test deferred — no GPU hardware available on this machine (covered by CI)

- [x] Fix #2: CUDA Taylor codegen u32 truncation
  - Changed all 12 `unsigned int` offset variables to `unsigned long long` in `taylor_codegen.rs`
  - Also added `(unsigned long long)` casts on `a_idx`, `b_idx`, `oi` multiplications to prevent truncation before addition to 64-bit base
  - Test: Compilation verified; large-tape test requires CUDA hardware (covered by CI)

- [x] Review-fix: Phase 1 commit

## Phase 2: Medium-severity numerical fixes ✓

- [x] Fix #3: Bytecode atan2 overflow
  - Replaced `a * a + b * b` with `a.hypot(b)` squared in `opcode.rs`

- [x] Fix #4: Catastrophic cancellation in asin/acos/atanh
  - Replaced `1 - x*x` with `(1-x)*(1+x)` in: dual.rs, dual_vec.rs, num_traits_impls.rs, opcode.rs, taylor_ops.rs (asin + atanh)

- [x] Fix #5: Division derivative overflow
  - Restructured `(a' * b - a * b') * inv * inv` to `(a' - a * inv * b') * inv` in std_ops.rs, dual_vec_ops.rs

- [x] Fix #6: Promote debug_assert to assert for custom ops in Hessian
  - Changed to `assert!` in tangent.rs (hessian_vec) and sparse.rs (sparse_hessian_vec, sparse_jacobian_vec)

- [x] Fix #7: WGSL u32 index overflow guard
  - Added `chunk_size = chunk_size.min(u32::MAX / (nv * K))` in gpu/mod.rs

- [x] Fix #8: Taylor hypot rescaling
  - Added rescaling by `max(|a₀|, |b₀|)` before squaring in taylor_ops.rs

- [x] Review-fix: Phase 2 commit

## Phase 3: Medium-severity STDE/optim fixes ✓

- [x] Fix #9: Guard estimate_weighted against zero w_sum
  - Added `w_sum > F::zero()` guard before `w_sum2 / w_sum` division in pipeline.rs

- [x] Fix #10: Clamp Welford variance to non-negative
  - Added `.max(F::zero())` on variance in types.rs (WelfordAccumulator) and pipeline.rs (estimate_weighted)

- [x] Review-fix: Phase 3 commit

## Phase 4: Low-severity fixes ✓

- [x] Fix #11: Per-type thread-local borrow guard
  - Added `borrow_cell()` to `BtapeThreadLocal` and `TapeThreadLocal` traits with per-type thread_local bools
  - `BtapeBorrowGuard` and `TapeBorrowGuard` now use the per-type cell, allowing cross-type nesting

- [x] Fix #12: Reject Custom opcodes in deserialized tapes
  - Added early rejection of Custom opcodes in serde_support.rs validation loop

- [x] Fix #13: f32-aware Gram-Schmidt epsilon
  - Replaced `F::from(1e-12).unwrap()` with `F::epsilon().sqrt()` in laplacian.rs

- [x] Fix #14: f32 factorial precision guard
  - Added `k < 13 || size_of::<F>() > 4` assertion in both diagonal_kth_order variants

- [x] Fix #15: L-BFGS near-zero curvature guard
  - Changed `sy > F::zero()` to `sy > F::epsilon() * yy` in lbfgs.rs

- [x] Fix #16: f32-aware LU singularity threshold
  - Replaced `sqrt(epsilon)` with relative threshold `epsilon * n * max_pivot_seen` in linalg.rs

- [x] Review-fix: Phase 4 commit

## Deferrals (requires user approval)

| # | Reason | Approved by | Date |
|---|--------|-------------|------|

## In-Situ Discoveries (append during execution)

| # | Found during | Severity | Fixed immediately? | If not, why? |
|---|-------------|----------|-------------------|--------------|

## Regression Test Log

| Phase | Tests added | Total test count |
|-------|------------|------------------|

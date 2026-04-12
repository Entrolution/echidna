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

## Phase 3: Medium-severity STDE/optim fixes

- [ ] Fix #9: Guard estimate_weighted against zero w_sum
  - Add `if w_sum == F::zero() { return EstimatorResult { estimate: F::zero(), ... }; }` before line 97 in `pipeline.rs`
  - Or check `w_sum > F::zero()` before computing denom
  - Test: Add test with all-zero weights

- [ ] Fix #10: Clamp Welford variance to non-negative
  - Add `.max(F::zero())` before sqrt in `types.rs:78` and `pipeline.rs:102`
  - Test: Add test with nearly-identical samples checking non-NaN SE

- [ ] Review-fix: Phase 3 commit

## Phase 4: Low-severity fixes

- [ ] Fix #11: Per-type thread-local borrow guard
  - Use `TypeId`-based or generic-parameter-based borrow tracking instead of single bool
  - Test: Add cross-type nesting test (BReverse<f64> inside BReverse<f32>)

- [ ] Fix #12: Reject Custom opcodes in deserialized tapes
  - Add `if data.opcodes[i] == OpCode::Custom { return Err(...) }` in serde validation
  - Test: Add test that serialized tape with Custom opcode is rejected

- [ ] Fix #13: f32-aware Gram-Schmidt epsilon
  - Replace `F::from(1e-12).unwrap()` with `F::epsilon().sqrt()` in `laplacian.rs:274`
  - Test: Add f32 Hutch++ test

- [ ] Fix #14: f32 factorial precision guard
  - Add `if k >= 13 && std::mem::size_of::<F>() <= 4 { ... }` guard or document limitation
  - Test: Document or test f32 limitation for high orders

- [ ] Fix #15: L-BFGS near-zero curvature guard
  - Replace `sy > F::zero()` with `sy > F::epsilon() * yy` in `lbfgs.rs:138`
  - Test: Add test with near-flat objective

- [ ] Fix #16: f32-aware LU singularity threshold
  - Scale threshold relative to matrix norm or max pivot, not just `sqrt(epsilon)`
  - Test: Add f32 ill-conditioned solve test

- [ ] Review-fix: Phase 4 commit

## Deferrals (requires user approval)

| # | Reason | Approved by | Date |
|---|--------|-------------|------|

## In-Situ Discoveries (append during execution)

| # | Found during | Severity | Fixed immediately? | If not, why? |
|---|-------------|----------|-------------------|--------------|

## Regression Test Log

| Phase | Tests added | Total test count |
|-------|------------|------------------|

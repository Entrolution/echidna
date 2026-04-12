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

## Phase 2: Medium-severity numerical fixes

- [ ] Fix #3: Bytecode atan2 overflow
  - Replace `a * a + b * b` with `let h = a.hypot(b); h * h` in `opcode.rs:342`
  - Test: Add test with atan2(1e200, 1e200) verifying non-zero derivatives

- [ ] Fix #4: Catastrophic cancellation in asin/acos/atanh
  - Replace `F::one() - x * x` with `(F::one() - x) * (F::one() + x)` in:
    - `dual.rs` asin/acos
    - `dual_vec.rs` asin/acos
    - `num_traits_impls.rs` asin/acos (Reverse mode)
    - `opcode.rs` Asin/Acos/Atanh reverse_partials
    - `taylor_ops.rs` taylor_asin leading coefficient
  - Test: Add test with asin(1.0 - 1e-15) checking derivative precision

- [ ] Fix #5: Division derivative overflow
  - Restructure `(self.eps * rhs.re - self.re * rhs.eps) * inv * inv` to avoid squaring inv
  - Use `(self.eps - self.re * rhs.eps * inv) * inv` or direct `/ (rhs.re * rhs.re)`
  - Apply to `std_ops.rs`, `dual_vec_ops.rs`
  - Test: Add test with division by 1e-200 checking finite derivative

- [ ] Fix #6: Promote debug_assert to assert for custom ops in Hessian
  - Change `debug_assert!(self.custom_ops.is_empty(), ...)` to `assert!(...)` in:
    - `bytecode_tape/tangent.rs:325`
    - `bytecode_tape/sparse.rs:72`
  - Test: Existing tests should continue to pass

- [ ] Fix #7: WGSL u32 index overflow guard
  - In chunking logic (`gpu/mod.rs`), add constraint: `chunk_size <= u32::MAX / (num_variables * K)`
  - Test: Add assertion test that large num_variables triggers smaller chunks

- [ ] Fix #8: Taylor hypot rescaling
  - Rescale inputs by `max(|a[0]|, |b[0]|)` before computing `a²+b²` in `taylor_ops.rs:652-670`
  - Multiply result by scale factor after sqrt
  - Test: Add Taylor hypot test with (1e200, 1e200) and (1e-200, 1e-200)

- [ ] Review-fix: Phase 2 commit

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

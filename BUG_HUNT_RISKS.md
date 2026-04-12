# Structural Risk Assessment

| Risk Area | Likelihood | Impact | Evidence | Mitigation |
|-----------|-----------|--------|----------|------------|
| CPU-GPU derivative parity | Medium | High | CBRT HVP bug (#1) — GPU second derivative is wrong while CPU is correct. No automated cross-check between CPU and GPU derivative implementations. | Add property-based tests that compare CPU and GPU HVP/Hessian results for all opcodes. |
| GPU codegen index overflow | Medium | High | CUDA u32 truncation (#2) and WGSL u32 overflow (#7) — generated GPU code has different integer width handling than handwritten kernels. | Audit all generated kernel code for consistent 64-bit indexing. Add large-tape stress tests. |
| f32 as a second-class citizen | High | Medium | Findings #13, #14, #16 — hardcoded constants, factorial precision, and LU thresholds assume f64. The library is generic over `Float` but numerical guards are tuned for f64. | Audit all hardcoded constants/thresholds for f32 compatibility. Consider `F::epsilon()`-relative thresholds throughout. |
| Numerical edge cases at domain boundaries | Medium | Medium | Findings #3, #4, #5, #8 — asin near ±1, atan2 with large inputs, division with tiny denominators, hypot overflow. All produce silently wrong derivatives. | Add boundary-value derivative tests. Use numerically stable formulations ((1-x)(1+x), hypot-based denominators, rescaled intermediates). |
| debug_assert vs assert for correctness guards | Medium | Medium | Finding #6 — custom ops in Hessian computations guarded by `debug_assert!` which is stripped in release. | Promote correctness-critical assertions from `debug_assert!` to `assert!` or return `Result::Err`. |
| Welford accumulator FP edge cases | Low | Medium | Findings #9, #10 — all-zero weights and negative variance from cancellation. Both produce silently wrong statistical estimates. | Add `m2.max(0.0)` clamp and guard `w_sum > 0` before division. |

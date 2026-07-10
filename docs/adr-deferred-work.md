# ADR: Deferred and Rejected Work

**Date**: 2026-02-25
**Status**: Accepted
**Last updated**: 2026-07-10 (absorbed the WS1–9 workstream ledger; this file is now the single source of truth for planning state)

All core roadmap items (R1–R13), Phases 1–5, the STDE deferred items plan, and the WS1–9 workstream track are complete. This ADR captures every evaluated-but-not-implemented item with explicit reasoning, so future planning doesn't re-investigate the same paths. Nothing is actively planned; new work should be planned against current `main` and recorded here when it is deferred or rejected.

---

## Blocked (monitor, revisit when blockers resolve)

| Item | Blocker | Revisit When |
|------|---------|--------------|
| RUSTSEC-2024-0436 (paste/simba) | Upstream simba must release with paste alternative | simba publishes a new release |

---

## Done (previously deferred, now completed)

| Item | Completed In | Notes |
|------|-------------|-------|
| nalgebra 0.33 → 0.34 | v0.4.0 | MSRV blocker resolved (1.93 ≥ 1.87+). Upgrade completed with no API breakage. |

---

## Completed workstreams (WS track)

Workstreams scoped and deferred during development, formerly tracked in the root
`ROADMAP.md` (now a pointer here). All nine closed; every issue they surfaced was
fixed in-PR, so the track left no follow-up deferrals.

| WS | Title | Merged | One-line summary |
|----|-------|--------|------------------|
| WS 1 | R2 full migration (CPU SSOT expansion) | PR #63 | `Dual` / `DualVec` / `Reverse` route atan/atan2/asinh/acosh/hypot through `src/kernels/`; coordinated CPU + WGSL + CUDA + Taylor-codegen acosh factored-form upgrade |
| WS 2 | Higher-order Taylor HYPOT GPU rescale | PR #64 | WGSL + CUDA Taylor jet HYPOT max-rescale; explicit IEEE NaN-propagation guard on both backends |
| WS 3 | Richer error types for piggyback / sparse_implicit | PR #62 | `Option<T>` → `Result<T, PiggybackError \| SparseImplicitError>`; per-module `#[non_exhaustive]` enums; `Send + Sync` compile-time-asserted |
| WS 4 | Solver diagnostics (L-BFGS / Newton silent-filter surface) | PR #60 | `OptimResult.diagnostics: SolverDiagnostics` per-solver counters |
| WS 5 | Optim error-API enrichment | PR #67 | `PiggybackError::*Divergence` gains `last_norm`; `SparseImplicitError::Residual` gains `tolerance` + `dimension`; `SparseImplicitError::{StructuralFailure, FactorFailed}` gain `source: Box<dyn Error>` with `source()` chain; `MaxIterations` Display no longer leaks `Some(...)` / `None`; echidna-optim 0.10.0 → 0.11.0 |
| WS 6 | Optim error-API consistency polish | PR #68 | `MaxIterations` typestate-split into three `IterationsExhausted*` variants; `SparseImplicitError` renames (`*Failure/Failed/Residual → *Singular/Singular/Exceeded`); `echidna::assert_send_sync!` macro hoisted and applied to `ClarkeError` + `GpuError` (previously unguarded); `DimensionMismatch` variant added to all three optim error enums (15 asserts → `Err`); echidna-optim 0.11.0 → 0.12.0 |
| WS 7 | Dense `implicit.rs` Result migration | PR #66 | `implicit_{tangent,adjoint,jacobian,hvp,hessian}` → `Result<T, ImplicitError>`; `lu_factor` non-finite-pivot guard; post-solve non-finite guards on all five publics; echidna-optim 0.9.0 → 0.10.0 |
| WS 8 | `Laurent::hypot` kernel migration | PR #69 | `Laurent::hypot` routed through `taylor_ops::taylor_hypot` (the shared CPU HYPOT kernel); last CPU HYPOT implementation outside the shared kernel eliminated; rebase-to-`min(pole_order)` prelude aligns mismatched-pole operands; `scale == 0` with non-zero higher-order seeds now produces correct recursive shift-and-square output instead of `Self::zero()` |
| WS 9 | GPU Taylor edge cases at the function-domain boundary | PR #70 | WGSL + CUDA Taylor jet `HYPOT` Inf-finite path now emits NaN higher-order (matches CPU `Inf * 0 = NaN`); zero-origin-with-non-zero-seed now emits the one-level shift-and-square unroll (matches CPU `\|t\| · hypot(a/t, b/t)`); deeper-order-zero now emits `0` primal + `Inf` higher (matches CPU `taylor_sqrt` at zero leading); `#[ignore]`-d WS2 divergence tests un-ignored and renamed to `ws9_*_matches_cpu` |

---

## Deferred (valuable, not yet needed)

| Item | Reasoning | Revisit When |
|------|-----------|--------------|
| Indefinite dense STDE | Eigendecomposition-based approach for indefinite coefficient matrices C (6 parameters, sign-splitting into C⁺ - C⁻) adds significant API complexity. The positive-definite Cholesky case (`dense_stde_2nd`) covers most PDE use cases (Fokker-Planck, Black-Scholes, HJB). Users with indefinite C can manually split into C⁺ - C⁻, compute Cholesky factors for each, and call `dense_stde_2nd` twice. | A concrete user need for indefinite C arises |
| General-K GPU Taylor kernels | GPU Taylor kernel is hardcoded to K=3 (second-order). Hardcoding allows complete loop unrolling — critical for GPU performance. General K would need dynamic loops or a family of K-specialized kernels. | Need for GPU-accelerated 3rd+ order derivatives |
| Chunked GPU Taylor dispatch | Working buffer is `3 * num_variables * batch_size * 4` bytes. WebGPU's 128 MB limit caps `num_variables * batch_size ≤ ~10M`. For larger problems, the dispatch function should chunk the batch and accumulate results. | Users hit the buffer limit in practice |
| CUDA `laplacian_with_control_gpu` | `laplacian_with_control_gpu` is currently wgpu-only. CUDA equivalent (`laplacian_with_control_gpu_cuda`) is straightforward to add — same CPU-side Welford aggregation, just dispatches through `CudaContext`. | CUDA users need variance-reduced Laplacian |
| `taylor_forward_2nd_batch` in `GpuBackend` trait | Currently an inherent method on each backend, not part of the `GpuBackend` trait. Adding to the trait would enable generic code over backends but requires an associated type for Taylor results. | Multiple backends need to be used generically for Taylor |

---

## Rejected (evaluated, explicit reasoning)

### Core AD

| Item | Reasoning | What exists instead |
|------|-----------|---------------------|
| Constant deduplication (5.2) | `FloatBits` orphan rule blocks impl for `Dual<F>`. CSE handles ops over duplicate constants. | CSE pass in bytecode tape |
| Cross-checkpoint DCE (5.3) | Segments use ephemeral per-step tapes by design. Cross-segment analysis requires a global tape architecture that contradicts segment isolation. Multi-output DCE (R13) covers the common case. Checkpoint steps typically produce fully-consumed state vectors — dead computation is rare. | `dead_code_elimination_for_outputs()` |
| SIMD vectorization (5.6) | Profiling shows bottleneck is opcode dispatch, not FP throughput. Would only help batched forward sweeps, not the dispatch loop. Requires dispatch overhead to be solved first (trace compilation/JIT). | GPU backends for batch throughput |
| no_std / embedded (5.7) | Requires removing heap allocation and the thread-local tape/arena storage — a ground-up rewrite. Architecture fundamentally depends on dynamic allocation. | — |
| Source transformation / proc-macro AD | Orthogonal approach. Enzyme covers LLVM-level AD. Would be a separate project. | — |
| Preaccumulation of straight-line segments | Superseded by cross-country Markowitz elimination (R5), which achieves the same goal more generally. | `jacobian_cross_country()` |
| Trait impl macros for num_traits | ~3,200 lines are mechanical but readable. Macros hurt error messages, IDE support, and debuggability. Low maintenance cost since impls rarely change. | Manual trait impls |
| DiffOperator trait abstraction | Concrete estimators work well as standalone functions. `Estimator` trait (5.1) already provides the needed abstraction. Another trait layer would over-abstract. | `Estimator` trait + concrete functions |

### STDE Paper Items

| Item | Paper Section | Reasoning | What exists instead |
|------|---------------|-----------|---------------------|
| Multi-pushforward correction | App F, Eq 48-52 | Current collision-free prime-window approach in `diagonal_kth_order` is simpler and equally effective for practical operators. The correction addresses collisions in multi-index sampling that don't occur with the prime-window design. | `diagonal_kth_order`, `diagonal_kth_order_const` |
| Amortized PINN training | §5.1, Eq 25 | Application-level integration — amortizing STDE samples across training steps is a training loop concern, not a core AD primitive. Users can implement this in their training code using existing `stde::*` functions. | `stde::laplacian`, `stde::stde_sparse` |
| Weight sharing across network layers | App G | Network architecture concern — sharing Taylor jet computation across layers with shared weights requires network-level orchestration, not AD-level support. | — |

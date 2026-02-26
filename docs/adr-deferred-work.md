# ADR: Deferred and Rejected Work

**Date**: 2026-02-25
**Status**: Accepted
**Last updated**: 2026-02-25 (added STDE deferred items)

All core roadmap items (R1–R13), Phases 1–5, and the STDE deferred items plan are complete. This ADR captures every evaluated-but-not-implemented item with explicit reasoning, so future planning doesn't re-investigate the same paths.

---

## Blocked (monitor, revisit when blockers resolve)

| Item | Blocker | Revisit When |
|------|---------|--------------|
| RUSTSEC-2024-0436 (paste/simba) | Upstream simba must release with paste alternative | simba publishes a new release |

---

## Deferred (valuable, not yet needed)

| Item | Reasoning | Revisit When |
|------|-----------|--------------|
| nalgebra 0.33 → 0.34 | MSRV blocker resolved (now 1.93, nalgebra 0.34 requires 1.87+). May have API changes in `NalgebraVec` and sparse matrix types. | When convenient; test thoroughly for API changes. |
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
| no_std / embedded (5.7) | Requires removing heap allocation, thread-local tapes, bumpalo arenas — a ground-up rewrite. Architecture fundamentally depends on dynamic allocation. | — |
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

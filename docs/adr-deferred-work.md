# ADR: Deferred and Rejected Work

**Date**: 2026-02-25
**Status**: Accepted

All core roadmap items (R1–R13) and Phases 1–5 are complete. This ADR captures every evaluated-but-not-implemented item with explicit reasoning, so future planning doesn't re-investigate the same paths.

---

## Blocked (monitor, revisit when blockers resolve)

| Item | Blocker | Revisit When |
|------|---------|--------------|
| RUSTSEC-2024-0436 (paste/simba) | Upstream simba must release with paste alternative | simba publishes a new release |
| nalgebra 0.33 → 0.34 | nalgebra 0.34 requires Rust 1.87+; MSRV is 1.80 | MSRV is raised to 1.87+ |

---

## Rejected (evaluated, explicit reasoning)

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

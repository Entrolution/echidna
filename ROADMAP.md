# echidna Roadmap

Outstanding and future work for the echidna automatic differentiation library. All core roadmap items (R1-R13) are complete and the crate is published on crates.io. See [`docs/roadmap.md`](docs/roadmap.md) for the historical record of completed work.

---

## Phase 0: Foundation (blocked on external factors)

Monitor these — don't action until blockers resolve.

| # | Item | Type | Notes |
|---|------|------|-------|
| 0.1 | **RUSTSEC-2024-0436** (paste/simba) | security | Tracked in `deny.toml`. Blocked on upstream simba releasing with paste alternative. |
| 0.2 | **nalgebra 0.33 → 0.34** | dependency | Blocked on MSRV — nalgebra 0.34 requires Rust 1.87+. Track for when MSRV is raised. |

---

## ~~Phase 1: Polish~~ Complete

All Phase 1 items are done. Items 1.4, 1.5, 1.7-1.11 were already addressed in prior cleanup PRs. Items 1.1-1.3 and 1.6 were completed in the Phase 1 CI hardening PR.

<details>
<summary>Completed items</summary>

### CI Hardening

| # | Item | Status |
|---|------|--------|
| 1.1 | `cargo check --features gpu-wgpu` in CI lint job | Done |
| 1.2 | echidna-optim `--features sparse-implicit` in CI test matrix | Done |
| 1.3 | `cargo-semver-checks` | Done (run locally — too slow for CI) |
| 1.4 | Harden CI `cargo-audit` install | Already done |
| 1.5 | Review lint suppressions | Already done (ndarray/nalgebra suppressions removed; bytecode_tape one justified) |
| 1.6 | Bump `num-dual` 0.10 → 0.11 | Done |

### Documentation

| # | Item | Status |
|---|------|--------|
| 1.7 | Missing APIs in README table | Already done |
| 1.8 | `CustomOp`/`CustomOpHandle` in crate docs | Already done |
| 1.9 | echidna-optim features in README | Already done |
| 1.10 | `//!` module doc for `src/traits/mod.rs` | Already done |
| 1.11 | Jacobian sparsity in `src/sparse.rs` module doc | Already done |

</details>

---

## ~~Phase 2: Quality~~ Complete

Coverage baseline established (49.42% overall) and cross-country unit tests added.

<details>
<summary>Completed items</summary>

| # | Item | Status |
|---|------|--------|
| 2.1 | Coverage baseline via `cargo tarpaulin` | Done — 49.42% (3201/6477 lines). Key: api 97.7%, bytecode_tape 75.6%, stde 97.9%, sparse 84.3%. Low coverage in trait impls is expected (mechanical boilerplate). |
| 2.2 | Unit tests for `cross_country.rs` Markowitz ordering and fill-in | Done — 11 unit tests covering edge accumulation, Markowitz ordering, vertex elimination, fill-in creation, and end-to-end Jacobian extraction. |

</details>

---

## ~~Phase 3: CustomOp Second-Order Derivatives~~ Complete

Added `eval_dual` and `partials_dual` default methods to `CustomOp<F>`, enabling correct second-order derivatives (HVP, Hessian) through custom ops. The original plan items (3.1 and 3.2) were merged: instead of a separate `CustomOpSecondOrder` trait or generic eval, concrete `Dual<F>` methods on the existing trait preserve object safety while avoiding code duplication.

<details>
<summary>Completed items</summary>

| # | Item | Status |
|---|------|--------|
| 3.1+3.2 | **`eval_dual` / `partials_dual` on `CustomOp<F>`** — Default methods use first-order chain rule (backward compatible). Override to propagate tangent through custom ops for correct HVP/Hessian. Covers `Dual<F>` path (HVP, Hessian, sparse Hessian). `DualVec`, `Dual<Dual<F>>`, and `Taylor` paths remain first-order for custom ops. | Done |

</details>

---

## ~~Phase 4: Architecture~~ Complete

All Phase 4 items are done: GpuBackend trait, reverse-wrapping-forward composition, and thread-local tape pooling.

<details>
<summary>Completed items</summary>

| # | Item | Type | Effort | Notes |
|---|------|------|--------|-------|
| 4.1 | **GpuBackend trait** — Unify wgpu and CUDA backends behind a common trait. f32 methods moved to `GpuBackend` trait in `gpu/mod.rs`; CUDA f64 methods remain inherent. CUDA `sparse_jacobian` compilation bugs fixed. | feature | medium | Done |
| 4.2 | **Reverse-wrapping-forward composition** — `BReverse<Dual<f64>>` via `BtapeThreadLocal` impls for `Dual<f32>` and `Dual<f64>`. Enables reverse-over-forward HVP and column-by-column Hessian. | feature | small | Done |
| 4.3 | **Thread-local Adept tape pooling** — Replaced per-call `Tape` allocation in `grad()`/`vjp()` with thread-local pool. Cleared tapes retain Vec capacity across calls. Zero API changes. Arena approach was rejected (lifetime cascading). | tech-debt | small | Done |

</details>

---

## Phase 5: Aspirational / Research

Require motivation from concrete use cases before committing.

| # | Item | Type | Effort | Notes |
|---|------|------|--------|-------|
| 5.1 | ~~**Estimator abstraction + divergence**~~ — `Estimator` trait generalising per-direction sample computation (`Laplacian`, `GradientSquaredNorm`), generic `estimate`/`estimate_weighted` pipeline, and Hutchinson divergence estimator for vector fields via `Dual<F>` forward mode. | feature | medium | **Done** |
| 5.2 | **Constant deduplication at record time** — Identical literal constants get separate tape entries. Blocked on `FloatBits` trait orphan rule for `Dual<F>`. CSE already handles ops over duplicate constants, limiting the benefit. | tech-debt | medium | Architecturally blocked |
| 5.3 | **Cross-checkpoint DCE** — Dead code elimination across checkpoint boundaries. Targeted multi-output DCE (R13) covers most cases; cross-checkpoint adds complexity for diminishing returns. | tech-debt | large | |
| 5.4 | ~~**Advanced STDE variance reduction**~~ — Importance-weighted estimation (West's 1979 algorithm) and Hutch++ (Meyer et al. 2021) O(1/S²) trace estimator via sketch + residual decomposition. Antithetic and stratified sampling subsumed by existing diagonal control variate and Hutch++. | feature | medium | **Done** |
| 5.5 | ~~**Extended nonsmooth operators**~~ — Added `signum`, `floor`, `ceil`, `round`, `trunc` to kink detection. Two-tier classification: all 8 ops tracked for proximity detection via `is_nonsmooth()`; only `abs`/`min`/`max` enumerated in Clarke Jacobian via `has_nontrivial_subdifferential()`. Fixed `Signed::signum()` for `BReverse` to record to tape. | feature | small-medium | **Done** |
| 5.6 | **SIMD vectorization** — `wide` crate integration for batched elemental kernels. Evaluated and deferred: profiling shows the bottleneck is opcode dispatch, not FP throughput. SIMD would help batched forward sweeps but not the dispatch overhead. | feature | large | Deferred |
| 5.7 | **no_std / embedded** — Explicitly a non-goal currently. Would require removing dynamic allocation and thread-local tape infrastructure. | feature | epic | Non-goal |
| 5.8 | ~~**ad-trait comparison benchmarks**~~ — Added `ad_trait` (v0.2.2) forward-mode and reverse-mode gradient benchmarks alongside existing num-dual comparisons. Correctness cross-check tests verify agreement with echidna. | quality | small | **Done** |

---

## Deferred / Won't Do

| Item | Reason |
|------|--------|
| Source transformation / proc-macro AD | Orthogonal approach; Enzyme covers LLVM-level AD |
| Preaccumulation of straight-line segments | Superseded by cross-country Markowitz elimination (R5) |
| Trait impl macros for num_traits boilerplate | ~3,200 lines are mechanical but readable. Macros would hurt error messages and IDE support. Not worth the trade-off. |

---

## Dependency Summary

```
0.1 (RUSTSEC-2024-0436)       ← upstream simba
0.2 (nalgebra 0.34)            ← MSRV bump decision
```

Phases 1–4 are complete. Phase 5 items 5.1, 5.4, 5.5, and 5.8 are done. Remaining items (5.2 blocked, 5.3 deferred, 5.6 deferred, 5.7 non-goal) are independent and require no specific ordering.

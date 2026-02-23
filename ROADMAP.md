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

## Phase 2: Quality

Establish a test baseline, then fill coverage gaps. Do this before any breaking changes.

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 2.1 | Run `cargo tarpaulin` to establish per-module coverage baseline | small | Do before any breaking changes |
| 2.2 | Unit tests for `cross_country.rs` Markowitz ordering and fill-in | small | |

---

## Phase 3: CustomOp Extensibility

Breaking API changes — requires semver-minor or semver-major bump.

| # | Item | Type | Effort | Notes |
|---|------|------|--------|-------|
| 3.2 | **CustomOp generic eval** — current `eval` takes `F: Float`. Forward-over-reverse (`Dual<BReverse<f64>>`) needs eval to accept `T: Scalar` to propagate tangent components through custom ops. Add a new method with a default impl to avoid semver-major. | tech-debt | medium | Prerequisite for 3.1 and 4.2 |
| 3.1 | **CustomOpSecondOrder trait** — doc comment at `bytecode_tape.rs:38` references it but it doesn't exist. Needed for HVP/Hessian computation through user-defined ops. Without it, second-order derivatives silently fail for custom ops. | feature | medium | Depends on 3.2 |

### Dependencies

```
3.2 (CustomOp generic eval)
 ├──▶ 3.1 (CustomOpSecondOrder trait)
 └──▶ 4.2 (reverse-wrapping-forward composition)
```

---

## Phase 4: Architecture (deferred tech debt)

Larger refactors with clear motivation but no urgency.

| # | Item | Type | Effort | Notes |
|---|------|------|--------|-------|
| 4.1 | **GpuBackend trait** — Unify wgpu and CUDA backends behind a common trait. Currently both backends exist (`wgpu_backend.rs`, `cuda_backend.rs`) with parallel but separate APIs. Natural next step if a third backend emerges (e.g., WebGPU, ROCm). | feature | large | Deliberately deferred |
| 4.2 | **Reverse-wrapping-forward composition** — `BReverse<Dual<f64>>` is unsupported. Would require new thread-local tapes per composed type. Forward-wrapping-reverse (`Dual<BReverse<f64>>`) works. | feature | large | Depends on 3.2 |
| 4.3 | **Bumpalo arena for Adept tape** — Replace Vec-based Adept tape allocation with arena. Deferred due to lifetime cascading: `Reverse<'a, F>` would propagate lifetimes through all APIs. Questionable value since bytecode tape covers the reuse case. | tech-debt | large | Low motivation |

---

## Phase 5: Aspirational / Research

Require motivation from concrete use cases before committing.

| # | Item | Type | Effort | Notes |
|---|------|------|--------|-------|
| 5.1 | **DiffOperator abstraction** — General combinator for differential operators (Laplacian, divergence, curl). Current STDE module exposes individual functions; a trait would allow user-defined operators with the same estimation pipeline. Deferred pending motivation from variance reduction research. | feature | medium | |
| 5.2 | **Constant deduplication at record time** — Identical literal constants get separate tape entries. Blocked on `FloatBits` trait orphan rule for `Dual<F>`. CSE already handles ops over duplicate constants, limiting the benefit. | tech-debt | medium | Architecturally blocked |
| 5.3 | **Cross-checkpoint DCE** — Dead code elimination across checkpoint boundaries. Targeted multi-output DCE (R13) covers most cases; cross-checkpoint adds complexity for diminishing returns. | tech-debt | large | |
| 5.4 | **Advanced STDE variance reduction** — Importance sampling, stratified sampling for stochastic Laplacian/Hessian diagonal estimators. Antithetic sampling proven unnecessary (`v^T H v` is even in `v`). | feature | medium | Research-dependent |
| 5.5 | **Extended nonsmooth operators** — `sign(·)`, `floor`/`ceil` step functions, broader Clarke subdifferential enumeration. Current support covers `abs`, `min`, `max`, and user-defined piecewise functions. | feature | small-medium | Incremental |
| 5.6 | **SIMD vectorization** — `wide` crate integration for batched elemental kernels. Requires benchmarking to justify complexity vs benefit. | feature | large | |
| 5.7 | **no_std / embedded** — Explicitly a non-goal currently. Would require removing dynamic allocation and thread-local tape infrastructure. | feature | epic | Non-goal |
| 5.8 | **ad-trait comparison benchmarks** — Expand comparison suite beyond num-dual. | quality | small | Low priority |

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

3.2 (CustomOp generic eval)
 ├──▶ 3.1 (CustomOpSecondOrder)
 └──▶ 4.2 (reverse-wrapping-forward)
```

All other items are independent and can be worked in any order within their phase.

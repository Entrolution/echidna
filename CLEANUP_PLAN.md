# Codebase Cleanup Plan

**Generated**: 2026-03-13 | **Project**: echidna v0.4.0 | **LOC**: ~23,400 source, ~15,700 tests

## Executive Summary

Three specialist agents (Infrastructure, Design, Documentation) reviewed the echidna codebase. The project is in strong shape — zero lint errors, zero dead code, zero debug artifacts, clean architecture with correct dependency direction, and comprehensive test coverage across 37 integration test files.

The main findings are:
- **13 unsafe blocks missing SAFETY comments** (carried forward from prior audit, still unaddressed)
- **8 lint suppressions missing explanatory comments**
- **CI gaps**: `diffop` feature has zero CI coverage; MSRV job only tests default features
- **5 code duplication sites** totalling ~217 lines (after prior cleanup resolved the larger duplications)
- **4 documentation files** need minor updates (~30 min of text edits)
- **1 security advisory** (paste via simba) — blocked upstream, already ignored in deny.toml

No critical issues. No breaking changes required. All items are incremental improvements.

## Current State

- **Architecture**: Clean. Dependencies point inward. Domain logic free of I/O (except disk checkpointing, appropriately isolated). No circular dependencies. Thread-local tape is hidden global state but well-mitigated by RAII guards and API layer.
- **Test Coverage**: Comprehensive integration tests across all major subsystems. cargo-tarpaulin runs in CI. Gaps in echidna-optim solvers and cross-country full-tape integration.
- **Documentation**: High quality overall. README, rustdoc, CHANGELOG, and algorithms.md are thorough. 4 files have minor staleness from the v0.4.0 bytecode_tape decomposition.
- **Dependency Health**: 1 advisory (paste/simba, blocked upstream). cudarc 2 minor versions behind (breaking changes). All other deps at latest.
- **Lint Health**: 0 errors, 0 default-level warnings. 686 pedantic warnings (mostly `must_use_candidate` and doc suggestions). 13 suppressions, all still needed, 8 lacking comments.

## Memory Context

- **Decisions from History**: Thread-local tape design for `Copy` on `Reverse<F>`, dual tape architecture (Adept + BytecodeTape), speed-first philosophy (NaN propagation, no Result branching) — all still current and load-bearing.
- **Known Tech Debt**: Welford duplication (RESOLVED — consolidated into `WelfordAccumulator`), reverse sweep duplication (RESOLVED — `reverse_sweep_core`), DCE duplication (RESOLVED — `dce_compact`), output_indices boilerplate (RESOLVED — `all_output_indices()`). Remaining: num_traits macro consolidation (deferred, unfavorable trade-off), GPU trait extraction (deferred, large refactor).
- **Past Attempts**: Cleanup PRs #12-16 completed Phases 0-4. Phase 5 (duplication): items 5.5, 5.8 done; 5.6 deliberately dropped. Memory specialist runs failed 3+ times due to context overflow.
- **Dependency History**: RUSTSEC-2024-0436 (paste/simba) blocked upstream since 2024, ignored in deny.toml. RUSTSEC-2025-0141 (bincode) resolved by replacing with postcard/bitcode. cudarc 0.18 breaking change deferred. nalgebra bumped to 0.34 in v0.4.0.
- **Lint/Suppression History**: All 13 `#[allow]` suppressions were reviewed during Phase 3 cleanup. `clippy::suspicious_arithmetic_impl` suppressions are justified (AD math where Mul impl does addition in tangent). MSRV updated from 1.80 to 1.93.

## Dependency Health

### Security Fixes (Priority)

| Dependency | Current | Fix Version | Vulnerability | Severity |
|-----------|---------|-------------|---------------|----------|
| paste (transitive, via simba) | 1.0.15 | N/A | RUSTSEC-2024-0436 — unmaintained | Low (advisory only) |

Already ignored in `deny.toml` with explanatory comment. Blocked on simba upstream migration. No action required.

### At-Risk Dependencies

| Dependency | Risk | Issue | Action | Alternative / Notes |
|-----------|------|-------|--------|---------------------|
| paste (via simba) | Medium | Unmaintained since Oct 2024 | Wait for simba upstream | Blocked — no action possible |
| cudarc | Medium | 0.17 pinned, 0.19.3 available; 0.18 has breaking API | Bump when ready for API migration | Memory confirms upgrade was previously deferred |
| simba | Medium | Pulls in unmaintained paste; single org (dimforge) | Monitor | No alternative for nalgebra ecosystem |

### Version Bumps

| Dependency | Current | Latest | Breaking | Notes |
|-----------|---------|--------|----------|-------|
| cudarc | 0.17 | 0.19.3 | Yes (0.18) | gpu-cuda feature; deferred from prior audit |
| num-dual | 0.11 | 0.13.6 | Minor | Dev-dep only; comparison benchmarks |

12 transitive crates have duplicate versions in lockfile (from faer/wgpu sub-trees). Not directly addressable.

## Lint & Static Analysis

### Errors

None. Zero errors across all lint levels.

### Warnings (by category)

Default-level clippy (enforced in CI with `-D warnings`) produces zero warnings.

Pedantic-level findings (686 total, not currently enforced):

| Category | Count | Action |
|----------|-------|--------|
| `must_use_candidate` | ~267 | Consider bulk-adding `#[must_use]` to pure functions in a future pass |
| Missing doc sections (`# Panics`, backticks) | ~124 | Incremental doc improvement |
| Cast portability (`usize` to `u32`) | ~68 | Audit CUDA/GPU paths for truncation risk |
| `inline_always` suggestions | ~35 | Review if `#[inline]` alone suffices (simba trait impls) |
| Auto-fixable style | ~50 | `cargo clippy --fix` for format args, redundant closures |
| Similar names | ~8 | Review for clarity |

### Suppression Audit

| File:Line | Suppression | Verdict | Action |
|-----------|------------|---------|--------|
| `bytecode_tape/jacobian.rs:142` | `needless_range_loop` | Valid | Keep (has comment) |
| `traits/dual_vec_ops.rs:31` | `suspicious_arithmetic_impl` | Valid | Keep (has comment) |
| `traits/laurent_std_ops.rs:79,91,212` | `suspicious_arithmetic_impl` | Valid | **Add comments** — AD math where Mul does addition in tangent |
| `traits/taylor_std_ops.rs:36,48,190,272,282` | `suspicious_arithmetic_impl` | Valid | **Add comments** — same pattern |
| `cross_country.rs:39` | `too_many_arguments` | Valid | Keep (has comment) |
| `diffop.rs:582` | `needless_range_loop` | Valid | **Add comment** |
| `float.rs:54` | `cfg_attr(dead_code)` | Valid | Keep (has comment) |

**Summary**: 13 suppressions, all still needed. **8 need explanatory comments added.**

## Dead Code & Artifact Removal

### Immediate Removal

None. Zero dead code, zero debug artifacts, zero TODO/FIXME/HACK comments in source.

### Verify Before Removal

| Item | Location | Verification Needed |
|------|----------|---------------------|
| Packaging artifacts | `target/package/echidna-0.1.0/Cargo.toml.orig` | Gitignored; harmless. `cargo clean` removes. |
| Unused deny.toml license entries | `deny.toml` L14-24 | BSD-3-Clause, BSL-1.0 may be unused. Verify with `cargo deny check`. Low priority. |

## Documentation Consolidation

### Documents to Update

| Document | Updates Required |
|----------|-----------------|
| `CONTRIBUTING.md` L154-211 | **Most impactful**: Replace `bytecode_tape.rs` single-file entry with `bytecode_tape/` directory listing (11 submodules). Misleads new contributors. |
| `docs/algorithms.md` L529 | Fix "43 opcodes" to "44 opcodes" (matches L75 and L513 in same doc, and verified OpCode enum). |
| `docs/adr-deferred-work.md` L23 | Move nalgebra 0.33->0.34 from Deferred to Done (completed in v0.4.0 per CHANGELOG). |
| `docs/roadmap.md` | Update 6 stale `src/bytecode_tape.rs` references to point to correct submodule files. |

### Documents to Remove/Merge

| Document | Action | Target |
|----------|--------|--------|
| `docs/plans/README.md` | Consider merging | `docs/roadmap.md` — 15-line file that only points to roadmap |

## Refactoring Roadmap

### Phase 0: Safety & Compliance (~1-2 hours)

| Task | Impact | Effort | Files Affected |
|------|--------|--------|----------------|
| Add SAFETY comments to 13 unsafe blocks | High | Small | `checkpoint.rs` (2), `cuda_backend.rs` (6), `simba_impls.rs` (4), `taylor_dyn.rs` (1) |
| Add explanatory comments to 8 `#[allow]` suppressions | Medium | Trivial | `laurent_std_ops.rs` (3), `taylor_std_ops.rs` (5), `diffop.rs` (1) |

### Phase 1: CI & Infrastructure (~30 min)

| Task | Impact | Effort | Files Affected |
|------|--------|--------|----------------|
| Add `diffop` feature to CI test and lint jobs | High | Trivial | `.github/workflows/ci.yml` |
| Add `parallel` feature to `publish.yml` pre-publish validation | Medium | Trivial | `.github/workflows/publish.yml` |
| Expand MSRV job to test key feature combinations (bytecode, taylor, stde) | Medium | Trivial | `.github/workflows/ci.yml` |

### Phase 2: Documentation (~30 min)

| Task | Impact | Effort | Files Affected |
|------|--------|--------|----------------|
| Update CONTRIBUTING.md architecture tree | High | Trivial | `CONTRIBUTING.md` |
| Fix algorithms.md opcode count (43->44) | Low | Trivial | `docs/algorithms.md` |
| Move nalgebra entry to Done in ADR | Low | Trivial | `docs/adr-deferred-work.md` |
| Update roadmap.md bytecode_tape paths | Low | Trivial | `docs/roadmap.md` |

### Phase 3: Duplication Consolidation (~2-3 hours)

| Task | Impact | Effort | Files Affected |
|------|--------|--------|----------------|
| Consolidate `greedy_coloring` → delegate to `greedy_distance1_coloring` | Medium | Small | `src/sparse.rs` (~72 lines) |
| Consolidate `sparse_hessian` → call `sparse_hessian_with_pattern` | Medium | Small | `src/bytecode_tape/sparse.rs` (~50 lines) |
| Extract shared opcode dispatch from `forward`/`forward_into` | Medium | Small | `src/bytecode_tape/forward.rs` (~40 lines) |
| Consolidate `column_coloring`/`row_coloring` → generic helper | Low | Small | `src/sparse.rs` (~30 lines) |
| Extract helper from `GpuTapeData::from_tape`/`from_tape_f64_lossy` | Low | Trivial | `src/gpu/mod.rs` (~25 lines) |

### Phase 4: Code Quality (~optional, future)

| Task | Impact | Effort | Components |
|------|--------|--------|------------|
| Decompose `stde.rs` (1409 lines) into sub-modules | Medium | Medium | `src/stde/` directory |
| Add `#![warn(missing_docs)]` and fill gaps | Medium | Large | All public items |
| Bulk-add `#[must_use]` to pure functions | Low | Medium | ~267 sites |
| Audit `usize` to `u32` casts in GPU paths | Medium | Small | `src/gpu/`, `src/bytecode_tape/` |
| Bump cudarc 0.17 → 0.19 (breaking API changes) | Medium | Medium | `src/gpu/cuda_backend.rs` |

## Testing Strategy

**Current state is strong.** 37 integration test files cover all major subsystems. Known gaps:

| Gap | Priority | Action |
|-----|----------|--------|
| `echidna-optim` solver convergence edge cases | High | Add convergence tests for ill-conditioned problems |
| `cross_country.rs` full-tape integration | Medium | Add end-to-end test with real BytecodeTape |
| `bytecode_tape/optimize.rs` CSE edge cases | Medium | Add tests for pathological CSE remapping scenarios |
| Sparse Jacobian reverse-mode path | Medium | Ensure auto-selection exercises reverse path in tests |

## Target State

- **Test Coverage**: Maintain current breadth; fill 4 identified gaps
- **Architecture**: No changes needed — already clean
- **Documentation**: All docs accurate; SAFETY comments on all unsafe blocks; suppressions documented
- **Key Improvements**: CI covers all features; zero undocumented unsafe; zero undocumented suppressions

## Risks & Considerations

- **paste/simba advisory**: Blocked upstream. Monitor simba releases for pastey migration. No action possible now.
- **cudarc upgrade**: Breaking API changes in 0.18. Defer until GPU backend is actively developed.
- **stde.rs decomposition**: Medium effort, risk of breaking the delicate Taylor jet propagation logic. Only pursue if the file continues to grow.
- **`#![warn(missing_docs)]`**: Large effort (all public items need docs). Consider enabling per-module incrementally.
- **Pedantic clippy**: 686 warnings. Do NOT enable `-W clippy::pedantic` in CI without triaging — many are noise for a math library. Cherry-pick useful categories (must_use, cast truncation).

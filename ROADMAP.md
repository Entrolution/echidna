# Echidna Roadmap

Forward-looking workstreams explicitly scoped and deferred from prior
bug-hunt and review-fix cycles. Each item here carries enough context
to re-enter cold, has a concrete "done" condition, and is independent
enough to ship on its own branch.

When starting an item, check the "Prior context" for the commit(s)
or PR(s) that introduced the partial fix so you can diff against
what was left behind.

---

## Completed

| WS | Title | Merged | One-line summary |
|----|-------|--------|------------------|
| WS 1 | R2 full migration (CPU SSOT expansion) | PR #63 | `Dual` / `DualVec` / `Reverse` route atan/atan2/asinh/acosh/hypot through `src/kernels/`; coordinated CPU + WGSL + CUDA + Taylor-codegen acosh factored-form upgrade |
| WS 2 | Higher-order Taylor HYPOT GPU rescale | PR #64 | WGSL + CUDA Taylor jet HYPOT max-rescale; explicit IEEE NaN-propagation guard on both backends |
| WS 3 | Richer error types for piggyback / sparse_implicit | PR #62 | `Option<T>` → `Result<T, PiggybackError | SparseImplicitError>`; per-module `#[non_exhaustive]` enums; `Send + Sync` compile-time-asserted |
| WS 4 | Solver diagnostics (L-BFGS / Newton silent-filter surface) | PR #60 | `OptimResult.diagnostics: SolverDiagnostics` per-solver counters |
| WS 5 | Optim error-API enrichment | PR #67 | `PiggybackError::*Divergence` gains `last_norm`; `SparseImplicitError::Residual` gains `tolerance` + `dimension`; `SparseImplicitError::{StructuralFailure, FactorFailed}` gain `source: Box<dyn Error>` with `source()` chain; `MaxIterations` Display no longer leaks `Some(...)` / `None`; echidna-optim 0.10.0 → 0.11.0 |
| WS 6 | Optim error-API consistency polish | PR #68 | `MaxIterations` typestate-split into three `IterationsExhausted*` variants; `SparseImplicitError` renames (`*Failure/Failed/Residual → *Singular/Singular/Exceeded`); `echidna::assert_send_sync!` macro hoisted and applied to `ClarkeError` + `GpuError` (previously unguarded); `DimensionMismatch` variant added to all three optim error enums (15 asserts → `Err`); echidna-optim 0.11.0 → 0.12.0 |
| WS 7 | Dense `implicit.rs` Result migration | PR #66 | `implicit_{tangent,adjoint,jacobian,hvp,hessian}` → `Result<T, ImplicitError>`; `lu_factor` non-finite-pivot guard; post-solve non-finite guards on all five publics; echidna-optim 0.9.0 → 0.10.0 |
| WS 8 | `Laurent::hypot` kernel migration | PR #69 | `Laurent::hypot` routed through `taylor_ops::taylor_hypot` (the shared CPU HYPOT kernel); last CPU HYPOT implementation outside the shared kernel eliminated; rebase-to-`min(pole_order)` prelude aligns mismatched-pole operands; `scale == 0` with non-zero higher-order seeds now produces correct recursive shift-and-square output instead of `Self::zero()` |
| WS 9 | GPU Taylor edge cases at the function-domain boundary | _pending merge_ | WGSL + CUDA Taylor jet `HYPOT` Inf-finite path now emits NaN higher-order (matches CPU `Inf * 0 = NaN`); zero-origin-with-non-zero-seed now emits the one-level shift-and-square unroll (matches CPU `|t| · hypot(a/t, b/t)`); deeper-order-zero now emits `0` primal + `Inf` higher (matches CPU `taylor_sqrt` at zero leading); `#[ignore]`-d WS2 divergence tests un-ignored and renamed to `ws9_*_matches_cpu` |

---

## Active workstreams

**None.** All nine ROADMAP workstreams are closed. The entire
echidna + echidna-optim deferral backlog surfaced during the WS1–4
review-fix cycles has shipped without scope creep spillage — every
finding raised during review-fix landed in-PR rather than spawning a
follow-up workstream.

Any new work should be planned against the current `main` and tracked
separately rather than re-opening this document.

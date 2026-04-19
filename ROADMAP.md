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
| WS 7 | Dense `implicit.rs` Result migration | PR #66 | `implicit_{tangent,adjoint,jacobian,hvp,hessian}` → `Result<T, ImplicitError>`; `lu_factor` non-finite-pivot guard; post-solve non-finite guards on all five publics; echidna-optim 0.9.0 → 0.10.0 |
| WS 5 | Optim error-API enrichment | PR #67 | `PiggybackError::*Divergence` gains `last_norm`; `SparseImplicitError::Residual` gains `tolerance` + `dimension`; `SparseImplicitError::{StructuralFailure, FactorFailed}` gain `source: Box<dyn Error>` with `source()` chain; `MaxIterations` Display no longer leaks `Some(...)` / `None`; echidna-optim 0.10.0 → 0.11.0 |
| WS 6 | Optim error-API consistency polish | PR #68 | `MaxIterations` typestate-split into three `IterationsExhausted*` variants; `SparseImplicitError` renames (`*Failure/Failed/Residual → *Singular/Singular/Exceeded`); `echidna::assert_send_sync!` macro hoisted and applied to `ClarkeError` + `GpuError` (previously unguarded); `DimensionMismatch` variant added to all three optim error enums (15 asserts → `Err`); echidna-optim 0.11.0 → 0.12.0 |
| WS 8 | `Laurent::hypot` kernel migration | _pending merge_ | `Laurent::hypot` routed through `taylor_ops::taylor_hypot` (the shared CPU HYPOT kernel); last CPU HYPOT implementation outside the shared kernel eliminated; rebase-to-`min(pole_order)` prelude aligns mismatched-pole operands; `scale == 0` with non-zero higher-order seeds now produces correct recursive shift-and-square output instead of `Self::zero()` |

---

## Active workstreams

The remainder are deferrals surfaced during WS1–4 review-fix cycles
that were intentionally not folded into those PRs — either because
they were out of scope, would have required GPU recoordination after
the in-flight fix, or because the verified score was below the
auto-fix threshold. None block any current work.

---


## WS 9 — GPU Taylor edge cases at the function-domain boundary

**Deferred from**: WS2 (PR #64), pinned by `#[ignore]`-d test +
documented in codegen comments.

**Prior context**: WS2 applied jet-wide max-rescale to GPU Taylor
HYPOT. Two corner cases remain where GPU output diverges from CPU
`taylor_ops::taylor_hypot` — both at the boundary of the function
domain where derivatives are mathematically undefined:

1. `hypot(0, 0)` with non-zero higher-order seeds (e.g. JVP through
   the origin along a non-zero direction): CPU recursively
   shift-and-square unwinds to extract a `|t|` factor; GPU returns
   the zero jet. Pinned by
   `tests/gpu_stde.rs::ws2_*_hypot_zero_origin_with_nonzero_seed_diverges_from_cpu`
   (`#[ignore]`-d).
2. Finite/Inf inputs with non-zero higher-order seeds: CPU produces
   NaN higher-order coefficients via `Inf * 0 = NaN` in the rescale
   path; GPU returns 0. Documented in codegen comments at the
   relevant emission sites.

**Problem**: Neither divergence is a bug per IEEE — derivatives at
the function-domain boundary are conventionally undefined. But
having GPU and CPU disagree silently means downstream code that
relies on either convention is platform-dependent.

**Approach**:
1. **For zero-origin recursion**: K-bounded unroll of the
   shift-and-square in the codegen — emit a separate code block
   per `K` value the user requests. ~30 lines per `K`, K ≤ 6
   currently supported, so ~180 lines of WGSL + same in CUDA.
   Verify against the existing `#[ignore]`-d test (un-ignore it
   when the implementation lands).
2. **For finite-Inf NaN**: change GPU Inf branch to set
   `r.v[i] = NaN` for `i >= 1`. WGSL: `bitcast<f32>(0x7fc00000u)`.
   CUDA: `(F)(0.0/0.0)` or NVRTC bit-cast. Add a parity test that
   asserts `c1.is_nan() && c2.is_nan()` at `hypot(Inf, finite)`.

**Done when**: Either (a) both divergences fixed and the
`#[ignore]`-d test passes; or (b) explicit decision to keep the
divergence with the rationale captured in a follow-up note (e.g.
"performance not worth it — function-domain boundary is
ill-conditioned anyway").

**Effort**: Medium for (1) (codegen complexity + GPU runtime
verification on vast.ai); Small for (2) (one branch per backend +
one test).
**Risk**: Moderate for (1) — recursive unwinding logic is the kind
of subtle codegen that can hide bugs from simple parity tests, same
risk profile as the original WS2.
**Priority**: Low. Both cases are at the function-domain boundary;
no realistic user has reported either. Track to ensure they don't
regress further or surprise a future user.

---

## Suggested order

1. **WS 9** — academic / boundary cases; only if a user hits one or
   if it bundles cheaply with another GPU-codegen workstream
   (vast.ai required for CUDA verification).

WS 9 is the sole remaining ROADMAP item and doesn't block any
current work.

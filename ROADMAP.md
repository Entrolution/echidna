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
| WS 5 | Optim error-API enrichment | _pending merge_ | `PiggybackError::*Divergence` gains `last_norm`; `SparseImplicitError::Residual` gains `tolerance` + `dimension`; `SparseImplicitError::{StructuralFailure, FactorFailed}` gain `source: Box<dyn Error>` with `source()` chain; `MaxIterations` Display no longer leaks `Some(...)` / `None`; echidna-optim 0.10.0 → 0.11.0 |

---

## Active workstreams

The remainder are deferrals surfaced during WS1–4 review-fix cycles
that were intentionally not folded into those PRs — either because
they were out of scope, would have required GPU recoordination after
the in-flight fix, or because the verified score was below the
auto-fix threshold. None block any current work.

---


## WS 6 — Optim error-API consistency polish (P2)

**Deferred from**: WS3 review-fix (PR #62) architecture review.

**Prior context**: WS3's two error enums were designed independently
(`PiggybackError` first, then `SparseImplicitError`) and use
divergent naming axes and structural choices. None of these are
correctness bugs; they're cleanup that becomes harder once external
callers depend on the names.

**Problem**:
- `PiggybackError::MaxIterations { z_norm: Option<f64>, lam_norm: Option<f64> }` allows the impossible `(None, None)` state in the type system. WS4's `SolverDiagnostics` enum solved the analogous problem by per-solver variants — WS3 chose differently.
- `MaxIterations` Display uses `{:?}` on `Option<f64>`, leaking Rust syntax (`"z_norm = Some(0.0034)"`) into user-facing error messages.
- `PiggybackError` uses `*Divergence` / `MaxIterations` (failure-class noun); `SparseImplicitError` uses `*Failure` / `*Failed` / `*Singular` / `Residual` (mixed: noun-noun, verb-past-participle, adjective, data-named). No single naming axis.
- `assert_send_sync` block (7 lines) is copy-pasted in `piggyback.rs` and `sparse_implicit.rs`. Other workspace error types (`ClarkeError`, `GpuError`) lack the guard entirely.

**Approach**:
1. Split `MaxIterations` into `MaxIterationsTangent { z_norm: f64 }`,
   `MaxIterationsAdjoint { lam_norm: f64 }`,
   `MaxIterationsForwardAdjoint { z_norm: f64, lam_norm: f64 }`.
   `#[non_exhaustive]` absorbs the variant proliferation.
2. Rewrite `MaxIterations*` Display to emit only the populated
   norm(s) without `Some(_)`/`None` syntax, e.g.
   `"piggyback: max_iter exceeded (z_norm = 3.4e-3); raise max_iter or relax tol"`.
3. Pick one naming axis (suggested:
   `<Mode><FailureClass>`) and rename across both enums:
   - `SparseImplicitError`: `StructuralFailure` →
     `StructuralSingular`, `FactorFailed` → `FactorSingular`,
     `Residual` → `ResidualExceeded`.
   - `PiggybackError`: `MaxIterations*` → `IterationsExhausted*`.
4. Hoist `assert_send_sync!` to a workspace macro in
   `echidna-optim/src/lib.rs` (or a small `util` module). Apply to
   `ClarkeError` and `GpuError` while there.
5. Promote the dimension-mismatch `assert!` calls in `implicit.rs`,
   `sparse_implicit.rs`, and `piggyback.rs` to an `Err` variant
   (e.g. `*Error::DimensionMismatch { expected, actual }`). Today
   these panic on bad inputs while every other failure mode returns
   an error — asymmetric and unfriendly to library consumers.
   Deferred from WS7 as out-of-scope; fits naturally with the
   naming-axis sweep so callers only adopt one breaking change.

**Done when**: Both enums use one naming axis; `MaxIterations` is
typestate-impossible to construct in a meaningless shape; Display
output contains no Rust internal syntax; one shared macro covers all
four error types; dimension-mismatch inputs return `Err` rather than
panic.

**Effort**: Small-to-medium. ~70 lines + test updates. Breaking API
(variant renames are visible to callers).
**Risk**: Low. Caught entirely by `cargo check` if any caller
pattern-matches the renamed variants.

---

## WS 8 — `Laurent::hypot` kernel migration

**Deferred from**: WS1 (PR #63), explicit defer in the plan.

**Prior context**: WS1 migrated 15 inline derivative formulas
across `Dual`, `DualVec`, and `Reverse` (via `num_traits_impls.rs`)
to call `src/kernels/`. The `Laurent::hypot` implementation
(`src/laurent.rs` lines 761–799) was intentionally not migrated —
it operates on jet-coefficient arrays with a max-rescale that
isn't expressible via the existing scalar `kernels::hypot_partials`
helper.

**Problem**: `Laurent::hypot` is the last CPU-side HYPOT
implementation that doesn't route through `kernels`. A future
correctness fix to `hypot` semantics on CPU has to be applied in
two places: the kernel (which propagates to Dual/DualVec/Reverse
via WS1 routing) and `Laurent::hypot` separately. WS1 closed three
of four CPU drift surfaces; this is the fourth.

**Approach**:
1. Define `kernels::hypot_jet_rescale<F: Float>(a_coeffs: &[F],
   b_coeffs: &[F], out: &mut [F])` (or similar shape — consult
   existing `taylor_ops::taylor_hypot` for the canonical jet-array
   API). The function takes the two coefficient arrays, applies
   max-rescale, computes `sqrt(sum_sq)` jet-wide, and writes the
   result.
2. `Laurent::hypot` calls the new helper.
3. Optionally: `Taylor::hypot` already delegates to
   `taylor_ops::taylor_hypot` — confirm whether it should also
   route through `kernels::hypot_jet_rescale` for SSOT (probably
   yes; the two implementations differ slightly in how they handle
   the `scale == 0` case).

**Done when**: `Laurent::hypot` body is a kernel call; no inline
max-rescale arithmetic remains in `src/laurent.rs` for hypot.

**Effort**: Small. ~40 lines + 1-2 test points.
**Risk**: Low. `tests/laurent_*` already exercises `Laurent::hypot`.
**Priority**: Low — single call site, no known bug, future-proofing
only.

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

1. **WS 6** — naming/typestate cleanup plus the dimension-mismatch
   `assert!` → `Err` promotion; do before WS3's and WS5's APIs
   freeze in any downstream consumer (cheaper to rename now than
   later).
2. **WS 8** — close the last CPU drift surface; future-proofing only.
3. **WS 9** — academic / boundary cases; only if a user hits one or
   if it bundles cheaply with another GPU-codegen workstream
   (vast.ai required for CUDA verification).

Each can ship as an independent PR. Finishing any one doesn't block
the others.

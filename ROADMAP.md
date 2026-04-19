# Echidna Roadmap

Forward-looking workstreams explicitly scoped and deferred from prior
bug-hunt cycles. Each item here carries enough context to re-enter
cold, has a concrete "done" condition, and is independent enough to
ship on its own branch.

When starting an item, check the "Prior context" for the commit(s)
that introduced the partial fix so you can diff against what was left
behind.

---

## WS 1 — R2 full migration (CPU SSOT expansion)

**Deferred from**: Cycle 6 Phase 9 (commit `9b17193`).

**Prior context**: `src/kernels/mod.rs` was introduced with
`hypot_partials`, `atan2_partials`, `atan_deriv`, `asinh_deriv`,
`acosh_deriv`. Only `src/opcode.rs` was migrated to call them.

**Problem**: `src/dual.rs`, `src/dual_vec.rs`, `src/laurent.rs`,
`src/breverse.rs`, `src/reverse.rs`, and `src/traits/num_traits_impls.rs`
all still carry their own inline copies of these formulas. A CPU-side
fix landed in any one of them will drift from the others — exactly
the class of bug Phase 7 found three times (atan large-|a|, div
small-|b|, hypot Inf). `tests/gpu_cpu_parity.rs` guards GPU drift
against CPU but does not guard CPU drift against itself.

**Approach**:
1. One commit per AD type. Order: `Dual` → `DualVec` → `BReverse` +
   `traits/num_traits_impls.rs` → `Reverse` → `Laurent`.
2. Pattern: call the kernel helper for the partial-derivative pair,
   keep per-type tangent/adjoint composition as-is. Example:
   ```rust
   impl<F: Float> Dual<F> {
       pub fn hypot(self, other: Self) -> Self {
           let h = self.re.hypot(other.re);
           let (da, db) = kernels::hypot_partials(self.re, other.re, h);
           Dual { re: h, eps: da * self.eps + db * other.eps }
       }
   }
   ```
3. `Laurent::hypot` uses jet rescaling; may need a `kernels::hypot_jet`
   variant. Handle last so the simpler cases pin the pattern.

**Done when**: `grep -n 'a.hypot\|atan2\|atan_deriv\|asinh_deriv\|acosh_deriv'`
across `src/{dual,dual_vec,laurent,breverse,reverse}.rs` and
`src/traits/num_traits_impls.rs` shows only kernel calls, not inline
formulas.

**Effort**: Medium. 5 commits, ~20-30 function bodies.
**Risk**: Low. `tests/gpu_cpu_parity.rs` + existing per-type unit
tests catch any regression.

---

## WS 2 — Higher-order Taylor HYPOT rescale on GPU

**Deferred from**: Cycle 6 Phase 7 Commit 1 (commit `a4ed834`).

**Prior context**: The primal (`r.v[0]`) of Taylor HYPOT was fixed
with max-rescale (WGSL) / `hypot()` builtin (CUDA). The comment in
`src/gpu/taylor_codegen.rs` explicitly flags higher-order coefficients
as a follow-up.

**Problem**: Higher-order jet coefficients still pass through
`jet_mul(a, a)` / `jet_add` / `jet_sqrt` without rescaling. For
`a.v[0] ~ 1e20` or similar, `a.v[0] * a.v[0] = Inf` in f32, so
`v[1]..v[K-1]` can produce spurious Inf / NaN.

**Approach**:
1. Compute `h = max(|a.v[0]|, |b.v[0]|)` once.
2. Build scaled jets `a_s = a * (1/h)`, `b_s = b * (1/h)` via a
   `jet_scale` helper (new).
3. Compute `sum_sq = jet_add(jet_mul(a_s, a_s), jet_mul(b_s, b_s))`
   on the scaled jets — leading-order magnitude is bounded ≤ 2.
4. `r_s = jet_sqrt(sum_sq)`, then scale back: `r = r_s * h`.
5. Primal stays patched as today (the primal path never overflows
   after the Phase 7 fix).

**Done when**: A GPU parity test at `hypot(1e20, 1e19)` reads
`v[1] ≈ 0.95` and `v[2]..v[K-1]` finite, matching CPU Taylor to the
documented ULP budget.

**Effort**: Medium. ~100 lines of codegen per backend (WGSL + CUDA).
Needs runtime verification on both.
**Risk**: Moderate. Codegen complexity; high-order Taylor failures
can hide from simple parity tests.

---

## WS 3 — Richer error types for piggyback / sparse_implicit

**Deferred from**: Cycle 6 Phase 6 (commit `f300f4a`) and Phase 8
review. Both the Phase 6 review-fix and the Phase 7/8 retroactive
review flagged the Option-collapse issue.

**Prior context**: `piggyback_tangent_solve`,
`piggyback_adjoint_solve`, `piggyback_forward_adjoint_solve`, and
`build_fz_and_factor` + the three `implicit_*_sparse` callers all
return `Option<T>`. Failure modes that should be distinguishable
(primal divergence, tangent divergence, adjoint divergence, structural
singularity, numeric singularity, residual exceeds tolerance, max-iter
exhausted) collapse to a single `None`.

**Problem**: Callers can't decide whether to retry (numeric blip with
different seed), re-formulate (structural singularity), or give up
(non-contractive).

**Approach**:
1. Define `enum PiggybackError { PrimalDivergence, TangentDivergence,
   AdjointDivergence, MaxIterations }` in
   `echidna-optim/src/piggyback.rs`.
2. Define `enum SparseImplicitError { StructuralSingular,
   NumericSingular { residual: f64 }, NumericBlowup, FactorFailed }`
   in `echidna-optim/src/sparse_implicit.rs`.
3. Change return types from `Option<T>` to `Result<T, E>`. Update
   internal `return None` sites to `return Err(Error::Variant)`.
4. Update existing tests (`.is_none()` → `.is_err()` in most cases).

**Done when**: No `Option<_>` return on public functions in those two
files; error enums exported; piggyback + sparse_implicit tests use
`.is_err()` or pattern-match specific variants.

**Effort**: Small-to-medium. ~30-50 lines of types + method signature
changes. Breaking API (minor version bump in echidna-optim).
**Risk**: Low. Pure API enrichment.

---

## WS 4 — Solver diagnostics (L-BFGS / Newton silent-filter surface) ✓ DONE

Merged via PR #60 (commit `ee95611`). `OptimResult.diagnostics` now
exposes per-solver counters via the `SolverDiagnostics` enum:
`LbfgsDiagnostics` (pairs accepted/rejected/evicted, gamma clamps,
line-search backtracks), `NewtonDiagnostics` (fallback steps, line-
search backtracks), `TrustRegionDiagnostics` (CG iters, two split
radius-shrink branches). `OptimResult` is `#[non_exhaustive]`.

**Original problem statement:**

**Deferred from**: Cycle 6 Phase 6 (commit `f300f4a`), error-handling
audit.

**Prior context**: L-BFGS silently drops curvature pairs that fail
`sy > eps * sqrt(ss * yy)` and silently clamps gamma to
`[1e-3, 1e3]`. Newton silently substitutes steepest-descent when the
LU solve returns a non-descent direction. None of these are visible
to the caller.

**Problem**: A solver that filters every pair reports
`TerminationReason::GradientNorm` but has actually been running
steepest-descent-with-gamma-1-clamp the whole time. A solver hitting
Newton fallback on every iteration reports success but has been
doing something suboptimal. No way for a user to tell.

**Approach**:
1. Add `pub struct SolverDiagnostics { pub pairs_accepted: usize,
   pub pairs_rejected: usize, pub gamma_clamp_hits: usize,
   pub fallback_steps: usize }` in `echidna-optim/src/result.rs`.
2. Add `pub diagnostics: SolverDiagnostics` to `OptimResult`. Default
   all-zero for solvers that don't populate it (trust_region, etc.).
3. Wire L-BFGS to increment `pairs_accepted` / `pairs_rejected` /
   `gamma_clamp_hits`. Wire Newton to increment `fallback_steps`.
4. Add a regression test that constructs an adversarial L-BFGS
   problem (e.g. pure steepest-descent by saturating the curvature
   filter) and asserts `pairs_rejected >= N`.

**Done when**: `OptimResult.diagnostics` exposes the counts for the
relevant solvers; regression test pins the adversarial case.

**Effort**: Small. ~30 lines across result.rs + lbfgs.rs + newton.rs.
**Risk**: Low. Non-breaking (new field is ignored by callers that
don't read it).

---

## Suggested order

1. ~~WS 4~~ — done (PR #60).
2. **WS 3** — small-to-medium, breaking but contained.
3. **WS 1** — medium, mechanical, well-guarded.
4. **WS 2** — medium, needs GPU instance up; schedule around vast.ai
   availability.

Each can ship as an independent PR. Finishing any one doesn't
block the others.

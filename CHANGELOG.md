# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **simba traits for DualVec**: implemented traits `SimdValue`, `PrimitiveSimdValue`, `SubsetOf`, `AbsDiffEq`, `RelativeEq`, `UlpsEq`, `Field`, `ComplexField`, `RealField` for `DualVec<F, N>`.

## [0.10.0] - 2026-04-24

**Coordinated release:** `echidna` 0.10.0 and `echidna-optim` 0.13.0.
`echidna-optim`'s dep on `echidna` updated from `0.9.0` to `0.10.0`.

Minor bump (rather than patch) because `wgpu` 28 → 29 is a transitive
major for anyone enabling the `gpu-wgpu` feature. No echidna public-API
types changed; the bump reflects the wgpu API-break that downstream
`gpu-wgpu` consumers will see in their own `Cargo.lock`.

### Changed (echidna)

- **`wgpu` dependency bumped to 29** (`gpu-wgpu` feature). Backend
  migration: `PipelineLayoutDescriptor::bind_group_layouts` now takes
  `&[Option<&BindGroupLayout>]` instead of `&[&BindGroupLayout]`, so
  each entry is wrapped in `Some(...)` at the five pipeline creation
  sites in `src/gpu/wgpu_backend.rs`. Additionally, wgpu 29 tightened
  `Limits::downlevel_defaults()` — `max_storage_buffers_per_shader_stage`
  dropped to a level that no longer fits our tangent-reverse pipeline's
  13 storage buffers (5 tape + 8 I/O). The backend now requests the
  limit explicitly in `DeviceDescriptor` and resolves it against the
  adapter's actual capability via `using_resolution(adapter.limits())`.
  Older drivers at the floor keep working; modern drivers get the
  13-buffer pipeline. No public `WgpuContext` API changes.
- `src/gpu/mod.rs::compute_chunk_size`: replaced manual guarded
  division (`if nv_k > 0 { chunk.min(u32::MAX as u64 / nv_k) }`)
  with `u64::checked_div` for identical behaviour and a cleaner
  expression.

### Added (infrastructure)

- **TLA+ spec-to-code sync enforcement** (`.github/workflows/specs.yml`,
  `specs/verify_anchors.sh`, `tests/spec_invariants_*.rs`). Every TLA+
  invariant in `specs/README.md` now carries a `// SPEC: <InvariantName>`
  anchor at the Rust line upholding it (21 anchors in `src/checkpoint.rs`,
  `src/bytecode_tape/optimize.rs`, `src/bytecode_tape/mod.rs`). A new
  `verify-anchors` CI job parses the cross-reference table and greps
  sources, failing on any missing anchor. Two new test files
  (`tests/spec_invariants_checkpoint.rs`,
  `tests/spec_invariants_tape_optimize.rs`) exercise the specs'
  properties against the real Rust implementation over deterministic
  bounded input spaces (gradient equality across checkpointing
  strategies; `optimize ∘ optimize = optimize`; post-optimise structural
  assertions gated on `cfg(debug_assertions)`; explicit high-N/low-C
  online-thinning witness).
- `README.md` and `CONTRIBUTING.md` document the three gates (source
  anchors, semantic property tests, TLC model checking) with local
  run commands for each.

### Changed (infrastructure)

- CI `Lint` job now runs `cargo clippy --all-targets -- -D warnings`
  on both `echidna` and `echidna-optim`. Previously `--all-targets` was
  omitted, letting benches and integration-test crates accumulate lint
  debt (~147 warnings + several hard compile errors) invisibly. The
  backlog was cleared first (bench `black_box` deprecation, redundant
  closures, a `num-dual`-gradient API break in `benches/comparison.rs`,
  scientific-idiom `needless_range_loop` allows, a stale `3.14`
  literal flagged as approximating PI), then CI was tightened.
- Dependabot maintenance: `codecov/codecov-action` v5 → v6,
  `actions/setup-java` v4 → v5, `softprops/action-gh-release` v2 → v3.

### Changed (echidna-optim, 0.13.0)

- `echidna` dep updated from `0.9.0` to `0.10.0` (follows the
  coordinated release; `echidna-optim`'s public API is unchanged).
- Test hygiene: `assign_op_pattern` and `field_reassign_with_default`
  clippy fixes in `tests/sparse_implicit.rs` and
  `tests/ws4_diagnostics.rs`. No runtime behaviour change.

## [0.9.0] - 2026-04-19

**Coordinated release:** `echidna` 0.9.0 and `echidna-optim` 0.12.0.
`echidna-optim`'s dep on `echidna` updated from `0.8.2` to `0.9.0`.

### Added (echidna)
- `echidna::assert_send_sync!` macro (`#[macro_export]`) for
  compile-time `Send + Sync + 'static` assertions on error types.
  Applied to `ClarkeError` (`src/nonsmooth.rs`) and `GpuError`
  (`src/gpu/mod.rs`), both of which previously had no guard —
  future variants carrying non-`Send` / non-`Sync` / non-`'static`
  payloads now produce a build-time failure at the type definition
  rather than at the (often distant) call site where the error
  meets an `async` boundary or a threadpool. Also adopted by
  `echidna-optim`'s three error types in place of copy-pasted
  inline `const _: fn() = || { ... }` blocks. The macro tightens
  the previously-inline `Send + Sync`-only bound by also
  requiring `'static`; all five error types satisfy it today and
  no caller relied on a non-`'static` payload, but callers adding
  an error type with a borrowed field must now use a different
  pattern.

### Changed (echidna)
- `Laurent::hypot` now routes its rescale + sum-of-squares + sqrt
  recipe through `taylor_ops::taylor_hypot`, the shared CPU HYPOT
  kernel already used by `Taylor::hypot` and `TaylorDyn::hypot`.
  Eliminates the last CPU HYPOT implementation outside the shared
  kernel. Public signature unchanged; the common matched-pole-order
  case produces identical output. Mismatched pole orders are
  handled by an explicit rebase-to-`min(pole_order_a, pole_order_b)`
  prelude that shifts the higher-pole operand's coefficient array
  right by the delta (zero-filling leading slots, truncating past
  index `K-1`) — matches the Laurent `*` / `+` truncation semantics.
  The cone-singularity case `hypot(zero, zero)` short-circuits
  ahead of the kernel call so the zero Laurent stays all-zero
  (delegating through the kernel would produce a pole-of-order-1
  with Inf coefficients after `normalize()` — nonsense for the
  cone point). Note: the kernel's `scale == 0` recursive shift-
  and-square branch is unreachable from normalized Laurent inputs
  (`Laurent::new`'s normalization absorbs leading zeros into
  `pole_order`), so it has no Laurent-level behavioural effect.
- **GPU Taylor jet `HYPOT` higher-order coefficients** now match CPU
  `Taylor::hypot` via max-rescale across both WGSL and CUDA codegen
  (`src/gpu/taylor_codegen.rs`). Pre-WS2, the primal was patched
  (Phase 7) but `v[1]..v[K-1]` still passed through unscaled
  `jet_mul(a, a) + jet_mul(b, b)`, overflowing to Inf/NaN at
  `a.v[0] ~ 1e20` in f32 — silently corrupting GPU Hessians and
  higher Taylor coefficients of `hypot` at extreme magnitudes.
  The new emission rescales by `h = max(|a.v[0]|, |b.v[0]|)`,
  computes the sum-of-squares on the bounded scaled jets, then
  scales the result back, mirroring `taylor_ops::taylor_hypot`.
  Both branches explicitly zero higher-order coefficients before
  early-return (uninitialised `var r: JetK` hazard).
- **GPU Taylor jet `HYPOT` function-domain-boundary cases** now match
  CPU `taylor_ops::taylor_hypot` across both WGSL and CUDA codegen
  (`src/gpu/taylor_codegen.rs`). Three boundary conventions closed:
    - `hypot(Inf, finite)` — primal stays Inf (IEEE); higher-order
      coefficients switched from zero to NaN, matching CPU's
      `Inf * 0 = NaN` rescale-path output. NaN synthesised at
      runtime via `inf - inf` (WGSL / CUDA) rather than a compile-
      time bitcast, to stay portable across driver fast-math
      settings.
    - `hypot(0, 0)` with non-zero first-order seed — GPU previously
      returned the zero jet; now emits a one-level shift-and-square
      unroll that reproduces CPU's `|t| · hypot(a/t, b/t)` expansion.
      CPU recursion is at most one level deep (the entry check
      `a[1] != 0 || b[1] != 0` guarantees the recursive call lands
      in the general `scale > 0` path), so the unroll is ~30 WGSL /
      CUDA lines per backend rather than a `K`-bounded loop.
    - `hypot(0, 0)` with all higher-order seeds also zero — GPU
      previously returned the zero jet; now emits 0 primal + Inf
      higher-order, matching CPU's `taylor_sqrt` at a zero leading
      coefficient (`taylor_ops.rs:146-152`). Primal override brings
      `c[0]` back to 0 from the `taylor_sqrt`'s implicit Inf.
  Pinned by `ws9_{wgpu,cuda}_hypot_*` tests in `tests/gpu_stde.rs`
  (zero-origin shift-and-square parity, Inf-finite NaN propagation,
  deeper-order-zero Inf-higher convention). Callers that were
  defensively treating GPU zero higher-order as a "singularity
  detected" sentinel may see the new Inf/NaN outputs; CPU callers
  at those inputs already observed the new values, so CPU+GPU now
  behave identically.
- **Internal**: `Dual`, `DualVec`, and `Reverse` now route per-component
  partial-derivative computation for `atan`, `atan2`, `asinh`, `acosh`,
  and `hypot` through the shared `src/kernels/` module — eliminating
  the CPU-vs-CPU drift surface that produced three Phase 7 bugs (atan
  large-|a|, hypot Inf, atan2 overflow). No public API change.
- **Numerics**: `acosh` derivative (and the WGSL `acosh_f32` / CUDA
  `acosh_f` primal helpers) uniformly switched from the unfactored
  `sqrt(a²-1)` form to the factored `sqrt((a-1)·(a+1))` form, which
  retains the `ε²` contribution at `a = 1 + ε` (the unfactored form
  rounds it away in both f32 and f64). Applied across CPU
  (`kernels::acosh_deriv`), the three WGSL shaders (`forward.wgsl`,
  `tangent_forward.wgsl`, `tangent_reverse.wgsl`, plus the WGSL
  reverse derivative in `reverse.wgsl`), the CUDA kernel
  (`tape_eval.cu` at all three derivative sites), and the Taylor jet
  codegen for both backends (`taylor_codegen.rs`, including the
  primal `acosh_f` helper). All sweeps (forward, reverse,
  tangent_forward, tangent_reverse) updated. Pinned by a CPU unit test
  in `kernels::tests` that compares factored vs unfactored output at
  `a = 1 + 1e-12` and asserts they differ — any swap back trips the
  test. f32 GPU precision near `a = 1` remains dominated by the
  input quantisation rather than formula choice, so
  `tests/gpu_cpu_parity.rs` continues to probe at moderate inputs
  (`a ∈ {1.5, 2.0, 10.0}`); near-one regression coverage lives in
  the f64 kernel unit test.

### Added (echidna-optim)
- `OptimResult.diagnostics: SolverDiagnostics` exposes per-solver
  internal counts that were previously silent — curvature pair
  acceptance / rejection / memory eviction in L-BFGS, gamma clamp
  hits, line-search backtracks, Newton steepest-descent fallback
  steps, and trust-region CG iterations and radius-shrink branch
  counts (split into `bad_model` vs `low_rho`). Use these to detect
  when a solver reports `GradientNorm` convergence but actually spent
  most of its work in fallback or filter paths.
- New public types: `SolverDiagnostics` (enum), `LbfgsDiagnostics`,
  `NewtonDiagnostics`, `TrustRegionDiagnostics`.
- `PiggybackError` enum (`piggyback` module) with variants
  `PrimalDivergence { iteration, last_norm }`,
  `TangentDivergence { iteration, last_norm }`,
  `AdjointDivergence { iteration, last_norm }`,
  `IterationsExhaustedTangent { iteration, z_norm }`,
  `IterationsExhaustedAdjoint { iteration, lam_norm }`,
  `IterationsExhaustedForwardAdjoint { iteration, z_norm, lam_norm }`,
  and `DimensionMismatch { field: &'static str, expected, actual }`.
  Implements `Display` + `std::error::Error`, `Send + Sync + 'static`,
  `#[non_exhaustive]`. `last_norm` surfaces the iteration-level
  relative norm at the detecting point (non-finite on norm-check
  paths; finite in ratio-converging componentwise cases). The three
  typestate-split exhaustion variants replace the earlier
  `MaxIterations { Option<f64>, Option<f64> }` shape; each carries
  exactly the norm(s) its solver tracks.
- `SparseImplicitError` enum (`sparse_implicit` module, gated on the
  `sparse-implicit` feature) with variants
  `StructuralSingular { source }`, `FactorSingular { source }`,
  `NumericSingular`, `ResidualExceeded { relative_residual,
  tolerance, dimension }`, and `DimensionMismatch { field, expected,
  actual }`. Implements `Display` + `std::error::Error` (with
  `source()` returning the underlying
  `faer::sparse::{CreationError, linalg::LuError}` for the two
  sourced variants), `Send + Sync + 'static`, `#[non_exhaustive]`.
  Does not derive `Clone` (the `Box<dyn Error + Send + Sync>`
  sources aren't `Clone`).
- `ImplicitError` enum (`implicit` module) with a `Singular` variant.
  Implements `Display` + `std::error::Error`, `Send + Sync`,
  `#[non_exhaustive]`. The single variant collapses three failure
  modes that the dense LU does not currently distinguish:
  exactly-zero pivot (structural), pivot below `ε·n·‖F_z‖∞`
  (numerical), or non-finite pivot / second-order RHS (NaN / ±Inf
  caught by the newly added finite-pivot guard in `lu_factor` and by
  post-solve guards in `implicit_hvp` / `implicit_hessian`). Sparse's
  richer variant set doesn't carry over because dense runs only one
  pivot-level check; future subdivision remains additive.

### Fixed (echidna-optim)
- `linalg::lu_factor` now rejects non-finite pivots (NaN / ±Inf) up
  front. IEEE comparisons against NaN return `false`, so without this
  guard a NaN pivot silently passed both the exact-zero and
  tolerance checks and propagated through the stored LU factors as a
  successful solve — callers downstream (including the dense
  `implicit_*` entry points and the Newton solver) then returned
  NaN-tainted results under an `Ok` / `Some` label. A NaN / ±Inf
  pivot now short-circuits to `None`, surfaced to `implicit_*`
  callers as `ImplicitError::Singular`. All five dense implicit
  entry points (`implicit_tangent`, `implicit_adjoint`,
  `implicit_jacobian`, `implicit_hvp`, `implicit_hessian`)
  additionally check their back-solve output for non-finite entries,
  catching NaN that reaches the solve RHS without `F_z` itself being
  non-finite — e.g. a NaN `x_dot` / `z_bar`, a tape where `∂F/∂x`
  goes non-finite independently of `∂F/∂z`, or a nested-dual forward
  pass producing non-finite higher-order coefficients at a
  function-domain boundary.
### Changed (echidna-optim) — BREAKING
- `OptimResult` is now `#[non_exhaustive]` so future field additions
  don't keep breaking downstream pattern-destructures. Construct
  results via the solver entry points (`lbfgs`, `newton`,
  `trust_region`); never with a struct literal.
- `piggyback_tangent_solve`, `piggyback_adjoint_solve`, and
  `piggyback_forward_adjoint_solve` now return
  `Result<T, PiggybackError>` instead of `Option<T>`. Failure modes
  that previously collapsed to `None` (primal divergence, tangent
  divergence, adjoint divergence, max-iter exhaustion) are now
  distinguishable so callers can decide whether to retry, re-formulate,
  or give up. Migration: `.is_none()` → `.is_err()`,
  `.unwrap()` / `.expect(...)` continue to work; callers that pattern-
  matched on `Some(_) | None` now match on `Ok(_) | Err(_)`.
- `implicit_tangent_sparse`, `implicit_adjoint_sparse`, and
  `implicit_jacobian_sparse` now return
  `Result<T, SparseImplicitError>` instead of `Option<T>`. The error
  variant pinpoints whether failure was structural (matrix
  construction), factor-failure (faer LU), numeric singularity (probe
  produced non-finite output), or residual exceedance (probe finite
  but `||F_z·x − rhs|| / ||rhs||` over `sqrt(eps)·sqrt(m)`). Same
  migration path as the piggyback functions.
- `implicit_tangent`, `implicit_adjoint`, `implicit_jacobian`,
  `implicit_hvp`, and `implicit_hessian` now return
  `Result<T, ImplicitError>` instead of `Option<T>`, closing the API
  asymmetry with the sparse counterparts shipped alongside. Same
  migration mechanic as the piggyback / sparse_implicit conversions:
  `.is_none()` → `.is_err()`; `.unwrap()` / `.expect(...)` continue
  to work; callers that pattern-matched on `Some(_) | None` now
  match on `Ok(_) | Err(_)`.
- `PiggybackError::{PrimalDivergence, TangentDivergence, AdjointDivergence}`
  gain `last_norm: f64` — the relative-convergence norm at the
  detecting iteration. Distinguishes Inf-overflow from
  NaN-cancellation when non-finite; surfaces a finite bounded value
  in the ratio-converging case (tangent / `lambda_new` componentwise
  overflow while the primal/adjoint norm stayed bounded). Captured
  inline at each `Err` return so the reported scalar reflects the
  failing iteration, not the previous one.
- `SparseImplicitError::ResidualExceeded` (renamed from `Residual`)
  gains `tolerance: f64, dimension: usize` — the probe threshold
  (`sqrt(eps)·sqrt(m)`) and the state-block dimension `m`
  (= `num_states`). Callers can now see how far over tolerance the
  probe solve landed without re-deriving the threshold.
- `SparseImplicitError::{StructuralSingular, FactorSingular}`
  (renamed from `StructuralFailure` / `FactorFailed`) gain
  `source: Box<dyn std::error::Error + Send + Sync + 'static>`;
  `SparseImplicitError` now implements
  `std::error::Error::source()` returning the underlying
  `faer::sparse::CreationError` / `LuError`. Previously
  `map_err(|_| ...)` discarded the faer diagnostic; callers
  hitting `FactorSingular` typically get `SymbolicSingular { index }`
  from the chained source, pinpointing the failing column.
- `SparseImplicitError` no longer derives `Clone` (the new
  `Box<dyn Error>` field on two variants is not `Clone`). No
  workspace consumer cloned the type.
- `PiggybackError::MaxIterations` replaced by three typed variants
  (typestate split):
  `IterationsExhaustedTangent { iteration, z_norm }`,
  `IterationsExhaustedAdjoint { iteration, lam_norm }`,
  `IterationsExhaustedForwardAdjoint { iteration, z_norm, lam_norm }`.
  The impossible `(None, None)` state is now unrepresentable; each
  variant carries exactly the norm(s) its solver tracks, as plain
  `f64` rather than `Option<f64>`. Each variant also adds
  `iteration: usize` (= `max_iter`) for parity with the three
  `*Divergence` variants.
- `SparseImplicitError` variant renames for naming-axis unity:
  `StructuralFailure → StructuralSingular`,
  `FactorFailed → FactorSingular`,
  `Residual → ResidualExceeded`. Payload fields unchanged; only the
  names move onto the `<Mode><FailureClass>` axis already used by
  `NumericSingular`.
- `PiggybackError`, `SparseImplicitError`, and `ImplicitError` each
  gain a `DimensionMismatch { field: &'static str, expected: usize,
  actual: usize }` variant. 14 existing + 1 new operation-time
  runtime-argument `assert_eq!` sites in public `*_solve` /
  `implicit_*` fns now surface as this `Err` variant (4 implicit +
  8 sparse_implicit + 2 piggyback existing + 1 new inline check in
  `piggyback_tangent_solve`). Construction-time
  (`SparseImplicitContext::new`), tape-shape (`validate_*`
  helpers), and step-fn (`piggyback_tangent_step[_with_buf]`)
  assertions remain panics per the programmer-contract convention.
- `echidna-optim` version bumped to `0.12.0` to reflect the
  breaking variant renames, `MaxIterations` typestate split, and
  new `DimensionMismatch` variant on three error enums.

### Removed (echidna-optim)
- Inline `const _: fn() = || { ... assert_send_sync ... }` blocks
  in `echidna-optim/src/piggyback.rs`,
  `echidna-optim/src/sparse_implicit.rs`, and
  `echidna-optim/src/implicit.rs` — replaced by the hoisted
  `echidna::assert_send_sync!` macro invoked once per error type.
- `PiggybackError::MaxIterations` (see `### Changed` — replaced by
  three typestate variants).
- The WS5-introduced 4-arm `MaxIterations` Display match in
  `piggyback.rs` — superseded by per-variant Display impls in the
  typestate split; each new variant has a single-arm Display, so
  the defensive `(None, None)` case no longer exists.

## [0.8.2] - 2026-04-12

### Fixed

#### Core AD (all modes)
- **atan derivative overflow**: `1/(1+x²)` overflows to 0 for `|x| > ~1.34e154` (f64). Now uses `(1/x)²/(1+(1/x)²)` for large inputs across Dual, DualVec, Reverse, and bytecode reverse_partials.
- **powf derivative underflow**: when `x^b` underflows to 0 but `x ≠ 0`, the derivative `b·x^b/x` silently returned 0. Now falls back to direct `b·x^(b-1)` computation. Fixed in all 4 AD modes.
- **powi(i32::MIN) f32 precision**: converting `i32::MIN - 1` to f32 rounds the exponent. Now uses `x^n/x` to compute `x^(n-1)` without float conversion. Fixed across 7 code sites (dual, dual_vec, Reverse, opcode, bytecode reverse, bytecode tangent, cross-country).
- **taylor_acosh cancellation**: `a²-1` near `a=1` suffered catastrophic cancellation. Replaced with factored form `(a-1)(a+1)`, matching sister functions asin/atanh.
- **Taylor/TaylorDyn max/min NaN handling**: `max(valid, NaN)` returned NaN instead of the valid value. Added IEEE 754 fmax/fmin NaN guard matching Dual/Laurent implementations.
- **Taylor::derivative factorial overflow**: standalone `k!` computation overflows f64 at k=171. Now interleaves multiplication with the coefficient, extending the usable range.

#### GPU (WGSL + CUDA codegen)
- **WGSL u32 index overflow**: `forward_batch`, `gradient_batch`, and `hvp_batch` now assert `batch_size × num_variables ≤ u32::MAX` before GPU dispatch. Previously, large workloads silently produced corrupted results.
- **POWI Taylor codegen at x=0**: `0^n` for n=2,3,4 now uses repeated `jet_mul` instead of `jet_const(0)`, preserving higher-order Taylor coefficients (e.g., Hessian of `x²` at zero). Both WGSL and CUDA codegen.
- **WGSL abs reverse NaN divergence**: `abs` reverse derivative returned 0 for NaN inputs instead of propagating NaN. Now matches CUDA/CPU behavior.
- **WGSL signum(-0.0)**: now uses `bitcast<u32>` sign-bit check to correctly return -1 for negative zero, matching Rust's `f32::signum`.

#### Checkpointing
- **Revolve checkpoint memory budget**: `revolve_schedule` produced O(num_steps) checkpoint positions instead of O(num_checkpoints). Forward pass now truncates to the budget. Gradients were always correct; only memory usage was affected. Fixed in all 3 call sites (standard, hints, disk).

#### STDE
- **diagonal_kth_order_const f32 guard**: const-generic variant now rejects k ≥ 13 for f32, matching the dynamic version's precision guard.
- **Weighted estimator finiteness**: `estimate_weighted` now asserts sample finiteness, matching `WelfordAccumulator` behavior.
- **Factorial guard comment**: corrected misleading "overflows f64 for k > 20" (actual overflow is k=171; precision loss begins at k=19).

#### echidna-optim
- **Trust region boundary_tau cancellation**: quadratic formula now uses Vieta's formula to avoid catastrophic cancellation when `|b| ≈ √discriminant`.
- **L-BFGS step vector cancellation**: `s = (x + α·d) - x` replaced with `s = α·d` to avoid cancellation when `‖x‖ ≫ α·‖d‖`.

### Added
- 9 new regression tests covering all fixed issues.
- Clarifying comments at 8 code sites frequently flagged as false positives during review (zero-adjoint skip, primal patching, powi encoding, Rem derivative, atan2 at origin, sqrt/cbrt singularity, custom op linearization).

## [0.8.1] - 2026-04-12

### Fixed

#### GPU kernels (CUDA + WGSL)
- **cbrt HVP second derivative**: tangent_reverse kernels computed `-2at/(9r³)` instead of `-2at/(9r⁵)`, producing wrong Hessian-vector products through `cbrt`. Fixed in both CUDA and WGSL.
- **asin/acos/atanh cancellation in GPU shaders**: GPU derivative formulas used `1-a*a` which loses ~15 digits near |a|→1. Replaced with `(1-a)*(1+a)` across all GPU kernels (reverse, tangent_forward, tangent_reverse) for CPU-GPU parity.
- **CUDA Taylor codegen u32 truncation**: generated code assigned 64-bit `j_base` to `unsigned int` intermediates, silently truncating for large tapes. All offset variables now use `unsigned long long`.

#### Core AD (all modes)
- **asin/acos/atanh catastrophic cancellation**: `1 - x*x` replaced with `(1-x)*(1+x)` in Dual, DualVec, Reverse, bytecode reverse_partials, and Taylor recurrences to preserve precision near domain boundaries.
- **atan2 bytecode overflow**: `a*a + b*b` replaced with `hypot(a,b)²` in bytecode reverse_partials, preventing zero derivatives for large inputs (|a| or |b| > ~1.34e154).
- **Division derivative overflow**: quotient rule restructured from `(a'b - ab') * inv²` to `(a' - a*inv*b') * inv`, avoiding intermediate overflow for small denominators.
- **Taylor hypot overflow/underflow**: inputs rescaled by `max(|a₀|, |b₀|)` before squaring, preventing silent zeroing of derivative coefficients for large inputs and infinity for small inputs.

#### Bytecode tape
- **Custom ops in Hessian (release builds)**: `debug_assert!` promoted to `assert!` for custom-ops guards in `hessian_vec`, `sparse_hessian_vec`, and `sparse_jacobian_vec`, preventing silently wrong second derivatives in release builds.
- **Serde Custom opcode rejection**: deserialization now rejects tapes containing Custom opcodes (which have no serializable callback) instead of silently accepting them.
- **Per-type thread-local borrow guards**: borrow guards for `with_active_tape` and `with_active_btape` are now per-type instead of global, preventing false reentrance panics when nesting different float types on the same thread.

#### STDE
- **Welford negative variance**: `m2.max(0.0)` clamp prevents NaN standard errors from floating-point cancellation in nearly-identical samples.
- **estimate_weighted zero-weight division**: guarded `w_sum2 / w_sum` against all-zero weights producing NaN.
- **Gram-Schmidt epsilon**: replaced hardcoded `1e-12` with `F::epsilon().sqrt()` in Hutch++, fixing f32 compatibility.
- **Higher-order f32 precision guard**: `diagonal_kth_order` now rejects k ≥ 13 for f32 (k! exceeds f32 mantissa precision).

#### GPU infrastructure
- **WGSL u32 index overflow**: chunking logic now caps `chunk_size` so `bid * num_variables * K` cannot exceed `u32::MAX`.

#### echidna-optim
- **L-BFGS rho overflow**: curvature pair acceptance tightened from `sy > 0` to `sy > epsilon * yy`, preventing infinite `rho = 1/sy` from near-zero curvature.
- **LU singularity threshold**: replaced `sqrt(epsilon)` with relative threshold `epsilon * n * max_pivot`, adapting to both f32 and matrix scale. Explicit zero-pivot check added.

### Added
- 5 additional `debug_assert!` → `assert!` promotions for correctness-critical guards (Welford finite-sample, Laurent pole guard, GPU dispatch u32 bounds).
- 35 new regression/structural tests:
  - 8 boundary-value derivative tests (asin/acos/atanh near ±1, atan2 large inputs, division small denominator, Taylor hypot extremes)
  - 2 Welford accumulator edge case tests (nearly-identical samples, all-zero weights)
  - 5 GPU chunking safety tests (u32 overflow prevention)
  - 4 f32 derivative correctness tests (cross-validation + diagonal_kth_order)
  - 15 per-opcode GPU HVP parity tests (exp, log, sqrt, cbrt, recip, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, powf)
  - 1 serde Custom opcode rejection test (existing assert test updated)

## [0.8.0] - 2026-04-11

### Fixed

#### GPU Taylor kernels (CUDA + wgpu, codegen)
- **POWI at x=0**: GPU now returns `jet_const(0)` for `0^n` (n>=2) instead of NaN from `ln(0)`. Negative exponents at x=0 correctly produce Inf.
- **REM higher-order coefficients**: GPU REM now loads the full b jet and computes `r[k] = a[k] - q * b[k]`, matching CPU Taylor behavior for non-constant divisors.
- **POWF a<=0 higher-order coefficients**: GPU now propagates explicit NaN for c2+ when base is non-positive, matching CPU `powf` IEEE semantics. Previously silently zeroed.
- **POWF WGSL K=1 out-of-bounds**: guarded `r.v[1]` write with `if k > 1`, matching CUDA.
- **ATAN2 b=0 higher-order coefficients**: GPU now computes full Taylor composition via `jet_div(b,a)` + `jet_atan` + negate, matching CPU for K>=3.
- **`_sign` consistency**: CUDA codegen and `tape_eval.cu` now use `copysign(1, x)` matching Rust's `signum` for `-0.0`. WGSL uses `select` (cannot distinguish `-0.0`).
- **u32 index overflow**: all CUDA kernel index arithmetic widened to `unsigned long long` to support batch_size * num_variables > 2^32.

#### Core AD
- **atan2 underflow**: `Dual`, `DualVec`, and `Reverse` now use `hypot` for the denominator in `atan2`, preventing underflow for very small inputs (~1e-200).
- **taylor_cbrt**: uses `c.len()` (output length) for iteration, consistent with all other Taylor functions.
- **Nonsmooth NaN consistency**: `is_smooth` and `active_kinks` now handle NaN switching values consistently (NaN = not smooth, appears in active kinks).
- **Fract kink tracking**: `OpCode::Fract` added to `is_nonsmooth` and `forward_nonsmooth` for kink proximity detection.

#### Laurent series
- **Sub assertion parity**: `Laurent::Sub` now has the same pole-order gap assertion as `Add`, preventing silent truncation.
- **`is_zero` semantics**: `Laurent::is_zero()` now checks all coefficients via `is_all_zero_pub()`, correctly returning `false` for series with `pole_order > 0` and nonzero coefficients.
- **max/min NaN handling**: `Laurent::max/min` now return the non-NaN argument, matching `Dual`/`Reverse`/`BReverse`.
- **powi pole_order overflow**: uses `checked_mul` instead of unchecked `i32` multiplication.

#### echidna-optim
- **Piggyback forward-adjoint stale x_bar**: `piggyback_forward_adjoint_solve` now performs a final reverse pass with the converged lambda, matching `piggyback_adjoint_solve` and eliminating O(tol) gradient bias.
- **NaN gradient detection**: all three solvers (trust region, L-BFGS, Newton) now detect NaN/Inf in gradient or function value and terminate with `NumericalError`.
- **Trust region negative predicted reduction**: rejects step and shrinks radius when the quadratic model predicts worsening, preventing spurious expansion.
- **Steihaug CG tolerance**: uses relative tolerance (`sqrt(epsilon) * ||g||`) instead of absolute `epsilon`, improving CG convergence for both large and small gradients.
- **Trust region radius shrink**: uses `0.25 * radius` (standard algorithm) instead of `0.25 * step_norm`.
- **L-BFGS gamma overflow**: guards against subnormal `y^T y` causing infinite scaling.

#### Bytecode tape
- **`sparse_jacobian_par` custom ops**: forward-mode path now uses `forward_tangent_dual` for correct primal evaluation at fresh inputs.
- **Deserialization validation**: validates structural consistency (lengths, bounds, opcode types) on deserialization, preventing panics from malformed tapes.
- **Hessian custom ops**: `debug_assert!` at entry of `hessian_vec`, `sparse_hessian_vec`, and `sparse_jacobian_vec` warns about approximate second derivatives through custom ops.

### Added
- **Reentrant borrow guards**: `with_active_tape` and `with_active_btape` now detect reentrant calls via RAII guards, panicking instead of creating aliased `&mut` references.
- **u32 overflow guard**: `debug_assert` on tape variable count prevents silent index wrapping.
- 17 regression tests covering all fixed bugs.

## [0.7.0] - 2026-04-11

### Added

- **CUDA kth-order Taylor evaluation**: `CudaContext::taylor_forward_kth_batch` and `taylor_forward_kth_batch_f64` support K=1..5 Taylor jet evaluation, bringing CUDA to parity with wgpu. Kernels are lazy-compiled on first use via `taylor_codegen::generate_taylor_cuda`.
- **`taylor_forward_kth_batch` on `GpuBackend` trait**: promoted from inherent method to trait method, enabling generic GPU STDE code to use arbitrary-order Taylor evaluation on either backend.
- **`taylor_forward_2nd_batch` default trait impl**: delegates to `taylor_forward_kth_batch(order=3)`, eliminating duplicated logic across both backends.
- **CUDA Taylor opcode test parity**: `gpu_stde.rs` refactored with `opcode_tests_for_backend!` macro to run all per-opcode Taylor tests on both wgpu and CUDA.

### Removed

- **Handwritten `taylor_eval.cu`** (527 lines): replaced by codegen K=3 output. The CUDA backend now uses `taylor_codegen::generate_taylor_cuda` for all Taylor kernels.
- **Handwritten `taylor_forward_2nd.wgsl`** (570 lines): replaced by codegen K=3 output. The wgpu backend now uses `taylor_codegen::generate_taylor_wgsl` for all Taylor shaders.
- **`cuda_taylor_fwd_2nd_body!` macro**: no longer needed since `taylor_forward_2nd_batch` delegates to the kth-order path.

### Changed

- **`TrustRegionConfig`** (echidna-optim): new `min_radius: F` field added in 0.6.0 (breaking for direct struct construction; `Default` impls provide it).

## [0.6.0] - 2026-04-11

### Fixed

#### Core AD (all modes: Dual, DualVec, Reverse, BReverse, bytecode tape)
- **Max/Min NaN handling**: `BReverse::max/min` and bytecode `eval_forward`/`reverse_partials` now return the non-NaN argument when one input is NaN, matching `Float::max`/`Float::min` semantics and the behavior of Dual/Reverse modes.
- **atan2(0,0) derivative**: returns zero gradient (matching JAX/PyTorch convention) instead of NaN from division by zero in `x²+y²`.
- **powf(x, 0) derivative**: correctly returns `d/db = ln(x)` for `x > 0` instead of silently dropping the exponent derivative. `d/da = 0` was already correct.
- **powf(x, y) at x=0**: Reverse and bytecode modes now use the `y * x^(y-1)` form (matching Dual) instead of `y * val / x` which produces `0/0 = NaN`.
- **powi_exp_decode**: deleted broken generic float round-trip decoder (failed for f32 negative exponents); replaced with `powi_exp_decode_raw` at both call sites.

#### Taylor / Laurent series
- **Taylor abs(0)**: `Taylor::abs` and `TaylorDyn::abs` now use the first nonzero coefficient's sign instead of `signum(+0.0) = 0`, preventing the entire jet from being annihilated.
- **taylor_cbrt at zero**: returns `[0, Inf, ...]` (signaling the vertical tangent) instead of NaN from `ln(0)`.
- **taylor_sqrt at zero**: runtime guard returns `[0, Inf, ...]` instead of silent `Inf/NaN` from division by `2*sqrt(0)`.
- **Laurent Add/Sub truncation**: promoted from `debug_assert` to `assert!` — silent coefficient loss when pole-order gap exceeds `K-1` is now caught at runtime.

#### Bytecode tape
- **Checkpoint thinning**: online checkpoint thinning now maintains uniform spacing after doubling (`skip(1).step_by(2)` instead of `step_by(2)`), fixing O(N) recomputation degradation for small checkpoint budgets.
- **Abs sparse Hessian**: reclassified from `UnaryNonlinear` to `ZeroDerivative` — `d²|x|/dx² = 0` a.e., reducing spurious Hessian sparsity pattern entries.
- Added `debug_assert` for tape variable count overflow (`u32::MAX`) and contiguous input opcodes.

#### GPU Taylor kernels (CUDA + wgpu, codegen)
- **POWI negative bases**: GPU Taylor mode now handles negative bases via `sign(a)^n * exp(n * ln(|a|))` instead of `exp(n * ln(a))` which produced NaN for `ln(negative)`.
- **POWF at a≤0**: added first-order chain-rule fallback when `a ≤ 0` (where `ln(a)` is undefined).
- **REM Taylor coefficients**: `c1`/`c2` now pass through `a`'s jet (`d(a%b)/da = 1` a.e.) instead of being zeroed.
- **`_sign` in taylor_eval.cu**: now matches `tape_eval.cu` — returns `1` for `+0.0` and `NaN` for `NaN` (Rust `signum` semantics).
- **ATAN2 at b=0**: uses direct derivative formula instead of `jet_div(a, b)` which divided by zero.
- **CBRT at a=0**: returns `[0, Inf, ...]` instead of NaN from `ln(0)`.
- **wgpu expm1/ln1p**: polynomial approximation for `|x| < 1e-4` to avoid catastrophic cancellation.

#### echidna-optim
- **Trust-region min radius**: added `min_radius` field to `TrustRegionConfig` (default `F::epsilon()`), preventing silent stall from geometric radius collapse.
- **boundary_tau zero direction**: guards `||d||² < epsilon` to prevent NaN from degenerate CG directions.
- **LU singularity threshold**: changed from hardcoded `1e-12` to `F::epsilon().sqrt()`, fixing false negatives for f32 near-singular matrices.
- **Piggyback adjoint**: extra reverse pass with converged `lambda_new` eliminates O(tol × ||G_x||) error in parameter gradients.
- **Weighted STDE**: `estimate_weighted` skips zero-weight samples to prevent division by zero in West's algorithm.
- **Welford accumulator**: `debug_assert!(sample.is_finite())` catches NaN/Inf inputs in debug builds.

### Changed

- **GPU STDE test parity**: refactored `tests/gpu_stde.rs` to use a generic `opcode_tests_for_backend!` macro, running all per-opcode Taylor tests on both wgpu and CUDA backends instead of wgpu-only.
- **`TrustRegionConfig`**: added `min_radius: F` field (breaking change for direct struct construction; `Default` impls updated).

## [0.5.0] - 2026-03-14

### Added

- **GPU cast safety audit**: SAFETY comments on all `as u32` casts in GPU paths (`mod.rs`, `cuda_backend.rs`, `wgpu_backend.rs`, `stde_gpu.rs`). Added `debug_assert!` guards on user-provided direction/batch counts in `stde_gpu.rs`.
- **`#[must_use]` annotations**: 19 pure functions now carry `#[must_use]` (support module helpers, GPU codegen, solver wrappers, `Laurent::zero`/`one`).
- **`#![warn(missing_docs)]`**: enabled crate-wide. All public items — 35 `OpCode` variants, ~190 elemental methods across `Dual`, `DualVec`, `Taylor`, `TaylorDyn`, `Laurent`, struct fields, and trait methods — now have doc comments.

### Changed

- **Test decomposition**: split `tests/stde.rs` (1630 lines, 76 tests) into 5 focused files: `stde_core`, `stde_stats`, `stde_pipeline`, `stde_higher_order`, `stde_dense`. All 76 tests preserved.
- Removed `ROADMAP.md` — all phases (0–5) complete.

## [0.4.1] - 2026-03-14

### Fixed

- **Powi f32 exponent encoding**: `powi(n)` on f32 bytecode tapes silently produced wrong values and gradients for negative exponents (`n <= -2`). The `i32` exponent was stored as `u32` then round-tripped through `f32`, which loses precision for values > 2^24 (all negative exponents). All 5 dispatch sites (forward, reverse, tangent forward, tangent reverse, cross-country) now decode the exponent directly from the raw `u32` via `powi_exp_decode_raw`, bypassing the float conversion entirely.
- **taylor_powi negative base**: `Taylor::powi` and bytecode Taylor-mode produced NaN for negative base values (e.g. `(-2)^3`) because the implementation used `exp(n * ln(a))` which fails for `ln(negative)`. Added `taylor_powi_squaring` using binary exponentiation with `taylor_mul`, dispatched when `a[0] < 0` or `|n| <= 8`.
- **Checkpoint position lookup**: `grad_checkpointed`, `grad_checkpointed_disk`, and `grad_checkpointed_with_hints` used `Vec::contains()` for checkpoint position lookups (O(n) per step). Converted to `HashSet` for O(1) lookups.
- **Nonsmooth Round kink detection**: `forward_nonsmooth` now correctly detects Round kinks at half-integers (0.5, 1.5, ...) instead of at integers, matching the actual discontinuity locations of the `round` function. Updated test to match.

## [0.4.0] - 2026-02-26

### Changed

#### Internal Architecture
- **BytecodeTape decomposition**: split 2,689-line monolithic `bytecode_tape.rs` into a directory module with 10 focused submodules (`forward.rs`, `reverse.rs`, `tangent.rs`, `jacobian.rs`, `sparse.rs`, `optimize.rs`, `taylor.rs`, `parallel.rs`, `serde_support.rs`, `thread_local.rs`). Zero public API changes; benchmarks confirm no performance impact.
- Deduplicated reverse sweep in `gradient_with_buf()` and `sparse_jacobian_par()` — both now call shared `reverse_sweep_core()` instead of inlining the loop. `gradient_with_buf` gains the zero-adjoint skip optimization it was previously missing.
- Bumped `nalgebra` dependency from 0.33 to 0.34

### Fixed
- Corrected opcode variant count in documentation (44 variants, not 38/43)
- Fixed CONTRIBUTING.md MSRV reference (1.93, not 1.80)

## [0.3.0] - 2026-02-25

### Added

#### Differential Operator Evaluation (`diffop` feature)
- `diffop::mixed_partial(tape, x, orders)` — compute any mixed partial derivative via jet coefficient extraction
- `diffop::hessian(tape, x)` — full Hessian via jet extraction (cross-validated against `tape.hessian()`)
- `MultiIndex` — specify which mixed partial to compute (e.g., `[2, 0, 1]` = ∂³u/∂x₀²∂x₂)
- `JetPlan::plan(n, indices)` — precompute slot assignments and extraction prefactors; reuse across evaluation points
- `diffop::eval_dyn(plan, tape, x)` — evaluate a plan at a new point using `TaylorDyn`
- Pushforward grouping: multi-indices with different active variable sets get separate forward passes to avoid slot contamination
- Prime window sliding for collision-free slot assignment up to high derivative orders

## [0.2.0] - 2026-02-25

### Added

#### Bytecode Tape (Graph-Mode AD)
- `BytecodeTape` SoA graph-mode AD with opcode dispatch and tape optimization (CSE, DCE, constant folding)
- `BReverse<F>` tape-recording reverse-mode variable
- `record()` / `record_multi()` to build tapes from closures
- Hessian computation via forward-over-reverse (`hessian`, `hvp`)
- `DualVec<F, N>` batched forward-mode with N tangent lanes for vectorized Hessians (`hessian_vec`)

#### Sparse Derivatives
- Sparsity pattern detection via bitset propagation
- Graph coloring: greedy distance-2 for Jacobians, star bicoloring for Hessians
- `sparse_jacobian`, `sparse_hessian`, `sparse_hessian_vec`
- CSR storage (`CsrPattern`, `JacobianSparsityPattern`, `SparsityPattern`)

#### Taylor Mode AD
- `Taylor<F, K>` const-generic Taylor coefficients with Cauchy product propagation
- `TaylorDyn<F>` arena-based dynamic Taylor (runtime degree)
- `taylor_grad` / `taylor_grad_with_buf` — reverse-over-Taylor for gradient + HVP + higher-order adjoints
- `ode_taylor_step` / `ode_taylor_step_with_buf` — ODE Taylor series integration via coefficient bootstrapping

#### Stochastic Taylor Derivative Estimators (STDE)
- `laplacian` — Hutchinson trace estimator for Laplacian approximation
- `hessian_diagonal` — exact Hessian diagonal via coordinate basis
- `directional_derivatives` — batched second-order directional derivatives
- `laplacian_with_stats` — Welford's online variance tracking
- `laplacian_with_control` — diagonal control variate variance reduction
- `Estimator` trait generalizing per-direction sample computation (`Laplacian`, `GradientSquaredNorm`)
- `estimate` / `estimate_weighted` generic pipeline
- Hutchinson divergence estimator for vector fields via `Dual<F>` forward mode
- Hutch++ (Meyer et al. 2021) O(1/S²) trace estimator via sketch + residual decomposition
- Importance-weighted estimation (West's 1979 algorithm)

#### Cross-Country Elimination
- `jacobian_cross_country` — Markowitz vertex elimination on linearized computational graph

#### Custom Operations
- `eval_dual` / `partials_dual` default methods on `CustomOp<F>` for correct second-order derivatives (HVP, Hessian) through custom ops

#### Nonsmooth AD
- `forward_nonsmooth` — branch tracking and kink detection for abs/min/max/signum/floor/ceil/round/trunc
- `clarke_jacobian` — Clarke generalized Jacobian via limiting Jacobian enumeration
- `has_nontrivial_subdifferential()` — two-tier classification: all 8 nonsmooth ops tracked for proximity detection; only abs/min/max enumerated in Clarke Jacobian
- `KinkEntry`, `NonsmoothInfo`, `ClarkeError` types

#### Laurent Series
- `Laurent<F, K>` — singularity analysis with pole tracking, flows through `BytecodeTape::forward_tangent`

#### Checkpointing
- `grad_checkpointed` — binomial Revolve checkpointing
- `grad_checkpointed_online` — periodic thinning for unknown step count
- `grad_checkpointed_disk` — disk-backed for large state vectors
- `grad_checkpointed_with_hints` — user-controlled checkpoint placement

#### GPU Acceleration
- wgpu backend: batched forward, gradient, sparse Jacobian, HVP, sparse Hessian (f32, Metal/Vulkan/DX12)
- CUDA backend: same operations with f32 + f64 support (NVRTC runtime compilation)
- `GpuBackend` trait unifying wgpu and CUDA backends behind a common interface

#### Composable Mode Nesting
- Type-level AD composition: `Dual<BReverse<f64>>`, `Taylor<BReverse<f64>, K>`, `DualVec<BReverse<f64>, N>`
- `composed_hvp` convenience function for forward-over-reverse HVP
- `BReverse<Dual<f64>>` reverse-wrapping-forward composition via `BtapeThreadLocal` impls for `Dual<f32>` and `Dual<f64>`

#### Serialization
- `serde` support for `BytecodeTape`, `Laurent<F, K>`, `KinkEntry`, `NonsmoothInfo`, `ClarkeError`
- JSON and bincode roundtrip support

#### Linear Algebra Integrations
- `faer_support`: HVP, sparse Hessian, dense/sparse solvers (LU, Cholesky)
- `nalgebra_support`: gradient, Hessian, Jacobian with nalgebra types
- `ndarray_support`: HVP, sparse Hessian, sparse Jacobian with ndarray types

#### Optimization Solvers (`echidna-optim`)
- L-BFGS solver with two-loop recursion
- Newton solver with Cholesky factorization
- Trust-region solver with Steihaug-Toint CG
- Armijo line search
- Implicit differentiation: `implicit_tangent`, `implicit_adjoint`, `implicit_jacobian`, `implicit_hvp`, `implicit_hessian`
- Piggyback differentiation: tangent, adjoint, and interleaved forward-adjoint modes
- Sparse implicit differentiation via faer sparse LU (`sparse-implicit` feature)

#### Benchmarking
- Criterion benchmarks for Taylor mode, STDE, cross-country, sparse derivatives, nonsmooth
- Comparison benchmarks against num-dual and ad-trait (forward + reverse gradient)
- Correctness cross-check tests verifying ad-trait gradient agreement with echidna
- CI regression detection via criterion-compare-action

### Changed

- Tape optimization: algebraic simplification at recording time (identity, absorbing, powi patterns)
- Tape optimization: targeted multi-output DCE (`dead_code_elimination_for_outputs`)
- Thread-local Adept tape pooling — `grad()`/`vjp()` reuse cleared tapes via thread-local pool instead of per-call allocation
- `Signed::signum()` for `BReverse<F>` now records `OpCode::Signum` to tape (was returning a constant)
- MSRV raised from 1.80 to 1.93
- `WelfordAccumulator` struct extracted, deduplicating Welford's algorithm across 4 STDE functions
- `cuda_err` helper extracted, replacing 72 inline `.map_err` closures in CUDA backend
- `create_tape_bind_group` method extracted, replacing 4 duplicated bind group blocks in wgpu backend

## [0.1.0] - 2026-02-21

### Added

#### Core Types
- `Dual<F>` forward-mode dual number with all 30+ elemental operations
- `Reverse<F>` reverse-mode AD variable (12 bytes for f64, `Copy`)
- `Float` marker trait for `f32`/`f64`
- `Scalar` trait for writing AD-generic code
- Type aliases: `Dual64`, `Dual32`, `Reverse64`, `Reverse32`

#### Tape
- Adept-style two-stack tape with precomputed partial derivatives
- Thread-local active tape with RAII guard (`TapeGuard`)
- Constant sentinel (`u32::MAX`) to avoid tape bloat from literals
- Zero-adjoint skipping in the reverse sweep

#### API
- `grad(f, x)` — gradient via reverse mode
- `jvp(f, x, v)` — Jacobian-vector product via forward mode
- `vjp(f, x, w)` — vector-Jacobian product via reverse mode
- `jacobian(f, x)` — full Jacobian via forward mode

#### Elemental Operations
- Powers: `recip`, `sqrt`, `cbrt`, `powi`, `powf`
- Exp/Log: `exp`, `exp2`, `exp_m1`, `ln`, `log2`, `log10`, `ln_1p`, `log`
- Trig: `sin`, `cos`, `tan`, `sin_cos`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Misc: `abs`, `signum`, `floor`, `ceil`, `round`, `trunc`, `fract`, `mul_add`, `hypot`

#### Trait Implementations
- `num-traits`: `Float`, `Zero`, `One`, `Num`, `Signed`, `FloatConst`, `FromPrimitive`, `ToPrimitive`, `NumCast`
- `std::ops`: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Rem` with assign variants
- Mixed scalar ops (`Dual<f64> + f64`, `f64 * Reverse<f64>`, etc.)

#### Testing
- 94 tests: forward mode, reverse mode, API, and cross-validation
- Every elemental validated against central finite differences
- Forward-vs-reverse cross-validation on Rosenbrock, Beale, Ackley, Booth, and more
- Criterion benchmarks for forward overhead and reverse gradient

[Unreleased]: https://github.com/Entrolution/echidna/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Entrolution/echidna/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Entrolution/echidna/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Entrolution/echidna/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Entrolution/echidna/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Entrolution/echidna/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Entrolution/echidna/releases/tag/v0.1.0

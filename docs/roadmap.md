# echidna Roadmap

**Status**: Phases 1-4 (core AD), Phase 8 partial (implicit IFT), R4a (piggyback differentiation), R4b (interleaved forward-adjoint piggyback), R4c (second-order implicit derivatives), R4d (sparse F_z exploitation), R1a+R1c+R1d+R3 (Taylor mode AD), R2a+R2c (STDE), R2b (variance reduction), R5 (cross-country elimination), R6 (nonsmooth extensions), R7 (tape serialization), R8 (benchmarking infrastructure), R9 (checkpointing improvements), and R10 (integration improvements) are complete. 510 tests passing (455 core + 55 optim).

This roadmap synthesizes:
- Deferred items from all implementation phases to date
- Remaining phases from the [book breakdown](book-breakdown.md)
- Taylor mode AD and stochastic estimators from the STDE paper (Schatz et al., NeurIPS 2024)
- Organic improvements identified during implementation

---

## What's Done

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Forward mode (`Dual<F>`), reverse mode (`Reverse<F>`), `Scalar` trait, closure-based API, 30+ elementals | Complete |
| 2 | BytecodeTape (SoA), BReverse recording, tape optimization (CSE, dead-code, constant folding), binomial checkpointing | Complete |
| 3 | Hessians (forward-over-reverse), HVP, DualVec batched tangents, full/sparse Hessians | Complete |
| 4 | Sparse Jacobians, sparsity pattern detection, graph coloring (greedy distance-2, star bicoloring), CSR storage | Complete |
| 8 (partial) | Implicit tangent/adjoint/Jacobian via IFT, LU factorization reuse | Complete |
| optim | L-BFGS, Newton, trust-region solvers, Armijo line search, convergence control | Complete |
| R3 | Public `forward_tangent` on BytecodeTape, `output_index()` accessor | Complete |
| R1a | `Taylor<F, K>` (const-generic) + `TaylorDyn<F>` (arena-based) + shared `taylor_ops` propagation rules, 35+ elementals, `Float`/`Scalar`/`num_traits::Float` impls, 33 tests | Complete |
| R1c | `taylor_grad` / `taylor_grad_with_buf` — reverse-over-Taylor for gradient + HVP + higher-order directional adjoints in a single pass, 11 tests | Complete |
| R1d | `ode_taylor_step` / `ode_taylor_step_with_buf` — ODE Taylor series integration via coefficient bootstrapping, `eval_at` Horner evaluation, 8 tests | Complete |
| R2a+R2c | STDE module: jet propagation, Laplacian estimator, Hessian diagonal, directional derivatives, TaylorDyn variants, 20 tests | Complete |
| R2b | STDE variance reduction: `EstimatorResult` with sample statistics, `laplacian_with_stats` (Welford's online variance), `laplacian_with_control` (diagonal control variate), 13 tests | Complete |
| R4a | Piggyback differentiation: tangent step/solve via dual-number forward pass, adjoint solve via iterative VJP, `reverse_seeded` optimization (fused single-pass), `all_output_indices()` accessor, 8 tests | Complete |
| R4b | Interleaved forward-adjoint piggyback: `piggyback_forward_adjoint_solve` runs primal + adjoint in one loop, cutting iterations from K_primal + K_adjoint to max(K_primal, K_adjoint), 4 tests | Complete |
| R4c | Second-order implicit derivatives: `implicit_hvp` via nested `Dual<Dual<F>>` forward passes, `implicit_hessian` for full m×n×n tensor with LU factor reuse, 6 tests | Complete |
| R4d | Sparse F_z exploitation: `SparseImplicitContext` precomputes sparsity + coloring, `implicit_tangent_sparse`/`implicit_adjoint_sparse`/`implicit_jacobian_sparse` via faer sparse LU, feature-gated behind `sparse-implicit`, 9 tests | Complete |
| R5 | Cross-country elimination: `LinearizedGraph` + Markowitz vertex elimination for Jacobian computation, `jacobian_cross_country` on BytecodeTape, 10 tests | Complete |
| R6 | Nonsmooth extensions: branch tracking + kink detection (`forward_nonsmooth`), Clarke generalized Jacobian via limiting Jacobian enumeration (`clarke_jacobian`), `Laurent<F, K>` singularity analysis type with full `num_traits::Float`/`Scalar` integration, 31 tests | Complete |
| R7 | Tape serialization: `BytecodeTape` serde support (manual Serialize/Deserialize impls), `OpCode`/`SparsityPattern` serde derives, R6 types (`KinkEntry`, `NonsmoothInfo`, `ClarkeError`) serde derives, `Laurent<F, K>` manual serde impls (const-generic array handling), JSON + bincode roundtrip, f32/f64/multi-output coverage, 9 tests | Complete |
| R8 | Benchmarking infrastructure: shared test functions (Rosenbrock, Rastrigin, nn_layer, PDE Poisson), refactored existing benches to common module, new benches for Taylor mode, STDE estimators, cross-country/sparse Jacobian/nonsmooth, comparison against num-dual, CI regression detection via criterion-compare-action | Complete |
| R9 | Checkpointing improvements: online checkpointing (periodic thinning for unknown step count), disk-backed checkpointing (raw byte I/O with panic-safe cleanup), user-controlled checkpoint placement hints (Revolve sub-interval distribution), shared backward pass extraction, 18 new tests, online benchmarks | Complete |

**Deferred from completed phases** (carried forward below):
- Custom elemental derivatives registration (CustomOp exists but has no reverse-mode derivative hook)

---

## Tier 1: High-Impact, Natural Next Steps

These build directly on what exists and deliver the most value per effort.

### R1. Taylor Mode AD and Jets
**Book Phase 6 (Ch 13) + STDE paper**

The core primitive that unlocks higher-order derivatives, ODE integration, and stochastic estimators.

**R1a. Taylor number type and UTP engine** — **COMPLETE**

Implemented `Taylor<F, K>` (const-generic, stack-allocated) and `TaylorDyn<F>` (arena-based, `Copy`, degree chosen at runtime). Both implement `Float`, `Scalar`, and full `num_traits::Float` (35+ elemental methods). Shared propagation logic in `taylor_ops.rs` (Cauchy product, recursive division, Griewank Ch 13 logarithmic derivative technique for transcendentals). Feature-gated behind `taylor` flag. 33 tests covering known series, cross-validation with Dual, arithmetic, scalar generics, bytecode tape integration, and edge cases.

Key files: `src/taylor.rs`, `src/taylor_dyn.rs`, `src/taylor_ops.rs`, `src/traits/taylor_std_ops.rs`, `src/traits/taylor_num_traits.rs`.

**R1b. Tape-based Taylor propagation** — **DELIVERED via R1a+R3**

`Taylor<F, K>` and `TaylorDyn<F>` both implement `num_traits::Float`, so they flow through `BytecodeTape::forward_tangent` automatically. No tape modifications needed — existing opcode dispatch calls `num_traits::Float` methods which dispatch to Taylor propagation rules. Verified by integration tests pushing Taylor inputs through recorded tapes.

**R1c. Higher-order adjoint (Taylor reverse)** — **COMPLETE**

Reverse-over-Taylor: `taylor_grad` builds Taylor inputs `x_i(t) = x_i + v_i·t`, runs `forward_tangent`, then `reverse_tangent` to get Taylor-valued adjoints. `adjoint[i].coeff(0)` = gradient, `adjoint[i].coeff(1)` = HVP, `adjoint[i].derivative(k)` = k-th order directional adjoint. For K=2 equivalent to `hvp()`; for K≥3 yields third-order and higher. Feature-gated behind `taylor`. `taylor_grad_with_buf` variant reuses buffers. 11 tests cross-validating against `hvp`, `gradient`, `hessian`, and `third_order_hvvp`.

Key file: `src/bytecode_tape.rs`.

**R1d. ODE Taylor series integration** — **COMPLETE**

Given `y' = f(y)`, computes the Taylor expansion of `y(t)` to order K via coefficient bootstrapping: `y_{k+1} = coeff_k(f(y(t))) / (k+1)`. Uses K-1 forward passes through the tape, building up Taylor inputs incrementally. Returns `Vec<Taylor<F, K>>` so users get `eval_at(h)` for stepping, `coeff(k)` for error estimation, and `derivative(k)` for the solution derivatives. Feature-gated behind `taylor`. Follows the `taylor_grad` / `hvp` dual-method pattern (allocating + buffer-reusing). 8 tests covering exponential growth/decay, quadratic blowup, 2D rotation, step evaluation, minimal K=2, buffer reuse, and standalone `eval_at`.

Key file: `src/bytecode_tape.rs`.

### R2. Stochastic Taylor Derivative Estimators (STDE)
**STDE paper (Schatz et al., NeurIPS 2024)**

Builds on R1. Uses random jets to estimate differential operators without computing full derivative tensors.

**R2a. Jet construction for differential operators** — **COMPLETE**

Implemented `echidna::stde` module with jet propagation and operator estimation. Uses `Taylor<F, 3>` (const-generic, stack-allocated) for second-order operators and `TaylorDyn` for runtime-determined order. Feature-gated behind `stde = ["bytecode", "taylor"]`.

API: `taylor_jet_2nd` / `taylor_jet_2nd_with_buf` (single-direction), `directional_derivatives` (batch), `laplacian` (Hutchinson trace estimator), `hessian_diagonal` (exact via coordinate basis), `taylor_jet_dyn` / `laplacian_dyn` (TaylorDyn variants). No `rand` dependency — users provide direction vectors. 20 tests covering known Hessians, Rademacher convergence, cross-validation with `hessian()` and `grad()`, TaylorDyn/const-generic parity, edge cases, and transcendental functions.

Key file: `src/stde.rs`.

**R2b. Variance reduction** — **COMPLETE**

Implemented `EstimatorResult<F>` struct (value, estimate, sample_variance, standard_error, num_samples), `laplacian_with_stats` (Hutchinson with Welford's online variance), and `laplacian_with_control` (diagonal control variate that reduces Gaussian variance to match Rademacher). Antithetic sampling was analyzed and skipped — `v^T H v` is even in v, so pairing +v/-v provides zero variance reduction. Stratified sampling is subsumed by the control variate approach. 13 tests covering correctness, variance properties, edge cases, and cross-validation.

Key file: `src/stde.rs`.

**R2c. PDE operator estimation** — **COMPLETE** (delivered via R2a)

Convenience API for common operators delivered as part of R2a: `laplacian`, `hessian_diagonal`, `directional_derivatives`. A general `DiffOperator` abstraction is deferred to R2b when variance reduction motivates it.

### R3. Public Forward Tangent API — **COMPLETE**
**Deferred from Phase 8 (implicit diff)**

`BytecodeTape::forward_tangent` is now public, along with a new `output_index()` accessor. Enables forward-mode JVP through recorded tapes, efficient single-direction implicit tangent, and Taylor propagation (R1b).

---

## Tier 2: Important Extensions

### R4. Remaining Implicit/Iterative Features
**Book Phase 8 (Ch 15), remainder**

**R4a. Piggyback differentiation** — **COMPLETE**

Tangent piggyback propagates `ż_{k+1} = G_z · ż_k + G_x · ẋ` alongside the primal fixed-point iteration via a single dual-number forward pass per step. Adjoint piggyback iterates `λ_{k+1} = G_z^T · λ_k + z̄` at converged z* via reverse sweeps. Optimized `reverse_seeded` to use fused single-pass (`reverse_seeded_full`) — O(tape_length) instead of O(m × tape_length). Added `all_output_indices()` zero-allocation accessor. 8 tests including cross-validation with IFT.

Key file: `echidna-optim/src/piggyback.rs`.

**R4b. Interleaved forward-adjoint piggyback** — **COMPLETE**

`piggyback_forward_adjoint_solve` runs the primal fixed-point `z_{k+1} = G(z_k, x)` and adjoint equation `λ_{k+1} = G_z^T · λ_k + z̄` in a single interleaved loop. Total iterations drop from `K_primal + K_adjoint` to `max(K_primal, K_adjoint)`. Pre-allocated input buffer avoids per-iteration allocation. Returns `(z_star, x_bar, iterations)`. 4 tests including cross-validation against sequential tangent+adjoint.

Key file: `echidna-optim/src/piggyback.rs`.

**R4c. Second-order implicit derivatives** — **COMPLETE**

`implicit_hvp` computes `d²z*/dx² · v · w` using nested `Dual<Dual<F>>` forward passes through `forward_tangent`. A single O(tape_length) pass extracts the second-order correction `ṗ^T · Hess(F_i) · ẇ` from the `.eps.eps` component, then solves `F_z · h = -RHS`. `implicit_hessian` builds the full m×n×n tensor by iterating over `n(n+1)/2` direction pairs with LU factor reuse. 6 tests including finite-difference cross-validation.

Key file: `echidna-optim/src/implicit.rs`.

**R4d. Sparse F_z exploitation** — **COMPLETE**

Exploits structural sparsity in F_z via `SparseImplicitContext` (precomputes full m×(m+n) sparsity pattern, graph coloring, COO index partitions). Three public functions mirror the dense API: `implicit_tangent_sparse`, `implicit_adjoint_sparse`, `implicit_jacobian_sparse`. Uses echidna's sparse Jacobian infrastructure for compressed evaluation and faer's sparse LU for the F_z solve. f64-only, feature-gated behind `sparse-implicit`. 9 tests including cross-validation with dense implementations, tridiagonal/block-diagonal structure verification, context reuse, singular detection, and dimension-mismatch panics.

Key file: `echidna-optim/src/sparse_implicit.rs`.

### R5. Cross-Country Elimination — **COMPLETE**
**Book Phase 5 (Ch 10)**

Vertex elimination on the linearized computational graph with Markowitz-based ordering (Griewank & Walther, Chapter 10). Builds a linearized DAG from the tape with edge weights = local partial derivatives, then eliminates intermediate vertices in greedy Markowitz order (smallest |preds| × |succs| first), accumulating fill-in edges. After elimination, remaining edges connect inputs to outputs directly, yielding the full m×n Jacobian.

Public API: `BytecodeTape::jacobian_cross_country(&mut self, inputs: &[F]) -> Vec<Vec<F>>`. Handles identity/passthrough (output-is-input), custom ops, constant nodes, and all standard opcodes. Generic over `F: Float`. 10 tests cross-validating against `jacobian()` (reverse mode) and `jacobian_forward()`.

Key files: `src/cross_country.rs`, `src/bytecode_tape.rs`.

### R6. Nonsmooth Extensions — **COMPLETE**
**Book Phase 7 (Ch 14)**

**R6a. Branch tracking and kink detection** — **COMPLETE**

`BytecodeTape::forward_nonsmooth` detects abs/min/max kinks during forward evaluation, records `KinkEntry` with tape index, opcode, switching value, and branch taken. `NonsmoothInfo` provides `active_kinks(tol)`, `is_smooth(tol)`, and `signature()`. Feature-gated behind `bytecode`. 8 tests.

Key files: `src/nonsmooth.rs`, `src/bytecode_tape.rs`.

**R6b. Clarke generalized Jacobian** — **COMPLETE**

`BytecodeTape::clarke_jacobian` enumerates all 2^k sign combinations for k active kinks, computing limiting Jacobians via `reverse_with_forced_signs`. `jacobian_limiting` allows explicit forced branch choices. `forced_reverse_partials` in `opcode.rs` overrides nonsmooth op partials. Configurable kink limit (default 20) returns `ClarkeError::TooManyKinks`. 8 tests.

Key files: `src/opcode.rs`, `src/bytecode_tape.rs`.

**R6c. Laurent numbers** — **COMPLETE**

`Laurent<F, K>` — const-generic, stack-allocated, `Copy`. Arithmetic reuses `taylor_ops` (Cauchy product, division, reciprocal) with separate pole_order tracking. Always normalized (leading coefficient nonzero). Essential singularities (exp/sin/cos of pole) → NaN. Division by zero → NaN. `value()` returns ±infinity for poles. Full `num_traits::Float`, `Float`, `Scalar` integration — flows through `BytecodeTape::forward_tangent` for singularity detection. Feature-gated behind `laurent = ["taylor"]`. 15 tests.

Key files: `src/laurent.rs`, `src/traits/laurent_std_ops.rs`, `src/traits/laurent_num_traits.rs`.

---

## Tier 3: Infrastructure and Polish

### R7. Tape Serialization — **COMPLETE**
**Deferred from design principles**

`serde` support for `BytecodeTape` with manual Serialize/Deserialize implementations. Custom ops are rejected at serialization time (must be re-registered after deserialization). R6 types (`KinkEntry`, `NonsmoothInfo`, `ClarkeError`) and `Laurent<F, K>` also support serde (Laurent uses manual impls due to const-generic array limitations in serde's derive). Feature-gated behind `serde` flag. JSON and bincode roundtrip verified. 9 tests covering f32/f64, single/multi-output tapes, binary format, custom-op rejection, sparsity patterns, and nonsmooth info.

Key files: `src/bytecode_tape.rs`, `src/nonsmooth.rs`, `src/laurent.rs`, `tests/serde.rs`.

### R8. Benchmarking Infrastructure — **COMPLETE**

Expanded from 3 Criterion bench files (forward, reverse, bytecode — Rosenbrock only) to 7 bench files covering all AD modes. Shared test functions in `benches/common/mod.rs` (Rosenbrock, Rastrigin, nn_layer, PDE Poisson residual). New bench files: `taylor.rs` (Taylor reverse, buffer reuse, higher-order), `stde.rs` (Laplacian estimator, Hessian diagonal, jet buffer reuse), `advanced.rs` (cross-country Jacobian, sparse Jacobian, nonsmooth overhead, Clarke subdifferential), `comparison.rs` (echidna vs num-dual for gradient, Jacobian, Hessian). CI regression detection via `boa-dev/criterion-compare-action` posts comparison as PR comment. ad-trait deferred (num-dual only).

Key files: `benches/common/mod.rs`, `benches/taylor.rs`, `benches/stde.rs`, `benches/advanced.rs`, `benches/comparison.rs`, `.github/workflows/bench.yml`.

### R9. Checkpointing Improvements — **COMPLETE**
**Deferred from Phase 2**

Three extensions to the binomial Revolve checkpointing:

**R9a. Online checkpointing** — `grad_checkpointed_online` uses periodic thinning for unknown iteration count. Maintains a fixed-size checkpoint buffer; when full, discards every other entry and doubles spacing. O(log N) recomputation overhead for N steps. Stop predicate determines termination.

**R9b. Disk-backed checkpointing** — `grad_checkpointed_disk` stores checkpoint states as raw binary files on disk for large state vectors (state_dim × num_checkpoints × 8 bytes > available memory). Uses unsafe byte transmutation (safe for all `Float` types: `Copy + Sized`, no pointers). `DiskCheckpointGuard` Drop guard ensures cleanup on both normal completion and panic.

**R9c. Checkpoint placement hints** — `grad_checkpointed_with_hints` accepts required checkpoint positions. Distributes remaining slots proportionally across sub-intervals using the largest-remainder method, then runs Revolve on each sub-interval. With empty required list, behaves identically to `grad_checkpointed`.

Shared backward pass extracted into `backward_from_checkpoints` helper, reused by the original `grad_checkpointed` and all three new variants. 18 new tests, 1 new benchmark group (online vs offline comparison). Feature-gated behind `bytecode`.

Key file: `src/checkpoint.rs`.

### R10. Integration Improvements — **COMPLETE**

Three sub-features deepening echidna's integration with Rust linear algebra crates:

**R10a. Deepen faer support** — `faer_support` extended with: `hvp_faer`/`tape_hvp_faer` (HVP wrappers), `sparse_hessian_faer`/`tape_sparse_hessian_faer` (sparse Hessian wrappers), `sparsity_to_faer_symmetric` (COO→SparseColMat conversion with symmetrization), `solve_dense_lu_faer`/`solve_dense_cholesky_faer` (dense solver convenience wrappers), `solve_sparse_cholesky_faer`/`solve_sparse_lu_faer` (sparse solvers with `catch_unwind` for faer's panicking API). 8 new tests.

**R10b. nalgebra integration** — New `nalgebra_support` module with `grad_nalgebra`, `grad_nalgebra_val`, `hessian_nalgebra`, `jacobian_nalgebra`, `tape_gradient_nalgebra`, `tape_hessian_nalgebra`. Uses `DVector::as_slice()` (no unwrap needed) and `DMatrix::from_row_slice` for correct row-major→column-major layout. Feature-gated behind `nalgebra = ["dep:nalgebra", "bytecode"]`. 6 new tests.

**R10c. ndarray extensions** — `ndarray_support` extended with: `hvp_ndarray`/`tape_hvp_ndarray`, `sparse_hessian_ndarray`/`tape_sparse_hessian_ndarray`, `sparse_jacobian_ndarray`. 3 new tests.

Key files: `src/faer_support.rs`, `src/nalgebra_support.rs`, `src/ndarray_support.rs`, `tests/faer_tests.rs`, `tests/nalgebra_support_tests.rs`, `tests/ndarray_tests.rs`.

---

## Tier 4: Aspirational / Research

### R6. Nonsmooth Extensions — **COMPLETE** (moved from Tier 4)

See Tier 2 section above for full details.

### R11. GPU Acceleration
**Design principles mention this; significant engineering effort**

- Batched forward mode on GPU (many inputs, same tape) — embarrassingly parallel
- Taylor coefficient propagation maps well to GPU (convolution-heavy)
- Sparse Jacobian assembly on GPU
- Requires kernel authoring in WGSL/CUDA, not just Rust

### R12. Composable Mode Nesting

True type-level composition: `Dual<BReverse<f64>>` for forward-over-reverse, `Taylor<Dual<f64>>` for Taylor-over-forward. Currently forward-over-reverse works via the tape-based HVP path but isn't composable at the type level.

### R13. Source-Level Optimizations

- Preaccumulation of straight-line segments (Griewank Ch 7)
- Common subexpression elimination improvements
- Dead code elimination across tape boundaries
- These improve the constant factors but don't change algorithmic complexity

---

## Suggested Implementation Order

```
R3 (public forward_tangent)          ─── ✓ DONE
  │
  ▼
R1a (Taylor type + UTP rules)       ─── ✓ DONE
  │
  ├──▶ R1b (tape-based Taylor)      ─── ✓ DONE (delivered via R1a+R3)
  │      │
  │      ├──▶ R1c (Taylor reverse)  ─── ✓ DONE
  │      │
  │      └──▶ R1d (ODE Taylor)    ─── ✓ DONE
  │
  └──▶ R2a (jet construction)       ─── ✓ DONE
         │
         ├──▶ R2c (PDE operators)   ─── ✓ DONE (delivered via R2a)
         │
         └──▶ R2b (variance reduction) ─── ✓ DONE

R4a (piggyback differentiation)      ─── ✓ DONE
R4b (interleaved forward-adjoint)    ─── ✓ DONE
R4c (second-order implicit)          ─── ✓ DONE
R4d (sparse F_z exploitation)        ─── ✓ DONE
R5 (cross-country elimination)       ─── ✓ DONE
R6a (branch tracking)                ─── ✓ DONE
R6b (Clarke subdifferential)         ─── ✓ DONE
R6c (Laurent numbers)                ─── ✓ DONE
R7 (serde)                           ─── ✓ DONE
R8 (benchmarks)                      ─── ✓ DONE
R9a (online checkpointing)           ─── ✓ DONE
R9b (disk-backed checkpointing)      ─── ✓ DONE
R9c (checkpoint hints)               ─── ✓ DONE
R10a (deepen faer)                   ─── ✓ DONE
R10b (nalgebra integration)          ─── ✓ DONE
R10c (ndarray extensions)            ─── ✓ DONE
```

All R1 items (Taylor mode AD) are now complete.

All R4 items (implicit/iterative features) are now complete.

R5 (cross-country elimination) is now complete.

R6 (nonsmooth extensions) is now complete.

R7 (serde) is now complete.

R8 (benchmarking infrastructure) is now complete.

R9 (checkpointing improvements) is now complete.

R10 (integration improvements) is now complete.

R11+ are lower priority and can be scheduled opportunistically.

---

## Complexity Estimates

| Item | Scope | New code (rough) | Actual |
|------|-------|-----------------|--------|
| R3 | Expose existing private method | ~20 lines | ~15 lines |
| R1a | New type + all elemental rules | ~800-1200 lines | ~2500 lines (incl. both types + tests) |
| R1b | Tape integration | ~400-600 lines | 0 (free via R1a+R3) |
| R1c | Taylor reverse sweep | ~300-500 lines |
| R1d | ODE bootstrapping | ~200-300 lines |
| R2a | Jet builder + sampling | ~400-600 lines |
| R2b | Variance reduction | ~200-300 lines |
| R2c | PDE operator API | ~300-400 lines |
| R4a | Piggyback | ~300-400 lines |
| R4b | Fixed-point adjoint | ~200-300 lines |
| R4c | Second-order implicit | ~200-300 lines |
| R5 | Cross-country | ~600-1000 lines | ~450 lines (234 impl + 212 tests) |
| R6a-b | Nonsmooth | ~500-800 lines | ~320 lines (impl) + ~250 lines (tests) |
| R6c | Laurent numbers | ~350-500 lines | ~880 lines (impl) + ~200 lines (tests) |

---

## Design Decisions

1. **Taylor type location**: `Taylor<F, K>` lives in **echidna core**, behind a `taylor` feature flag. It's a fundamental numeric type like `Dual<F>` and `Reverse<F>`, and tape-based Taylor propagation needs direct access to `BytecodeTape`.

2. **STDE location**: STDE also lives in **echidna core**, under the same `taylor` feature flag (e.g. `echidna::stde` module). It's derivative estimation, not optimization. Accepts user-provided random vectors to avoid a `rand` dependency.

3. **Taylor degree representation**: **Both const-generic and dynamic** from the start. `Taylor<F, const K: usize>` (stack-allocated, monomorphized) for STDE's typical K=2..3. `TaylorDyn<F>` (heap-allocated) for ODE integration and cases where K is a runtime parameter. Shared elemental propagation rules, different storage backends.

4. **Sparse F_z reuse**: **Defer until R4d** is implemented. The existing sparse Jacobian machinery (graph coloring, CSR) is a natural fit but the right abstraction boundary will be clearer with concrete requirements. No speculative refactoring now.

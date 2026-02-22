# echidna Roadmap

**Status**: Phases 1-4 (core AD), Phase 8 partial (implicit IFT), and R1a+R3 (Taylor mode AD) are complete. 326 tests passing.

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

**Deferred from completed phases** (carried forward below):
- Custom elemental derivatives registration (CustomOp exists but has no reverse-mode derivative hook)
- Tape serialization (serde)
- `faer` / `ndarray` integration modules exist but are thin wrappers; no deep integration with tape operations
- Checkpointing is binomial only; no online checkpointing for unknown-length computations

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

**R1c. Higher-order adjoint (Taylor reverse)**

Reverse mode applied to Taylor coefficients. Needed for:
- Higher-order gradient computation
- Combining with forward Taylor to get mixed partial information
- The "reverse-over-Taylor" pattern from Griewank Ch 13.4

**R1d. ODE Taylor series integration**

Given `y' = f(y)`, compute the Taylor expansion of `y(t)` to order K by bootstrapping:
1. `y_0` = initial condition
2. `y_1 = f(y_0)` (evaluate f)
3. `y_k` for k >= 2: push `y_0, y_1, ..., y_{k-1}` as a jet through f, read off the degree-k coefficient

This gives an automatic high-order ODE integrator with adaptive step control via coefficient monitoring.

### R2. Stochastic Taylor Derivative Estimators (STDE)
**STDE paper (Schatz et al., NeurIPS 2024)**

Builds on R1. Uses random jets to estimate differential operators without computing full derivative tensors.

**R2a. Jet construction for differential operators**

The STDE paper's key insight: for any linear differential operator `L = sum c_alpha * D^alpha`, there exists a sparse jet structure that, when pushed through Taylor mode, yields an unbiased estimator of `L[f]`.

Implement a `JetBuilder` that, given a differential operator specification, produces the right input jets:

| Operator | Jet structure | Reference |
|----------|--------------|-----------|
| Laplacian `sum d^2f/dx_i^2` | Standard basis vectors `e_j` sampled uniformly, scaled by `d` | STDE Sec 4.3 |
| Full Hessian trace | Same as Laplacian | |
| Mixed partials `d^2f/dx_i dx_j` | Specific 2-jet tangent pairs from partition analysis | STDE Appendix F |
| Dense Hutchinson-style | Gaussian random vectors | STDE Sec 4.4 |

**R2b. Variance reduction**

- Antithetic sampling (pair +v with -v)
- Control variates using lower-order derivative information
- Stratified sampling over coordinate directions
- Batch estimation with confidence intervals

**R2c. PDE operator estimation**

High-level API for common PDE operators:
```rust
// Estimate the Laplacian at a point
let laplacian = stde::laplacian(&tape, &x, num_samples)?;

// Estimate a general differential operator
let op = DiffOperator::new()
    .add_term(1.0, &[2, 0, 0])  // d^2f/dx^2
    .add_term(1.0, &[0, 2, 0])  // d^2f/dy^2
    .add_term(1.0, &[0, 0, 2]); // d^2f/dz^2
let result = stde::estimate(&tape, &x, &op, num_samples)?;
```

This is the >1000x speedup path for PINNs and physics-informed applications.

### R3. Public Forward Tangent API — **COMPLETE**
**Deferred from Phase 8 (implicit diff)**

`BytecodeTape::forward_tangent` is now public, along with a new `output_index()` accessor. Enables forward-mode JVP through recorded tapes, efficient single-direction implicit tangent, and Taylor propagation (R1b).

---

## Tier 2: Important Extensions

### R4. Remaining Implicit/Iterative Features
**Book Phase 8 (Ch 15), remainder**

**R4a. Piggyback differentiation**

Propagate derivatives alongside solver iterations rather than post-hoc via IFT. For iterative solvers (Newton, fixed-point), record only the last iteration's local derivatives and "piggyback" the adjoint/tangent accumulation onto the convergence loop.

Advantages over current IFT approach:
- No need for the solver to have fully converged
- Works naturally with early termination
- Memory-efficient (no need to store the full Jacobian of the residual)

**R4b. Fixed-point iteration adjoint**

For `z = G(z, x)` (contraction mapping), the adjoint satisfies `lambda = G_z^T * lambda + z_bar`. This is itself a fixed-point iteration that can run alongside the forward iteration (Griewank Rule 26).

Only needs to record a single step of G, not the entire iteration history.

**R4c. Second-order implicit derivatives**

Compose IFT with forward-over-reverse to get implicit Hessians. Needed for second-order optimization through implicit layers (e.g., bilevel optimization).

**R4d. Sparse F_z exploitation**

For large-scale systems (PDE discretizations), F_z is typically sparse. Use graph coloring (already implemented in Phase 4) to compute the sparse Jacobian of F_z efficiently, then use a sparse LU solver.

### R5. Cross-Country Elimination
**Book Phase 5 (Ch 10)**

Vertex/edge/face elimination on the computational graph with Markowitz-based ordering. Computes Jacobians with fewer operations than either pure forward or pure reverse mode.

Most valuable for functions with comparable input/output dimensions where neither forward nor reverse has a clear advantage. Lower priority than Taylor mode but completes the theoretical picture.

### R6. Nonsmooth Extensions
**Book Phase 7 (Ch 14)**

**R6a. Correct abs/min/max handling**

Currently `abs`, `min`, `max` use standard derivatives which are undefined at kinks. Implement:
- Branch tracking during forward evaluation
- Active set identification at kink points
- Subgradient selection rules (lexicographic, steepest descent)

**R6b. Generalized gradients (Clarke subdifferential)**

For piecewise smooth functions, compute elements of the Clarke generalized gradient. Important for ReLU networks, `max(0, x)` constraints, and L1 regularization.

**R6c. Laurent numbers (optional)**

Singularity analysis via Laurent series arithmetic. Detects and characterizes singularities in the computation. Specialized — defer unless there's a concrete use case.

---

## Tier 3: Infrastructure and Polish

### R7. Tape Serialization
**Deferred from design principles**

`serde` support for `BytecodeTape`. Enables:
- Saving compiled tapes to disk (avoid re-recording)
- Sending tapes across network boundaries
- Caching tapes in long-running applications

Feature-gated behind `serde` flag.

### R8. Benchmarking Infrastructure

Expand beyond the existing Criterion benchmarks:
- Benchmark suite covering all AD modes (forward, reverse, Hessian, sparse, Taylor when available)
- Standard test functions (Rosenbrock, Rastrigin, neural network layers, PDE residuals)
- Comparison framework against `num-dual`, `ad-trait` for the modes they support
- CI-integrated regression detection (catch performance regressions in PRs)

### R9. Checkpointing Improvements
**Deferred from Phase 2**

- Online checkpointing for computations with unknown iteration count (Griewank's online Revolve)
- Disk-backed checkpointing for very long computations
- User-controlled checkpoint placement hints

### R10. Integration Improvements

- Deepen `faer` support: sparse LU/Cholesky via faer (faster than hand-rolled LU for large systems)
- `nalgebra` integration: direct construction of `DMatrix`/`DVector` from tape outputs
- `ndarray` integration: array views over tape buffers

---

## Tier 4: Aspirational / Research

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
  │      ├──▶ R1c (Taylor reverse)
  │      │
  │      └──▶ R1d (ODE Taylor)
  │
  └──▶ R2a (jet construction)       ─── NEXT: can start now
         │
         ├──▶ R2c (PDE operators)
         │
         └──▶ R2b (variance reduction)

R4a-d (implicit extensions)          ─── Independent of R1, can parallelize
R7 (serde)                           ─── Independent, any time
R8 (benchmarks)                      ─── Independent, any time
```

The critical path remaining is **R2a → R2c**. R1c (Taylor reverse) and R1d (ODE Taylor) are also unblocked.

R4 (remaining implicit features), R7 (serde), and R8 (benchmarks) are independent and can be interleaved as needed.

R5 (cross-country), R6 (nonsmooth), and R9+ are lower priority and can be scheduled opportunistically.

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
| R5 | Cross-country | ~600-1000 lines |
| R6a-b | Nonsmooth | ~500-800 lines |

---

## Design Decisions

1. **Taylor type location**: `Taylor<F, K>` lives in **echidna core**, behind a `taylor` feature flag. It's a fundamental numeric type like `Dual<F>` and `Reverse<F>`, and tape-based Taylor propagation needs direct access to `BytecodeTape`.

2. **STDE location**: STDE also lives in **echidna core**, under the same `taylor` feature flag (e.g. `echidna::stde` module). It's derivative estimation, not optimization. Accepts user-provided random vectors to avoid a `rand` dependency.

3. **Taylor degree representation**: **Both const-generic and dynamic** from the start. `Taylor<F, const K: usize>` (stack-allocated, monomorphized) for STDE's typical K=2..3. `TaylorDyn<F>` (heap-allocated) for ODE integration and cases where K is a runtime parameter. Shared elemental propagation rules, different storage backends.

4. **Sparse F_z reuse**: **Defer until R4d** is implemented. The existing sparse Jacobian machinery (graph coloring, CSR) is a natural fit but the right abstraction boundary will be clearer with concrete requirements. No speculative refactoring now.

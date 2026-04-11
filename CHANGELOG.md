# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **simba traits for DualVec**: implemented traits `SimdValue`, `PrimitiveSimdValue`, `SubsetOf`, `AbsDiffEq`, `RelativeEq`, `UlpsEq`, `Field`, `ComplexField`, `RealField` for `DualVec<F, N>`.

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

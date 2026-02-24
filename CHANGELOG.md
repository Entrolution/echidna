# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Entrolution/echidna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Entrolution/echidna/releases/tag/v0.1.0

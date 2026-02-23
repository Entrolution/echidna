# echidna

[![CI](https://github.com/Entrolution/echidna/actions/workflows/ci.yml/badge.svg)](https://github.com/Entrolution/echidna/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/echidna.svg)](https://crates.io/crates/echidna)
[![Docs.rs](https://docs.rs/echidna/badge.svg)](https://docs.rs/echidna)
[![License](https://img.shields.io/crates/l/echidna.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.80-blue.svg)](https://www.rust-lang.org)

A high-performance automatic differentiation library for Rust with forward-mode dual numbers, reverse-mode (Adept-style two-stack tape), bytecode graph-mode AD with tape optimization, Taylor-mode higher-order derivatives, sparse Jacobian/Hessian computation via graph coloring, GPU acceleration (wgpu and CUDA), nonsmooth AD with Clarke generalized Jacobians, gradient checkpointing, cross-country elimination, stochastic Taylor derivative estimators, and composable type nesting for arbitrary-order differentiation.

## Feature Overview

| Technique | Description |
|-----------|-------------|
| Forward mode | `Dual<F>`, `DualVec<F, N>` -- tangent propagation, JVP, batched tangents |
| Reverse mode | `Reverse<F>` -- Adept-style two-stack tape, `Copy` type (12 bytes for f64) |
| Bytecode tape | `BytecodeTape` -- record-once evaluate-many SoA tape with CSE, DCE, constant folding, algebraic simplification |
| Taylor mode | `Taylor<F, K>`, `TaylorDyn<F>` -- const-generic and arena-based dynamic higher-order derivatives |
| Sparse derivatives | Auto sparsity detection + graph coloring for Jacobians and Hessians |
| Cross-country elimination | Markowitz vertex elimination for optimal Jacobian accumulation |
| Checkpointing | Binomial, online, disk-backed, and hint-guided gradient checkpointing |
| STDE | Stochastic Taylor Derivative Estimators -- Laplacian, Hessian diagonal, variance reduction |
| Nonsmooth AD | Branch tracking, kink detection, Clarke generalized Jacobian |
| Laurent series | `Laurent<F, K>` -- singularity analysis via Laurent expansion |
| GPU acceleration | wgpu (Metal/Vulkan/DX12, f32) and CUDA (NVIDIA, f32+f64) batch evaluation |
| Composable nesting | `Dual<BReverse<f64>>`, `Taylor<BReverse<f64>, K>`, `Dual<Dual<f64>>` for higher-order |
| Optimization | L-BFGS, Newton, trust-region solvers; implicit differentiation and piggyback (via `echidna-optim`) |

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
echidna = "0.1"
```

### Gradient via reverse mode

```rust
use echidna::Scalar;

// Gradient of f(x) = x0^2 + x1^2
let g = echidna::grad(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);
assert!((g[0] - 6.0).abs() < 1e-10);
assert!((g[1] - 8.0).abs() < 1e-10);

// Write generic code that works with f64, Dual, and Reverse
fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(1.0);
    let hundred = T::from_f(100.0);
    let t1 = one - x[0];
    let t2 = x[1] - x[0] * x[0];
    t1 * t1 + hundred * t2 * t2
}

let g = echidna::grad(|x| rosenbrock(x), &[1.5, 2.0]);
```

### BytecodeTape: record once, evaluate many

```rust,ignore
use echidna::{record, record_multi};

// Record a scalar function to tape
let (tape, value) = echidna::record(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);

// Re-evaluate at new points without re-recording
let g = tape.gradient(&[1.0, 2.0]);

// Hessian and Hessian-vector product
let (val, grad, hess) = tape.hessian(&[1.0, 2.0]);
let (grad, hvp) = tape.hvp(&[1.0, 2.0], &[1.0, 0.0]);

// Record a multi-output function, compute sparse Jacobian
let (mut tape, vals) = echidna::record_multi(
    |x| vec![x[0] * x[1], x[0] + x[1] * x[1]],
    &[1.0, 2.0],
);
let (vals, pattern, jac_vals) = tape.sparse_jacobian(&[1.0, 2.0]);
```

### Taylor mode: higher-order derivatives

```rust,ignore
use echidna::{Taylor64, record};

// Record function, then compute Taylor coefficients via taylor_grad
let (tape, _) = echidna::record(|x| x[0].sin() + x[1].exp(), &[0.0, 0.0]);

// K=4 Taylor coefficients along direction v
let (output, adjoints) = tape.taylor_grad::<4>(&[0.0, 0.0], &[1.0, 0.0]);
// output.coeffs() gives [f(x), f'(x)*v, f''(x)*v^2/2!, ...]
```

## Feature Flags

| Flag | Description | Dependencies |
|------|-------------|--------------|
| `default` | simba/approx integration | -- |
| `bytecode` | BytecodeTape graph-mode AD (SoA tape, opcode dispatch, optimization passes) | -- |
| `parallel` | Rayon parallel tape evaluation | `bytecode` |
| `serde` | Serialization for BytecodeTape, Laurent, nonsmooth types | -- |
| `taylor` | Taylor-mode AD (const-generic + arena-based dynamic) | -- |
| `laurent` | Laurent series for singularity analysis | `taylor` |
| `stde` | Stochastic Taylor Derivative Estimators | `bytecode`, `taylor` |
| `ndarray` | ndarray integration | `bytecode` |
| `faer` | faer linear algebra integration | `bytecode` |
| `nalgebra` | nalgebra integration | `bytecode` |
| `gpu-wgpu` | GPU via wgpu (Metal/Vulkan/DX12, f32) | `bytecode` |
| `gpu-cuda` | GPU via CUDA (NVIDIA, f32+f64) | `bytecode` |

Enable features in `Cargo.toml`:

```toml
echidna = { version = "0.1", features = ["bytecode", "taylor"] }
```

## API

### Core (always available)

| Function | Signature | Description |
|----------|-----------|-------------|
| `grad` | `(f: FnOnce(&[Reverse<F>]) -> Reverse<F>, x: &[F]) -> Vec<F>` | Gradient via reverse mode |
| `jvp` | `(f: Fn(&[Dual<F>]) -> Vec<Dual<F>>, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>)` | Jacobian-vector product |
| `vjp` | `(f: FnOnce(&[Reverse<F>]) -> Vec<Reverse<F>>, x: &[F], w: &[F]) -> (Vec<F>, Vec<F>)` | Vector-Jacobian product |
| `jacobian` | `(f: Fn(&[Dual<F>]) -> Vec<Dual<F>>, x: &[F]) -> (Vec<F>, Vec<Vec<F>>)` | Full Jacobian |

### Bytecode Tape (requires `bytecode`)

| Function | Description |
|----------|-------------|
| `record(f, x)` / `record_multi(f, x)` | Record scalar / multi-output function to `BytecodeTape` |
| `tape.gradient(x)` | Gradient via reverse sweep |
| `tape.hessian(x)` | Full Hessian via forward-over-reverse |
| `tape.hvp(x, v)` | Hessian-vector product |
| `tape.sparse_jacobian(x)` | Sparse Jacobian with auto sparsity detection + coloring |
| `tape.sparse_hessian(x)` | Sparse Hessian with auto sparsity detection + coloring |
| `tape.jacobian_cross_country(x)` | Jacobian via Markowitz vertex elimination |
| `tape.forward_nonsmooth(x)` | Branch tracking and kink detection |
| `tape.clarke_jacobian(x, tol)` | Clarke generalized Jacobian |
| `composed_hvp(f, x, v)` | One-shot forward-over-reverse Hessian-vector product |
| `tape.hessian_vec::<N>(x)` | Batched Hessian computation (N directions per pass) |
| `tape.sparse_hessian_vec::<N>(x)` | Batched sparse Hessian (N directions per pass) |
| `grad_checkpointed(f, x, segments)` | Binomial gradient checkpointing |
| `grad_checkpointed_online(f, x, budget)` | Online checkpointing |
| `grad_checkpointed_disk(f, x, segments, dir)` | Disk-backed checkpointing |
| `grad_checkpointed_with_hints(f, x, hints)` | User-controlled checkpoint placement |

### Higher-Order (requires `taylor` or `stde`)

| Function | Description |
|----------|-------------|
| `tape.taylor_grad::<K>(x, v)` | Taylor-mode reverse: K-th order derivatives along direction v |
| `tape.ode_taylor_step::<K>(y0)` | ODE Taylor integration step |
| `stde::laplacian(tape, x, dirs)` | Stochastic Laplacian estimate |
| `stde::hessian_diagonal(tape, x)` | Stochastic Hessian diagonal |
| `stde::directional_derivatives(tape, x, dirs)` | Batched 1st and 2nd order directional derivatives |
| `stde::laplacian_with_control(tape, x, dirs, ctrl)` | Laplacian with control variate variance reduction |

### GPU (requires `gpu-wgpu` or `gpu-cuda`)

| Method | Description |
|--------|-------------|
| `ctx.forward_batch(bufs, inputs)` | Batched forward evaluation |
| `ctx.gradient_batch(bufs, inputs)` | Batched gradient computation |
| `ctx.sparse_jacobian(bufs, inputs)` | Sparse Jacobian on GPU |
| `ctx.hvp_batch(bufs, inputs, dirs)` | Batched Hessian-vector products |
| `ctx.sparse_hessian(bufs, inputs)` | Sparse Hessian on GPU |

CUDA additionally supports `_f64` variants of all methods.

## Architecture

echidna uses a two-tier AD architecture:

1. **Eager mode (Adept-style)**: `Dual<F>` and `Reverse<F>` use operator overloading with a thread-local tape managed by RAII. `Reverse<F>` stores precomputed partial derivatives (no opcode dispatch); the reverse sweep is a single multiply-accumulate loop. `Reverse<f64>` is `Copy` and 12 bytes.

2. **Graph mode (BytecodeTape)**: `BReverse<F>` records operations to an SoA bytecode tape. The tape is recorded once and evaluated many times at different inputs. Optimization passes (CSE, DCE, algebraic simplification, constant folding) reduce tape size. The bytecode tape enables Hessians, sparse derivatives, GPU acceleration, checkpointing, nonsmooth AD, and cross-country elimination.

Types compose via nesting: `Dual<BReverse<f64>>` gives forward-over-reverse for Hessian-vector products, `Taylor<BReverse<f64>, K>` gives Taylor-over-reverse, and `Dual<Dual<f64>>` gives forward-over-forward.

## Comparison

| Library | Forward | Reverse | Graph tape | Sparse | Taylor | GPU | Nonsmooth |
|---------|---------|---------|------------|--------|--------|-----|-----------|
| **echidna** | Yes | Yes | Yes (bytecode, optimized) | Yes (auto coloring) | Yes (const-generic + dynamic) | Yes (wgpu + CUDA) | Yes (Clarke) |
| num-dual | Yes | No | No | No | No | No | No |
| autodiff (crate) | Yes | No | No | No | No | No | No |
| Enzyme (via LLVM) | Yes | Yes | LLVM IR | No | No | No | No |

Enzyme operates at the LLVM IR level and is not a Rust-native library; it requires compiler integration. num-dual and the autodiff crate are pure Rust but limited to forward mode.

## Performance

Benchmarks use Criterion and cover forward mode, reverse mode, bytecode tape, Taylor mode, STDE, GPU, and comparison with num-dual. Run with:

```bash
cargo bench                                         # Forward + reverse
cargo bench --features bytecode --bench bytecode    # Bytecode tape
cargo bench --features stde --bench taylor          # Taylor mode
cargo bench --features gpu-wgpu --bench gpu         # GPU
```

## echidna-optim

The [`echidna-optim`](echidna-optim/) crate provides optimization solvers and implicit differentiation built on `echidna`:

**Solvers**: L-BFGS, Newton (Cholesky), trust-region (Steihaug-Toint CG), with Armijo line search.

**Implicit differentiation**: Differentiate through the fixed point of an optimization problem or nonlinear solve -- `implicit_tangent`, `implicit_adjoint`, `implicit_jacobian`, `implicit_hvp`, `implicit_hessian`.

**Piggyback differentiation**: `piggyback_tangent_solve`, `piggyback_adjoint_solve`, `piggyback_forward_adjoint_solve`.

**Sparse implicit** (with faer): `implicit_tangent_sparse`, `implicit_adjoint_sparse`, `implicit_jacobian_sparse`.

```toml
[dependencies]
echidna-optim = "0.1"
```

Optional features: `parallel` (enables rayon parallelism via `echidna/parallel`), `sparse-implicit` (sparse implicit differentiation via faer).

## Development

```bash
cargo test                                    # Core tests
cargo test --features bytecode,taylor,stde    # All features
cargo bench                                   # Run benchmarks
cargo clippy                                  # Lint
cargo fmt                                     # Format
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

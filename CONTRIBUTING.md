# Contributing to echidna

Thank you for your interest in contributing to echidna! This document provides guidelines and information for contributors.

## Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/echidna.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Run lints: `cargo clippy && cargo fmt --check`
7. Commit your changes
8. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Rust 1.93 or later (install via [rustup](https://rustup.rs/))
- Cargo (included with Rust)

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

### Testing

```bash
# Run core tests (no feature flags)
cargo test

# Run with all CPU features
cargo test --features "bytecode,taylor,laurent,stde,diffop,serde,faer,nalgebra,ndarray"

# Run GPU tests (requires hardware)
cargo test --features "bytecode,gpu-wgpu"
cargo test --features "bytecode,gpu-cuda"

# Run echidna-optim tests
cargo test -p echidna-optim

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_name
```

### Code Quality

Before submitting a PR, ensure:

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps
```

### Benchmarks

```bash
# Run core benchmarks (forward + reverse)
cargo bench

# Bytecode tape
cargo bench --features bytecode --bench bytecode

# Taylor mode
cargo bench --features stde --bench taylor

# STDE estimators
cargo bench --features stde --bench stde

# Differential operators
cargo bench --features diffop --bench diffop

# Cross-country, sparse, nonsmooth
cargo bench --features bytecode --bench advanced

# GPU
cargo bench --features gpu-wgpu --bench gpu

# Comparison vs other libraries
cargo bench --features bytecode --bench comparison
```

### Security Audits

Run dependency audits before submitting PRs:

```bash
# Install audit tools (one-time)
cargo install cargo-audit cargo-deny

# Check for known vulnerabilities
cargo audit

# Check licenses and advisories
cargo deny check
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted with `cargo fmt`
- [ ] No clippy warnings
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for user-facing changes

### PR Description

Please include:

- **What**: Brief description of the change
- **Why**: Motivation for the change
- **How**: High-level approach (if not obvious)
- **Testing**: How you tested the changes

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Architecture Overview

```
src/
├── lib.rs                 # Re-exports, crate-level docs
├── float.rs               # Float trait (f32/f64 marker)
├── scalar.rs              # Scalar trait (AD-generic bound)
├── dual.rs                # Dual<F> forward-mode type + elementals
├── dual_vec.rs            # DualVec<F, N> batched forward-mode
├── tape.rs                # Adept-style two-stack tape
├── reverse.rs             # Reverse<F> reverse-mode type
├── api.rs                 # Public API: grad, jvp, vjp, jacobian, hessian, ...
├── breverse.rs            # BReverse<F> bytecode-tape reverse variable [bytecode]
├── bytecode_tape.rs       # BytecodeTape SoA representation [bytecode]
├── opcode.rs              # Opcode definitions and dispatch (44 opcodes) [bytecode]
├── sparse.rs              # Sparsity detection and graph coloring [bytecode]
├── cross_country.rs       # Markowitz vertex elimination [bytecode]
├── nonsmooth.rs           # Branch tracking, Clarke Jacobian [bytecode]
├── checkpoint.rs          # Revolve + online + disk checkpointing [bytecode]
├── taylor.rs              # Taylor<F, K> const-generic type [taylor]
├── taylor_dyn.rs          # TaylorDyn<F> arena-based type [taylor]
├── taylor_ops.rs          # Shared Taylor propagation rules [taylor]
├── laurent.rs             # Laurent<F, K> singularity analysis [laurent]
├── stde.rs                # Stochastic derivative estimators [stde]
├── diffop.rs              # Arbitrary differential operators via jets [diffop]
├── gpu/
│   ├── mod.rs             # GpuBackend trait, GpuTapeData, GpuError
│   ├── wgpu_backend.rs    # WgpuContext (Metal/Vulkan/DX12, f32) [gpu-wgpu]
│   ├── cuda_backend.rs    # CudaContext (NVIDIA, f32+f64) [gpu-cuda]
│   ├── stde_gpu.rs        # GPU-accelerated STDE functions [stde]
│   ├── shaders/           # 5 WGSL compute shaders [gpu-wgpu]
│   └── kernels/           # CUDA kernels (tape_eval.cu, taylor_eval.cu) [gpu-cuda]
├── faer_support.rs        # faer integration [faer]
├── nalgebra_support.rs    # nalgebra integration [nalgebra]
├── ndarray_support.rs     # ndarray integration [ndarray]
└── traits/
    ├── mod.rs
    ├── std_ops.rs         # Add/Sub/Mul/Div/Neg for all AD types
    ├── num_traits_impls.rs # Zero, One, Num, Float, etc.
    ├── taylor_std_ops.rs  # Taylor arithmetic
    ├── taylor_num_traits.rs # Taylor num_traits
    ├── laurent_std_ops.rs # Laurent arithmetic
    └── laurent_num_traits.rs # Laurent num_traits

echidna-optim/src/
├── lib.rs                 # Re-exports
├── convergence.rs         # Convergence parameters
├── line_search.rs         # Armijo line search
├── objective.rs           # Objective/TapeObjective traits
├── result.rs              # OptimResult, TerminationReason
├── implicit.rs            # Implicit differentiation (IFT)
├── piggyback.rs           # Piggyback differentiation
├── sparse_implicit.rs     # Sparse implicit diff [sparse-implicit]
├── linalg.rs              # Linear algebra utilities
└── solvers/
    ├── mod.rs
    ├── lbfgs.rs           # L-BFGS
    ├── newton.rs          # Newton
    └── trust_region.rs    # Trust-region
```

## Adding New Features

1. **Discuss first**: Open an issue to discuss significant changes
2. **Backward compatibility**: Avoid breaking changes unless necessary
3. **Testing**: Add tests for new functionality — cross-validate forward vs reverse vs finite differences
4. **Documentation**: Update rustdoc and README as needed
5. **Benchmarks**: Add benchmarks for performance-sensitive code

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions

Thank you for contributing!

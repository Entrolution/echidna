# Contributing to echidna

Thank you for your interest in contributing to echidna! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

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

- Rust 1.80 or later (install via [rustup](https://rustup.rs/))
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
# Run all tests
cargo test

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
# Run all benchmarks
cargo bench

# Run a specific benchmark
cargo bench --bench forward
cargo bench --bench reverse
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
├── lib.rs                 # Re-exports, top-level docs
├── float.rs               # Float trait (f32/f64 marker)
├── scalar.rs              # Scalar trait (AD-generic bound)
├── dual.rs                # Dual<F> forward-mode type + elementals
├── tape.rs                # Adept-style two-stack tape
├── reverse.rs             # Reverse<F> reverse-mode type
├── api.rs                 # Public closure API: grad, jvp, vjp, jacobian
└── traits/
    ├── mod.rs
    ├── std_ops.rs         # Add/Sub/Mul/Div/Neg for Dual and Reverse
    └── num_traits_impls.rs # Zero, One, Num, Float, etc.
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

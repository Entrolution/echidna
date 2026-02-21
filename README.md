# echidna

[![CI](https://github.com/Entrolution/echidna/actions/workflows/ci.yml/badge.svg)](https://github.com/Entrolution/echidna/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/echidna.svg)](https://crates.io/crates/echidna)
[![Docs.rs](https://docs.rs/echidna/badge.svg)](https://docs.rs/echidna)
[![License](https://img.shields.io/crates/l/echidna.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.80-blue.svg)](https://www.rust-lang.org)

A high-performance automatic differentiation library for Rust.

echidna provides forward-mode and reverse-mode AD with an ergonomic closure-based API. Write standard Rust numeric code and get exact derivatives automatically.

## Quick Start

```rust
use echidna::{grad, Scalar};

// Gradient of a scalar function via reverse mode
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

## Features

| Mode | Type | Use Case |
|------|------|----------|
| Forward | `Dual<F>` | Few inputs, many outputs (JVP) |
| Reverse | `Reverse<F>` | Many inputs, few outputs (gradient) |

### API

| Function | Description |
|----------|-------------|
| `grad(f, x)` | Gradient of `f : R^n -> R` via reverse mode |
| `jvp(f, x, v)` | Jacobian-vector product via forward mode |
| `vjp(f, x, w)` | Vector-Jacobian product via reverse mode |
| `jacobian(f, x)` | Full Jacobian of `f : R^n -> R^m` via forward mode |

### Design

- **Adept-style two-stack tape** — precomputed partial derivatives, no opcode dispatch. The reverse sweep is a single multiply-accumulate loop.
- **Thread-local tape with RAII** — `Reverse<F>` is `Copy` (12 bytes for f64).
- **Constant sentinel** — literal values don't bloat the tape.
- **`Scalar` trait** — write generic code once, differentiate with any mode.
- **Full `num-traits` interop** — `Float`, `Zero`, `One`, `Num`, `Signed`, `FloatConst`, etc.

## Development

```bash
cargo test              # Run all tests
cargo bench             # Run benchmarks
cargo clippy            # Lint
cargo fmt               # Format
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

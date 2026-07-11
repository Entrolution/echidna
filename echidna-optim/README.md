# echidna-optim

[![Crates.io](https://img.shields.io/crates/v/echidna-optim.svg)](https://crates.io/crates/echidna-optim)
[![Docs.rs](https://docs.rs/echidna-optim/badge.svg)](https://docs.rs/echidna-optim)

Optimization solvers and implicit differentiation for [echidna](https://crates.io/crates/echidna).

## Installation

```toml
[dependencies]
echidna = "0.14"
echidna-optim = "0.14"
```

Requires Rust 1.93 or later, matching the workspace MSRV.

| Feature | Default | Enables |
|---------|---------|---------|
| `parallel` | no | Rayon-parallel objective evaluation paths |
| `sparse-implicit` | no | Sparse implicit differentiation via faer (`implicit_*_sparse`) |

## Quick Start

Record the objective as a bytecode tape, wrap it, and hand it to a solver:

```rust
use echidna::BReverse;
use echidna_optim::{lbfgs, LbfgsConfig, TapeObjective, TerminationReason};

// f(x) = (x0 - 1)^2 + (x1 + 2)^2, minimized at (1, -2).
let (tape, _) = echidna::record(
    |x| {
        let a = x[0] - BReverse::constant(1.0);
        let b = x[1] + BReverse::constant(2.0);
        a * a + b * b
    },
    &[0.0_f64, 0.0],
);

let mut objective = TapeObjective::new(tape);
let result = lbfgs(&mut objective, &[0.0, 0.0], &LbfgsConfig::default());

assert_eq!(result.termination, TerminationReason::GradientNorm);
assert!((result.x[0] - 1.0).abs() < 1e-6);
assert!((result.x[1] + 2.0).abs() < 1e-6);
```

`newton` and `trust_region` take the same objective with their own configs.

## Solvers

Three unconstrained optimizers operating on bytecode-tape objectives:

- **L-BFGS** — limited-memory quasi-Newton (default choice for smooth, large-scale problems)
- **Newton** — exact Hessian with LU factorization (partial pivoting, steepest-descent fallback on indefinite Hessians; quadratic convergence, moderate `n`)
- **Trust-region** — Steihaug-Toint CG subproblem (robust on indefinite/ill-conditioned Hessians)

All solvers use Armijo backtracking line search.

## Implicit Differentiation

Differentiate through solutions of `F(z, x) = 0` via the Implicit Function Theorem:

| Function | Description |
|----------|-------------|
| `implicit_tangent` | Forward-mode: `dz/dx · v` |
| `implicit_adjoint` | Reverse-mode: `(dz/dx)^T · w` |
| `implicit_jacobian` | Full Jacobian `dz/dx` |
| `implicit_hvp` | Hessian-vector product of composed loss |
| `implicit_hessian` | Full Hessian of composed loss |

With the `sparse-implicit` feature (requires faer): `implicit_tangent_sparse`, `implicit_adjoint_sparse`, `implicit_jacobian_sparse`.

## Piggyback Differentiation

Differentiate through fixed-point iterations `z = G(z, x)`:

- `piggyback_tangent_solve` / `piggyback_adjoint_solve` — sequential tangent/adjoint propagation
- `piggyback_forward_adjoint_solve` — interleaved primal + adjoint in one loop

## Stability

Pre-1.0: minor releases (0.x) may contain breaking changes, always listed in
the [CHANGELOG](../CHANGELOG.md). See the
[echidna README's Stability section](../README.md#stability) for the shared
policy.

## License

Licensed under either of Apache License 2.0 or MIT license, at your option.

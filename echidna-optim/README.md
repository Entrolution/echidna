# echidna-optim

[![Crates.io](https://img.shields.io/crates/v/echidna-optim.svg)](https://crates.io/crates/echidna-optim)
[![Docs.rs](https://docs.rs/echidna-optim/badge.svg)](https://docs.rs/echidna-optim)

Optimization solvers and implicit differentiation for [echidna](https://crates.io/crates/echidna).

## Solvers

Three unconstrained optimizers operating on bytecode-tape objectives:

- **L-BFGS** — limited-memory quasi-Newton (default choice for smooth, large-scale problems)
- **Newton** — exact Hessian with Cholesky factorization (quadratic convergence, moderate `n`)
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

## License

Licensed under either of Apache License 2.0 or MIT license, at your option.

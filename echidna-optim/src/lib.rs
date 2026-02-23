//! Optimization solvers with automatic differentiation support, plus implicit
//! differentiation tools for differentiating through fixed-point equations.
//!
//! This crate depends on [`echidna`] with the `bytecode` feature enabled,
//! giving it access to bytecode tapes, forward-over-reverse Hessians, and
//! sparse derivative machinery.
//!
//! # Solvers
//!
//! Three unconstrained optimizers, all operating on a bytecode-tape
//! [`Objective`]:
//!
//! - **L-BFGS** ([`lbfgs`]) — two-loop recursion limited-memory quasi-Newton.
//!   Low per-iteration cost; the default choice for smooth, large-scale problems.
//! - **Newton** ([`newton`]) — exact Hessian with Cholesky factorization.
//!   Quadratic convergence near the solution; practical when `n` is moderate.
//! - **Trust-region** ([`trust_region`]) — Steihaug-Toint conjugate-gradient
//!   subproblem. Robust on indefinite or ill-conditioned Hessians.
//!
//! # Line search
//!
//! All solvers use **Armijo backtracking** ([`ArmijoParams`]) to enforce
//! sufficient decrease along the search direction.
//!
//! # Implicit differentiation
//!
//! Differentiate through solutions of `F(z, x) = 0` via the Implicit Function
//! Theorem without unrolling the solver:
//!
//! - [`implicit_tangent`] — tangent (forward) mode: `dz/dx · v`
//! - [`implicit_adjoint`] — adjoint (reverse) mode: `(dz/dx)^T · w`
//! - [`implicit_jacobian`] — full Jacobian `dz/dx`
//! - [`implicit_hvp`] — Hessian-vector product of a loss composed with the
//!   implicit solution
//! - [`implicit_hessian`] — full Hessian of a loss composed with the implicit
//!   solution
//!
//! # Piggyback differentiation
//!
//! Differentiate through fixed-point iterations `z = G(z, x)` by
//! interleaving derivative accumulation with the primal iteration:
//!
//! - [`piggyback_tangent_solve`] — tangent mode (forward)
//! - [`piggyback_adjoint_solve`] — adjoint mode (reverse)
//! - [`piggyback_forward_adjoint_solve`] — interleaved forward-adjoint for
//!   second-order derivatives
//! - [`piggyback_tangent_step`] / [`piggyback_tangent_step_with_buf`] —
//!   single-step building blocks for custom loops
//!
//! # Sparse implicit differentiation
//!
//! With the **`sparse-implicit`** feature, [`sparse_implicit`] exploits
//! structural sparsity in `F_z` for efficient implicit differentiation via
//! `faer` sparse LU factorization. See [`SparseImplicitContext`],
//! [`implicit_tangent_sparse`], [`implicit_adjoint_sparse`], and
//! [`implicit_jacobian_sparse`].

pub mod convergence;
pub mod implicit;
pub mod linalg;
pub mod line_search;
pub mod objective;
pub mod piggyback;
pub mod result;
pub mod solvers;

#[cfg(feature = "sparse-implicit")]
pub mod sparse_implicit;

pub use convergence::ConvergenceParams;
pub use implicit::{
    implicit_adjoint, implicit_hessian, implicit_hvp, implicit_jacobian, implicit_tangent,
};

pub use line_search::ArmijoParams;
pub use objective::{Objective, TapeObjective};
pub use piggyback::{
    piggyback_adjoint_solve, piggyback_forward_adjoint_solve, piggyback_tangent_solve,
    piggyback_tangent_step, piggyback_tangent_step_with_buf,
};
pub use result::{OptimResult, TerminationReason};
pub use solvers::lbfgs::{lbfgs, LbfgsConfig};
pub use solvers::newton::{newton, NewtonConfig};
pub use solvers::trust_region::{trust_region, TrustRegionConfig};
#[cfg(feature = "sparse-implicit")]
pub use sparse_implicit::{
    implicit_adjoint_sparse, implicit_jacobian_sparse, implicit_tangent_sparse,
    SparseImplicitContext,
};

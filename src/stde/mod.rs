//! Stochastic Taylor Derivative Estimators (STDE).
//!
//! Estimate differential operators (Laplacian, Hessian diagonal, directional
//! derivatives) by pushing random direction vectors through Taylor-mode AD.
//!
//! # How it works
//!
//! For f: R^n -> R at point x, define g(t) = f(x + t*v) where v is a
//! direction vector. The Taylor coefficients of g at t=0 are:
//!
//! - c0 = f(x)
//! - c1 = nabla f(x) . v   (directional first derivative)
//! - c2 = v^T H_f(x) v / 2 (half the directional second derivative)
//!
//! By choosing v appropriately (Rademacher, Gaussian, coordinate basis),
//! we can estimate operators like the Laplacian in O(S*K*L) time instead
//! of O(n^2*L) for the full Hessian.
//!
//! # Variance properties
//!
//! The Hutchinson estimator `(1/S) sum_s v_s^T H v_s` is unbiased when
//! E\[vv^T\] = I. Its variance depends on the distribution of v:
//!
//! - **Rademacher** (entries ±1): Var = `2 sum_{i≠j} H_ij^2`. The diagonal
//!   contributes zero variance since `v_i^2 = 1` always.
//! - **Gaussian** (v ~ N(0,I)): Var = `2 ||H||_F^2`. Higher variance than
//!   Rademacher because `v_i^2 ~ chi-squared(1)` introduces diagonal noise.
//!
//! The [`laplacian_with_control`] function reduces Gaussian variance to match
//! Rademacher by subtracting the exact diagonal contribution (a control
//! variate). This requires the diagonal from [`hessian_diagonal`] (n extra
//! evaluations). For Rademacher directions, the control variate has no effect
//! since the diagonal variance is already zero.
//!
//! **Antithetic sampling** (pairing +v with -v) does **not** reduce variance
//! for trace estimation. The Hutchinson sample `v^T H v` is quadratic (even)
//! in v, so `(-v)^T H (-v) = v^T H v` — antithetic pairs are identical.
//!
//! # Design
//!
//! - **No `rand` dependency**: all functions accept user-provided direction
//!   vectors. The library stays pure; users bring their own RNG.
//! - **`Taylor<F, 3>`** for second-order operators: stack-allocated, Copy,
//!   monomorphized. The order K=3 is statically known.
//! - **`TaylorDyn`** variants for runtime-determined order.
//! - **Panics on misuse**: dimension mismatches panic, following existing
//!   API conventions (`record`, `grad`, `hvp`).
//!
//! # Const-Generic Higher-Order Diagonal
//!
//! [`diagonal_kth_order_const`] is a stack-allocated variant of
//! [`diagonal_kth_order`] for compile-time-known derivative order. It uses
//! `Taylor<F, ORDER>` directly (no `TaylorDyn` arena), which is faster when
//! the order is statically known. `ORDER = k + 1` where k is the derivative
//! order. For f32, practical limit is `ORDER ≤ 14` (k ≤ 13) since `k! > 2^23`
//! causes precision loss.
//!
//! # Higher-Order Estimation
//!
//! [`diagonal_kth_order`] generalises [`hessian_diagonal`] from k=2 to
//! arbitrary k, computing exact `[∂^k u/∂x_j^k]` for all coordinates via
//! `TaylorDyn` jets of order k+1. The stochastic variant
//! [`diagonal_kth_order_stochastic`] subsamples coordinates for an unbiased
//! estimate of `Σ_j ∂^k u/∂x_j^k`.
//!
//! # Dense STDE for Positive-Definite Operators
//!
//! [`dense_stde_2nd`] estimates `tr(C · H_u(x)) = Σ_{ij} C_{ij} ∂²u/∂x_i∂x_j`
//! for a positive-definite coefficient matrix C. The caller provides a Cholesky
//! factor L (such that `C = L L^T`) and standard Gaussian vectors z. Internally,
//! `v = L · z` is computed, then pushed through a second-order Taylor jet. When
//! `L = I`, this reduces to the Hutchinson Laplacian estimator.
//!
//! # Parabolic PDE σ-Transform
//!
//! [`parabolic_diffusion`] computes `½ tr(σσ^T · Hess u)` for parabolic PDEs
//! (Fokker-Planck, Black-Scholes, HJB) by pushing columns of σ through
//! second-order Taylor jets. This avoids forming the off-diagonal Hessian
//! entries that a naïve `tr(A·H)` computation would require.
//!
//! # Sparse STDE (requires `stde` + `diffop`)
//!
//! `stde_sparse` estimates arbitrary differential operators `Lu(x)` by
//! sampling sparse k-jets from a `SparseSamplingDistribution` built via
//! `DiffOp::sparse_distribution`. Each sample is a single forward
//! pushforward; the per-sample estimator is `sign(C_α) · Z · D^α u` where
//! Z = Σ|C_α| is the normalization constant. This implements the core
//! contribution of Shi et al. (NeurIPS 2024).

mod estimator;
mod higher_order;
mod jet;
mod laplacian;
mod pde;
mod pipeline;
#[cfg(feature = "diffop")]
mod sparse;
mod types;

pub use estimator::{Estimator, GradientSquaredNorm, Laplacian};
pub use higher_order::{
    diagonal_kth_order, diagonal_kth_order_const, diagonal_kth_order_const_with_buf,
    diagonal_kth_order_stochastic, diagonal_kth_order_with_buf, laplacian_dyn, taylor_jet_dyn,
};
pub use jet::{directional_derivatives, taylor_jet_2nd, taylor_jet_2nd_with_buf};
pub use laplacian::{
    hessian_diagonal, hessian_diagonal_with_buf, laplacian, laplacian_hutchpp,
    laplacian_with_control, laplacian_with_stats,
};
pub use pde::{
    dense_stde_2nd, divergence, parabolic_diffusion, parabolic_diffusion_stochastic,
};
pub use pipeline::{estimate, estimate_weighted};
pub use types::{DivergenceResult, EstimatorResult};

#[cfg(feature = "diffop")]
pub use sparse::stde_sparse;

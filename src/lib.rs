#![cfg_attr(docsrs, feature(doc_cfg))]
//! # echidna
//!
//! A high-performance automatic differentiation (AD) library for Rust.
//!
//! echidna provides forward-mode, reverse-mode, bytecode-graph-mode, and Taylor-mode
//! automatic differentiation. Write standard Rust numeric code and get exact derivatives
//! automatically -- no symbolic rewriting, no numerical finite differences.
//!
//! # Examples
//!
//! Compute the gradient of f(x, y) = x^2 + y^2:
//!
//! ```
//! let g = echidna::grad(|x: &[echidna::Reverse<f64>]| {
//!     x[0] * x[0] + x[1] * x[1]
//! }, &[3.0, 4.0]);
//! assert!((g[0] - 6.0).abs() < 1e-10);
//! assert!((g[1] - 8.0).abs() < 1e-10);
//! ```
//!
//! Forward-mode with `Dual<f64>`:
//!
//! ```
//! use echidna::{Dual, Float};
//!
//! let x = Dual::new(2.0_f64, 1.0); // seed derivative = 1
//! let y = x.sin() + x * x;
//! assert!((y.eps - (2.0_f64.cos() + 4.0)).abs() < 1e-10);
//! ```
//!
//! # AD Modes
//!
//! | Mode | Type | Best for | Description |
//! |------|------|----------|-------------|
//! | Forward | [`Dual<F>`] | Few inputs, many outputs | Propagates tangent vectors (JVP) |
//! | Reverse | [`Reverse<F>`] | Many inputs, few outputs | Adept-style two-stack tape, `Copy`, 12 bytes for f64 |
//! | Bytecode tape | [`BytecodeTape`] + [`BReverse<F>`] | Record-once evaluate-many | Graph mode enabling Hessians, sparse derivatives, GPU acceleration, checkpointing. Requires `bytecode` feature. |
//! | Taylor | [`Taylor<F, K>`] | Higher-order derivatives | Univariate Taylor propagation. Requires `taylor` feature. |
//! | Batched forward | [`DualVec<F, N>`] | Vectorised Jacobians/Hessians | N tangent directions simultaneously |
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `simba` | yes | simba / approx trait integration |
//! | `bytecode` | no | [`BytecodeTape`] graph-mode AD |
//! | `parallel` | no | Rayon parallel evaluation (implies `bytecode`) |
//! | `taylor` | no | Taylor-mode AD ([`Taylor<F, K>`], [`TaylorDyn<F>`]) |
//! | `laurent` | no | Laurent series for singularity analysis (implies `taylor`) |
//! | `stde` | no | Stochastic Taylor Derivative Estimators (implies `bytecode` + `taylor`) |
//! | `serde` | no | Serialization support via serde |
//! | `faer` | no | faer linear algebra integration (implies `bytecode`) |
//! | `nalgebra` | no | nalgebra linear algebra integration (implies `bytecode`) |
//! | `ndarray` | no | ndarray integration (implies `bytecode`) |
//! | `gpu-wgpu` | no | GPU acceleration via wgpu (implies `bytecode`) |
//! | `gpu-cuda` | no | GPU acceleration via CUDA (implies `bytecode`) |
//!
//! # Core Types
//!
//! **Always available:**
//! - [`Dual<F>`] -- forward-mode dual number
//! - [`Reverse<F>`] -- reverse-mode AD variable
//! - [`DualVec<F, N>`] -- batched forward-mode (N tangent directions)
//! - [`Scalar`] -- trait for writing AD-generic code
//! - [`Float`] -- trait for underlying floating-point types
//! - Type aliases: [`Dual64`], [`Dual32`], [`DualVec64`], [`DualVec32`],
//!   [`Reverse64`], [`Reverse32`]
//!
//! **With `bytecode`:**
//! - [`BytecodeTape`] -- recorded computation graph
//! - [`BReverse<F>`] -- reverse-mode variable for bytecode evaluation
//! - Type aliases: [`BReverse64`], [`BReverse32`]
//!
//! **With `taylor`:**
//! - [`Taylor<F, K>`] -- fixed-order Taylor coefficients
//! - [`TaylorDyn<F>`] -- dynamic-order Taylor coefficients
//! - Type aliases: [`Taylor64`], [`Taylor32`], [`TaylorDyn64`], [`TaylorDyn32`]
//!
//! **With `laurent`:**
//! - [`Laurent<F, K>`] -- Laurent series with negative-order terms
//!
//! # Getting Started
//!
//! - **Gradients**: use [`grad()`] -- it manages the reverse-mode tape automatically.
//! - **Jacobians**: use [`jacobian()`] (forward mode) or [`sparse_jacobian()`]
//!   (sparse, requires `bytecode`).
//! - **Hessians**: use [`hessian()`] or [`sparse_hessian()`] (requires `bytecode`).
//! - **AD-generic code**: write functions as `fn foo<T: Scalar>(x: T) -> T` and they
//!   work with any AD type.
//! - **Tape reuse (performance)**: use [`record()`] to build a [`BytecodeTape`], then
//!   call methods on it for repeated evaluation (requires `bytecode`).

pub mod api;
pub mod dual;
pub mod dual_vec;
pub mod float;
pub mod reverse;
pub mod scalar;
pub mod tape;
mod traits;

#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod breverse;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod bytecode_tape;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod checkpoint;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod cross_country;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod nonsmooth;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod opcode;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub mod sparse;

#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub mod taylor;
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub mod taylor_dyn;
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub mod taylor_ops;

#[cfg_attr(docsrs, doc(cfg(feature = "laurent")))]
#[cfg(feature = "laurent")]
pub mod laurent;

#[cfg_attr(docsrs, doc(cfg(feature = "stde")))]
#[cfg(feature = "stde")]
pub mod stde;

#[cfg_attr(docsrs, doc(cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))))]
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
pub mod gpu;

#[cfg_attr(docsrs, doc(cfg(feature = "faer")))]
#[cfg(feature = "faer")]
pub mod faer_support;
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
pub mod nalgebra_support;
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
pub mod ndarray_support;

pub use api::{grad, jacobian, jvp, vjp};
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use float::Float;
pub use reverse::Reverse;
pub use scalar::Scalar;

#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use api::{
    composed_hvp, hessian, hessian_vec, hvp, record, record_multi, sparse_hessian,
    sparse_hessian_vec, sparse_jacobian,
};
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use breverse::BReverse;
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use bytecode_tape::{BytecodeTape, CustomOp, CustomOpHandle};
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use checkpoint::{
    grad_checkpointed, grad_checkpointed_disk, grad_checkpointed_online,
    grad_checkpointed_with_hints,
};
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use nonsmooth::{ClarkeError, KinkEntry, NonsmoothInfo};
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub use sparse::{CsrPattern, JacobianSparsityPattern, SparsityPattern};

#[cfg_attr(docsrs, doc(cfg(feature = "laurent")))]
#[cfg(feature = "laurent")]
pub use laurent::Laurent;

#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub use taylor::Taylor;
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub use taylor_dyn::{TaylorArena, TaylorDyn, TaylorDynGuard};

/// Type alias for forward-mode dual numbers over `f64`.
pub type Dual64 = Dual<f64>;
/// Type alias for forward-mode dual numbers over `f32`.
pub type Dual32 = Dual<f32>;
/// Type alias for batched forward-mode dual numbers over `f64`.
pub type DualVec64<const N: usize> = DualVec<f64, N>;
/// Type alias for batched forward-mode dual numbers over `f32`.
pub type DualVec32<const N: usize> = DualVec<f32, N>;
/// Type alias for reverse-mode variables over `f64`.
pub type Reverse64 = Reverse<f64>;
/// Type alias for reverse-mode variables over `f32`.
pub type Reverse32 = Reverse<f32>;

/// Type alias for bytecode-tape reverse-mode variables over `f64`.
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub type BReverse64 = BReverse<f64>;
/// Type alias for bytecode-tape reverse-mode variables over `f32`.
#[cfg_attr(docsrs, doc(cfg(feature = "bytecode")))]
#[cfg(feature = "bytecode")]
pub type BReverse32 = BReverse<f32>;

/// Type alias for Taylor coefficients over `f64` with K coefficients.
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub type Taylor64<const K: usize> = Taylor<f64, K>;
/// Type alias for Taylor coefficients over `f32` with K coefficients.
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub type Taylor32<const K: usize> = Taylor<f32, K>;
/// Type alias for dynamic Taylor coefficients over `f64`.
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub type TaylorDyn64 = TaylorDyn<f64>;
/// Type alias for dynamic Taylor coefficients over `f32`.
#[cfg_attr(docsrs, doc(cfg(feature = "taylor")))]
#[cfg(feature = "taylor")]
pub type TaylorDyn32 = TaylorDyn<f32>;

pub mod api;
pub mod dual;
pub mod dual_vec;
pub mod float;
pub mod reverse;
pub mod scalar;
pub mod tape;
mod traits;

#[cfg(feature = "bytecode")]
pub mod breverse;
#[cfg(feature = "bytecode")]
pub mod bytecode_tape;
#[cfg(feature = "bytecode")]
pub mod checkpoint;
#[cfg(feature = "bytecode")]
pub mod cross_country;
#[cfg(feature = "bytecode")]
pub mod nonsmooth;
#[cfg(feature = "bytecode")]
pub mod opcode;
#[cfg(feature = "bytecode")]
pub mod sparse;

#[cfg(feature = "taylor")]
pub mod taylor;
#[cfg(feature = "taylor")]
pub mod taylor_dyn;
#[cfg(feature = "taylor")]
pub mod taylor_ops;

#[cfg(feature = "laurent")]
pub mod laurent;

#[cfg(feature = "stde")]
pub mod stde;

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
pub mod gpu;

#[cfg(feature = "faer")]
pub mod faer_support;
#[cfg(feature = "nalgebra")]
pub mod nalgebra_support;
#[cfg(feature = "ndarray")]
pub mod ndarray_support;

pub use api::{grad, jacobian, jvp, vjp};
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use float::Float;
pub use reverse::Reverse;
pub use scalar::Scalar;

#[cfg(feature = "bytecode")]
pub use api::{
    composed_hvp, hessian, hessian_vec, hvp, record, record_multi, sparse_hessian,
    sparse_hessian_vec, sparse_jacobian,
};
#[cfg(feature = "bytecode")]
pub use breverse::BReverse;
#[cfg(feature = "bytecode")]
pub use bytecode_tape::{BytecodeTape, CustomOp, CustomOpHandle};
#[cfg(feature = "bytecode")]
pub use checkpoint::{
    grad_checkpointed, grad_checkpointed_disk, grad_checkpointed_online,
    grad_checkpointed_with_hints,
};
#[cfg(feature = "bytecode")]
pub use nonsmooth::{ClarkeError, KinkEntry, NonsmoothInfo};
#[cfg(feature = "bytecode")]
pub use sparse::{CsrPattern, JacobianSparsityPattern, SparsityPattern};

#[cfg(feature = "laurent")]
pub use laurent::Laurent;

#[cfg(feature = "taylor")]
pub use taylor::Taylor;
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
#[cfg(feature = "bytecode")]
pub type BReverse64 = BReverse<f64>;
/// Type alias for bytecode-tape reverse-mode variables over `f32`.
#[cfg(feature = "bytecode")]
pub type BReverse32 = BReverse<f32>;

/// Type alias for Taylor coefficients over `f64` with K coefficients.
#[cfg(feature = "taylor")]
pub type Taylor64<const K: usize> = Taylor<f64, K>;
/// Type alias for Taylor coefficients over `f32` with K coefficients.
#[cfg(feature = "taylor")]
pub type Taylor32<const K: usize> = Taylor<f32, K>;
/// Type alias for dynamic Taylor coefficients over `f64`.
#[cfg(feature = "taylor")]
pub type TaylorDyn64 = TaylorDyn<f64>;
/// Type alias for dynamic Taylor coefficients over `f32`.
#[cfg(feature = "taylor")]
pub type TaylorDyn32 = TaylorDyn<f32>;

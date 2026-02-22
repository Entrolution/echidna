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
pub mod opcode;
#[cfg(feature = "bytecode")]
pub mod sparse;

pub use api::{grad, jacobian, jvp, vjp};
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use float::Float;
pub use reverse::Reverse;
pub use scalar::Scalar;

#[cfg(feature = "bytecode")]
pub use api::{hessian, hessian_vec, hvp, record, sparse_hessian};
#[cfg(feature = "bytecode")]
pub use breverse::BReverse;
#[cfg(feature = "bytecode")]
pub use bytecode_tape::BytecodeTape;
#[cfg(feature = "bytecode")]
pub use checkpoint::grad_checkpointed;
#[cfg(feature = "bytecode")]
pub use sparse::SparsityPattern;

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

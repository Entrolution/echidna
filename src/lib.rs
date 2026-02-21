pub mod float;
pub mod scalar;
pub mod dual;
pub mod tape;
pub mod reverse;
pub mod api;
mod traits;

pub use float::Float;
pub use scalar::Scalar;
pub use dual::Dual;
pub use reverse::Reverse;
pub use api::{grad, jvp, vjp, jacobian};

/// Type alias for forward-mode dual numbers over `f64`.
pub type Dual64 = Dual<f64>;
/// Type alias for forward-mode dual numbers over `f32`.
pub type Dual32 = Dual<f32>;
/// Type alias for reverse-mode variables over `f64`.
pub type Reverse64 = Reverse<f64>;
/// Type alias for reverse-mode variables over `f32`.
pub type Reverse32 = Reverse<f32>;

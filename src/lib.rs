pub mod api;
pub mod dual;
pub mod float;
pub mod reverse;
pub mod scalar;
pub mod tape;
mod traits;

pub use api::{grad, jacobian, jvp, vjp};
pub use dual::Dual;
pub use float::Float;
pub use reverse::Reverse;
pub use scalar::Scalar;

/// Type alias for forward-mode dual numbers over `f64`.
pub type Dual64 = Dual<f64>;
/// Type alias for forward-mode dual numbers over `f32`.
pub type Dual32 = Dual<f32>;
/// Type alias for reverse-mode variables over `f64`.
pub type Reverse64 = Reverse<f64>;
/// Type alias for reverse-mode variables over `f32`.
pub type Reverse32 = Reverse<f32>;

pub mod convergence;
pub mod implicit;
pub mod linalg;
pub mod line_search;
pub mod objective;
pub mod result;
pub mod solvers;

pub use convergence::ConvergenceParams;
pub use implicit::{implicit_adjoint, implicit_jacobian, implicit_tangent};
pub use line_search::ArmijoParams;
pub use objective::{Objective, TapeObjective};
pub use result::{OptimResult, TerminationReason};
pub use solvers::lbfgs::{lbfgs, LbfgsConfig};
pub use solvers::newton::{newton, NewtonConfig};
pub use solvers::trust_region::{trust_region, TrustRegionConfig};

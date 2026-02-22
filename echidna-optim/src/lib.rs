pub mod convergence;
pub mod linalg;
pub mod line_search;
pub mod objective;
pub mod result;
pub mod solvers;

pub use convergence::ConvergenceParams;
pub use line_search::ArmijoParams;
pub use objective::{Objective, TapeObjective};
pub use result::{OptimResult, TerminationReason};
pub use solvers::lbfgs::{lbfgs, LbfgsConfig};
pub use solvers::newton::{newton, NewtonConfig};
pub use solvers::trust_region::{trust_region, TrustRegionConfig};

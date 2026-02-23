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

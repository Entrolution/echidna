use std::fmt;

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimResult<F> {
    /// Solution point.
    pub x: Vec<F>,
    /// Objective value at the solution.
    pub value: F,
    /// Gradient at the solution.
    pub gradient: Vec<F>,
    /// Norm of the gradient at the solution.
    pub gradient_norm: F,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Total number of objective function evaluations.
    pub func_evals: usize,
    /// Reason for termination.
    pub termination: TerminationReason,
}

/// Why the optimizer stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Gradient norm fell below tolerance.
    GradientNorm,
    /// Step size fell below tolerance.
    StepSize,
    /// Change in objective value fell below tolerance.
    FunctionChange,
    /// Reached the maximum number of iterations.
    MaxIterations,
    /// Line search could not find a sufficient decrease.
    LineSearchFailed,
    /// A numerical error occurred (e.g. singular Hessian, NaN).
    NumericalError,
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminationReason::GradientNorm => write!(f, "gradient norm below tolerance"),
            TerminationReason::StepSize => write!(f, "step size below tolerance"),
            TerminationReason::FunctionChange => write!(f, "function change below tolerance"),
            TerminationReason::MaxIterations => write!(f, "maximum iterations reached"),
            TerminationReason::LineSearchFailed => write!(f, "line search failed"),
            TerminationReason::NumericalError => write!(f, "numerical error"),
        }
    }
}

use std::fmt;

/// Result of an optimization run.
///
/// Marked `#[non_exhaustive]` so we can add fields without further
/// breaking-change releases. Construct via the solver entry points
/// (`lbfgs`, `newton`, `trust_region`, ...) — never with a struct
/// literal.
#[non_exhaustive]
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
    /// Per-solver diagnostic counters surfacing internal events that
    /// would otherwise be silent (curvature pair filtering, gamma
    /// clamps, line-search backtracks, Newton fallback steps, trust-
    /// region radius shrinks, CG inner iterations).
    ///
    /// Use this to detect when a solver reports `GradientNorm`
    /// convergence but actually spent most of its work in fallback or
    /// filtering paths — a sign that the problem doesn't suit the
    /// chosen solver.
    pub diagnostics: SolverDiagnostics,
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

/// Per-solver diagnostic counters.
///
/// Each variant carries the counters that solver tracks. The enum
/// shape (rather than a flat struct with optional fields) makes it
/// impossible to confuse "this solver doesn't track this counter"
/// with "this counter genuinely observed zero".
///
/// Marked `#[non_exhaustive]` so future solver additions don't keep
/// breaking downstream `match` exhaustiveness.
///
/// # Example
///
/// ```ignore
/// use echidna_optim::{lbfgs, LbfgsConfig, SolverDiagnostics, TerminationReason};
/// let result = lbfgs(&mut obj, &x0, &LbfgsConfig::default());
/// if let SolverDiagnostics::Lbfgs(d) = &result.diagnostics {
///     if result.termination == TerminationReason::GradientNorm
///        && d.pairs_curvature_rejected > d.pairs_accepted
///     {
///         eprintln!("L-BFGS converged but ran mostly as steepest descent — \
///                    consider a different solver or rescale the problem");
///     }
/// }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SolverDiagnostics {
    /// L-BFGS-specific counters.
    Lbfgs(LbfgsDiagnostics),
    /// Newton-specific counters.
    Newton(NewtonDiagnostics),
    /// Trust-region-specific counters.
    TrustRegion(TrustRegionDiagnostics),
    /// Fallback for solver paths that don't yet emit specific counters.
    Other,
}

impl SolverDiagnostics {
    /// Returns the L-BFGS counters if this result came from `lbfgs`.
    #[must_use]
    pub fn as_lbfgs(&self) -> Option<&LbfgsDiagnostics> {
        match self {
            SolverDiagnostics::Lbfgs(d) => Some(d),
            _ => None,
        }
    }

    /// Returns the Newton counters if this result came from `newton`.
    #[must_use]
    pub fn as_newton(&self) -> Option<&NewtonDiagnostics> {
        match self {
            SolverDiagnostics::Newton(d) => Some(d),
            _ => None,
        }
    }

    /// Returns the trust-region counters if this result came from `trust_region`.
    #[must_use]
    pub fn as_trust_region(&self) -> Option<&TrustRegionDiagnostics> {
        match self {
            SolverDiagnostics::TrustRegion(d) => Some(d),
            _ => None,
        }
    }
}

/// Counters surfaced by the L-BFGS solver.
#[derive(Debug, Clone, Default)]
pub struct LbfgsDiagnostics {
    /// Number of (s, y) curvature pairs that passed the Cauchy-Schwarz
    /// filter `sy > F::epsilon() · sqrt(ss · yy)` and entered the
    /// history buffer.
    pub pairs_accepted: usize,
    /// Number of curvature pairs rejected by the filter
    /// `sy > F::epsilon() · sqrt(ss · yy)` (negative or near-zero
    /// curvature, i.e. cosine angle near 0 between `s` and `y`).
    pub pairs_curvature_rejected: usize,
    /// Number of evict-then-push events: a new accepted pair was added
    /// while the history buffer was already at `config.memory`, so the
    /// oldest pair was dropped. With the FIFO eviction policy used here,
    /// the invariant `pairs_evicted_by_memory == max(0, pairs_accepted
    /// - config.memory)` holds exactly at termination.
    pub pairs_evicted_by_memory: usize,
    /// Number of iterations where the initial L-BFGS gamma was clamped
    /// to the open range `(1e-3, 1e3)` (i.e. `raw_gamma` was strictly
    /// outside) or substituted with `1.0` because `sy/yy` was non-finite.
    /// A `raw_gamma` exactly equal to a clamp boundary is not counted.
    pub gamma_clamp_hits: usize,
    /// Total Armijo line-search trial points beyond the first per outer
    /// iteration, summed across all iterations. A high value relative
    /// to `iterations` signals the search direction is poorly scaled.
    pub line_search_backtracks: usize,
}

/// Counters surfaced by the Newton solver.
#[derive(Debug, Clone, Default)]
pub struct NewtonDiagnostics {
    /// Number of iterations where the LU solve failed or returned a
    /// non-descent direction, forcing the steepest-descent fallback.
    pub fallback_steps: usize,
    /// Total Armijo line-search trial points beyond the first.
    pub line_search_backtracks: usize,
}

/// Counters surfaced by the trust-region solver.
#[derive(Debug, Clone, Default)]
pub struct TrustRegionDiagnostics {
    /// Sum of inner Steihaug-CG iterations across all outer iterations.
    pub cg_inner_iters: usize,
    /// Trust-region radius shrinks because the predicted reduction was
    /// non-positive (the quadratic model itself is unreliable).
    pub radius_shrinks_bad_model: usize,
    /// Trust-region radius shrinks because `actual / predicted < 1/4`
    /// (the model over-predicted reduction).
    pub radius_shrinks_low_rho: usize,
}

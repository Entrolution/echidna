//! Sparse implicit differentiation via the Implicit Function Theorem.
//!
//! Exploits structural sparsity in the Jacobian `∂F/∂z` (the state block) to compute
//! implicit derivatives more efficiently for large-scale systems where F_z is sparse
//! (e.g., PDE discretizations, banded systems).
//!
//! Uses echidna's sparse Jacobian infrastructure (sparsity detection, graph coloring,
//! compressed evaluation) combined with faer's sparse LU solver.
//!
//! # Overview
//!
//! Given a residual function `F: R^(m+n) → R^m` with `F(z*, x) = 0`, the three public
//! functions mirror their dense counterparts in [`crate::implicit`]:
//!
//! - [`implicit_tangent_sparse`]: `dz*/dx · ẋ` via sparse LU solve
//! - [`implicit_adjoint_sparse`]: `(dz*/dx)^T · z̄` via sparse transpose solve
//! - [`implicit_jacobian_sparse`]: full `dz*/dx` matrix via column-wise solves
//!
//! The [`SparseImplicitContext`] precomputes sparsity pattern, graph coloring, and
//! COO index partitions so they can be reused across multiple evaluation points.

use echidna::sparse::{column_coloring, row_coloring, JacobianSparsityPattern};
use echidna::BytecodeTape;

use faer::linalg::solvers::SpSolver;
use faer::sparse::SparseColMat;
use faer::Col;

/// Precomputed context for sparse implicit differentiation.
///
/// Stores the sparsity pattern, graph coloring, and index partitions for the full
/// m×(m+n) Jacobian. Construct once per tape, reuse across evaluation points.
pub struct SparseImplicitContext {
    /// Full m×(m+n) Jacobian sparsity pattern.
    pattern: JacobianSparsityPattern,
    /// Column colors (forward mode) or row colors (reverse mode).
    colors: Vec<u32>,
    /// Number of distinct colors.
    num_colors: u32,
    /// Whether to use forward mode (column coloring) for Jacobian computation.
    forward_mode: bool,
    /// Number of state variables (m).
    num_states: usize,
    /// Number of parameters (n = num_inputs - num_states).
    num_params: usize,
    /// COO indices where col < m (F_z entries).
    fz_indices: Vec<usize>,
    /// COO indices where col >= m (F_x entries).
    fx_indices: Vec<usize>,
    /// `fx_indices` grouped by parameter column: `fx_by_col[j]` contains indices
    /// into the COO arrays where `col == m + j`.
    fx_by_col: Vec<Vec<usize>>,
}

impl SparseImplicitContext {
    /// Create a new sparse implicit context from a multi-output tape.
    ///
    /// # Arguments
    ///
    /// * `tape` - A bytecode tape representing `F: R^(m+n) → R^m`
    /// * `num_states` - Number of state variables `m` (first `m` inputs are states)
    ///
    /// # Panics
    ///
    /// Panics if `tape.num_outputs() != num_states` or `tape.num_inputs() <= num_states`.
    pub fn new(tape: &BytecodeTape<f64>, num_states: usize) -> Self {
        let m = num_states;
        assert_eq!(
            tape.num_outputs(),
            m,
            "tape.num_outputs() ({}) must equal num_states ({})",
            tape.num_outputs(),
            m
        );
        assert!(
            tape.num_inputs() > m,
            "tape.num_inputs() ({}) must be greater than num_states ({})",
            tape.num_inputs(),
            m
        );
        let n = tape.num_inputs() - m;

        // Detect full Jacobian sparsity
        let pattern = tape.detect_jacobian_sparsity();

        // Try both colorings, pick fewer colors
        let (col_colors, col_ncolors) = column_coloring(&pattern);
        let (row_colors, row_ncolors) = row_coloring(&pattern);

        let (colors, num_colors, forward_mode) = if col_ncolors <= row_ncolors {
            (col_colors, col_ncolors, true)
        } else {
            (row_colors, row_ncolors, false)
        };

        // Partition COO indices into F_z and F_x blocks
        let mut fz_indices = Vec::new();
        let mut fx_indices = Vec::new();
        let mut fx_by_col: Vec<Vec<usize>> = vec![Vec::new(); n];

        for k in 0..pattern.nnz() {
            let col = pattern.cols[k] as usize;
            if col < m {
                fz_indices.push(k);
            } else {
                fx_indices.push(k);
                fx_by_col[col - m].push(k);
            }
        }

        SparseImplicitContext {
            pattern,
            colors,
            num_colors,
            forward_mode,
            num_states: m,
            num_params: n,
            fz_indices,
            fx_indices,
            fx_by_col,
        }
    }

    /// Number of state variables (m).
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Number of parameters (n).
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Number of structural non-zeros in the full Jacobian pattern.
    pub fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    /// Number of structural non-zeros in the F_z block.
    pub fn fz_nnz(&self) -> usize {
        self.fz_indices.len()
    }

    /// Number of structural non-zeros in the F_x block.
    pub fn fx_nnz(&self) -> usize {
        self.fx_indices.len()
    }
}

// ══════════════════════════════════════════════
//  Private helpers
// ══════════════════════════════════════════════

/// Extract F_z triplets from COO Jacobian values.
fn extract_fz_triplets(
    ctx: &SparseImplicitContext,
    jac_values: &[f64],
) -> Vec<(usize, usize, f64)> {
    ctx.fz_indices
        .iter()
        .map(|&k| {
            (
                ctx.pattern.rows[k] as usize,
                ctx.pattern.cols[k] as usize,
                jac_values[k],
            )
        })
        .collect()
}

/// Build sparse F_z and compute LU factorization.
///
/// Returns `None` if the matrix is singular or construction fails.
/// Uses `catch_unwind` because faer's sparse LU panics on singular matrices
/// rather than returning an error.
fn build_fz_and_factor(
    ctx: &SparseImplicitContext,
    jac_values: &[f64],
) -> Option<faer::sparse::linalg::solvers::Lu<usize, f64>> {
    let m = ctx.num_states;
    let triplets = extract_fz_triplets(ctx, jac_values);
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(m, m, &triplets).ok()?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mat.sp_lu().ok()))
        .ok()
        .flatten()
}

/// Compute F_x · v by iterating COO entries.
fn fx_matvec(ctx: &SparseImplicitContext, jac_values: &[f64], v: &[f64]) -> Vec<f64> {
    let m = ctx.num_states;
    let mut result = vec![0.0; m];
    for &k in &ctx.fx_indices {
        let row = ctx.pattern.rows[k] as usize;
        let col = ctx.pattern.cols[k] as usize - m;
        result[row] += jac_values[k] * v[col];
    }
    result
}

/// Compute F_x^T · v by iterating COO entries.
fn fx_transpose_matvec(ctx: &SparseImplicitContext, jac_values: &[f64], v: &[f64]) -> Vec<f64> {
    let m = ctx.num_states;
    let n = ctx.num_params;
    let mut result = vec![0.0; n];
    for &k in &ctx.fx_indices {
        let row = ctx.pattern.rows[k] as usize;
        let col = ctx.pattern.cols[k] as usize - m;
        result[col] += jac_values[k] * v[row];
    }
    result
}

/// Compute the sparse Jacobian at the given evaluation point.
///
/// Returns `(outputs, jac_values)` where `jac_values` are in COO order matching `ctx.pattern`.
fn compute_sparse_jacobian(
    tape: &mut BytecodeTape<f64>,
    z_star: &[f64],
    x: &[f64],
    ctx: &SparseImplicitContext,
) -> (Vec<f64>, Vec<f64>) {
    let mut inputs = Vec::with_capacity(ctx.num_states + ctx.num_params);
    inputs.extend_from_slice(z_star);
    inputs.extend_from_slice(x);

    let (outputs, jac_values) = tape.sparse_jacobian_with_pattern(
        &inputs,
        &ctx.pattern,
        &ctx.colors,
        ctx.num_colors,
        ctx.forward_mode,
    );

    // Debug check: warn if residual is not near zero
    #[cfg(debug_assertions)]
    {
        let norm_sq: f64 = outputs.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-6 {
            eprintln!(
                "WARNING: sparse implicit differentiation called with ||F(z*, x)|| = {:.6e} > 1e-6. \
                 Derivatives may be meaningless if z* is not a root.",
                norm
            );
        }
    }

    (outputs, jac_values)
}

/// Convert a `Col<f64>` to a `Vec<f64>`.
fn col_to_vec(col: &Col<f64>, len: usize) -> Vec<f64> {
    (0..len).map(|i| col[i]).collect()
}

// ══════════════════════════════════════════════
//  Public API
// ══════════════════════════════════════════════

/// Compute the implicit tangent `dz*/dx · ẋ` using sparse Jacobian evaluation and sparse LU.
///
/// This is the sparse analogue of [`crate::implicit_tangent`]. It solves:
///
///   `F_z · ż = -(F_x · ẋ)`
///
/// where F_z and F_x are computed via graph-coloring-compressed forward/reverse passes,
/// and F_z is factorized using faer's sparse LU.
///
/// Returns `None` if F_z is singular.
///
/// # Panics
///
/// Panics if input dimensions don't match the context.
pub fn implicit_tangent_sparse(
    tape: &mut BytecodeTape<f64>,
    z_star: &[f64],
    x: &[f64],
    x_dot: &[f64],
    ctx: &SparseImplicitContext,
) -> Option<Vec<f64>> {
    let m = ctx.num_states;
    let n = ctx.num_params;
    assert_eq!(z_star.len(), m, "z_star length ({}) must equal num_states ({})", z_star.len(), m);
    assert_eq!(x.len(), n, "x length ({}) must equal num_params ({})", x.len(), n);
    assert_eq!(x_dot.len(), n, "x_dot length ({}) must equal num_params ({})", x_dot.len(), n);

    let (_outputs, jac_values) = compute_sparse_jacobian(tape, z_star, x, ctx);

    // F_x · ẋ
    let fx_xdot = fx_matvec(ctx, &jac_values, x_dot);

    // Sparse LU factorize F_z
    let lu = build_fz_and_factor(ctx, &jac_values)?;

    // Solve F_z · ż = -(F_x · ẋ)
    let rhs = Col::<f64>::from_fn(m, |i| -fx_xdot[i]);
    let sol = lu.solve(&rhs);

    Some(col_to_vec(&sol, m))
}

/// Compute the implicit adjoint `(dz*/dx)^T · z̄` using sparse Jacobian evaluation and sparse LU.
///
/// This is the sparse analogue of [`crate::implicit_adjoint`]. It solves:
///
///   `F_z^T · λ = z̄`
///
/// then computes `x̄ = -F_x^T · λ`.
///
/// Returns `None` if F_z is singular.
///
/// # Panics
///
/// Panics if input dimensions don't match the context.
pub fn implicit_adjoint_sparse(
    tape: &mut BytecodeTape<f64>,
    z_star: &[f64],
    x: &[f64],
    z_bar: &[f64],
    ctx: &SparseImplicitContext,
) -> Option<Vec<f64>> {
    let m = ctx.num_states;
    let n = ctx.num_params;
    assert_eq!(z_star.len(), m, "z_star length ({}) must equal num_states ({})", z_star.len(), m);
    assert_eq!(x.len(), n, "x length ({}) must equal num_params ({})", x.len(), n);
    assert_eq!(z_bar.len(), m, "z_bar length ({}) must equal num_states ({})", z_bar.len(), m);

    let (_outputs, jac_values) = compute_sparse_jacobian(tape, z_star, x, ctx);

    // Sparse LU factorize F_z
    let lu = build_fz_and_factor(ctx, &jac_values)?;

    // Solve F_z^T · λ = z̄
    let rhs = Col::<f64>::from_fn(m, |i| z_bar[i]);
    let lambda = lu.solve_transpose(&rhs);
    let lambda_vec = col_to_vec(&lambda, m);

    // x̄ = -F_x^T · λ
    let fx_t_lambda = fx_transpose_matvec(ctx, &jac_values, &lambda_vec);
    let x_bar: Vec<f64> = fx_t_lambda.iter().map(|&v| -v).collect();

    Some(x_bar)
}

/// Compute the full implicit Jacobian `dz*/dx` (m × n) using sparse LU.
///
/// This is the sparse analogue of [`crate::implicit_jacobian`]. It solves:
///
///   `F_z · col_j = -F_x[:, j]`   for j = 0..n
///
/// using a single sparse LU factorization of F_z, with column extraction via
/// the pre-grouped `fx_by_col` index.
///
/// Returns a dense m×n matrix since `dz*/dx` has no sparsity guarantee.
/// Returns `None` if F_z is singular.
///
/// # Panics
///
/// Panics if input dimensions don't match the context.
pub fn implicit_jacobian_sparse(
    tape: &mut BytecodeTape<f64>,
    z_star: &[f64],
    x: &[f64],
    ctx: &SparseImplicitContext,
) -> Option<Vec<Vec<f64>>> {
    let m = ctx.num_states;
    let n = ctx.num_params;
    assert_eq!(z_star.len(), m, "z_star length ({}) must equal num_states ({})", z_star.len(), m);
    assert_eq!(x.len(), n, "x length ({}) must equal num_params ({})", x.len(), n);

    let (_outputs, jac_values) = compute_sparse_jacobian(tape, z_star, x, ctx);

    // Single factorization
    let lu = build_fz_and_factor(ctx, &jac_values)?;

    // Solve for each column of -F_x
    let mut result = vec![vec![0.0; n]; m];
    for (j, fx_col_indices) in ctx.fx_by_col.iter().enumerate() {
        // Build -F_x[:, j] using pre-grouped indices
        let mut neg_col = vec![0.0; m];
        for &k in fx_col_indices {
            let row = ctx.pattern.rows[k] as usize;
            neg_col[row] -= jac_values[k];
        }

        let rhs = Col::<f64>::from_fn(m, |i| neg_col[i]);
        let sol = lu.solve(&rhs);

        for i in 0..m {
            result[i][j] = sol[i];
        }
    }

    Some(result)
}

//! WS3: variant-pinning tests for the new `Result`-based piggyback /
//! sparse_implicit error API.
//!
//! These tests are deliberately pessimistic: they construct adversarial
//! tapes whose failure mode is *known* (not guessed) and pin the exact
//! `PiggybackError` / `SparseImplicitError` variant returned. A future
//! refactor that silently reclassifies (e.g. promotes a divergence to
//! a max-iter exhaustion) will trip these.

use echidna::record_multi;
use echidna_optim::{
    piggyback_adjoint_solve, piggyback_forward_adjoint_solve, piggyback_tangent_solve,
    PiggybackError,
};

// ── 1. AdjointDivergence: lambda overflows under aggressive blow-up ──
//
// G(z, x) = 100*z + x. Adjoint iteration is `λ_{k+1} = G_z^T · λ + z̄ =
// 100·λ + 1`. λ_k ≈ 100^k, so it crosses f64::MAX (~1.8e308) at k ≈ 154.
// max_iter = 200 ensures the overflow check fires before max_iter exhausts.
#[test]
fn variant_adjoint_divergence_pins_to_adjoint_divergence() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            // 100 = (1+1+1+1+1+1+1+1+1+1) * (1+1+1+1+1+1+1+1+1+1) / (x/x)
            // built from x to keep coefficients data-dependent on inputs.
            let one = x / x;
            let ten = one + one + one + one + one + one + one + one + one + one;
            let hundred = ten * ten;
            vec![hundred * z + x]
        },
        &[1.0_f64, 1.0],
    );

    let err = piggyback_adjoint_solve(&mut tape, &[1.0], &[1.0], &[1.0], 1, 200, 1e-12)
        .expect_err("expected divergence");
    assert!(
        matches!(err, PiggybackError::AdjointDivergence { .. }),
        "expected AdjointDivergence, got {err:?}"
    );
}

// ── 2. PrimalDivergence: quadratic primal blowup in forward_adjoint ──
//
// G(z, x) = z² + x. Starting at z=0, x=1 this gives 0, 1, 2, 5, 26, 677,
// 458330, 2.1e11, 4.5e22, 2.0e45, 4.0e90, 1.6e181, ∞ — overflow at k ≈ 12.
// max_iter = 50 ensures the primal-divergence check fires.
#[test]
fn variant_primal_divergence_pins_to_primal_divergence() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z + x]
        },
        &[0.0_f64, 1.0],
    );

    let err = piggyback_forward_adjoint_solve(&mut tape, &[0.0], &[1.0], &[1.0], 1, 50, 1e-12)
        .expect_err("expected primal blowup");
    assert!(
        matches!(err, PiggybackError::PrimalDivergence { .. }),
        "expected PrimalDivergence, got {err:?}"
    );
}

// ── 3. MaxIterations on tangent_solve: slow contraction ──
//
// G(z, x) = 0.99*z + x. Contraction ratio 0.99, so reaching tol=1e-12 from
// z₀=0 needs ~ln(1e-12)/ln(0.99) ≈ 2750 iterations. max_iter = 50 ensures
// the loop exhausts; the test pins MaxIterations { z_norm: Some(_),
// lam_norm: None } — the shape unique to `piggyback_tangent_solve`.
#[test]
fn variant_max_iterations_tangent_pins_z_norm_only() {
    let (tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            // 0.99 = 99 / 100. Built from inputs so the coefficient is taped.
            let one = x / x;
            let ten = one + one + one + one + one + one + one + one + one + one;
            let hundred = ten * ten;
            let ninety_nine = hundred - one;
            vec![(ninety_nine / hundred) * z + x]
        },
        &[0.0_f64, 1.0],
    );

    let err = piggyback_tangent_solve(&tape, &[0.0], &[1.0], &[1.0], 1, 50, 1e-12)
        .expect_err("slow contraction must not converge in 50 steps");
    assert!(
        matches!(
            err,
            PiggybackError::MaxIterations {
                z_norm: Some(_),
                lam_norm: None
            }
        ),
        "expected MaxIterations with z_norm only, got {err:?}"
    );
}

// ── 4. Display + std::error::Error: smoke test that the trait impls compile
//      and produce non-empty messages for each variant.
#[test]
fn variant_display_smoke_test() {
    let cases = vec![
        PiggybackError::PrimalDivergence { iteration: 7 },
        PiggybackError::TangentDivergence { iteration: 3 },
        PiggybackError::AdjointDivergence { iteration: 11 },
        PiggybackError::MaxIterations {
            z_norm: Some(1e-3),
            lam_norm: None,
        },
    ];
    for err in &cases {
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "empty Display for {err:?}");
        assert!(msg.contains("piggyback"), "Display missing prefix: {msg}");
        // Confirm std::error::Error is actually implemented (compile-time check
        // wrapped in a runtime call so the trait bound is exercised).
        let _: &dyn std::error::Error = err;
    }
}

// ── 5. SparseImplicitError coverage notes ──
//
// `tests/sparse_implicit.rs::sparse_singular_returns_numeric_singular`
// pins `NumericSingular` for all three sparse entry points (rank-1 F_z
// whose LU completes but whose mixed-sign probe produces non-finite
// output). The test below pins `FactorFailed` via a structurally-
// degenerate F_z (zero column triggers faer's symbolic singularity
// detection during `sp_lu()`). The remaining two variants —
// `StructuralFailure` and `Residual` — have only Display-level coverage
// (see `sparse_variant_display_smoke_test`). Reliably triggering
// `Residual` requires a tape whose F_z passes faer's pivot check yet
// produces a finite-but-inaccurate solution: an empirically narrow
// regime not worth a fragile test fixture. `StructuralFailure` only
// fires if `try_new_from_triplets` errors, which would indicate a bug
// in the upstream sparsity-pattern computation rather than user input.

#[cfg(feature = "sparse-implicit")]
#[test]
fn variant_factor_failed_pins_to_factor_failed() {
    use echidna_optim::{implicit_jacobian_sparse, SparseImplicitContext, SparseImplicitError};

    // F(z, x) = [z1 - x, -z1 + x] → F_z = [[0, 1], [0, -1]].
    // Column 0 is identically zero — F_z is structurally singular.
    // faer's sparse LU detects this during symbolic factorization
    // (no candidate pivot for column 0) and returns Err, which maps
    // to `FactorFailed` before the mixed-sign probe ever runs.
    let (mut tape, _) = record_multi(
        |v| {
            let z1 = v[1];
            let x = v[2];
            vec![z1 - x, x - z1]
        },
        &[1.0_f64, 1.0, 1.0],
    );

    let z_star = [1.0, 1.0];
    let x = [1.0];
    let ctx = SparseImplicitContext::new(&tape, 2);

    let err = implicit_jacobian_sparse(&mut tape, &z_star, &x, &ctx)
        .expect_err("structurally singular F_z must error");
    assert!(
        matches!(err, SparseImplicitError::FactorFailed),
        "expected FactorFailed for zero-column F_z, got {err:?}"
    );
}

#[cfg(feature = "sparse-implicit")]
#[test]
fn sparse_variant_display_smoke_test() {
    use echidna_optim::SparseImplicitError;

    let cases = vec![
        SparseImplicitError::StructuralFailure,
        SparseImplicitError::FactorFailed,
        SparseImplicitError::NumericSingular,
        SparseImplicitError::Residual {
            relative_residual: 1.234e-3,
        },
    ];
    for err in &cases {
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "empty Display for {err:?}");
        assert!(
            msg.contains("sparse_implicit"),
            "Display missing prefix: {msg}"
        );
        let _: &dyn std::error::Error = err;
    }

    // Confirm the `Residual` Display branch interpolates the value in
    // scientific notation (regression guard against accidental `:.3` →
    // `1.234` instead of `:.3e` → `1.234e-3`).
    let residual = SparseImplicitError::Residual {
        relative_residual: 1.234e-3,
    };
    let msg = format!("{residual}");
    assert!(
        msg.contains("1.234e-3") || msg.contains("1.234e-03"),
        "Residual Display lost scientific-notation formatting: {msg}"
    );
}

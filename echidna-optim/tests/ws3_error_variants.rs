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
    match err {
        PiggybackError::AdjointDivergence { last_norm, .. } => {
            // The `100·λ` blow-up path crosses `f64::MAX`, so the
            // adjoint-delta norm itself goes non-finite (Inf or NaN)
            // before any componentwise check runs — this binds the
            // WS5 `last_norm` field to the norm-check detection path.
            assert!(
                !last_norm.is_finite(),
                "AdjointDivergence via norm check must carry a non-finite last_norm (got {last_norm})"
            );
        }
        other => panic!("expected AdjointDivergence, got {other:?}"),
    }
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
        PiggybackError::PrimalDivergence {
            iteration: 7,
            last_norm: f64::INFINITY,
        },
        PiggybackError::TangentDivergence {
            iteration: 3,
            last_norm: 1.25e-2,
        },
        PiggybackError::AdjointDivergence {
            iteration: 11,
            last_norm: f64::NAN,
        },
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
        matches!(err, SparseImplicitError::FactorFailed { .. }),
        "expected FactorFailed for zero-column F_z, got {err:?}"
    );
}

#[cfg(feature = "sparse-implicit")]
#[test]
fn sparse_variant_display_smoke_test() {
    use echidna_optim::SparseImplicitError;

    // Dummy boxed source for the literal-construction paths that now
    // carry a `source: Box<dyn Error>` field. `std::io::Error` is
    // `Send + Sync + 'static + std::error::Error` — satisfies the
    // trait-object bounds without pulling in test-only dev-deps.
    let dummy_source = || -> Box<dyn std::error::Error + Send + Sync + 'static> {
        Box::new(std::io::Error::other("dummy"))
    };

    let cases = vec![
        SparseImplicitError::StructuralFailure {
            source: dummy_source(),
        },
        SparseImplicitError::FactorFailed {
            source: dummy_source(),
        },
        SparseImplicitError::NumericSingular,
        SparseImplicitError::Residual {
            relative_residual: 1.234e-3,
            tolerance: 1.5e-8,
            dimension: 42,
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

    // Confirm the `Residual` Display branch interpolates the relative
    // residual, tolerance, and dimension in scientific-notation /
    // decimal form (regression guard against accidental format-spec
    // regressions).
    let residual = SparseImplicitError::Residual {
        relative_residual: 1.234e-3,
        tolerance: 1.5e-8,
        dimension: 42,
    };
    let msg = format!("{residual}");
    assert!(
        msg.contains("1.234e-3") || msg.contains("1.234e-03"),
        "Residual Display lost relative_residual scientific-notation: {msg}"
    );
    assert!(
        msg.contains("1.500e-8") || msg.contains("1.500e-08"),
        "Residual Display lost tolerance scientific-notation: {msg}"
    );
    assert!(
        msg.contains("dim = 42") || msg.contains("42"),
        "Residual Display missing dimension field: {msg}"
    );
}

// ══════════════════════════════════════════════════════════════
//  WS5 — payload enrichment regressions
// ══════════════════════════════════════════════════════════════

// ── TangentDivergence solver-path test ──
//
// Contraction `G(z, x) = 0.5·z + x` — primal ratio 0.5, converges
// from any finite start. With `x_dot = [f64::INFINITY]` the tangent
// forward pass sees `eps = Inf` on the input `x`; propagating that
// through the tape's operations — including the `x / x` trick used
// to materialise tape-side `1.0` — generates non-finite tangent
// coefficients (the specific value is `NaN` rather than `Inf`, via
// `(Inf - 1·1·Inf) / (1·1) = NaN` in dual division, but the branch
// we care about fires on *any* non-finite tangent entry). Meanwhile
// `.re` — the primal — stays finite because `z_new = 0.5·z_0 + x`
// doesn't reference the tangent stream. The primal-delta norm is
// therefore finite; the componentwise-finite check on `z_dot_new`
// fires → `TangentDivergence { iteration: 0, last_norm: finite }`.
// This was noted in the WS3 plan as hard-but-tractable; Inf-x_dot
// makes it trivial and deterministic.
#[test]
fn variant_tangent_divergence_pins_to_tangent_divergence() {
    #[allow(clippy::eq_op)]
    let (tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            // 0.5 = 1 / (1 + 1). `x / x` materialises a tape-side `1.0`.
            let one = x / x;
            let two = one + one;
            vec![(one / two) * z + x]
        },
        &[0.0_f64, 1.0],
    );

    let err = piggyback_tangent_solve(&tape, &[0.0], &[1.0], &[f64::INFINITY], 1, 50, 1e-12)
        .expect_err("Inf x_dot must make tangent diverge on first step");
    match err {
        PiggybackError::TangentDivergence {
            iteration,
            last_norm,
        } => {
            assert_eq!(iteration, 0, "tangent divergence must fire at iteration 0");
            assert!(
                last_norm.is_finite(),
                "primal-delta norm should be finite in the ratio-converging case (got {last_norm})"
            );
        }
        other => panic!("expected TangentDivergence, got {other:?}"),
    }
}

// ── PrimalDivergence carries a non-finite last_norm ──
//
// Quadratic blow-up from `G(z, x) = z² + x` at `z=0, x=1` overflows
// primal to Inf by ~k=12; `!norm.is_finite()` fires and `last_norm`
// should be captured as the non-finite value at that iteration (not
// the previous iteration's finite value — see WS5 plan Finding #1).
#[test]
fn variant_primal_divergence_carries_non_finite_last_norm() {
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
    match err {
        PiggybackError::PrimalDivergence { last_norm, .. } => {
            assert!(
                !last_norm.is_finite(),
                "PrimalDivergence via norm check must carry a non-finite last_norm (got {last_norm})"
            );
        }
        other => panic!("expected PrimalDivergence, got {other:?}"),
    }
}

// ── FactorFailed surfaces the underlying faer LuError via source() ──
//
// Extends `variant_factor_failed_pins_to_factor_failed`: the zero-
// column F_z triggers faer's symbolic LU detection. The WS5 migration
// preserves the faer error as `SparseImplicitError::FactorFailed`'s
// `source` so callers get the exact failure point
// (`SymbolicSingular { index }`) rather than the generic wrapper
// message. faer's `LuError` Display delegates to Debug, so the
// variant name appears verbatim in the formatted source.
#[cfg(feature = "sparse-implicit")]
#[test]
fn variant_factor_failed_exposes_faer_lu_error_via_source() {
    use echidna_optim::{implicit_jacobian_sparse, SparseImplicitContext, SparseImplicitError};

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
        matches!(err, SparseImplicitError::FactorFailed { .. }),
        "expected FactorFailed, got {err:?}"
    );

    // Walk the source chain. `err.source()` must not collapse to None.
    let src = std::error::Error::source(&err).expect("FactorFailed must carry a source error");
    let src_msg = format!("{src}");
    assert!(
        src_msg.contains("SymbolicSingular"),
        "source message should surface faer's SymbolicSingular variant, got {src_msg:?}"
    );
}

// ── StructuralFailure carries a source too (construction-literal) ──
//
// `SparseImplicitError::StructuralFailure` only fires when
// `try_new_from_triplets` errors, which in this codebase would
// indicate a bug in the sparsity-pattern computation rather than
// user input — noted as empirically rare in the code comment. Rather
// than hand-inject invalid triplets (which would require exposing
// internals), pin the `source()` plumbing via construction-literal.
#[cfg(feature = "sparse-implicit")]
#[test]
fn variant_structural_failure_exposes_source_via_source_method() {
    use echidna_optim::SparseImplicitError;

    let err = SparseImplicitError::StructuralFailure {
        source: Box::new(std::io::Error::other("synthetic triplet-builder failure")),
    };
    let src = std::error::Error::source(&err).expect("StructuralFailure must carry a source error");
    let src_msg = format!("{src}");
    assert!(
        src_msg.contains("synthetic triplet-builder failure"),
        "source message should surface the boxed error's Display, got {src_msg:?}"
    );
}

// ── MaxIterations Display no longer leaks Rust-internal syntax ──
//
// Regression guard on the WS5 adjacent-fix. Pre-fix output used
// `{:?}` on `Option<f64>`, leaking `Some(0.0034)` / `None`. Post-fix
// emits only the populated norm(s), with scientific notation, and
// the `(None, None)` case degrades to a bare prefix (no `()`).
#[test]
fn piggyback_max_iterations_display_has_no_rust_syntax() {
    let cases = [
        PiggybackError::MaxIterations {
            z_norm: Some(3.4e-3),
            lam_norm: Some(1.2e-2),
        },
        PiggybackError::MaxIterations {
            z_norm: Some(3.4e-3),
            lam_norm: None,
        },
        PiggybackError::MaxIterations {
            z_norm: None,
            lam_norm: Some(1.2e-2),
        },
        // Impossible-by-construction today but guard the defensive arm.
        PiggybackError::MaxIterations {
            z_norm: None,
            lam_norm: None,
        },
    ];

    for err in &cases {
        let msg = format!("{err}");
        for leak in ["Some", "None", "Option"] {
            assert!(
                !msg.contains(leak),
                "Display leaks Rust-internal token {leak:?}: {msg:?}"
            );
        }
        // Defensive-arm sanity: no trailing empty parens when both norms are None.
        assert!(
            !msg.ends_with("()"),
            "Display ends with empty parens: {msg:?}"
        );
    }
}

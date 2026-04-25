//! WS7: variant-pinning tests for the new `Result`-based dense
//! `implicit` error API.
//!
//! Mirrors the shape of `tests/ws3_error_variants.rs` (sparse counterpart).
//! Not gated on `sparse-implicit` — exercises the dense path only and must
//! run under both `cargo test -p echidna-optim` and
//! `cargo test -p echidna-optim --features sparse-implicit`.
//!
//! Each public entry point gets its own `#[test]`; a single sweep would
//! stop at the first `expect_err` failure and leave the other four
//! unverified, so per-fn tests make the "all five covered" property
//! visible in the pass count and give clean per-fn failure messages.

use echidna::record_multi;
use echidna_optim::{
    implicit_adjoint, implicit_hessian, implicit_hvp, implicit_jacobian, implicit_tangent,
    ImplicitError,
};

/// Build a tape whose `F_z = [[1,1],[2,2]]` is exactly rank 1 — the
/// `max_val == 0` branch of `lu_factor`'s singularity check. Returned
/// from a shared helper so every per-fn test exercises the *same*
/// failure mode rather than five subtly different ones.
// `x / x` materialises a tape-side `1.0` constant using the input value,
// since the closure's `v` elements are the AD scalar type (not plain
// `f64`). Triggers `clippy::eq_op` under `--tests`; allowed here to keep
// the idiom identical to the rest of this crate's test suite (e.g.
// `tests/implicit.rs`, `tests/piggyback.rs`).
#[allow(clippy::eq_op)]
fn singular_rank1_tape() -> echidna::BytecodeTape<f64> {
    record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x = v[2];
            let one = x / x;
            let two = one + one;
            vec![z0 + z1 - x, two * z0 + two * z1 - two * x]
        },
        &[0.5_f64, 0.5, 1.0],
    )
    .0
}

#[test]
fn variant_singular_pins_implicit_jacobian() {
    let mut tape = singular_rank1_tape();
    let err = implicit_jacobian(&mut tape, &[0.5, 0.5], &[1.0], 2)
        .expect_err("rank-1 F_z must error from implicit_jacobian");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

#[test]
fn variant_singular_pins_implicit_tangent() {
    let mut tape = singular_rank1_tape();
    let err = implicit_tangent(&mut tape, &[0.5, 0.5], &[1.0], &[1.0], 2)
        .expect_err("rank-1 F_z must error from implicit_tangent");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

#[test]
fn variant_singular_pins_implicit_adjoint() {
    let mut tape = singular_rank1_tape();
    let err = implicit_adjoint(&mut tape, &[0.5, 0.5], &[1.0], &[1.0, 0.0], 2)
        .expect_err("rank-1 F_z must error from implicit_adjoint");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

#[test]
fn variant_singular_pins_implicit_hvp() {
    let mut tape = singular_rank1_tape();
    let err = implicit_hvp(&mut tape, &[0.5, 0.5], &[1.0], &[1.0], &[1.0], 2)
        .expect_err("rank-1 F_z must error from implicit_hvp");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

#[test]
fn variant_singular_pins_implicit_hessian() {
    let mut tape = singular_rank1_tape();
    let err = implicit_hessian(&mut tape, &[0.5, 0.5], &[1.0], 2)
        .expect_err("rank-1 F_z must error from implicit_hessian");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

// ── NaN input regression ──
//
// For nonlinear `F(z, x) = z² - x`, the Jacobian entry ∂F/∂z = 2·z picks up
// the NaN when `z*` is NaN. `F_z = [[NaN]]` must trip `lu_factor`'s new
// non-finite-pivot guard (before WS7 Cycle 2 this fell through both the
// `== 0` and `< tol` checks and produced `Ok(vec![NaN])`). This regression
// pins the contract: non-finite `F_z` → `Err(Singular)`, not `Ok(NaN)`.
#[test]
fn variant_singular_pins_on_nan_jacobian_input() {
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z - x]
        },
        &[2.0_f64, 4.0],
    );

    let z_nan = [f64::NAN];
    let x = [4.0_f64];

    assert!(
        matches!(
            implicit_jacobian(&mut tape, &z_nan, &x, 1),
            Err(ImplicitError::Singular)
        ),
        "NaN z_star must surface as ImplicitError::Singular from implicit_jacobian"
    );
    assert!(
        matches!(
            implicit_tangent(&mut tape, &z_nan, &x, &[1.0], 1),
            Err(ImplicitError::Singular)
        ),
        "NaN z_star must surface as ImplicitError::Singular from implicit_tangent"
    );
    assert!(
        matches!(
            implicit_adjoint(&mut tape, &z_nan, &x, &[1.0], 1),
            Err(ImplicitError::Singular)
        ),
        "NaN z_star must surface as ImplicitError::Singular from implicit_adjoint"
    );
    assert!(
        matches!(
            implicit_hvp(&mut tape, &z_nan, &x, &[1.0], &[1.0], 1),
            Err(ImplicitError::Singular)
        ),
        "NaN z_star must surface as ImplicitError::Singular from implicit_hvp"
    );
    assert!(
        matches!(
            implicit_hessian(&mut tape, &z_nan, &x, 1),
            Err(ImplicitError::Singular)
        ),
        "NaN z_star must surface as ImplicitError::Singular from implicit_hessian"
    );
}

// ── Finite F_z, non-finite RHS — exercises the post-solve guards ──
//
// When `F_z` stays finite but the solve RHS goes non-finite (e.g. NaN
// `x_dot` / `z_bar`), `lu_factor` accepts the matrix and `lu_back_solve`
// propagates NaN through the substitution. Without the guards added to
// `implicit_tangent` / `implicit_adjoint` / `implicit_hvp` /
// `implicit_hessian` these would escape as `Ok(vec![NaN, ...])`.
#[test]
fn variant_singular_pins_on_nan_x_dot() {
    // F(z, x) = z² - x, at z* = 2, x = 4. F_z = [[4]] — finite.
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z - x]
        },
        &[2.0_f64, 4.0],
    );
    let err = implicit_tangent(&mut tape, &[2.0], &[4.0], &[f64::NAN], 1)
        .expect_err("NaN x_dot must trip the post-solve non-finite guard");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

#[test]
fn variant_singular_pins_on_nan_z_bar() {
    // Same tape/point. NaN `z_bar` → NaN adjoint-solve RHS.
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z - x]
        },
        &[2.0_f64, 4.0],
    );
    let err = implicit_adjoint(&mut tape, &[2.0], &[4.0], &[f64::NAN], 1)
        .expect_err("NaN z_bar must trip the post-solve non-finite guard");
    assert!(
        matches!(err, ImplicitError::Singular),
        "expected Singular, got {err:?}"
    );
}

// ── Display + std::error::Error smoke test ──
//
// Iterates over a `vec![]` so a future variant is added to the test by
// appending one line; mirrors the shape of
// `sparse_variant_display_smoke_test`.
#[test]
fn variant_display_smoke_test() {
    let cases = vec![ImplicitError::Singular];
    for err in &cases {
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "empty Display for {err:?}");
        assert!(
            msg.contains("implicit"),
            "Display missing `implicit` prefix: {msg}"
        );
        // Compile-time check of the `std::error::Error` bound wrapped in a
        // runtime coercion.
        let _: &dyn std::error::Error = err;
    }
}

#![cfg(feature = "bytecode")]
//! The reverse-mode zero-multiplier convention (see `kernels/mod.rs`): an
//! exactly-zero partial absorbs any adjoint, including Inf/NaN adjoints from
//! a chained singularity downstream. A zero partial means the input does not
//! move the output locally — the adjoint's magnitude is irrelevant. NaN
//! partials (out-of-domain) are NOT zero and still propagate.

use echidna::record;
use echidna::BReverse;
use num_traits::Float as _;

/// sqrt'(0) = Inf sends an infinite adjoint into hypot's origin node, whose
/// partials are exactly (0, 0). The gradient must be [0, 0], and — unlike a
/// record-time constant fold — the SAME tape must still produce correct
/// gradients on replay away from the origin.
#[test]
fn bytecode_hypot_origin_absorbs_infinite_adjoint_and_replays() {
    let (mut tape, _) = record(|v: &[BReverse<f64>]| v[0].hypot(v[1]).sqrt(), &[3.0, 4.0]);

    let g0 = tape.gradient(&[0.0, 0.0]);
    assert_eq!(
        g0[0], 0.0,
        "zero partial absorbs the Inf adjoint, got {}",
        g0[0]
    );
    assert_eq!(g0[1], 0.0);

    // Replay integrity: d/dx sqrt(hypot(x,y)) = (x/h)·(1/(2·sqrt(h))).
    let g = tape.gradient(&[3.0, 4.0]);
    let expect_x = (3.0 / 5.0) * 0.5 / 5.0_f64.sqrt();
    let expect_y = (4.0 / 5.0) * 0.5 / 5.0_f64.sqrt();
    assert!(
        (g[0] - expect_x).abs() < 1e-12,
        "replay gradient x: {}",
        g[0]
    );
    assert!(
        (g[1] - expect_y).abs() < 1e-12,
        "replay gradient y: {}",
        g[1]
    );
}

#[test]
fn bytecode_atan2_origin_absorbs_infinite_adjoint() {
    let (mut tape, _) = record(|v: &[BReverse<f64>]| v[0].atan2(v[1]).sqrt(), &[3.0, 4.0]);
    let g0 = tape.gradient(&[0.0, 0.0]);
    assert_eq!(g0[0], 0.0);
    assert_eq!(g0[1], 0.0);
}

/// Regular-point zero partials follow the same rule: the losing branch of a
/// max and a mul-by-zero operand absorb an infinite adjoint.
#[test]
fn regular_zero_partials_absorb_infinite_adjoints() {
    // f = sqrt(max(x, y)) at (4, 1): loser y has partial 0; sqrt is regular
    // at 4, so make the adjoint infinite via sqrt(x - 4)... instead chain:
    // f = sqrt(max(x, y) - 4) at (4, 1): sqrt'(0) = Inf flows into max;
    // winner x gets Inf (live), loser y must get exactly 0.
    let (mut tape, _) = record(
        |v: &[BReverse<f64>]| (v[0].max(v[1]) - 4.0).sqrt(),
        &[5.0, 1.0],
    );
    let g = tape.gradient(&[4.0, 1.0]);
    assert!(
        g[0].is_infinite(),
        "winner keeps the live Inf, got {}",
        g[0]
    );
    assert_eq!(g[1], 0.0, "loser's zero partial absorbs the Inf adjoint");

    // f = sqrt(x·y) at (5, 0): ∂f/∂x = y/(2·sqrt(xy)) — partial of the mul
    // node w.r.t. x is y = 0, adjoint is Inf. Pointwise limit is 0.
    let (mut tape2, _) = record(|v: &[BReverse<f64>]| (v[0] * v[1]).sqrt(), &[2.0, 3.0]);
    let g2 = tape2.gradient(&[5.0, 0.0]);
    assert_eq!(
        g2[0], 0.0,
        "zero partial × Inf adjoint must be 0, got {}",
        g2[0]
    );
    assert!(g2[1].is_infinite(), "live partial keeps Inf, got {}", g2[1]);
}

/// Out-of-domain NaN partials are not zero: NaN still propagates.
#[test]
fn nan_partials_still_propagate() {
    let (mut tape, _) = record(|v: &[BReverse<f64>]| v[0].ln().sqrt(), &[2.0]);
    let g = tape.gradient(&[-1.0]);
    assert!(
        g[0].is_nan(),
        "out-of-domain partial must stay NaN, got {}",
        g[0]
    );
}

/// The documented singular-point conventions, pinned per elemental: a live
/// tangent keeps the non-finite derivative with the documented sign; a
/// structurally zero tangent stays exactly zero. (Describes current
/// behavior — the convention doc in kernels/mod.rs — it does not legislate.)
#[test]
fn singular_point_convention_table() {
    use echidna::Dual;

    type Elemental = fn(Dual<f64>) -> Dual<f64>;
    // (name, singular x, elemental, expected live-tangent derivative)
    let cases: &[(&str, f64, Elemental, f64)] = &[
        ("sqrt@0", 0.0, |x| x.sqrt(), f64::INFINITY),
        ("cbrt@0", 0.0, |x| x.cbrt(), f64::INFINITY),
        ("ln@+0", 0.0, |x| x.ln(), f64::INFINITY),
        ("ln@-0", -0.0, |x| x.ln(), f64::NEG_INFINITY),
        ("recip@+0", 0.0, |x| x.recip(), f64::NEG_INFINITY),
        ("asin@1", 1.0, |x| x.asin(), f64::INFINITY),
        ("acos@1", 1.0, |x| x.acos(), f64::NEG_INFINITY),
        ("acosh@1", 1.0, |x| x.acosh(), f64::INFINITY),
        ("atanh@1", 1.0, |x| x.atanh(), f64::INFINITY),
    ];
    for &(name, x0, f, expect) in cases {
        let live = f(Dual::new(x0, 1.0));
        assert_eq!(
            live.eps, expect,
            "{name}: live tangent must be {expect}, got {}",
            live.eps
        );
        let dead = f(Dual::new(x0, 0.0));
        assert_eq!(
            dead.eps, 0.0,
            "{name}: structurally zero tangent must stay 0, got {}",
            dead.eps
        );
    }
}

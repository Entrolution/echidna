//! Out-of-domain derivative convention.
//!
//! Domain-restricted ops (`ln`, `log2`, `log10`, `ln_1p`, `atanh`) must emit a
//! NaN partial *strictly* outside their real domain — across every AD mode, not
//! just the bytecode `OpCode` reference — so a caller who supplies out-of-domain
//! inputs sees NaN instead of a finite but semantically meaningless derivative
//! (e.g. `d/dx ln(x) = 1/x = -0.5` at `x = -2`). Boundary values (`x = 0` for
//! `ln`, `x = -1` for `ln_1p`, `|x| = 1` for `atanh`) keep the IEEE `1/0 = ±Inf`
//! one-sided limit.

use echidna::{grad, jvp, Dual, DualVec, Reverse};
use num_traits::Float;

// Reverse (reverse-mode gradient), Dual (forward-mode JVP), and DualVec
// (forward-mode vector) must all agree with the `OpCode` convention: NaN
// derivative at an out-of-domain point.
macro_rules! out_of_domain_is_nan {
    ($name:ident, $method:ident, $p:expr) => {
        #[test]
        fn $name() {
            let p: f64 = $p;

            let g = grad(|x: &[Reverse<f64>]| x[0].$method(), &[p]);
            assert!(
                g[0].is_nan(),
                "{}: reverse grad must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                g[0]
            );

            let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].$method()], &[p], &[1.0]);
            assert!(
                t[0].is_nan(),
                "{}: dual tangent must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                t[0]
            );

            let dv = DualVec::<f64, 1>::new(p, [1.0]).$method();
            assert!(
                dv.eps[0].is_nan(),
                "{}: dualvec tangent must be NaN at out-of-domain x={}, got {}",
                stringify!($method),
                p,
                dv.eps[0]
            );
        }
    };
}

out_of_domain_is_nan!(ln_out_of_domain_is_nan, ln, -2.0);
out_of_domain_is_nan!(log2_out_of_domain_is_nan, log2, -2.0);
out_of_domain_is_nan!(log10_out_of_domain_is_nan, log10, -2.0);
out_of_domain_is_nan!(ln_1p_out_of_domain_is_nan, ln_1p, -2.0);
out_of_domain_is_nan!(atanh_out_of_domain_is_nan, atanh, 1.5);

// Boundary + in-domain: the guard must NOT clobber the finite/±Inf values
// inside the domain (guards against over-correction that returns NaN there).
// `+0.0` is used deliberately — `-0.0` correctly yields `-Inf` via `1/-0.0` on
// every path and is not a boundary the convention special-cases.
#[test]
fn ln_boundary_and_in_domain_preserved() {
    let g = grad(|x: &[Reverse<f64>]| x[0].ln(), &[0.0]);
    assert!(
        g[0].is_infinite() && g[0] > 0.0,
        "ln reverse grad at x=+0 must be +Inf, got {}",
        g[0]
    );

    let (_, t) = jvp(|x: &[Dual<f64>]| vec![x[0].ln()], &[0.0], &[1.0]);
    assert!(
        t[0].is_infinite() && t[0] > 0.0,
        "ln dual tangent at x=+0 must be +Inf, got {}",
        t[0]
    );

    let g = grad(|x: &[Reverse<f64>]| x[0].ln(), &[2.0]);
    assert!(
        (g[0] - 0.5).abs() < 1e-15,
        "ln reverse grad at x=2 must be 0.5, got {}",
        g[0]
    );
}

#[test]
fn atanh_boundary_preserves_inf() {
    let g = grad(|x: &[Reverse<f64>]| x[0].atanh(), &[1.0]);
    assert!(
        g[0].is_infinite() && g[0] > 0.0,
        "atanh reverse grad at x=1 must be +Inf, got {}",
        g[0]
    );
}

// Cross-mode anchor: the bytecode forward-over-reverse path (gradient AND
// Hessian-vector product) already routes through the `OpCode` convention, so it
// is the canonical reference the scalar modes above must match. Green before and
// after the scalar-mode fix — it pins the target convention and cross-mode
// agreement in one place.
#[cfg(feature = "bytecode")]
#[test]
fn bytecode_hvp_out_of_domain_is_nan() {
    use echidna::{hvp, BReverse};

    type UnaryOp = fn(&[BReverse<f64>]) -> BReverse<f64>;
    let cases: [(&str, UnaryOp, f64); 5] = [
        ("ln", |x| x[0].ln(), -2.0),
        ("log2", |x| x[0].log2(), -2.0),
        ("log10", |x| x[0].log10(), -2.0),
        ("ln_1p", |x| x[0].ln_1p(), -2.0),
        ("atanh", |x| x[0].atanh(), 1.5),
    ];
    for (label, f, p) in cases {
        let (g, h) = hvp(f, &[p], &[1.0]);
        assert!(
            g[0].is_nan(),
            "{label}: bytecode gradient must be NaN at out-of-domain x={p}, got {}",
            g[0]
        );
        assert!(
            h[0].is_nan(),
            "{label}: bytecode HVP must be NaN at out-of-domain x={p}, got {}",
            h[0]
        );
    }
}

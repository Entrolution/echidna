#![cfg(feature = "bytecode")]
//! Opcode × mode parity matrix (structural guard against surface drift).
//!
//! Every unary elemental is evaluated at a grid of edge inputs across the
//! three CPU surfaces — `Dual` (forward), the bytecode tape's dual-tangent
//! sweep, and the eager `Reverse` gradient — asserting:
//!
//! - primals agree bitwise (or are both NaN),
//! - forward tangents agree (or are both NaN),
//! - the reverse derivative agrees with the forward tangent BY VALUE
//!   (signed zeros excepted: adjoint accumulation starts at +0.0 and IEEE
//!   +0.0 + (-0.0) = +0.0, so reverse mode cannot preserve a derivative's
//!   negative zero — an inherent property of accumulation, not drift).
//!   Kink conventions agree across modes today (abs/signum/floor/… all
//!   carry derivative 0 at their kinks in BOTH directions), so no
//!   exceptions are needed; if a mode-specific kink convention is ever
//!   introduced, this matrix will flag it for an explicit decision.
//!
//! GPU surfaces are not re-enumerated here: they run in f32 with tolerance
//! regimes of their own and are covered by `gpu_cpu_parity` / `gpu_stde` /
//! `gpu_kernel_parity`.

use echidna::{grad, record, BReverse, Dual};
use num_traits::Float as _;

const EDGES: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1e-300,
    1e300,
];

fn both_nan_or_eq(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

macro_rules! unary_parity {
    ($($name:ident),* $(,)?) => {$(
        #[test]
        fn $name() {
            for &x in EDGES {
                // Forward: Dual.
                let live = Dual::new(x, 1.0).$name();
                let dead = Dual::new(x, 0.0).$name();

                // Bytecode: record the op at a regular point, then run the
                // dual-tangent sweep at x (replay semantics).
                let (tape, _) = record(|v: &[BReverse<f64>]| v[0].$name(), &[0.7]);
                let mut buf = Vec::new();
                tape.forward_tangent_dual(&[Dual::new(x, 1.0)], &mut buf);
                let bt = buf[tape.output_index()];

                assert!(
                    both_nan_or_eq(live.re, bt.re),
                    "{}({x:?}): primal drift Dual {} vs bytecode {}",
                    stringify!($name), live.re, bt.re
                );
                assert!(
                    both_nan_or_eq(live.eps, bt.eps),
                    "{}({x:?}): tangent drift Dual {} vs bytecode {}",
                    stringify!($name), live.eps, bt.eps
                );
                assert_eq!(
                    dead.eps, 0.0,
                    "{}({x:?}): structural-zero tangent must stay 0, got {}",
                    stringify!($name), dead.eps
                );

                // Reverse: eager gradient (seed 1) vs the forward tangent.
                // Value equality (0 == -0): accumulation cannot preserve
                // derivative signed zeros — see the file doc.
                let g = grad(|v| v[0].$name(), &[x]);
                assert!(
                    (g[0].is_nan() && live.eps.is_nan()) || g[0] == live.eps,
                    "{}({x:?}): reverse {} vs forward tangent {}",
                    stringify!($name), g[0], live.eps
                );
            }
        }
    )*};
}

unary_parity!(
    sqrt, cbrt, exp, exp2, exp_m1, ln, ln_1p, log2, log10, sin, cos, tan, asin, acos, atan, sinh,
    cosh, tanh, asinh, acosh, atanh, recip, abs, signum, floor, ceil, round, trunc, fract,
);

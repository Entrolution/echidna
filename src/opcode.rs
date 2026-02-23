//! Bytecode opcodes for the bytecode tape.
//!
//! Each opcode represents an elementary operation. The [`eval_forward`] and
//! [`reverse_partials`] functions evaluate / differentiate a single opcode.

use num_traits::Float;

/// Sentinel used in `arg_indices[1]` for unary ops (the second argument slot is unused).
pub const UNUSED: u32 = u32::MAX;

/// Elementary operation codes for the bytecode tape.
///
/// Fits in a `u8` (~38 variants). Binary ops use both `arg_indices` slots;
/// unary ops use slot 0 only (slot 1 = [`UNUSED`], except for [`OpCode::Powi`]
/// which stores the `i32` exponent reinterpreted as `u32` in slot 1).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OpCode {
    // ── Structural ──
    /// Input variable (leaf node).
    Input,
    /// Scalar constant.
    Const,

    // ── Binary arithmetic ──
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Powf,
    Atan2,
    Hypot,
    Max,
    Min,

    // ── Unary ──
    Neg,
    Recip,
    Sqrt,
    Cbrt,
    /// Integer power. Exponent stored in `arg_indices[1]` as `exp as u32`.
    Powi,

    // ── Exp / Log ──
    Exp,
    Exp2,
    ExpM1,
    Ln,
    Log2,
    Log10,
    Ln1p,

    // ── Trig ──
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,

    // ── Hyperbolic ──
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,

    // ── Misc ──
    Abs,
    /// Zero derivative but needed for re-evaluation.
    Signum,
    /// Zero derivative but needed for re-evaluation.
    Floor,
    /// Zero derivative but needed for re-evaluation.
    Ceil,
    /// Zero derivative but needed for re-evaluation.
    Round,
    /// Zero derivative but needed for re-evaluation.
    Trunc,
    Fract,

    // ── Custom ──
    /// User-registered custom operation. The callback index is stored in
    /// `arg_indices[1]` (for unary ops) or in a side table (for binary ops).
    Custom,
}

/// Returns true if this opcode is a nonsmooth operation (abs, min, max).
///
/// These operations have kinks where the derivative is undefined. Signum is
/// excluded — it has zero derivative on both sides, so there's no useful
/// subdifferential information.
#[inline]
pub fn is_nonsmooth(op: OpCode) -> bool {
    matches!(op, OpCode::Abs | OpCode::Min | OpCode::Max)
}

/// Compute reverse-mode partials with a forced branch choice for nonsmooth ops.
///
/// For nonsmooth ops (`Abs`, `Min`, `Max`), uses the given `sign` to select
/// which branch's derivative to return, regardless of the actual operand values.
/// For all other ops, delegates to [`reverse_partials`].
///
/// - `Abs`: `sign >= 0` → `(+1, 0)`, `sign < 0` → `(-1, 0)`
/// - `Max`: `sign >= 0` → `(1, 0)` (first arg wins), `sign < 0` → `(0, 1)`
/// - `Min`: `sign >= 0` → `(1, 0)` (first arg wins), `sign < 0` → `(0, 1)`
#[inline]
pub fn forced_reverse_partials<T: Float>(op: OpCode, a: T, b: T, r: T, sign: i8) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    match op {
        OpCode::Abs => {
            if sign >= 0 {
                (one, zero)
            } else {
                (-one, zero)
            }
        }
        OpCode::Max | OpCode::Min => {
            if sign >= 0 {
                (one, zero)
            } else {
                (zero, one)
            }
        }
        _ => reverse_partials(op, a, b, r),
    }
}

/// Evaluate a single opcode in the forward direction.
///
/// Generic over `T: Float` so Phase 3 can call it with `Dual<F>`.
///
/// For binary ops, `a` and `b` are the two operand values.
/// For unary ops, `a` is the operand value and `b` is ignored
/// (except [`OpCode::Powi`] where `b` bits encode the `i32` exponent).
#[inline]
pub fn eval_forward<T: Float>(op: OpCode, a: T, b: T) -> T {
    match op {
        OpCode::Input | OpCode::Const => {
            // values are already set during tape setup
            unreachable!("Input/Const should not be re-evaluated via eval_forward")
        }

        // Binary arithmetic
        OpCode::Add => a + b,
        OpCode::Sub => a - b,
        OpCode::Mul => a * b,
        OpCode::Div => a / b,
        OpCode::Rem => a % b,
        OpCode::Powf => a.powf(b),
        OpCode::Atan2 => a.atan2(b),
        OpCode::Hypot => a.hypot(b),
        OpCode::Max => {
            if a >= b {
                a
            } else {
                b
            }
        }
        OpCode::Min => {
            if a <= b {
                a
            } else {
                b
            }
        }

        // Unary
        OpCode::Neg => -a,
        OpCode::Recip => a.recip(),
        OpCode::Sqrt => a.sqrt(),
        OpCode::Cbrt => a.cbrt(),
        OpCode::Powi => {
            let exp = powi_exp_decode(b);
            a.powi(exp)
        }

        // Exp/Log
        OpCode::Exp => a.exp(),
        OpCode::Exp2 => a.exp2(),
        OpCode::ExpM1 => a.exp_m1(),
        OpCode::Ln => a.ln(),
        OpCode::Log2 => a.log2(),
        OpCode::Log10 => a.log10(),
        OpCode::Ln1p => a.ln_1p(),

        // Trig
        OpCode::Sin => a.sin(),
        OpCode::Cos => a.cos(),
        OpCode::Tan => a.tan(),
        OpCode::Asin => a.asin(),
        OpCode::Acos => a.acos(),
        OpCode::Atan => a.atan(),

        // Hyperbolic
        OpCode::Sinh => a.sinh(),
        OpCode::Cosh => a.cosh(),
        OpCode::Tanh => a.tanh(),
        OpCode::Asinh => a.asinh(),
        OpCode::Acosh => a.acosh(),
        OpCode::Atanh => a.atanh(),

        // Misc
        OpCode::Abs => a.abs(),
        OpCode::Signum => a.signum(),
        OpCode::Floor => a.floor(),
        OpCode::Ceil => a.ceil(),
        OpCode::Round => a.round(),
        OpCode::Trunc => a.trunc(),
        OpCode::Fract => a.fract(),

        OpCode::Custom => unreachable!("Custom ops are dispatched separately in the tape"),
    }
}

/// Compute reverse-mode partial derivatives for a single opcode.
///
/// Returns `(∂result/∂arg0, ∂result/∂arg1)`.
/// For unary ops the second partial is `T::zero()`.
///
/// `a`, `b` are the operand values (at recording or re-evaluation time).
/// `r` is the result value.
///
/// Generic over `T: Float` so Phase 3 can call it with `Dual<F>` for
/// forward-over-reverse.
#[inline]
pub fn reverse_partials<T: Float>(op: OpCode, a: T, b: T, r: T) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    match op {
        OpCode::Input | OpCode::Const => (zero, zero),

        // Binary
        OpCode::Add => (one, one),
        OpCode::Sub => (one, -one),
        OpCode::Mul => (b, a),
        OpCode::Div => {
            let inv = one / b;
            (inv, -a * inv * inv)
        }
        OpCode::Rem => (one, zero),
        OpCode::Powf => {
            // d/da a^b = b * a^(b-1)
            // d/db a^b = a^b * ln(a)
            let da = b * a.powf(b - one);
            let db = r * a.ln();
            (da, db)
        }
        OpCode::Atan2 => {
            // atan2(a, b): d/da = b/(a²+b²), d/db = -a/(a²+b²)
            let denom = a * a + b * b;
            (b / denom, -a / denom)
        }
        OpCode::Hypot => {
            // hypot(a,b) = sqrt(a²+b²), d/da = a/r, d/db = b/r
            (a / r, b / r)
        }
        OpCode::Max => {
            if a >= b {
                (one, zero)
            } else {
                (zero, one)
            }
        }
        OpCode::Min => {
            if a <= b {
                (one, zero)
            } else {
                (zero, one)
            }
        }

        // Unary
        OpCode::Neg => (-one, zero),
        OpCode::Recip => {
            // d/da (1/a) = -1/a²
            let inv = one / a;
            (-inv * inv, zero)
        }
        OpCode::Sqrt => {
            let two = one + one;
            (one / (two * r), zero)
        }
        OpCode::Cbrt => {
            let three = T::from(3.0).unwrap();
            (one / (three * r * r), zero)
        }
        OpCode::Powi => {
            let exp = powi_exp_decode(b);
            let n = T::from(exp).unwrap();
            (n * a.powi(exp - 1), zero)
        }

        // Exp/Log
        OpCode::Exp => (r, zero), // d/da e^a = e^a = r
        OpCode::Exp2 => (r * T::ln(T::from(2.0).unwrap()), zero),
        OpCode::ExpM1 => (r + one, zero), // d/da (e^a - 1) = e^a = r+1
        OpCode::Ln => (one / a, zero),
        OpCode::Log2 => (one / (a * T::ln(T::from(2.0).unwrap())), zero),
        OpCode::Log10 => (one / (a * T::ln(T::from(10.0).unwrap())), zero),
        OpCode::Ln1p => (one / (one + a), zero),

        // Trig
        OpCode::Sin => (a.cos(), zero),
        OpCode::Cos => (-a.sin(), zero),
        OpCode::Tan => {
            let c = a.cos();
            (one / (c * c), zero)
        }
        OpCode::Asin => (one / (one - a * a).sqrt(), zero),
        OpCode::Acos => (-one / (one - a * a).sqrt(), zero),
        OpCode::Atan => (one / (one + a * a), zero),

        // Hyperbolic
        OpCode::Sinh => (a.cosh(), zero),
        OpCode::Cosh => (a.sinh(), zero),
        OpCode::Tanh => {
            let c = a.cosh();
            (one / (c * c), zero)
        }
        OpCode::Asinh => (one / (a * a + one).sqrt(), zero),
        OpCode::Acosh => (one / (a * a - one).sqrt(), zero),
        OpCode::Atanh => (one / (one - a * a), zero),

        // Misc
        OpCode::Abs => (a.signum(), zero),
        OpCode::Signum | OpCode::Floor | OpCode::Ceil | OpCode::Round | OpCode::Trunc => {
            (zero, zero)
        }
        OpCode::Fract => (one, zero),

        OpCode::Custom => unreachable!("Custom ops are dispatched separately in the tape"),
    }
}

/// Decode a `powi` exponent from `arg_indices[1]` (stored as `u32` bits → `i32`).
///
/// During recording, the exponent `n: i32` is stored as `n as u32`.
/// We recover it here via the reverse cast.
#[inline]
fn powi_exp_decode<T: Float>(b: T) -> i32 {
    // b holds the bits of the u32 reinterpreted from i32.
    // We round-trip through u32 to recover it.
    b.to_u32().unwrap_or(0) as i32
}

/// Encode a `powi` exponent as a value that can be stored in `arg_indices[1]`.
#[inline]
pub fn powi_exp_encode(exp: i32) -> u32 {
    exp as u32
}

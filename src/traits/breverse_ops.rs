//! `std::ops` implementations for [`BReverse<F>`].
//!
//! Each operator records an opcode to the active bytecode tape.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::breverse::BReverse;
use crate::bytecode_tape::{self, BtapeThreadLocal, BytecodeTape, CONSTANT};
use crate::float::Float;
use crate::opcode::{OpCode, UNUSED};

/// Ensure a BReverse operand has a valid tape index. If it's a constant
/// (index == CONSTANT), promote it to a `Const` entry on the tape.
#[inline]
fn ensure_on_tape<F: Float>(x: &BReverse<F>, tape: &mut BytecodeTape<F>) -> u32 {
    if x.index == CONSTANT {
        tape.push_const(x.value)
    } else {
        x.index
    }
}

/// Record a binary op, promoting constants as needed.
#[inline]
fn brev_binary_op<F: Float + BtapeThreadLocal>(
    lhs: BReverse<F>,
    rhs: BReverse<F>,
    op: OpCode,
    value: F,
) -> BReverse<F> {
    let index = bytecode_tape::with_active_btape(|t| {
        let li = ensure_on_tape(&lhs, t);
        let ri = ensure_on_tape(&rhs, t);
        t.push_op(op, li, ri, value)
    });
    BReverse { value, index }
}

/// Record a unary op, promoting constant as needed.
#[inline]
fn brev_unary_op<F: Float + BtapeThreadLocal>(x: BReverse<F>, op: OpCode, value: F) -> BReverse<F> {
    let index = bytecode_tape::with_active_btape(|t| {
        let xi = ensure_on_tape(&x, t);
        t.push_op(op, xi, UNUSED, value)
    });
    BReverse { value, index }
}

// ──────────────────────────────────────────────
//  BReverse<F> ↔ BReverse<F> operators
// ──────────────────────────────────────────────

impl<F: Float + BtapeThreadLocal> Add for BReverse<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        brev_binary_op(self, rhs, OpCode::Add, self.value + rhs.value)
    }
}

impl<F: Float + BtapeThreadLocal> Sub for BReverse<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        brev_binary_op(self, rhs, OpCode::Sub, self.value - rhs.value)
    }
}

impl<F: Float + BtapeThreadLocal> Mul for BReverse<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        brev_binary_op(self, rhs, OpCode::Mul, self.value * rhs.value)
    }
}

impl<F: Float + BtapeThreadLocal> Div for BReverse<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        brev_binary_op(self, rhs, OpCode::Div, self.value / rhs.value)
    }
}

impl<F: Float + BtapeThreadLocal> Neg for BReverse<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        brev_unary_op(self, OpCode::Neg, -self.value)
    }
}

impl<F: Float + BtapeThreadLocal> Rem for BReverse<F> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        brev_binary_op(self, rhs, OpCode::Rem, self.value % rhs.value)
    }
}

// Assign variants delegate to the binary ops.
impl<F: Float + BtapeThreadLocal> AddAssign for BReverse<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float + BtapeThreadLocal> SubAssign for BReverse<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float + BtapeThreadLocal> MulAssign for BReverse<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float + BtapeThreadLocal> DivAssign for BReverse<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float + BtapeThreadLocal> RemAssign for BReverse<F> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// ──────────────────────────────────────────────
//  Mixed ops: BReverse<F> with primitive floats
// ──────────────────────────────────────────────

// For mixed ops, the scalar is promoted to a Const entry on the bytecode tape.
macro_rules! impl_breverse_scalar_ops {
    ($f:ty) => {
        impl Add<$f> for BReverse<$f> {
            type Output = BReverse<$f>;
            #[inline]
            fn add(self, rhs: $f) -> BReverse<$f> {
                let value = self.value + rhs;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(rhs);
                    t.push_op(OpCode::Add, self.index, c, value)
                });
                BReverse { value, index }
            }
        }

        impl Add<BReverse<$f>> for $f {
            type Output = BReverse<$f>;
            #[inline]
            fn add(self, rhs: BReverse<$f>) -> BReverse<$f> {
                let value = self + rhs.value;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(self);
                    t.push_op(OpCode::Add, c, rhs.index, value)
                });
                BReverse { value, index }
            }
        }

        impl Sub<$f> for BReverse<$f> {
            type Output = BReverse<$f>;
            #[inline]
            fn sub(self, rhs: $f) -> BReverse<$f> {
                let value = self.value - rhs;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(rhs);
                    t.push_op(OpCode::Sub, self.index, c, value)
                });
                BReverse { value, index }
            }
        }

        impl Sub<BReverse<$f>> for $f {
            type Output = BReverse<$f>;
            #[inline]
            fn sub(self, rhs: BReverse<$f>) -> BReverse<$f> {
                let value = self - rhs.value;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(self);
                    t.push_op(OpCode::Sub, c, rhs.index, value)
                });
                BReverse { value, index }
            }
        }

        impl Mul<$f> for BReverse<$f> {
            type Output = BReverse<$f>;
            #[inline]
            fn mul(self, rhs: $f) -> BReverse<$f> {
                let value = self.value * rhs;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(rhs);
                    t.push_op(OpCode::Mul, self.index, c, value)
                });
                BReverse { value, index }
            }
        }

        impl Mul<BReverse<$f>> for $f {
            type Output = BReverse<$f>;
            #[inline]
            fn mul(self, rhs: BReverse<$f>) -> BReverse<$f> {
                let value = self * rhs.value;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(self);
                    t.push_op(OpCode::Mul, c, rhs.index, value)
                });
                BReverse { value, index }
            }
        }

        impl Div<$f> for BReverse<$f> {
            type Output = BReverse<$f>;
            #[inline]
            fn div(self, rhs: $f) -> BReverse<$f> {
                let value = self.value / rhs;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(rhs);
                    t.push_op(OpCode::Div, self.index, c, value)
                });
                BReverse { value, index }
            }
        }

        impl Div<BReverse<$f>> for $f {
            type Output = BReverse<$f>;
            #[inline]
            fn div(self, rhs: BReverse<$f>) -> BReverse<$f> {
                let value = self / rhs.value;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(self);
                    t.push_op(OpCode::Div, c, rhs.index, value)
                });
                BReverse { value, index }
            }
        }

        impl Rem<$f> for BReverse<$f> {
            type Output = BReverse<$f>;
            #[inline]
            fn rem(self, rhs: $f) -> BReverse<$f> {
                let value = self.value % rhs;
                let index = bytecode_tape::with_active_btape(|t| {
                    let c = t.push_const(rhs);
                    t.push_op(OpCode::Rem, self.index, c, value)
                });
                BReverse { value, index }
            }
        }

        impl Rem<BReverse<$f>> for $f {
            type Output = BReverse<$f>;
            #[inline]
            fn rem(self, rhs: BReverse<$f>) -> BReverse<$f> {
                let value = self % rhs.value;
                // Constant % variable has zero derivative — treat as constant.
                BReverse {
                    value,
                    index: CONSTANT,
                }
            }
        }
    };
}

impl_breverse_scalar_ops!(f32);
impl_breverse_scalar_ops!(f64);

// ── Comparison ──

impl<F: Float> PartialEq for BReverse<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: Float> PartialOrd for BReverse<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

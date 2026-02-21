use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::dual::Dual;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::{self, TapeThreadLocal, CONSTANT};

// ──────────────────────────────────────────────
//  Dual<F> operators
// ──────────────────────────────────────────────

impl<F: Float> Add for Dual<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<F: Float> Sub for Dual<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<F: Float> Mul for Dual<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Dual {
            re: self.re * rhs.re,
            eps: self.re * rhs.eps + self.eps * rhs.re,
        }
    }
}

impl<F: Float> Div for Dual<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = F::one() / rhs.re;
        Dual {
            re: self.re * inv,
            eps: (self.eps * rhs.re - self.re * rhs.eps) * inv * inv,
        }
    }
}

impl<F: Float> Neg for Dual<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl<F: Float> Rem for Dual<F> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Dual {
            re: self.re % rhs.re,
            eps: self.eps,
        }
    }
}

impl<F: Float> AddAssign for Dual<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float> SubAssign for Dual<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float> MulAssign for Dual<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float> DivAssign for Dual<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float> RemAssign for Dual<F> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: Dual<F> with primitive floats.
// We generate these for f32 and f64 via a macro.
macro_rules! impl_dual_scalar_ops {
    ($f:ty) => {
        impl Add<$f> for Dual<$f> {
            type Output = Dual<$f>;
            #[inline]
            fn add(self, rhs: $f) -> Dual<$f> {
                Dual {
                    re: self.re + rhs,
                    eps: self.eps,
                }
            }
        }

        impl Add<Dual<$f>> for $f {
            type Output = Dual<$f>;
            #[inline]
            fn add(self, rhs: Dual<$f>) -> Dual<$f> {
                Dual {
                    re: self + rhs.re,
                    eps: rhs.eps,
                }
            }
        }

        impl Sub<$f> for Dual<$f> {
            type Output = Dual<$f>;
            #[inline]
            fn sub(self, rhs: $f) -> Dual<$f> {
                Dual {
                    re: self.re - rhs,
                    eps: self.eps,
                }
            }
        }

        impl Sub<Dual<$f>> for $f {
            type Output = Dual<$f>;
            #[inline]
            fn sub(self, rhs: Dual<$f>) -> Dual<$f> {
                Dual {
                    re: self - rhs.re,
                    eps: -rhs.eps,
                }
            }
        }

        impl Mul<$f> for Dual<$f> {
            type Output = Dual<$f>;
            #[inline]
            fn mul(self, rhs: $f) -> Dual<$f> {
                Dual {
                    re: self.re * rhs,
                    eps: self.eps * rhs,
                }
            }
        }

        impl Mul<Dual<$f>> for $f {
            type Output = Dual<$f>;
            #[inline]
            fn mul(self, rhs: Dual<$f>) -> Dual<$f> {
                Dual {
                    re: self * rhs.re,
                    eps: self * rhs.eps,
                }
            }
        }

        impl Div<$f> for Dual<$f> {
            type Output = Dual<$f>;
            #[inline]
            fn div(self, rhs: $f) -> Dual<$f> {
                let inv = 1.0 / rhs;
                Dual {
                    re: self.re * inv,
                    eps: self.eps * inv,
                }
            }
        }

        impl Div<Dual<$f>> for $f {
            type Output = Dual<$f>;
            #[inline]
            fn div(self, rhs: Dual<$f>) -> Dual<$f> {
                let inv = 1.0 / rhs.re;
                Dual {
                    re: self * inv,
                    eps: -self * rhs.eps * inv * inv,
                }
            }
        }

        impl Rem<$f> for Dual<$f> {
            type Output = Dual<$f>;
            #[inline]
            fn rem(self, rhs: $f) -> Dual<$f> {
                Dual {
                    re: self.re % rhs,
                    eps: self.eps,
                }
            }
        }

        impl Rem<Dual<$f>> for $f {
            type Output = Dual<$f>;
            #[inline]
            fn rem(self, rhs: Dual<$f>) -> Dual<$f> {
                Dual {
                    re: self % rhs.re,
                    eps: <$f>::from(0.0),
                }
            }
        }
    };
}

impl_dual_scalar_ops!(f32);
impl_dual_scalar_ops!(f64);

impl<F: Float> PartialEq for Dual<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<F: Float> PartialOrd for Dual<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

// ──────────────────────────────────────────────
//  Reverse<F> operators
// ──────────────────────────────────────────────

impl<F: Float + TapeThreadLocal> Add for Reverse<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = self.value + rhs.value;
        let index =
            tape::with_active_tape(|t| t.push_binary(self.index, F::one(), rhs.index, F::one()));
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> Sub for Reverse<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = self.value - rhs.value;
        let index =
            tape::with_active_tape(|t| t.push_binary(self.index, F::one(), rhs.index, -F::one()));
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> Mul for Reverse<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let value = self.value * rhs.value;
        let index =
            tape::with_active_tape(|t| t.push_binary(self.index, rhs.value, rhs.index, self.value));
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> Div for Reverse<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = F::one() / rhs.value;
        let value = self.value * inv;
        let index = tape::with_active_tape(|t| {
            t.push_binary(self.index, inv, rhs.index, -self.value * inv * inv)
        });
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> Neg for Reverse<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let value = -self.value;
        let index = tape::with_active_tape(|t| t.push_unary(self.index, -F::one()));
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> Rem for Reverse<F> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let value = self.value % rhs.value;
        let index =
            tape::with_active_tape(|t| t.push_binary(self.index, F::one(), rhs.index, F::zero()));
        Reverse { value, index }
    }
}

impl<F: Float + TapeThreadLocal> AddAssign for Reverse<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float + TapeThreadLocal> SubAssign for Reverse<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float + TapeThreadLocal> MulAssign for Reverse<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float + TapeThreadLocal> DivAssign for Reverse<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float + TapeThreadLocal> RemAssign for Reverse<F> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: Reverse<F> with primitive floats.
macro_rules! impl_reverse_scalar_ops {
    ($f:ty) => {
        impl Add<$f> for Reverse<$f> {
            type Output = Reverse<$f>;
            #[inline]
            fn add(self, rhs: $f) -> Reverse<$f> {
                let value = self.value + rhs;
                let index = tape::with_active_tape(|t| t.push_unary(self.index, 1.0));
                Reverse { value, index }
            }
        }

        impl Add<Reverse<$f>> for $f {
            type Output = Reverse<$f>;
            #[inline]
            fn add(self, rhs: Reverse<$f>) -> Reverse<$f> {
                let value = self + rhs.value;
                let index = tape::with_active_tape(|t| t.push_unary(rhs.index, 1.0));
                Reverse { value, index }
            }
        }

        impl Sub<$f> for Reverse<$f> {
            type Output = Reverse<$f>;
            #[inline]
            fn sub(self, rhs: $f) -> Reverse<$f> {
                let value = self.value - rhs;
                let index = tape::with_active_tape(|t| t.push_unary(self.index, 1.0));
                Reverse { value, index }
            }
        }

        impl Sub<Reverse<$f>> for $f {
            type Output = Reverse<$f>;
            #[inline]
            fn sub(self, rhs: Reverse<$f>) -> Reverse<$f> {
                let value = self - rhs.value;
                let index = tape::with_active_tape(|t| t.push_unary(rhs.index, -1.0));
                Reverse { value, index }
            }
        }

        impl Mul<$f> for Reverse<$f> {
            type Output = Reverse<$f>;
            #[inline]
            fn mul(self, rhs: $f) -> Reverse<$f> {
                let value = self.value * rhs;
                let index = tape::with_active_tape(|t| t.push_unary(self.index, rhs));
                Reverse { value, index }
            }
        }

        impl Mul<Reverse<$f>> for $f {
            type Output = Reverse<$f>;
            #[inline]
            fn mul(self, rhs: Reverse<$f>) -> Reverse<$f> {
                let value = self * rhs.value;
                let index = tape::with_active_tape(|t| t.push_unary(rhs.index, self));
                Reverse { value, index }
            }
        }

        impl Div<$f> for Reverse<$f> {
            type Output = Reverse<$f>;
            #[inline]
            fn div(self, rhs: $f) -> Reverse<$f> {
                let inv: $f = 1.0 / rhs;
                let value = self.value * inv;
                let index = tape::with_active_tape(|t| t.push_unary(self.index, inv));
                Reverse { value, index }
            }
        }

        impl Div<Reverse<$f>> for $f {
            type Output = Reverse<$f>;
            #[inline]
            fn div(self, rhs: Reverse<$f>) -> Reverse<$f> {
                let inv: $f = 1.0 / rhs.value;
                let value = self * inv;
                let index = tape::with_active_tape(|t| t.push_unary(rhs.index, -self * inv * inv));
                Reverse { value, index }
            }
        }

        impl Rem<$f> for Reverse<$f> {
            type Output = Reverse<$f>;
            #[inline]
            fn rem(self, rhs: $f) -> Reverse<$f> {
                let value = self.value % rhs;
                let index = tape::with_active_tape(|t| t.push_unary(self.index, 1.0));
                Reverse { value, index }
            }
        }

        impl Rem<Reverse<$f>> for $f {
            type Output = Reverse<$f>;
            #[inline]
            fn rem(self, rhs: Reverse<$f>) -> Reverse<$f> {
                let value = self % rhs.value;
                // Constant remainder w.r.t. denominator has zero derivative.
                Reverse {
                    value,
                    index: CONSTANT,
                }
            }
        }
    };
}

impl_reverse_scalar_ops!(f32);
impl_reverse_scalar_ops!(f64);

impl<F: Float> PartialEq for Reverse<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: Float> PartialOrd for Reverse<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

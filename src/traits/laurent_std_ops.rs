//! `std::ops` implementations for `Laurent<F, K>`.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::float::Float;
use crate::laurent::Laurent;
use crate::taylor_ops;

// ══════════════════════════════════════════════
//  Laurent<F, K> ↔ Laurent<F, K>
// ══════════════════════════════════════════════

impl<F: Float, const K: usize> Add for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let p1 = self.pole_order();
        let p2 = rhs.pole_order();
        let p_out = p1.min(p2);
        // Align both to p_out by shifting
        let shift1 = (p1 - p_out) as usize;
        let shift2 = (p2 - p_out) as usize;

        let a: [F; K] = std::array::from_fn(|i| {
            if i >= shift1 && i - shift1 < K {
                self.coeff(p_out + i as i32)
            } else {
                F::zero()
            }
        });
        let b: [F; K] = std::array::from_fn(|i| {
            if i >= shift2 && i - shift2 < K {
                rhs.coeff(p_out + i as i32)
            } else {
                F::zero()
            }
        });

        let mut c = [F::zero(); K];
        taylor_ops::taylor_add(&a, &b, &mut c);
        Laurent::new(c, p_out)
    }
}

impl<F: Float, const K: usize> Sub for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let p1 = self.pole_order();
        let p2 = rhs.pole_order();
        let p_out = p1.min(p2);
        let shift1 = (p1 - p_out) as usize;
        let shift2 = (p2 - p_out) as usize;

        let a: [F; K] = std::array::from_fn(|i| {
            if i >= shift1 && i - shift1 < K {
                self.coeff(p_out + i as i32)
            } else {
                F::zero()
            }
        });
        let b: [F; K] = std::array::from_fn(|i| {
            if i >= shift2 && i - shift2 < K {
                rhs.coeff(p_out + i as i32)
            } else {
                F::zero()
            }
        });

        let mut c = [F::zero(); K];
        taylor_ops::taylor_sub(&a, &b, &mut c);
        Laurent::new(c, p_out)
    }
}

// Laurent Mul delegates to taylor_ops::taylor_mul (Cauchy product) which involves addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float, const K: usize> Mul for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_mul(&self.leading_coeffs(), &rhs.leading_coeffs(), &mut c);
        Laurent::new(c, self.pole_order() + rhs.pole_order())
    }
}

// Laurent Div delegates to taylor_ops::taylor_div which involves multiplication internally
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float, const K: usize> Div for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.is_all_zero_pub() {
            return Laurent::nan_pub();
        }
        let mut c = [F::zero(); K];
        taylor_ops::taylor_div(&self.leading_coeffs(), &rhs.leading_coeffs(), &mut c);
        Laurent::new(c, self.pole_order() - rhs.pole_order())
    }
}

impl<F: Float, const K: usize> Neg for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_neg(&self.leading_coeffs(), &mut c);
        Laurent::new(c, self.pole_order())
    }
}

impl<F: Float, const K: usize> Rem for Laurent<F, K> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // a % b = a - (a / b).trunc() * b
        let quotient = (self / rhs).trunc();
        self - quotient * rhs
    }
}

impl<F: Float, const K: usize> AddAssign for Laurent<F, K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float, const K: usize> SubAssign for Laurent<F, K> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float, const K: usize> MulAssign for Laurent<F, K> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float, const K: usize> DivAssign for Laurent<F, K> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float, const K: usize> RemAssign for Laurent<F, K> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: Laurent<F, K> with primitive floats.
macro_rules! impl_laurent_scalar_ops {
    ($f:ty) => {
        impl<const K: usize> Add<$f> for Laurent<$f, K> {
            type Output = Laurent<$f, K>;
            #[inline]
            fn add(self, rhs: $f) -> Laurent<$f, K> {
                self + Laurent::constant(rhs)
            }
        }

        impl<const K: usize> Add<Laurent<$f, K>> for $f {
            type Output = Laurent<$f, K>;
            #[inline]
            fn add(self, rhs: Laurent<$f, K>) -> Laurent<$f, K> {
                Laurent::constant(self) + rhs
            }
        }

        impl<const K: usize> Sub<$f> for Laurent<$f, K> {
            type Output = Laurent<$f, K>;
            #[inline]
            fn sub(self, rhs: $f) -> Laurent<$f, K> {
                self - Laurent::constant(rhs)
            }
        }

        impl<const K: usize> Sub<Laurent<$f, K>> for $f {
            type Output = Laurent<$f, K>;
            #[inline]
            fn sub(self, rhs: Laurent<$f, K>) -> Laurent<$f, K> {
                Laurent::constant(self) - rhs
            }
        }

        impl<const K: usize> Mul<$f> for Laurent<$f, K> {
            type Output = Laurent<$f, K>;
            #[inline]
            fn mul(self, rhs: $f) -> Laurent<$f, K> {
                self * Laurent::constant(rhs)
            }
        }

        impl<const K: usize> Mul<Laurent<$f, K>> for $f {
            type Output = Laurent<$f, K>;
            #[inline]
            fn mul(self, rhs: Laurent<$f, K>) -> Laurent<$f, K> {
                Laurent::constant(self) * rhs
            }
        }

        // Scalar Div delegates to Laurent Div (self / Laurent::constant(rhs)) to reuse pole arithmetic
        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<const K: usize> Div<$f> for Laurent<$f, K> {
            type Output = Laurent<$f, K>;
            #[inline]
            fn div(self, rhs: $f) -> Laurent<$f, K> {
                self / Laurent::constant(rhs)
            }
        }

        impl<const K: usize> Div<Laurent<$f, K>> for $f {
            type Output = Laurent<$f, K>;
            #[inline]
            fn div(self, rhs: Laurent<$f, K>) -> Laurent<$f, K> {
                Laurent::constant(self) / rhs
            }
        }

        impl<const K: usize> Rem<$f> for Laurent<$f, K> {
            type Output = Laurent<$f, K>;
            #[inline]
            fn rem(self, rhs: $f) -> Laurent<$f, K> {
                self % Laurent::constant(rhs)
            }
        }

        impl<const K: usize> Rem<Laurent<$f, K>> for $f {
            type Output = Laurent<$f, K>;
            #[inline]
            fn rem(self, rhs: Laurent<$f, K>) -> Laurent<$f, K> {
                Laurent::constant(self) % rhs
            }
        }
    };
}

impl_laurent_scalar_ops!(f32);
impl_laurent_scalar_ops!(f64);

impl<F: Float, const K: usize> PartialEq for Laurent<F, K> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl<F: Float, const K: usize> PartialOrd for Laurent<F, K> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

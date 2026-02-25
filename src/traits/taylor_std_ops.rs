//! `std::ops` implementations for `Taylor<F, K>` and `TaylorDyn<F>`.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::float::Float;
use crate::taylor::Taylor;
use crate::taylor_ops;

// ══════════════════════════════════════════════
//  Taylor<F, K> ↔ Taylor<F, K>
// ══════════════════════════════════════════════

impl<F: Float, const K: usize> Add for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_add(&self.coeffs, &rhs.coeffs, &mut c);
        Taylor { coeffs: c }
    }
}

impl<F: Float, const K: usize> Sub for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_sub(&self.coeffs, &rhs.coeffs, &mut c);
        Taylor { coeffs: c }
    }
}

// Taylor Mul delegates to taylor_ops::taylor_mul (Cauchy product) which involves addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float, const K: usize> Mul for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_mul(&self.coeffs, &rhs.coeffs, &mut c);
        Taylor { coeffs: c }
    }
}

// Taylor Div delegates to taylor_ops::taylor_div which involves multiplication internally
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float, const K: usize> Div for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_div(&self.coeffs, &rhs.coeffs, &mut c);
        Taylor { coeffs: c }
    }
}

impl<F: Float, const K: usize> Neg for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut c = [F::zero(); K];
        taylor_ops::taylor_neg(&self.coeffs, &mut c);
        Taylor { coeffs: c }
    }
}

impl<F: Float, const K: usize> Rem for Taylor<F, K> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Taylor {
            coeffs: std::array::from_fn(|k| {
                if k == 0 {
                    self.coeffs[0] % rhs.coeffs[0]
                } else {
                    self.coeffs[k]
                }
            }),
        }
    }
}

impl<F: Float, const K: usize> AddAssign for Taylor<F, K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float, const K: usize> SubAssign for Taylor<F, K> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float, const K: usize> MulAssign for Taylor<F, K> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float, const K: usize> DivAssign for Taylor<F, K> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float, const K: usize> RemAssign for Taylor<F, K> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: Taylor<F, K> with primitive floats.
macro_rules! impl_taylor_scalar_ops {
    ($f:ty) => {
        impl<const K: usize> Add<$f> for Taylor<$f, K> {
            type Output = Taylor<$f, K>;
            #[inline]
            fn add(self, rhs: $f) -> Taylor<$f, K> {
                let mut coeffs = self.coeffs;
                coeffs[0] += rhs;
                Taylor { coeffs }
            }
        }

        impl<const K: usize> Add<Taylor<$f, K>> for $f {
            type Output = Taylor<$f, K>;
            #[inline]
            fn add(self, rhs: Taylor<$f, K>) -> Taylor<$f, K> {
                let mut coeffs = rhs.coeffs;
                coeffs[0] += self;
                Taylor { coeffs }
            }
        }

        impl<const K: usize> Sub<$f> for Taylor<$f, K> {
            type Output = Taylor<$f, K>;
            #[inline]
            fn sub(self, rhs: $f) -> Taylor<$f, K> {
                let mut coeffs = self.coeffs;
                coeffs[0] -= rhs;
                Taylor { coeffs }
            }
        }

        impl<const K: usize> Sub<Taylor<$f, K>> for $f {
            type Output = Taylor<$f, K>;
            #[inline]
            fn sub(self, rhs: Taylor<$f, K>) -> Taylor<$f, K> {
                Taylor {
                    coeffs: std::array::from_fn(|k| {
                        if k == 0 {
                            self - rhs.coeffs[0]
                        } else {
                            -rhs.coeffs[k]
                        }
                    }),
                }
            }
        }

        impl<const K: usize> Mul<$f> for Taylor<$f, K> {
            type Output = Taylor<$f, K>;
            #[inline]
            fn mul(self, rhs: $f) -> Taylor<$f, K> {
                Taylor {
                    coeffs: std::array::from_fn(|k| self.coeffs[k] * rhs),
                }
            }
        }

        impl<const K: usize> Mul<Taylor<$f, K>> for $f {
            type Output = Taylor<$f, K>;
            #[inline]
            fn mul(self, rhs: Taylor<$f, K>) -> Taylor<$f, K> {
                Taylor {
                    coeffs: std::array::from_fn(|k| self * rhs.coeffs[k]),
                }
            }
        }

        // Scalar Div uses Mul internally (multiply by reciprocal for efficiency)
        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<const K: usize> Div<$f> for Taylor<$f, K> {
            type Output = Taylor<$f, K>;
            #[inline]
            fn div(self, rhs: $f) -> Taylor<$f, K> {
                let inv: $f = 1.0 / rhs;
                Taylor {
                    coeffs: std::array::from_fn(|k| self.coeffs[k] * inv),
                }
            }
        }

        impl<const K: usize> Div<Taylor<$f, K>> for $f {
            type Output = Taylor<$f, K>;
            #[inline]
            fn div(self, rhs: Taylor<$f, K>) -> Taylor<$f, K> {
                Taylor::constant(self) / rhs
            }
        }

        impl<const K: usize> Rem<$f> for Taylor<$f, K> {
            type Output = Taylor<$f, K>;
            #[inline]
            fn rem(self, rhs: $f) -> Taylor<$f, K> {
                let mut coeffs = self.coeffs;
                coeffs[0] %= rhs;
                Taylor { coeffs }
            }
        }

        impl<const K: usize> Rem<Taylor<$f, K>> for $f {
            type Output = Taylor<$f, K>;
            #[inline]
            fn rem(self, rhs: Taylor<$f, K>) -> Taylor<$f, K> {
                let mut coeffs = [<$f>::from(0.0); K];
                coeffs[0] = self % rhs.coeffs[0];
                Taylor { coeffs }
            }
        }
    };
}

impl_taylor_scalar_ops!(f32);
impl_taylor_scalar_ops!(f64);

impl<F: Float, const K: usize> PartialEq for Taylor<F, K> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.coeffs[0] == other.coeffs[0]
    }
}

impl<F: Float, const K: usize> PartialOrd for Taylor<F, K> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.coeffs[0].partial_cmp(&other.coeffs[0])
    }
}

// ══════════════════════════════════════════════
//  TaylorDyn<F> operators
// ══════════════════════════════════════════════

use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn};

impl<F: Float + TaylorArenaLocal> Add for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        TaylorDyn::binary_op(&self, &rhs, |a, b, c| taylor_ops::taylor_add(a, b, c))
    }
}

impl<F: Float + TaylorArenaLocal> Sub for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        TaylorDyn::binary_op(&self, &rhs, |a, b, c| taylor_ops::taylor_sub(a, b, c))
    }
}

// TaylorDyn Mul delegates to taylor_ops::taylor_mul (Cauchy product) which involves addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float + TaylorArenaLocal> Mul for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        TaylorDyn::binary_op(&self, &rhs, |a, b, c| taylor_ops::taylor_mul(a, b, c))
    }
}

// TaylorDyn Div delegates to taylor_ops::taylor_div which involves multiplication internally
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float + TaylorArenaLocal> Div for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        TaylorDyn::binary_op(&self, &rhs, |a, b, c| taylor_ops::taylor_div(a, b, c))
    }
}

impl<F: Float + TaylorArenaLocal> Neg for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        TaylorDyn::unary_op(&self, |a, c| taylor_ops::taylor_neg(a, c))
    }
}

impl<F: Float + TaylorArenaLocal> Rem for TaylorDyn<F> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        TaylorDyn::binary_op(&self, &rhs, |a, b, c| {
            c[0] = a[0] % b[0];
            c[1..].copy_from_slice(&a[1..]);
        })
    }
}

impl<F: Float + TaylorArenaLocal> AddAssign for TaylorDyn<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float + TaylorArenaLocal> SubAssign for TaylorDyn<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float + TaylorArenaLocal> MulAssign for TaylorDyn<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float + TaylorArenaLocal> DivAssign for TaylorDyn<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float + TaylorArenaLocal> RemAssign for TaylorDyn<F> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: TaylorDyn<F> with primitive floats.
macro_rules! impl_taylor_dyn_scalar_ops {
    ($f:ty) => {
        impl Add<$f> for TaylorDyn<$f> {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn add(self, rhs: $f) -> TaylorDyn<$f> {
                self + TaylorDyn::constant(rhs)
            }
        }

        impl Add<TaylorDyn<$f>> for $f {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn add(self, rhs: TaylorDyn<$f>) -> TaylorDyn<$f> {
                TaylorDyn::constant(self) + rhs
            }
        }

        impl Sub<$f> for TaylorDyn<$f> {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn sub(self, rhs: $f) -> TaylorDyn<$f> {
                self - TaylorDyn::constant(rhs)
            }
        }

        impl Sub<TaylorDyn<$f>> for $f {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn sub(self, rhs: TaylorDyn<$f>) -> TaylorDyn<$f> {
                TaylorDyn::constant(self) - rhs
            }
        }

        impl Mul<$f> for TaylorDyn<$f> {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn mul(self, rhs: $f) -> TaylorDyn<$f> {
                self * TaylorDyn::constant(rhs)
            }
        }

        impl Mul<TaylorDyn<$f>> for $f {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn mul(self, rhs: TaylorDyn<$f>) -> TaylorDyn<$f> {
                TaylorDyn::constant(self) * rhs
            }
        }

        impl Div<$f> for TaylorDyn<$f> {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn div(self, rhs: $f) -> TaylorDyn<$f> {
                self / TaylorDyn::constant(rhs)
            }
        }

        impl Div<TaylorDyn<$f>> for $f {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn div(self, rhs: TaylorDyn<$f>) -> TaylorDyn<$f> {
                TaylorDyn::constant(self) / rhs
            }
        }

        impl Rem<$f> for TaylorDyn<$f> {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn rem(self, rhs: $f) -> TaylorDyn<$f> {
                self % TaylorDyn::constant(rhs)
            }
        }

        impl Rem<TaylorDyn<$f>> for $f {
            type Output = TaylorDyn<$f>;
            #[inline]
            fn rem(self, rhs: TaylorDyn<$f>) -> TaylorDyn<$f> {
                TaylorDyn::constant(self) % rhs
            }
        }
    };
}

impl_taylor_dyn_scalar_ops!(f32);
impl_taylor_dyn_scalar_ops!(f64);

impl<F: Float> PartialEq for TaylorDyn<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: Float> PartialOrd for TaylorDyn<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

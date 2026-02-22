use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::dual_vec::DualVec;
use crate::float::Float;

impl<F: Float, const N: usize> Add for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        DualVec {
            re: self.re + rhs.re,
            eps: std::array::from_fn(|k| self.eps[k] + rhs.eps[k]),
        }
    }
}

impl<F: Float, const N: usize> Sub for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        DualVec {
            re: self.re - rhs.re,
            eps: std::array::from_fn(|k| self.eps[k] - rhs.eps[k]),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Float, const N: usize> Mul for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        DualVec {
            re: self.re * rhs.re,
            eps: std::array::from_fn(|k| self.re * rhs.eps[k] + self.eps[k] * rhs.re),
        }
    }
}

impl<F: Float, const N: usize> Div for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = F::one() / rhs.re;
        DualVec {
            re: self.re * inv,
            eps: std::array::from_fn(|k| (self.eps[k] * rhs.re - self.re * rhs.eps[k]) * inv * inv),
        }
    }
}

impl<F: Float, const N: usize> Neg for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        DualVec {
            re: -self.re,
            eps: std::array::from_fn(|k| -self.eps[k]),
        }
    }
}

impl<F: Float, const N: usize> Rem for DualVec<F, N> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        DualVec {
            re: self.re % rhs.re,
            eps: self.eps,
        }
    }
}

impl<F: Float, const N: usize> AddAssign for DualVec<F, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Float, const N: usize> SubAssign for DualVec<F, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Float, const N: usize> MulAssign for DualVec<F, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Float, const N: usize> DivAssign for DualVec<F, N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Float, const N: usize> RemAssign for DualVec<F, N> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Mixed ops: DualVec<F, N> with primitive floats.
macro_rules! impl_dual_vec_scalar_ops {
    ($f:ty) => {
        impl<const N: usize> Add<$f> for DualVec<$f, N> {
            type Output = DualVec<$f, N>;
            #[inline]
            fn add(self, rhs: $f) -> DualVec<$f, N> {
                DualVec {
                    re: self.re + rhs,
                    eps: self.eps,
                }
            }
        }

        impl<const N: usize> Add<DualVec<$f, N>> for $f {
            type Output = DualVec<$f, N>;
            #[inline]
            fn add(self, rhs: DualVec<$f, N>) -> DualVec<$f, N> {
                DualVec {
                    re: self + rhs.re,
                    eps: rhs.eps,
                }
            }
        }

        impl<const N: usize> Sub<$f> for DualVec<$f, N> {
            type Output = DualVec<$f, N>;
            #[inline]
            fn sub(self, rhs: $f) -> DualVec<$f, N> {
                DualVec {
                    re: self.re - rhs,
                    eps: self.eps,
                }
            }
        }

        impl<const N: usize> Sub<DualVec<$f, N>> for $f {
            type Output = DualVec<$f, N>;
            #[inline]
            fn sub(self, rhs: DualVec<$f, N>) -> DualVec<$f, N> {
                DualVec {
                    re: self - rhs.re,
                    eps: std::array::from_fn(|k| -rhs.eps[k]),
                }
            }
        }

        impl<const N: usize> Mul<$f> for DualVec<$f, N> {
            type Output = DualVec<$f, N>;
            #[inline]
            fn mul(self, rhs: $f) -> DualVec<$f, N> {
                DualVec {
                    re: self.re * rhs,
                    eps: std::array::from_fn(|k| self.eps[k] * rhs),
                }
            }
        }

        impl<const N: usize> Mul<DualVec<$f, N>> for $f {
            type Output = DualVec<$f, N>;
            #[inline]
            fn mul(self, rhs: DualVec<$f, N>) -> DualVec<$f, N> {
                DualVec {
                    re: self * rhs.re,
                    eps: std::array::from_fn(|k| self * rhs.eps[k]),
                }
            }
        }

        impl<const N: usize> Div<$f> for DualVec<$f, N> {
            type Output = DualVec<$f, N>;
            #[inline]
            fn div(self, rhs: $f) -> DualVec<$f, N> {
                let inv = 1.0 / rhs;
                DualVec {
                    re: self.re * inv,
                    eps: std::array::from_fn(|k| self.eps[k] * inv),
                }
            }
        }

        impl<const N: usize> Div<DualVec<$f, N>> for $f {
            type Output = DualVec<$f, N>;
            #[inline]
            fn div(self, rhs: DualVec<$f, N>) -> DualVec<$f, N> {
                let inv = 1.0 / rhs.re;
                DualVec {
                    re: self * inv,
                    eps: std::array::from_fn(|k| -self * rhs.eps[k] * inv * inv),
                }
            }
        }

        impl<const N: usize> Rem<$f> for DualVec<$f, N> {
            type Output = DualVec<$f, N>;
            #[inline]
            fn rem(self, rhs: $f) -> DualVec<$f, N> {
                DualVec {
                    re: self.re % rhs,
                    eps: self.eps,
                }
            }
        }

        impl<const N: usize> Rem<DualVec<$f, N>> for $f {
            type Output = DualVec<$f, N>;
            #[inline]
            fn rem(self, rhs: DualVec<$f, N>) -> DualVec<$f, N> {
                DualVec {
                    re: self % rhs.re,
                    eps: [<$f>::from(0.0); N],
                }
            }
        }
    };
}

impl_dual_vec_scalar_ops!(f32);
impl_dual_vec_scalar_ops!(f64);

impl<F: Float, const N: usize> PartialEq for DualVec<F, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<F: Float, const N: usize> PartialOrd for DualVec<F, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

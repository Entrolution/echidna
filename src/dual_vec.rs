//! Batched forward-mode dual numbers with `N` tangent lanes.
//!
//! [`DualVec<F, N>`] carries `N` independent tangent directions simultaneously,
//! enabling vectorized Jacobian columns or batched Hessian computation via
//! forward-over-reverse mode.

use std::fmt::{self, Display};

use crate::Float;

/// Batched forward-mode dual number: a value with N tangent lanes.
///
/// `DualVec { re, eps }` represents a value with N independent tangent directions,
/// enabling batched Hessian computation.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DualVec<F: Float, const N: usize> {
    /// Primal (real) value.
    pub re: F,
    /// Tangent (derivative) values — one per lane.
    pub eps: [F; N],
}

impl<F: Float, const N: usize> Default for DualVec<F, N> {
    fn default() -> Self {
        DualVec {
            re: F::zero(),
            eps: [F::zero(); N],
        }
    }
}

impl<F: Float, const N: usize> Display for DualVec<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.re)?;
        for (i, e) in self.eps.iter().enumerate() {
            write!(f, " + {}\u{03b5}{}", e, i)?;
        }
        Ok(())
    }
}

impl<F: Float, const N: usize> From<F> for DualVec<F, N> {
    #[inline]
    fn from(val: F) -> Self {
        DualVec::constant(val)
    }
}

impl<F: Float, const N: usize> DualVec<F, N> {
    /// Create a new batched dual number.
    #[inline]
    pub fn new(re: F, eps: [F; N]) -> Self {
        DualVec { re, eps }
    }

    /// Create a constant (zero derivatives in all lanes).
    #[inline]
    pub fn constant(re: F) -> Self {
        DualVec {
            re,
            eps: [F::zero(); N],
        }
    }

    /// Create a variable with unit derivative in the specified lane.
    #[inline]
    pub fn with_tangent(re: F, lane: usize) -> Self {
        DualVec {
            re,
            eps: std::array::from_fn(|k| if k == lane { F::one() } else { F::zero() }),
        }
    }

    /// Apply the chain rule: given `f(self.re)` and `f'(self.re)`, produce the dual result.
    #[inline(always)]
    fn chain(self, f_val: F, f_deriv: F) -> Self {
        DualVec {
            re: f_val,
            eps: std::array::from_fn(|k| self.eps[k] * f_deriv),
        }
    }

    // -- Powers --

    #[inline]
    pub fn recip(self) -> Self {
        let inv = F::one() / self.re;
        self.chain(inv, -inv * inv)
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        let two = F::one() + F::one();
        self.chain(s, F::one() / (two * s))
    }

    #[inline]
    pub fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        let three = F::from(3.0).unwrap();
        self.chain(c, F::one() / (three * c * c))
    }

    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let val = self.re.powi(n);
        let deriv = F::from(n).unwrap() * self.re.powi(n - 1);
        self.chain(val, deriv)
    }

    #[inline]
    pub fn powf(self, n: Self) -> Self {
        let val = self.re.powf(n.re);
        DualVec {
            re: val,
            eps: std::array::from_fn(|k| {
                val * (n.re * self.eps[k] / self.re + n.eps[k] * self.re.ln())
            }),
        }
    }

    // -- Exp/Log --

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    #[inline]
    pub fn exp2(self) -> Self {
        let e = self.re.exp2();
        self.chain(e, e * F::LN_2())
    }

    #[inline]
    pub fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }

    #[inline]
    pub fn ln(self) -> Self {
        self.chain(self.re.ln(), F::one() / self.re)
    }

    #[inline]
    pub fn log2(self) -> Self {
        self.chain(self.re.log2(), F::one() / (self.re * F::LN_2()))
    }

    #[inline]
    pub fn log10(self) -> Self {
        self.chain(self.re.log10(), F::one() / (self.re * F::LN_10()))
    }

    #[inline]
    pub fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), F::one() / (F::one() + self.re))
    }

    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    // -- Trig --

    #[inline]
    pub fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }

    #[inline]
    pub fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }

    #[inline]
    pub fn tan(self) -> Self {
        let c = self.re.cos();
        self.chain(self.re.tan(), F::one() / (c * c))
    }

    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (
            DualVec {
                re: s,
                eps: std::array::from_fn(|k| self.eps[k] * c),
            },
            DualVec {
                re: c,
                eps: std::array::from_fn(|k| self.eps[k] * (-s)),
            },
        )
    }

    #[inline]
    pub fn asin(self) -> Self {
        self.chain(
            self.re.asin(),
            F::one() / (F::one() - self.re * self.re).sqrt(),
        )
    }

    #[inline]
    pub fn acos(self) -> Self {
        self.chain(
            self.re.acos(),
            -F::one() / (F::one() - self.re * self.re).sqrt(),
        )
    }

    #[inline]
    pub fn atan(self) -> Self {
        self.chain(self.re.atan(), F::one() / (F::one() + self.re * self.re))
    }

    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let denom = self.re * self.re + other.re * other.re;
        DualVec {
            re: self.re.atan2(other.re),
            eps: std::array::from_fn(|k| (other.re * self.eps[k] - self.re * other.eps[k]) / denom),
        }
    }

    // -- Hyperbolic --

    #[inline]
    pub fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }

    #[inline]
    pub fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }

    #[inline]
    pub fn tanh(self) -> Self {
        let c = self.re.cosh();
        self.chain(self.re.tanh(), F::one() / (c * c))
    }

    #[inline]
    pub fn asinh(self) -> Self {
        self.chain(
            self.re.asinh(),
            F::one() / (self.re * self.re + F::one()).sqrt(),
        )
    }

    #[inline]
    pub fn acosh(self) -> Self {
        self.chain(
            self.re.acosh(),
            F::one() / (self.re * self.re - F::one()).sqrt(),
        )
    }

    #[inline]
    pub fn atanh(self) -> Self {
        self.chain(self.re.atanh(), F::one() / (F::one() - self.re * self.re))
    }

    // -- Misc --

    #[inline]
    pub fn abs(self) -> Self {
        self.chain(self.re.abs(), self.re.signum())
    }

    #[inline]
    pub fn signum(self) -> Self {
        DualVec {
            re: self.re.signum(),
            eps: [F::zero(); N],
        }
    }

    #[inline]
    pub fn floor(self) -> Self {
        DualVec {
            re: self.re.floor(),
            eps: [F::zero(); N],
        }
    }

    #[inline]
    pub fn ceil(self) -> Self {
        DualVec {
            re: self.re.ceil(),
            eps: [F::zero(); N],
        }
    }

    #[inline]
    pub fn round(self) -> Self {
        DualVec {
            re: self.re.round(),
            eps: [F::zero(); N],
        }
    }

    #[inline]
    pub fn trunc(self) -> Self {
        DualVec {
            re: self.re.trunc(),
            eps: [F::zero(); N],
        }
    }

    #[inline]
    pub fn fract(self) -> Self {
        DualVec {
            re: self.re.fract(),
            eps: self.eps,
        }
    }

    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        DualVec {
            re: self.re.mul_add(a.re, b.re),
            eps: std::array::from_fn(|k| self.eps[k] * a.re + self.re * a.eps[k] + b.eps[k]),
        }
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        DualVec {
            re: h,
            eps: std::array::from_fn(|k| (self.re * self.eps[k] + other.re * other.eps[k]) / h),
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.re >= other.re {
            self
        } else {
            other
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.re <= other.re {
            self
        } else {
            other
        }
    }
}

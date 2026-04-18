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

    /// Reciprocal (1/x).
    #[inline]
    pub fn recip(self) -> Self {
        let inv = F::one() / self.re;
        self.chain(inv, -inv * inv)
    }

    /// Square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        let two = F::one() + F::one();
        self.chain(s, F::one() / (two * s))
    }

    /// Cube root.
    #[inline]
    pub fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        let three = F::from(3.0).unwrap();
        self.chain(c, F::one() / (three * c * c))
    }

    /// Integer power.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return DualVec {
                re: F::one(),
                eps: [F::zero(); N],
            };
        }
        let val = self.re.powi(n);
        let deriv = if n == i32::MIN {
            F::from(n).unwrap() * val / self.re
        } else {
            F::from(n).unwrap() * self.re.powi(n - 1)
        };
        self.chain(val, deriv)
    }

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        if n.re == F::zero() {
            // a^0 = 1, d/da(a^0) = 0, d/db(a^b)|_{b=0} = ln(a) (for a > 0)
            let dy = if self.re > F::zero() {
                self.re.ln()
            } else {
                F::zero()
            };
            return DualVec {
                re: F::one(),
                eps: std::array::from_fn(|k| dy * n.eps[k]),
            };
        }
        let val = self.re.powf(n.re);
        let dx_factor = if self.re == F::zero() || val == F::zero() {
            // Use n*x^(n-1) form to avoid 0/0 when x=0 and to handle
            // underflow when x^n underflows to 0 but x != 0
            n.re * self.re.powf(n.re - F::one())
        } else {
            n.re * val / self.re
        };
        let dy_factor = if val == F::zero() {
            F::zero()
        } else {
            val * self.re.ln()
        };
        DualVec {
            re: val,
            eps: std::array::from_fn(|k| dx_factor * self.eps[k] + dy_factor * n.eps[k]),
        }
    }

    // -- Exp/Log --

    /// Natural exponential (e^x).
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    /// Base-2 exponential (2^x).
    #[inline]
    pub fn exp2(self) -> Self {
        let e = self.re.exp2();
        self.chain(e, e * F::LN_2())
    }

    /// e^x - 1, accurate near zero.
    #[inline]
    pub fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        self.chain(self.re.ln(), F::one() / self.re)
    }

    /// Base-2 logarithm.
    #[inline]
    pub fn log2(self) -> Self {
        self.chain(self.re.log2(), F::one() / (self.re * F::LN_2()))
    }

    /// Base-10 logarithm.
    #[inline]
    pub fn log10(self) -> Self {
        self.chain(self.re.log10(), F::one() / (self.re * F::LN_10()))
    }

    /// ln(1+x), accurate near zero.
    #[inline]
    pub fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), F::one() / (F::one() + self.re))
    }

    /// Logarithm with given base.
    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    // -- Trig --

    /// Sine.
    #[inline]
    pub fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }

    /// Cosine.
    #[inline]
    pub fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }

    /// Tangent.
    #[inline]
    pub fn tan(self) -> Self {
        let c = self.re.cos();
        self.chain(self.re.tan(), F::one() / (c * c))
    }

    /// Simultaneous sine and cosine.
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

    /// Arcsine.
    #[inline]
    pub fn asin(self) -> Self {
        self.chain(
            self.re.asin(),
            F::one() / ((F::one() - self.re) * (F::one() + self.re)).sqrt(),
        )
    }

    /// Arccosine.
    #[inline]
    pub fn acos(self) -> Self {
        self.chain(
            self.re.acos(),
            -F::one() / ((F::one() - self.re) * (F::one() + self.re)).sqrt(),
        )
    }

    /// Arctangent.
    #[inline]
    pub fn atan(self) -> Self {
        let deriv = if self.re.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.re;
            inv * inv / (F::one() + inv * inv)
        } else {
            F::one() / (F::one() + self.re * self.re)
        };
        self.chain(self.re.atan(), deriv)
    }

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        if h == F::zero() {
            return DualVec {
                re: self.re.atan2(other.re),
                eps: [F::zero(); N],
            };
        }
        // Factor ((x/h)·dy - (y/h)·dx)/h to avoid h² over/underflow; see
        // `Dual::atan2` for the numerical argument.
        let x_over_h = other.re / h;
        let y_over_h = self.re / h;
        DualVec {
            re: self.re.atan2(other.re),
            eps: std::array::from_fn(|k| {
                (x_over_h * self.eps[k] - y_over_h * other.eps[k]) / h
            }),
        }
    }

    // -- Hyperbolic --

    /// Hyperbolic sine.
    #[inline]
    pub fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }

    /// Hyperbolic cosine.
    #[inline]
    pub fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }

    /// Hyperbolic tangent.
    #[inline]
    pub fn tanh(self) -> Self {
        let c = self.re.cosh();
        self.chain(self.re.tanh(), F::one() / (c * c))
    }

    /// Inverse hyperbolic sine.
    #[inline]
    pub fn asinh(self) -> Self {
        // See `Dual::asinh` for the overflow rationale on the |x|>1e8 switch.
        let deriv = if self.re.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.re;
            inv.abs() / (F::one() + inv * inv).sqrt()
        } else {
            F::one() / (self.re * self.re + F::one()).sqrt()
        };
        self.chain(self.re.asinh(), deriv)
    }

    /// Inverse hyperbolic cosine.
    #[inline]
    pub fn acosh(self) -> Self {
        let deriv = if self.re.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.re;
            inv.abs() / (F::one() - inv * inv).sqrt()
        } else {
            F::one() / (self.re * self.re - F::one()).sqrt()
        };
        self.chain(self.re.acosh(), deriv)
    }

    /// Inverse hyperbolic tangent.
    #[inline]
    pub fn atanh(self) -> Self {
        self.chain(
            self.re.atanh(),
            F::one() / ((F::one() - self.re) * (F::one() + self.re)),
        )
    }

    // -- Misc --

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        self.chain(self.re.abs(), self.re.signum())
    }

    /// Sign function (zero derivative).
    #[inline]
    pub fn signum(self) -> Self {
        DualVec {
            re: self.re.signum(),
            eps: [F::zero(); N],
        }
    }

    /// Floor (zero derivative).
    #[inline]
    pub fn floor(self) -> Self {
        DualVec {
            re: self.re.floor(),
            eps: [F::zero(); N],
        }
    }

    /// Ceiling (zero derivative).
    #[inline]
    pub fn ceil(self) -> Self {
        DualVec {
            re: self.re.ceil(),
            eps: [F::zero(); N],
        }
    }

    /// Round to nearest integer (zero derivative).
    #[inline]
    pub fn round(self) -> Self {
        DualVec {
            re: self.re.round(),
            eps: [F::zero(); N],
        }
    }

    /// Truncate toward zero (zero derivative).
    #[inline]
    pub fn trunc(self) -> Self {
        DualVec {
            re: self.re.trunc(),
            eps: [F::zero(); N],
        }
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        DualVec {
            re: self.re.fract(),
            eps: self.eps,
        }
    }

    /// Fused multiply-add: self * a + b.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        DualVec {
            re: self.re.mul_add(a.re, b.re),
            eps: std::array::from_fn(|k| self.eps[k] * a.re + self.re * a.eps[k] + b.eps[k]),
        }
    }

    /// Euclidean distance: sqrt(self^2 + other^2).
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        DualVec {
            re: h,
            eps: if h == F::zero() {
                [F::zero(); N]
            } else {
                std::array::from_fn(|k| (self.re * self.eps[k] + other.re * other.eps[k]) / h)
            },
        }
    }

    /// Maximum of two values.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.re >= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }

    /// Minimum of two values.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.re <= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }
}

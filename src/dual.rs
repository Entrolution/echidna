//! Forward-mode dual numbers for automatic differentiation.
//!
//! [`Dual<F>`] pairs a value with its tangent (derivative), using the algebra of
//! dual numbers where epsilon^2 = 0. Best suited for functions with few inputs and many
//! outputs, or when computing a single directional derivative (JVP).

use std::fmt::{self, Display};

use crate::Float;

/// Forward-mode dual number: a value paired with its tangent (derivative).
///
/// `Dual { re, eps }` represents `re + eps·ε` where `ε² = 0`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dual<F: Float> {
    /// Primal (real) value.
    pub re: F,
    /// Tangent (derivative) value.
    pub eps: F,
}

impl<F: Float> Display for Dual<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl<F: Float> From<F> for Dual<F> {
    #[inline]
    fn from(val: F) -> Self {
        Dual::constant(val)
    }
}

impl<F: Float> Dual<F> {
    /// Create a new dual number.
    #[inline]
    pub fn new(re: F, eps: F) -> Self {
        Dual { re, eps }
    }

    /// Create a constant (zero derivative).
    #[inline]
    pub fn constant(re: F) -> Self {
        Dual { re, eps: F::zero() }
    }

    /// Create a variable (unit derivative) for differentiation.
    #[inline]
    pub fn variable(re: F) -> Self {
        Dual { re, eps: F::one() }
    }

    /// Apply the chain rule: given `f(self.re)` and `f'(self.re)`, produce the dual result.
    #[inline]
    fn chain(self, f_val: F, f_deriv: F) -> Self {
        Dual {
            re: f_val,
            eps: self.eps * f_deriv,
        }
    }

    // ── Powers ──

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
            return Dual {
                re: F::one(),
                eps: F::zero(),
            };
        }
        let val = self.re.powi(n);
        let deriv = if n == i32::MIN {
            // n - 1 would overflow i32; fall back to powf
            F::from(n).unwrap() * self.re.powf(F::from(n as i64 - 1).unwrap())
        } else {
            F::from(n).unwrap() * self.re.powi(n - 1)
        };
        self.chain(val, deriv)
    }

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        // d/dx (x^y) = y * x^(y-1) * dx + x^y * ln(x) * dy
        if n.re == F::zero() {
            // a^0 = 1, d/da(a^0) = 0, d/db(a^b)|_{b=0} = ln(a) (for a > 0)
            let dy = if self.re > F::zero() {
                self.re.ln()
            } else {
                F::zero()
            };
            return Dual {
                re: F::one(),
                eps: dy * n.eps,
            };
        }
        let val = self.re.powf(n.re);
        let dx = if self.re == F::zero() {
            // Avoid 0/0: use n*x^(n-1) form which IEEE handles correctly
            n.re * self.re.powf(n.re - F::one()) * self.eps
        } else {
            n.re * val / self.re * self.eps
        };
        let dy = if val == F::zero() {
            // lim_{x→0+} x^y * ln(x) = 0 for y > 0
            F::zero()
        } else {
            val * self.re.ln() * n.eps
        };
        Dual {
            re: val,
            eps: dx + dy,
        }
    }

    // ── Exp/Log ──

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

    // ── Trig ──

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
            Dual {
                re: s,
                eps: self.eps * c,
            },
            Dual {
                re: c,
                eps: self.eps * (-s),
            },
        )
    }

    /// Arcsine.
    #[inline]
    pub fn asin(self) -> Self {
        self.chain(
            self.re.asin(),
            F::one() / (F::one() - self.re * self.re).sqrt(),
        )
    }

    /// Arccosine.
    #[inline]
    pub fn acos(self) -> Self {
        self.chain(
            self.re.acos(),
            -F::one() / (F::one() - self.re * self.re).sqrt(),
        )
    }

    /// Arctangent.
    #[inline]
    pub fn atan(self) -> Self {
        self.chain(self.re.atan(), F::one() / (F::one() + self.re * self.re))
    }

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        // d/dx atan2(y,x) = x/(x²+y²) dy - y/(x²+y²) dx
        let h = self.re.hypot(other.re);
        let denom = h * h;
        if denom == F::zero() {
            return Dual {
                re: self.re.atan2(other.re),
                eps: F::zero(),
            };
        }
        Dual {
            re: self.re.atan2(other.re),
            eps: (other.re * self.eps - self.re * other.eps) / denom,
        }
    }

    // ── Hyperbolic ──

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
        self.chain(
            self.re.asinh(),
            F::one() / (self.re * self.re + F::one()).sqrt(),
        )
    }

    /// Inverse hyperbolic cosine.
    #[inline]
    pub fn acosh(self) -> Self {
        self.chain(
            self.re.acosh(),
            F::one() / (self.re * self.re - F::one()).sqrt(),
        )
    }

    /// Inverse hyperbolic tangent.
    #[inline]
    pub fn atanh(self) -> Self {
        self.chain(self.re.atanh(), F::one() / (F::one() - self.re * self.re))
    }

    // ── Misc ──

    /// Absolute value.
    ///
    /// Derivative uses `signum(x)`: returns 1 at x=+0 and -1 at x=-0
    /// (matching Rust's `f64::signum`). Both are valid subgradients of |x| at 0.
    /// Consistent across all AD modes and GPU backends.
    #[inline]
    pub fn abs(self) -> Self {
        self.chain(self.re.abs(), self.re.signum())
    }

    /// Sign function (zero derivative).
    #[inline]
    pub fn signum(self) -> Self {
        Dual {
            re: self.re.signum(),
            eps: F::zero(),
        }
    }

    /// Floor (zero derivative).
    #[inline]
    pub fn floor(self) -> Self {
        Dual {
            re: self.re.floor(),
            eps: F::zero(),
        }
    }

    /// Ceiling (zero derivative).
    #[inline]
    pub fn ceil(self) -> Self {
        Dual {
            re: self.re.ceil(),
            eps: F::zero(),
        }
    }

    /// Round to nearest integer (zero derivative).
    #[inline]
    pub fn round(self) -> Self {
        Dual {
            re: self.re.round(),
            eps: F::zero(),
        }
    }

    /// Truncate toward zero (zero derivative).
    #[inline]
    pub fn trunc(self) -> Self {
        Dual {
            re: self.re.trunc(),
            eps: F::zero(),
        }
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        Dual {
            re: self.re.fract(),
            eps: self.eps,
        }
    }

    /// Fused multiply-add: self * a + b.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // d(x*a + b) = a*dx + x*da + db
        Dual {
            re: self.re.mul_add(a.re, b.re),
            eps: self.eps * a.re + self.re * a.eps + b.eps,
        }
    }

    /// Euclidean distance: sqrt(self^2 + other^2).
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        Dual {
            re: h,
            eps: if h == F::zero() {
                F::zero()
            } else {
                (self.re * self.eps + other.re * other.eps) / h
            },
        }
    }

    /// Maximum of two values.
    ///
    /// Matches `num_traits::Float::max` semantics: returns the non-NaN argument.
    /// At tie points, returns `self` (standard AD convention for non-differentiable points).
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.re >= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }

    /// Minimum of two values.
    ///
    /// Matches `num_traits::Float::min` semantics: returns the non-NaN argument.
    /// At tie points, returns `self` (standard AD convention for non-differentiable points).
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.re <= other.re || other.re.is_nan() {
            self
        } else {
            other
        }
    }
}

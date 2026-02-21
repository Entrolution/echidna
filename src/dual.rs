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
        // d/dx (x^y) = y * x^(y-1) * dx + x^y * ln(x) * dy
        let val = self.re.powf(n.re);
        Dual {
            re: val,
            eps: val * (n.re * self.eps / self.re + n.eps * self.re.ln()),
        }
    }

    // ── Exp/Log ──

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

    // ── Trig ──

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
        // d/dx atan2(y,x) = x/(x²+y²) dy - y/(x²+y²) dx
        let denom = self.re * self.re + other.re * other.re;
        Dual {
            re: self.re.atan2(other.re),
            eps: (other.re * self.eps - self.re * other.eps) / denom,
        }
    }

    // ── Hyperbolic ──

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

    // ── Misc ──

    #[inline]
    pub fn abs(self) -> Self {
        self.chain(self.re.abs(), self.re.signum())
    }

    #[inline]
    pub fn signum(self) -> Self {
        Dual {
            re: self.re.signum(),
            eps: F::zero(),
        }
    }

    #[inline]
    pub fn floor(self) -> Self {
        Dual {
            re: self.re.floor(),
            eps: F::zero(),
        }
    }

    #[inline]
    pub fn ceil(self) -> Self {
        Dual {
            re: self.re.ceil(),
            eps: F::zero(),
        }
    }

    #[inline]
    pub fn round(self) -> Self {
        Dual {
            re: self.re.round(),
            eps: F::zero(),
        }
    }

    #[inline]
    pub fn trunc(self) -> Self {
        Dual {
            re: self.re.trunc(),
            eps: F::zero(),
        }
    }

    #[inline]
    pub fn fract(self) -> Self {
        Dual {
            re: self.re.fract(),
            eps: self.eps,
        }
    }

    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // d(x*a + b) = a*dx + x*da + db
        Dual {
            re: self.re.mul_add(a.re, b.re),
            eps: self.eps * a.re + self.re * a.eps + b.eps,
        }
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        Dual {
            re: h,
            eps: (self.re * self.eps + other.re * other.eps) / h,
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

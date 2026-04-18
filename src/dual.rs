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
        // At self.re = 0, inv = ±Inf. Skip the chain for eps = 0 so the
        // tangent is 0 (the "constant zero" convention) rather than the
        // IEEE `0 * Inf = NaN` we'd otherwise propagate. Non-zero eps at
        // the singularity keeps the Inf (the true derivative is unbounded).
        let eps = if self.eps == F::zero() {
            F::zero()
        } else {
            self.eps * (-inv * inv)
        };
        Dual { re: inv, eps }
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
            // n - 1 would overflow i32; use x^n / x to avoid precision loss
            // from converting n-1 to float (which rounds for f32)
            F::from(n).unwrap() * val / self.re
        } else {
            F::from(n).unwrap() * self.re.powi(n - 1)
        };
        self.chain(val, deriv)
    }

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        // d/dx (x^y) = y * x^(y-1) * dx + x^y * ln(x) * dy
        //
        // Constant integer exponent fast path: if `n` has no tangent and its
        // value is a losslessly representable integer, dispatch to `powi`.
        // This avoids computing `ln(x)` for `x < 0` where stdlib returns NaN —
        // that NaN would poison `eps` via `NaN * 0 = NaN` in IEEE 754, even
        // though `dy` is algebraically zero for a constant exponent.
        if n.eps == F::zero() {
            if let Some(ni) = n.re.to_i32() {
                if F::from(ni).unwrap() == n.re {
                    return self.powi(ni);
                }
            }
        }
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
        let dx = if self.re == F::zero() || val == F::zero() {
            // Use n*x^(n-1) form to avoid 0/0 when x=0 and to handle
            // underflow when x^n underflows to 0 but x != 0
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
        // For large |x|, 1 + x² overflows to inf, producing 1/inf = 0.
        // Use 1/x² form instead, which avoids the overflow.
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
        // d/dx atan2(y,x) = x/(x²+y²) dy - y/(x²+y²) dx
        let h = self.re.hypot(other.re);
        if h == F::zero() {
            return Dual {
                re: self.re.atan2(other.re),
                eps: F::zero(),
            };
        }
        // Factor as ((x/h)·dy - (y/h)·dx)/h instead of .../(h*h). Both x/h
        // and y/h are bounded by 1 (since h = hypot(x,y) ≥ |x|, |y|), so no
        // intermediate step overflows. The naive h*h form overflows for
        // |h| > sqrt(f64::MAX) ≈ 1.3e154 and underflows for |h| below
        // sqrt(f64::MIN_POSITIVE) — silently corrupting otherwise finite
        // partials.
        let x_over_h = other.re / h;
        let y_over_h = self.re / h;
        Dual {
            re: self.re.atan2(other.re),
            eps: (x_over_h * self.eps - y_over_h * other.eps) / h,
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
        // For |x| > 1e8, `x*x + 1` overflows in f64 at |x| > ~1.3e154 and the
        // derivative silently collapses to 0. Use the algebraically equivalent
        // |1/x| / sqrt(1 + (1/x)²) form, which stays in-range for any x.
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
        // Two-branch derivative:
        //   |x| > 1e8   → `|1/x| / sqrt(1 - (1/x)²)` avoids x²-1 overflow.
        //   |x| ≤ 1e8   → `1 / sqrt((x-1)·(x+1))` avoids catastrophic
        //                 cancellation near x = 1 that `x²-1` would hit
        //                 (at x = 1 + ε, `x² = 1 + 2ε + ε²` rounds to
        //                 `1 + 2ε`, losing the ε² contribution). The
        //                 factored form stays numerically distinct down
        //                 to the minimum representable positive number.
        let deriv = if self.re.abs() > F::from(1e8).unwrap() {
            let inv = F::one() / self.re;
            inv.abs() / (F::one() - inv * inv).sqrt()
        } else {
            F::one() / ((self.re - F::one()) * (self.re + F::one())).sqrt()
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
                // Factor as (x/h)·dx + (y/h)·dy to avoid numerator overflow:
                // `x·dx` alone can overflow for x and dx both large even when
                // the true tangent is representable. `hypot` does this trick
                // for the primal; the tangent needs it too.
                (self.re / h) * self.eps + (other.re / h) * other.eps
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

//! Batched forward-mode dual numbers with `N` tangent lanes.
//!
//! [`DualVec<F, N>`] carries `N` independent tangent directions simultaneously,
//! enabling vectorized Jacobian columns or batched Hessian computation via
//! forward-over-reverse mode.

use crate::dual::forward_elementary_methods;
use std::fmt::{self, Display};

use crate::kernels;
use crate::Float;

/// Batched forward-mode dual number: a value with N tangent lanes.
///
/// `DualVec { re, eps }` represents a value with N independent tangent directions,
/// enabling batched Hessian computation.
#[derive(Clone, Copy, Debug)]
// repr(C) pins the field order (re, then the eps lanes) so batched values have
// a byte-stable layout for FFI/GPU-style interop; no in-crate consumer depends
// on it today.
#[repr(C)]
pub struct DualVec<F: Float, const N: usize> {
    /// Primal (real) value.
    pub re: F,
    /// Tangent (derivative) values — one per lane.
    pub eps: [F; N],
}

impl<F: Float, const N: usize> Default for DualVec<F, N> {
    fn default() -> Self {
        Self::constant(F::zero())
    }
}

impl<F: Float, const N: usize> Display for DualVec<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.re)?;
        for (i, e) in self.eps.iter().enumerate() {
            write!(f, " + {e}\u{03b5}{i}")?;
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
    #[must_use]
    pub fn new(re: F, eps: [F; N]) -> Self {
        DualVec { re, eps }
    }

    /// Create a constant (zero derivatives in all lanes).
    #[inline]
    #[must_use]
    pub fn constant(re: F) -> Self {
        DualVec {
            re,
            eps: [F::zero(); N],
        }
    }

    /// Create a variable with unit derivative in the specified lane.
    #[inline]
    #[must_use]
    pub fn with_tangent(re: F, lane: usize) -> Self {
        DualVec {
            re,
            eps: std::array::from_fn(|k| if k == lane { F::one() } else { F::zero() }),
        }
    }

    /// Apply the chain rule: given `f(self.re)` and `f'(self.re)`, produce the dual result.
    ///
    /// A structurally zero tangent lane short-circuits to exactly zero: the
    /// derivative of a constant is 0 regardless of where `f` is evaluated,
    /// even where `f_deriv` is ±Inf (domain boundaries like `sqrt`/`ln` at 0,
    /// `atanh` at ±1) or NaN (out-of-domain primals) — IEEE `0 * Inf` would
    /// otherwise poison the lane with NaN. Matches `Dual::chain`; the check
    /// uses `is_all_zero()` rather than `==` (primal-only) so nested lanes like
    /// `DualVec<Dual<F>, N>` with live second-order components are kept.
    #[inline]
    fn chain(self, f_val: F, f_deriv: F) -> Self {
        DualVec {
            re: f_val,
            eps: std::array::from_fn(|k| {
                if self.eps[k].is_all_zero() {
                    F::zero()
                } else {
                    self.eps[k] * f_deriv
                }
            }),
        }
    }

    forward_elementary_methods!();

    // -- Powers --

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        // Constant integer exponent fast path (see `Dual::powf` for rationale):
        // avoids `ln(x)` NaN-poisoning the tangent when the exponent is a
        // constant and `x < 0`.
        if n.eps.iter().all(|e| e.is_all_zero()) {
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
            return DualVec {
                re: F::one(),
                eps: std::array::from_fn(|k| {
                    // Per-lane structural-zero short-circuit: dy = ln(a) is
                    // +Inf at an infinite base, and a constant lane must stay
                    // exactly 0 rather than Inf·0 = NaN. (The all-constant
                    // exponent is already captured by the powi fast path
                    // above; this branch is reached only with mixed lanes.)
                    if n.eps[k].is_all_zero() {
                        F::zero()
                    } else {
                        dy * n.eps[k]
                    }
                }),
            };
        }
        let val = self.re.powf(n.re);
        let dx_factor =
            if self.re == F::zero() || val == F::zero() || !self.re.is_finite() || !val.is_finite()
            {
                // Use n*x^(n-1) form to avoid 0/0 when x=0, to handle underflow
                // when x^n underflows to 0 but x != 0, and to avoid Inf/Inf = NaN
                // at an infinite base (matches `Reverse`/`opcode`).
                n.re * self.re.powf(n.re - F::one())
            } else {
                n.re * val / self.re
            };
        let dy_factor = if val == F::zero() || self.re <= F::zero() {
            // Negative/zero base: `ln(x)` is undefined for x <= 0, so the
            // exponent direction is treated as locally constant (matches
            // `Reverse`/`opcode`). The base-direction factor stays finite.
            F::zero()
        } else {
            val * self.re.ln()
        };
        DualVec {
            re: val,
            eps: std::array::from_fn(|k| {
                // Per-lane structural-zero short-circuits (see `Dual::powf`):
                // a constant lane contributes nothing even where its factor is
                // non-finite (x^0.5 at x = 0 for dx; overflowed `val` for dy).
                let dx = if self.eps[k].is_all_zero() {
                    F::zero()
                } else {
                    dx_factor * self.eps[k]
                };
                let dy = if n.eps[k].is_all_zero() {
                    F::zero()
                } else {
                    dy_factor * n.eps[k]
                };
                dx + dy
            }),
        }
    }

    // -- Trig --

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let (d_self, d_other) = kernels::atan2_partials(self.re, other.re);
        DualVec {
            re: self.re.atan2(other.re),
            eps: std::array::from_fn(|k| d_self * self.eps[k] + d_other * other.eps[k]),
        }
    }

    // -- Misc --

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
        if h == F::zero() {
            // See `Dual::hypot`: singular point — short-circuit to zero
            // tangent regardless of `eps` sign/finiteness.
            return Self::constant(h);
        }
        let (da, db) = kernels::hypot_partials(self.re, other.re, h);
        DualVec {
            re: h,
            eps: std::array::from_fn(|k| da * self.eps[k] + db * other.eps[k]),
        }
    }
}

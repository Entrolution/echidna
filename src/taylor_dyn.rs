//! Dynamic (arena-based) Taylor coefficient type: `TaylorDyn<F>`.
//!
//! Follows the `Reverse<F>` pattern: a lightweight `{ value, index }` struct
//! with coefficient storage in a thread-local arena. This makes `TaylorDyn`
//! `Copy`, enabling full `num_traits::Float` / echidna `Float` / `Scalar`.
//!
//! The degree (number of coefficients) is set at runtime when creating a
//! `TaylorDynGuard`, which initializes the arena.

use std::cell::Cell;
use std::fmt::{self, Display};

use crate::taylor_ops;
use crate::Float;

/// Sentinel index for constants (not stored in arena).
///
/// A `TaylorDyn` with `index == CONSTANT` implicitly has coefficients
/// `[value, 0, 0, ..., 0]`. This avoids arena allocation for literals
/// and constants from `forward_tangent`.
pub const CONSTANT: u32 = u32::MAX;

/// Flat arena for Taylor coefficient vectors.
///
/// All entries have the same `degree` (number of coefficients). Entry `i`
/// occupies `data[i*degree .. (i+1)*degree]`.
pub struct TaylorArena<F: Float> {
    data: Vec<F>,
    degree: usize,
    count: u32,
}

impl<F: Float> TaylorArena<F> {
    /// Create a new arena with the given degree.
    pub fn new(degree: usize) -> Self {
        TaylorArena {
            data: Vec::new(),
            degree,
            count: 0,
        }
    }

    /// Number of coefficients per entry.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Allocate a new entry (zeroed). Returns its index.
    #[inline]
    pub fn allocate(&mut self) -> u32 {
        let idx = self.count;
        self.count += 1;
        self.data
            .resize(self.count as usize * self.degree, F::zero());
        idx
    }

    /// Get the coefficient slice for entry `index`.
    #[inline]
    pub fn coeffs(&self, index: u32) -> &[F] {
        let start = index as usize * self.degree;
        &self.data[start..start + self.degree]
    }

    /// Get a mutable coefficient slice for entry `index`.
    #[inline]
    pub fn coeffs_mut(&mut self, index: u32) -> &mut [F] {
        let start = index as usize * self.degree;
        &mut self.data[start..start + self.degree]
    }

    /// Reset the arena (keeps capacity).
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }
}

// ── Thread-local arenas ──

thread_local! {
    static TAYLOR_ARENA_F32: Cell<*mut TaylorArena<f32>> = const { Cell::new(std::ptr::null_mut()) };
    static TAYLOR_ARENA_F64: Cell<*mut TaylorArena<f64>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Trait to select the correct thread-local arena for a given float type.
///
/// Mirrors `TapeThreadLocal` from `tape.rs`.
pub trait TaylorArenaLocal: Float {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut TaylorArena<Self>>>;
}

impl TaylorArenaLocal for f32 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut TaylorArena<Self>>> {
        &TAYLOR_ARENA_F32
    }
}

impl TaylorArenaLocal for f64 {
    fn cell() -> &'static std::thread::LocalKey<Cell<*mut TaylorArena<Self>>> {
        &TAYLOR_ARENA_F64
    }
}

/// Access the active Taylor arena for the current thread.
/// Panics if no arena is active (i.e., no `TaylorDynGuard` is in scope).
#[inline]
pub fn with_active_arena<F: TaylorArenaLocal, R>(f: impl FnOnce(&mut TaylorArena<F>) -> R) -> R {
    F::cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active Taylor arena. Create a TaylorDynGuard first."
        );
        let arena = unsafe { &mut *ptr };
        f(arena)
    })
}

/// RAII guard that activates a Taylor arena on the current thread.
///
/// Creates a new arena with the specified degree. Restores the previous
/// arena (if any) on drop.
pub struct TaylorDynGuard<F: TaylorArenaLocal> {
    arena: Box<TaylorArena<F>>,
    prev: *mut TaylorArena<F>,
}

impl<F: TaylorArenaLocal> TaylorDynGuard<F> {
    /// Create and activate a Taylor arena with the given `degree`
    /// (number of Taylor coefficients per variable).
    pub fn new(degree: usize) -> Self {
        let mut arena = Box::new(TaylorArena::new(degree));
        let prev = F::cell().with(|cell| {
            let prev = cell.get();
            cell.set(&mut *arena as *mut TaylorArena<F>);
            prev
        });
        TaylorDynGuard { arena, prev }
    }

    /// Access the underlying arena.
    pub fn arena(&self) -> &TaylorArena<F> {
        &self.arena
    }
}

impl<F: TaylorArenaLocal> Drop for TaylorDynGuard<F> {
    fn drop(&mut self) {
        F::cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

// ══════════════════════════════════════════════
//  TaylorDyn<F> type
// ══════════════════════════════════════════════

/// Dynamic Taylor coefficient variable.
///
/// `Copy`-friendly: stores only `{ value, index }`. Coefficient vectors
/// live in a thread-local [`TaylorArena`].
///
/// `value` = `coeffs[0]` (primal), kept inline for comparisons/branching.
/// `index` = arena slot, or [`CONSTANT`] sentinel for literals.
#[derive(Clone, Copy, Debug)]
pub struct TaylorDyn<F: Float> {
    pub(crate) value: F,
    pub(crate) index: u32,
}

impl<F: Float> TaylorDyn<F> {
    /// Create a constant (not stored in arena).
    #[inline]
    pub fn constant(value: F) -> Self {
        TaylorDyn {
            value,
            index: CONSTANT,
        }
    }
}

impl<F: Float + TaylorArenaLocal> TaylorDyn<F> {
    /// Create a variable: c₀ = val, c₁ = 1, rest zero.
    #[inline]
    pub fn variable(val: F) -> Self {
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let idx = arena.allocate();
            let coeffs = arena.coeffs_mut(idx);
            coeffs[0] = val;
            if coeffs.len() > 1 {
                coeffs[1] = F::one();
            }
            TaylorDyn {
                value: val,
                index: idx,
            }
        })
    }

    /// Create from explicit coefficients (copies into arena).
    #[inline]
    pub fn from_coeffs(coeffs: &[F]) -> Self {
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let idx = arena.allocate();
            let slot = arena.coeffs_mut(idx);
            let copy_len = coeffs.len().min(slot.len());
            slot[..copy_len].copy_from_slice(&coeffs[..copy_len]);
            TaylorDyn {
                value: coeffs[0],
                index: idx,
            }
        })
    }

    /// Primal value.
    #[inline]
    pub fn value(&self) -> F {
        self.value
    }

    /// Get arena index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Read all coefficients (copies from arena).
    pub fn coeffs(&self) -> Vec<F> {
        if self.index == CONSTANT {
            with_active_arena(|arena: &mut TaylorArena<F>| {
                let mut v = vec![F::zero(); arena.degree()];
                v[0] = self.value;
                v
            })
        } else {
            with_active_arena(|arena: &mut TaylorArena<F>| arena.coeffs(self.index).to_vec())
        }
    }

    /// Get the k-th derivative: `k! × coeffs[k]`.
    pub fn derivative(&self, k: usize) -> F {
        let ck = if k == 0 {
            self.value
        } else if self.index == CONSTANT {
            F::zero()
        } else {
            with_active_arena(|arena: &mut TaylorArena<F>| arena.coeffs(self.index)[k])
        };
        let mut factorial = F::one();
        for i in 2..=k {
            factorial = factorial * F::from(i).unwrap();
        }
        ck * factorial
    }

    // ── Operation helpers ──

    /// Helper: get coefficients as a slice, using a temporary buffer for constants.
    fn get_coeffs_vec(&self) -> Vec<F> {
        if self.index == CONSTANT {
            with_active_arena(|arena: &mut TaylorArena<F>| {
                let mut v = vec![F::zero(); arena.degree()];
                v[0] = self.value;
                v
            })
        } else {
            with_active_arena(|arena: &mut TaylorArena<F>| arena.coeffs(self.index).to_vec())
        }
    }

    /// Apply a unary operation that takes input coefficients and writes output coefficients.
    pub(crate) fn unary_op(x: &Self, f: impl FnOnce(&[F], &mut [F])) -> Self {
        let a = x.get_coeffs_vec();
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let deg = arena.degree();
            let idx = arena.allocate();
            let mut result = vec![F::zero(); deg];
            f(&a, &mut result);
            let slot = arena.coeffs_mut(idx);
            slot.copy_from_slice(&result);
            TaylorDyn {
                value: result[0],
                index: idx,
            }
        })
    }

    /// Apply a binary operation.
    pub(crate) fn binary_op(x: &Self, y: &Self, f: impl FnOnce(&[F], &[F], &mut [F])) -> Self {
        // Both constants: result is also a constant (optimize for forward_tangent)
        if x.index == CONSTANT && y.index == CONSTANT {
            let deg = with_active_arena(|arena: &mut TaylorArena<F>| arena.degree());
            let mut a = vec![F::zero(); deg];
            a[0] = x.value;
            let mut b = vec![F::zero(); deg];
            b[0] = y.value;
            let mut result = vec![F::zero(); deg];
            f(&a, &b, &mut result);
            // If result is constant-like (only c[0] nonzero), return as constant
            if result[1..].iter().all(|&c| c == F::zero()) {
                return TaylorDyn {
                    value: result[0],
                    index: CONSTANT,
                };
            }
            // Otherwise allocate
            return with_active_arena(|arena: &mut TaylorArena<F>| {
                let idx = arena.allocate();
                let slot = arena.coeffs_mut(idx);
                slot.copy_from_slice(&result);
                TaylorDyn {
                    value: result[0],
                    index: idx,
                }
            });
        }

        let a = x.get_coeffs_vec();
        let b = y.get_coeffs_vec();
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let deg = arena.degree();
            let idx = arena.allocate();
            let mut result = vec![F::zero(); deg];
            f(&a, &b, &mut result);
            let slot = arena.coeffs_mut(idx);
            slot.copy_from_slice(&result);
            TaylorDyn {
                value: result[0],
                index: idx,
            }
        })
    }

    // ── Elemental methods ──

    #[inline]
    pub fn recip(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_recip(a, c))
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_sqrt(a, c))
    }

    #[inline]
    pub fn cbrt(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_cbrt(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self::unary_op(&self, |a, c| {
            let deg = c.len();
            let mut s1 = vec![F::zero(); deg];
            let mut s2 = vec![F::zero(); deg];
            taylor_ops::taylor_powi(a, n, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn powf(self, n: Self) -> Self {
        let b = n.get_coeffs_vec();
        Self::unary_op(&self, |a, c| {
            let deg = c.len();
            let mut s1 = vec![F::zero(); deg];
            let mut s2 = vec![F::zero(); deg];
            taylor_ops::taylor_powf(a, &b, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn exp(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_exp(a, c))
    }

    #[inline]
    pub fn exp2(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s = vec![F::zero(); n];
            taylor_ops::taylor_exp2(a, c, &mut s);
        })
    }

    #[inline]
    pub fn exp_m1(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_exp_m1(a, c))
    }

    #[inline]
    pub fn ln(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_ln(a, c))
    }

    #[inline]
    pub fn log2(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_log2(a, c))
    }

    #[inline]
    pub fn log10(self) -> Self {
        Self::unary_op(&self, |a, c| taylor_ops::taylor_log10(a, c))
    }

    #[inline]
    pub fn ln_1p(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s = vec![F::zero(); n];
            taylor_ops::taylor_ln_1p(a, c, &mut s);
        })
    }

    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    pub fn sin(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut co = vec![F::zero(); n];
            taylor_ops::taylor_sin_cos(a, c, &mut co);
        })
    }

    #[inline]
    pub fn cos(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s = vec![F::zero(); n];
            taylor_ops::taylor_sin_cos(a, &mut s, c);
        })
    }

    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let a = self.get_coeffs_vec();
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let deg = arena.degree();
            let sin_idx = arena.allocate();
            let cos_idx = arena.allocate();
            let mut s = vec![F::zero(); deg];
            let mut co = vec![F::zero(); deg];
            taylor_ops::taylor_sin_cos(&a, &mut s, &mut co);
            arena.coeffs_mut(sin_idx).copy_from_slice(&s);
            arena.coeffs_mut(cos_idx).copy_from_slice(&co);
            (
                TaylorDyn {
                    value: s[0],
                    index: sin_idx,
                },
                TaylorDyn {
                    value: co[0],
                    index: cos_idx,
                },
            )
        })
    }

    #[inline]
    pub fn tan(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s = vec![F::zero(); n];
            taylor_ops::taylor_tan(a, c, &mut s);
        })
    }

    #[inline]
    pub fn asin(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_asin(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn acos(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_acos(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atan(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_atan(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let b = other.get_coeffs_vec();
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            let mut s3 = vec![F::zero(); n];
            taylor_ops::taylor_atan2(a, &b, c, &mut s1, &mut s2, &mut s3);
        })
    }

    #[inline]
    pub fn sinh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut ch = vec![F::zero(); n];
            taylor_ops::taylor_sinh_cosh(a, c, &mut ch);
        })
    }

    #[inline]
    pub fn cosh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut sh = vec![F::zero(); n];
            taylor_ops::taylor_sinh_cosh(a, &mut sh, c);
        })
    }

    #[inline]
    pub fn tanh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s = vec![F::zero(); n];
            taylor_ops::taylor_tanh(a, c, &mut s);
        })
    }

    #[inline]
    pub fn asinh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_asinh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn acosh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_acosh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn atanh(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_atanh(a, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self::unary_op(&self, |a, c| {
            let sign = a[0].signum();
            for k in 0..c.len() {
                c[k] = a[k] * sign;
            }
        })
    }

    #[inline]
    pub fn signum(self) -> Self {
        TaylorDyn::constant(self.value.signum())
    }

    #[inline]
    pub fn floor(self) -> Self {
        TaylorDyn::constant(self.value.floor())
    }

    #[inline]
    pub fn ceil(self) -> Self {
        TaylorDyn::constant(self.value.ceil())
    }

    #[inline]
    pub fn round(self) -> Self {
        TaylorDyn::constant(self.value.round())
    }

    #[inline]
    pub fn trunc(self) -> Self {
        TaylorDyn::constant(self.value.trunc())
    }

    #[inline]
    pub fn fract(self) -> Self {
        Self::unary_op(&self, |a, c| {
            c[0] = a[0].fract();
            c[1..].copy_from_slice(&a[1..]);
        })
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        Self::binary_op(&self, &other, |a, b, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            taylor_ops::taylor_hypot(a, b, c, &mut s1, &mut s2);
        })
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.value >= other.value {
            self
        } else {
            other
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.value <= other.value {
            self
        } else {
            other
        }
    }
}

impl<F: Float> Display for TaylorDyn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<F: Float> Default for TaylorDyn<F> {
    fn default() -> Self {
        TaylorDyn::constant(F::zero())
    }
}

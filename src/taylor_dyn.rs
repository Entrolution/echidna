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
    /// Reusable per-op scratch, lazily grown and zero-filled on loan.
    /// Living on the (thread-local) arena keeps the elemental closures free
    /// of per-op heap allocation; ops never nest, so one slab pool suffices.
    scratch: Vec<F>,
}

impl<F: Float> TaylorArena<F> {
    /// Create a new arena with the given degree.
    #[must_use]
    pub fn new(degree: usize) -> Self {
        TaylorArena {
            data: Vec::new(),
            degree,
            count: 0,
            scratch: Vec::new(),
        }
    }

    /// Number of coefficients per entry.
    #[inline]
    #[must_use]
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
    #[must_use]
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

    /// Reset the arena (keeps capacity; the scratch pool is per-op
    /// transient and needs no reset).
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }

    /// Borrow the freshly allocated output slot, every earlier entry, and
    /// `slabs × degree` of zeroed scratch — all from one `&mut self`, which
    /// is what lets a kernel read input entries while writing its output in
    /// place. `out_idx` must be the newest entry, and the split must be
    /// derived AFTER `allocate` (allocation may move the backing storage).
    fn split_output(&mut self, out_idx: u32, slabs: usize) -> (&[F], &mut [F], &mut [F]) {
        debug_assert_eq!(out_idx + 1, self.count, "output must be the newest entry");
        let deg = self.degree;
        let need = slabs * deg;
        if self.scratch.len() < need {
            self.scratch.resize(need, F::zero());
        }
        self.scratch[..need].fill(F::zero());
        let start = out_idx as usize * deg;
        let (before, out) = self.data.split_at_mut(start);
        (&*before, &mut out[..deg], &mut self.scratch[..need])
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
    /// Returns the thread-local cell holding a pointer to the active arena.
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
        // SAFETY: The pointer is non-null (asserted above) and was set by a
        // `TaylorDynGuard` that owns the `Box<TaylorArena<F>>` and keeps it alive
        // for the guard's lifetime. The thread-local cell ensures single-threaded
        // access, so no aliasing occurs.
        let arena = unsafe { &mut *ptr };
        f(arena)
    })
}

/// RAII guard that activates a Taylor arena on the current thread.
///
/// Creates a new arena with the specified degree. Restores the previous
/// arena (if any) on drop.
pub struct TaylorDynGuard<F: TaylorArenaLocal> {
    // Never read directly, but load-bearing: the thread-local cell holds a
    // raw pointer into this Box, and `with_active_arena`'s SAFETY contract
    // depends on the guard keeping the allocation alive until drop.
    #[allow(dead_code)]
    arena: Box<TaylorArena<F>>,
    prev: *mut TaylorArena<F>,
}

impl<F: TaylorArenaLocal> TaylorDynGuard<F> {
    /// Create and activate a Taylor arena with the given `degree`
    /// (number of Taylor coefficients per variable).
    #[must_use]
    pub fn new(degree: usize) -> Self {
        let mut arena = Box::new(TaylorArena::new(degree));
        let prev = F::cell().with(|cell| {
            let prev = cell.get();
            cell.set(&mut *arena as *mut TaylorArena<F>);
            prev
        });
        TaylorDynGuard { arena, prev }
    }
}

impl<F: TaylorArenaLocal> Drop for TaylorDynGuard<F> {
    fn drop(&mut self) {
        F::cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}

/// Seed one `TaylorDyn` jet per input: `coeffs[0] = x[i]`, plus
/// `coeffs[slot] = value` for every `(var, slot, value)` in `active` with
/// `var == i` and `slot` inside the jet order. Shared by the deterministic
/// (diffop plan groups) and stochastic (STDE sparse/diagonal) samplers, whose
/// only difference is where `active` comes from.
///
/// An active arena of degree `order` must be live (`TaylorDynGuard`).
#[cfg(any(feature = "stde", feature = "diffop"))]
pub(crate) fn seed_taylor_dyn_jets<F: Float + TaylorArenaLocal>(
    x: &[F],
    order: usize,
    active: &[(usize, usize, F)],
) -> Vec<TaylorDyn<F>> {
    (0..x.len())
        .map(|i| {
            let mut coeffs = vec![F::zero(); order];
            coeffs[0] = x[i];
            for &(var, slot, value) in active {
                if var == i && slot < order {
                    coeffs[slot] = value;
                }
            }
            TaylorDyn::from_coeffs(&coeffs)
        })
        .collect()
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

impl<F: Float> From<F> for TaylorDyn<F> {
    #[inline]
    fn from(val: F) -> Self {
        TaylorDyn::constant(val)
    }
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

/// Emits the uniform elemental wrappers around the `taylor_ops` kernels:
/// each is `unary_op` plus the kernel call, differing only in the kernel
/// name and how many scratch buffers it needs. The recurrence math stays in
/// `taylor_ops`; ops with extra arguments or reused two-output kernels
/// (`powi`, `powf`, `sin`/`cos`/`sin_cos`, `sinh`/`cosh`, `atan2`, `hypot`)
/// remain hand-written below. Closure bodies must not touch the arena —
/// they run inside `with_active_arena`, which does not tolerate reentrancy.
macro_rules! taylor_dyn_elementals {
    ($( $(#[$doc:meta])* $name:ident => $kernel:ident / $scratch:tt; )+) => {$(
        $(#[$doc])*
        #[inline]
        pub fn $name(self) -> Self {
            taylor_dyn_elementals!(@body $kernel, self, $scratch)
        }
    )+};
    (@body $kernel:ident, $self:ident, 0) => {
        Self::unary_op(&$self, |a, c| taylor_ops::$kernel(a, c))
    };
    (@body $kernel:ident, $self:ident, 1) => {
        Self::unary_op_scratch(&$self, 1, |a, c, s| taylor_ops::$kernel(a, c, s))
    };
    (@body $kernel:ident, $self:ident, 2) => {
        Self::unary_op_scratch(&$self, 2, |a, c, s| {
            let (s1, s2) = s.split_at_mut(c.len());
            taylor_ops::$kernel(a, c, s1, s2)
        })
    };
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
        // Interleave k! multiplication with coefficient to avoid standalone
        // factorial overflow (f64 overflows at k=171, f32 at k=35)
        let mut result = ck;
        for i in 2..=k {
            result = result * F::from(i).unwrap();
        }
        result
    }

    // ── Operation helpers ──

    /// Apply a unary operation that takes input coefficients and writes output coefficients.
    pub(crate) fn unary_op(x: &Self, f: impl FnOnce(&[F], &mut [F])) -> Self {
        Self::unary_op_scratch(x, 0, |a, c, _| f(a, c))
    }

    /// Like [`unary_op`](Self::unary_op), with `slabs × degree` of zeroed
    /// arena scratch passed to the closure — elemental kernels take their
    /// work buffers from here instead of allocating per op.
    ///
    /// Everything the closure touches comes from one borrow-split taken
    /// after the output allocation: the input entry is read in place (no
    /// staging copy), a constant operand is synthesized into an extra
    /// scratch slab, and the output slot is written directly. The closure
    /// must not touch the arena itself (`with_active_arena` does not
    /// tolerate reentrancy — see the borrow-flag notes on the tape
    /// accessors).
    pub(crate) fn unary_op_scratch(
        x: &Self,
        slabs: usize,
        f: impl FnOnce(&[F], &mut [F], &mut [F]),
    ) -> Self {
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let deg = arena.degree();
            let idx = arena.allocate();
            if x.index == CONSTANT {
                let (_, slot, scratch) = arena.split_output(idx, slabs + 1);
                let (fs, synth) = scratch.split_at_mut(slabs * deg);
                synth[0] = x.value;
                f(&*synth, slot, fs);
                TaylorDyn {
                    value: slot[0],
                    index: idx,
                }
            } else {
                let (before, slot, fs) = arena.split_output(idx, slabs);
                let start = x.index as usize * deg;
                f(&before[start..start + deg], slot, fs);
                TaylorDyn {
                    value: slot[0],
                    index: idx,
                }
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

        Self::binary_op_scratch(x, y, 0, |a, b, c, _| f(a, b, c))
    }

    /// Like [`binary_op`](Self::binary_op) with zeroed arena scratch, under
    /// the same one-borrow-split contract as
    /// [`unary_op_scratch`](Self::unary_op_scratch). Constant operands each
    /// synthesize into their own extra slab.
    pub(crate) fn binary_op_scratch(
        x: &Self,
        y: &Self,
        slabs: usize,
        f: impl FnOnce(&[F], &[F], &mut [F], &mut [F]),
    ) -> Self {
        // The both-CONSTANT case is fast-pathed by `binary_op` before this
        // runs; reaching here with both constant is correct, just slower.
        with_active_arena(|arena: &mut TaylorArena<F>| {
            let deg = arena.degree();
            let idx = arena.allocate();
            let const_slabs = usize::from(x.index == CONSTANT) + usize::from(y.index == CONSTANT);
            let (before, slot, scratch) = arena.split_output(idx, slabs + const_slabs);
            let (fs, synth) = scratch.split_at_mut(slabs * deg);
            let mut synth_chunks = synth.chunks_mut(deg);
            let a: &[F] = if x.index == CONSTANT {
                let s = synth_chunks.next().expect("const slab reserved");
                s[0] = x.value;
                &*s
            } else {
                let start = x.index as usize * deg;
                &before[start..start + deg]
            };
            let b: &[F] = if y.index == CONSTANT {
                let s = synth_chunks.next().expect("const slab reserved");
                s[0] = y.value;
                &*s
            } else {
                let start = y.index as usize * deg;
                &before[start..start + deg]
            };
            f(a, b, slot, fs);
            TaylorDyn {
                value: slot[0],
                index: idx,
            }
        })
    }

    // ── Elemental methods ──

    taylor_dyn_elementals! {
        /// Reciprocal (1/x).
        recip => taylor_recip / 0;
        /// Square root.
        sqrt => taylor_sqrt / 0;
        /// Cube root.
        cbrt => taylor_cbrt / 2;
        /// Natural exponential (e^x).
        exp => taylor_exp / 0;
        /// Base-2 exponential (2^x).
        exp2 => taylor_exp2 / 1;
        /// e^x - 1, accurate near zero.
        exp_m1 => taylor_exp_m1 / 0;
        /// Natural logarithm.
        ln => taylor_ln / 0;
        /// Base-2 logarithm.
        log2 => taylor_log2 / 0;
        /// Base-10 logarithm.
        log10 => taylor_log10 / 0;
        /// ln(1+x), accurate near zero.
        ln_1p => taylor_ln_1p / 1;
        /// Tangent.
        tan => taylor_tan / 1;
        /// Arcsine.
        asin => taylor_asin / 2;
        /// Arccosine.
        acos => taylor_acos / 2;
        /// Arctangent.
        atan => taylor_atan / 2;
        /// Hyperbolic tangent.
        tanh => taylor_tanh / 1;
        /// Inverse hyperbolic sine.
        asinh => taylor_asinh / 2;
        /// Inverse hyperbolic cosine.
        acosh => taylor_acosh / 2;
        /// Inverse hyperbolic tangent.
        atanh => taylor_atanh / 2;
    }

    /// Integer power.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self::unary_op_scratch(&self, 2, |a, c, s| {
            let (s1, s2) = s.split_at_mut(c.len());
            taylor_ops::taylor_powi(a, n, c, s1, s2);
        })
    }

    /// Floating-point power.
    #[inline]
    pub fn powf(self, n: Self) -> Self {
        let b = n.coeffs();
        Self::unary_op(&self, |a, c| {
            let deg = c.len();
            let mut s1 = vec![F::zero(); deg];
            let mut s2 = vec![F::zero(); deg];
            taylor_ops::taylor_powf(a, &b, c, &mut s1, &mut s2);
        })
    }

    /// Logarithm with given base.
    #[inline]
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    /// Sine.
    #[inline]
    pub fn sin(self) -> Self {
        Self::unary_op_scratch(&self, 1, |a, c, co| taylor_ops::taylor_sin_cos(a, c, co))
    }

    /// Cosine.
    #[inline]
    pub fn cos(self) -> Self {
        Self::unary_op_scratch(&self, 1, |a, c, s| taylor_ops::taylor_sin_cos(a, s, c))
    }

    /// Simultaneous sine and cosine.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let a = self.coeffs();
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

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let b = other.coeffs();
        Self::unary_op(&self, |a, c| {
            let n = c.len();
            let mut s1 = vec![F::zero(); n];
            let mut s2 = vec![F::zero(); n];
            let mut s3 = vec![F::zero(); n];
            taylor_ops::taylor_atan2(a, &b, c, &mut s1, &mut s2, &mut s3);
        })
    }

    /// Hyperbolic sine.
    #[inline]
    pub fn sinh(self) -> Self {
        Self::unary_op_scratch(&self, 1, |a, c, ch| taylor_ops::taylor_sinh_cosh(a, c, ch))
    }

    /// Hyperbolic cosine.
    #[inline]
    pub fn cosh(self) -> Self {
        Self::unary_op_scratch(&self, 1, |a, c, sh| taylor_ops::taylor_sinh_cosh(a, sh, c))
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::unary_op(&self, |a, c| {
            // Use first nonzero coefficient's sign to determine the branch direction
            // at zero, avoiding signum(+0.0) = 0 which would annihilate the jet.
            let sign = if a[0] != F::zero() {
                a[0].signum()
            } else if let Some(k) = (1..a.len()).find(|&k| a[k] != F::zero()) {
                a[k].signum()
            } else {
                F::one()
            };
            for k in 0..c.len() {
                c[k] = a[k] * sign;
            }
        })
    }

    /// Sign function (zero derivative).
    #[inline]
    pub fn signum(self) -> Self {
        TaylorDyn::constant(self.value.signum())
    }

    /// Floor (zero derivative).
    #[inline]
    pub fn floor(self) -> Self {
        TaylorDyn::constant(self.value.floor())
    }

    /// Ceiling (zero derivative).
    #[inline]
    pub fn ceil(self) -> Self {
        TaylorDyn::constant(self.value.ceil())
    }

    /// Round to nearest integer (zero derivative).
    #[inline]
    pub fn round(self) -> Self {
        TaylorDyn::constant(self.value.round())
    }

    /// Truncate toward zero (zero derivative).
    #[inline]
    pub fn trunc(self) -> Self {
        TaylorDyn::constant(self.value.trunc())
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        Self::unary_op(&self, |a, c| {
            c[0] = a[0].fract();
            c[1..].copy_from_slice(&a[1..]);
        })
    }

    /// Euclidean distance: sqrt(self^2 + other^2).
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        Self::binary_op_scratch(&self, &other, 2, |a, b, c, s| {
            let (s1, s2) = s.split_at_mut(c.len());
            taylor_ops::taylor_hypot(a, b, c, s1, s2);
        })
    }

    /// Maximum of two values.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        // NaN guard: return the non-NaN argument (IEEE 754 fmax semantics)
        if self.value >= other.value || other.value.is_nan() {
            self
        } else {
            other
        }
    }

    /// Minimum of two values.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        // NaN guard: return the non-NaN argument (IEEE 754 fmin semantics)
        if self.value <= other.value || other.value.is_nan() {
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

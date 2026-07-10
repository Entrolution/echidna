use num_traits::Float;

/// Parameters controlling convergence checks.
#[derive(Debug, Clone)]
pub struct ConvergenceParams<F> {
    /// Maximum number of iterations (default: 100).
    pub max_iter: usize,
    /// Gradient norm tolerance: stop when `||g|| < grad_tol` (default: 1e-8).
    pub grad_tol: F,
    /// Step size tolerance: stop when `||x_{k+1} - x_k|| < step_tol` (default: 1e-12).
    pub step_tol: F,
    /// Function change tolerance: stop when `|f_{k+1} - f_k| < func_tol` (default: 0, disabled).
    pub func_tol: F,
}

impl Default for ConvergenceParams<f64> {
    fn default() -> Self {
        ConvergenceParams {
            max_iter: 100,
            grad_tol: 1e-8,
            step_tol: 1e-12,
            func_tol: 0.0,
        }
    }
}

impl Default for ConvergenceParams<f32> {
    fn default() -> Self {
        ConvergenceParams {
            max_iter: 100,
            grad_tol: 1e-5,
            step_tol: 1e-7,
            func_tol: 0.0,
        }
    }
}

/// Compute the L2 norm of a vector.
///
/// For `len() >= KAHAN_THRESHOLD`, uses Neumaier/Kahan compensated
/// summation. Naive recursive summation of `len` terms accumulates
/// `O(len·eps)` relative error in the worst case; at `len = 10^4`
/// (f32) or `10^{13}` (f64) the ULP-level noise can leak into the
/// convergence test and make the optimizer oscillate. Kahan drops
/// the error to `O(eps)` independent of `len`.
pub fn norm<F: Float>(v: &[F]) -> F {
    kahan_sum(v.iter().map(|&x| x * x)).sqrt()
}

/// Compute the dot product of two vectors.
pub fn dot<F: Float>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    kahan_sum(a.iter().zip(b.iter()).map(|(&x, &y)| x * y))
}

/// The shared post-step convergence gate: gradient norm, then step size,
/// then relative function change — the order every solver used.
///
/// Relative func_tol: absolute `|f_prev - f_val| < tol` is scale-blind — a
/// tolerance of 1e-8 means ULP-precision on large-magnitude objectives
/// (|f| ≈ 1e8) and impossibly tight on tiny ones. Scaling by `(1 + |f|)`
/// makes the criterion track the problem. NOT for the solvers' pre-loop
/// gradient-only checks: with no step taken, `f_prev == f_val` would fire
/// the function-change predicate spuriously.
pub(crate) fn check_convergence<F: Float>(
    grad_norm: F,
    step_norm: F,
    f_prev: F,
    f_val: F,
    p: &ConvergenceParams<F>,
) -> Option<crate::result::TerminationReason> {
    use crate::result::TerminationReason;
    if grad_norm < p.grad_tol {
        return Some(TerminationReason::GradientNorm);
    }
    if step_norm < p.step_tol {
        return Some(TerminationReason::StepSize);
    }
    if p.func_tol > F::zero() && (f_prev - f_val).abs() < p.func_tol * (F::one() + f_val.abs()) {
        return Some(TerminationReason::FunctionChange);
    }
    None
}

/// Threshold above which compensated summation beats naive summation.
/// Below this, naive summation's runtime advantage dominates and the
/// precision gap is negligible.
const KAHAN_THRESHOLD: usize = 64;

/// Neumaier's improved Kahan summation. Handles arbitrary input
/// magnitudes (unlike plain Kahan, which struggles when a term is
/// larger than the running sum). Falls back to naive summation for
/// short sequences where the overhead isn't worth it.
#[inline]
fn kahan_sum<F: Float, I: Iterator<Item = F>>(iter: I) -> F {
    let mut it = iter;
    let mut s = F::zero();
    let mut c = F::zero();
    let mut n = 0usize;
    // Gather a small prefix to decide whether to use compensated summation.
    let mut prefix: [F; KAHAN_THRESHOLD] = [F::zero(); KAHAN_THRESHOLD];
    for slot in prefix.iter_mut() {
        if let Some(x) = it.next() {
            *slot = x;
            n += 1;
        } else {
            break;
        }
    }
    if n < KAHAN_THRESHOLD {
        // Short case: naive accumulation. Cheaper and precision loss is
        // bounded by `n·eps` — negligible for n < 64.
        for &x in prefix.iter().take(n) {
            s = s + x;
        }
        return s;
    }
    // Long case: Neumaier compensated summation.
    for x in prefix.iter().copied().chain(it) {
        let t = s + x;
        if s.abs() >= x.abs() {
            c = c + ((s - t) + x);
        } else {
            c = c + ((x - t) + s);
        }
        s = t;
    }
    s + c
}

#[cfg(test)]
mod tests {
    use super::*;

    // M36: Kahan summation — very long vectors should yield tight norms.
    #[test]
    fn m36_norm_kahan_tight_for_long_vector() {
        // 10_000 copies of 1.0 — exact norm is sqrt(10_000) = 100.0.
        let v: Vec<f64> = (0..10_000).map(|_| 1.0).collect();
        let n = norm(&v);
        assert!(
            (n - 100.0).abs() < 1e-12,
            "norm of 10k ones not near 100: got {}",
            n
        );
    }

    #[test]
    fn m36_norm_short_vector_still_works() {
        let v = vec![3.0_f64, 4.0];
        let n = norm(&v);
        assert!((n - 5.0).abs() < 1e-15);
    }
}

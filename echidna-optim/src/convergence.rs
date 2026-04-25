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
    for &x in prefix.iter() {
        let t = s + x;
        if s.abs() >= x.abs() {
            c = c + ((s - t) + x);
        } else {
            c = c + ((x - t) + s);
        }
        s = t;
    }
    for x in it {
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

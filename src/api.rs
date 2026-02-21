use crate::dual::Dual;
use crate::float::Float;
use crate::reverse::Reverse;
use crate::tape::{Tape, TapeGuard, TapeThreadLocal};

#[cfg(feature = "bytecode")]
use crate::breverse::BReverse;
#[cfg(feature = "bytecode")]
use crate::bytecode_tape::{BtapeGuard, BtapeThreadLocal, BytecodeTape};

/// Compute the gradient of a scalar function `f : R^n → R` using reverse mode.
///
/// ```
/// let g = echidna::grad(|x: &[echidna::Reverse<f64>]| {
///     x[0] * x[0] + x[1] * x[1]
/// }, &[3.0, 4.0]);
/// assert!((g[0] - 6.0).abs() < 1e-10);
/// assert!((g[1] - 8.0).abs() < 1e-10);
/// ```
pub fn grad<F: Float + TapeThreadLocal>(
    f: impl FnOnce(&[Reverse<F>]) -> Reverse<F>,
    x: &[F],
) -> Vec<F> {
    let n = x.len();
    let mut tape = Tape::with_capacity(n * 10);

    // Create input variables.
    let inputs: Vec<Reverse<F>> = x
        .iter()
        .map(|&val| {
            let (idx, v) = tape.new_variable(val);
            Reverse::from_tape(v, idx)
        })
        .collect();

    let _guard = TapeGuard::new(&mut tape);
    let output = f(&inputs);

    // Run reverse sweep.
    let adjoints = tape.reverse(output.index);

    // Extract gradients for input variables (indices 0..n).
    (0..n).map(|i| adjoints[i]).collect()
}

/// Jacobian-vector product (forward mode): `(f(x), J·v)`.
///
/// Evaluates `f` at `x` and computes the directional derivative in direction `v`.
pub fn jvp<F: Float>(f: impl Fn(&[Dual<F>]) -> Vec<Dual<F>>, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(x.len(), v.len(), "x and v must have the same length");
    let inputs: Vec<Dual<F>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Dual::new(xi, vi))
        .collect();
    let outputs = f(&inputs);
    let values = outputs.iter().map(|d| d.re).collect();
    let tangents = outputs.iter().map(|d| d.eps).collect();
    (values, tangents)
}

/// Vector-Jacobian product (reverse mode): `(f(x), wᵀ·J)`.
///
/// Evaluates `f` at `x` and computes the adjoint product with weights `w`.
pub fn vjp<F: Float + TapeThreadLocal>(
    f: impl FnOnce(&[Reverse<F>]) -> Vec<Reverse<F>>,
    x: &[F],
    w: &[F],
) -> (Vec<F>, Vec<F>) {
    let n = x.len();
    let mut tape = Tape::with_capacity(n * 10);

    let inputs: Vec<Reverse<F>> = x
        .iter()
        .map(|&val| {
            let (idx, v) = tape.new_variable(val);
            Reverse::from_tape(v, idx)
        })
        .collect();

    let _guard = TapeGuard::new(&mut tape);
    let outputs = f(&inputs);

    assert_eq!(
        outputs.len(),
        w.len(),
        "output length must match weight vector length"
    );

    let values: Vec<F> = outputs.iter().map(|r| r.value).collect();

    // Seed adjoints with weights.
    let seeds: Vec<(u32, F)> = outputs
        .iter()
        .zip(w.iter())
        .filter(|(r, _)| r.index != crate::tape::CONSTANT)
        .map(|(r, &wi)| (r.index, wi))
        .collect();
    let adjoints = tape.reverse_seeded(&seeds);

    let grad: Vec<F> = (0..n).map(|i| adjoints[i]).collect();
    (values, grad)
}

/// Compute the full Jacobian of `f : R^n → R^m` using forward mode.
///
/// Returns `(f(x), J)` where `J[i][j] = ∂f_i/∂x_j`.
pub fn jacobian<F: Float>(
    f: impl Fn(&[Dual<F>]) -> Vec<Dual<F>>,
    x: &[F],
) -> (Vec<F>, Vec<Vec<F>>) {
    let n = x.len();

    // First pass to get output dimension and values.
    let const_inputs: Vec<Dual<F>> = x.iter().map(|&xi| Dual::constant(xi)).collect();
    let const_outputs = f(&const_inputs);
    let m = const_outputs.len();
    let values: Vec<F> = const_outputs.iter().map(|d| d.re).collect();

    // One forward pass per input variable.
    let mut jac = vec![vec![F::zero(); n]; m];
    for j in 0..n {
        let inputs: Vec<Dual<F>> = x
            .iter()
            .enumerate()
            .map(|(k, &xi)| {
                if k == j {
                    Dual::variable(xi)
                } else {
                    Dual::constant(xi)
                }
            })
            .collect();
        let outputs = f(&inputs);
        for (row, out) in jac.iter_mut().zip(outputs.iter()) {
            row[j] = out.eps;
        }
    }

    (values, jac)
}

/// Record a function into a [`BytecodeTape`] that can be re-evaluated at
/// different inputs without re-recording.
///
/// Returns the tape and the output value from the recording pass.
///
/// # Limitations
///
/// The tape records one execution path. If `f` contains branches
/// (`if x > 0 { ... } else { ... }`), re-evaluating at inputs that take a
/// different branch produces **incorrect results**.
///
/// # Example
///
/// ```ignore
/// let (mut tape, val) = echidna::record(
///     |x| x[0] * x[0] + x[1] * x[1],
///     &[3.0, 4.0],
/// );
/// assert!((val - 25.0).abs() < 1e-10);
///
/// let g = tape.gradient(&[3.0, 4.0]);
/// assert!((g[0] - 6.0).abs() < 1e-10);
/// assert!((g[1] - 8.0).abs() < 1e-10);
/// ```
#[cfg(feature = "bytecode")]
pub fn record<F: Float + BtapeThreadLocal>(
    f: impl FnOnce(&[BReverse<F>]) -> BReverse<F>,
    x: &[F],
) -> (BytecodeTape<F>, F) {
    let n = x.len();
    let mut tape = BytecodeTape::with_capacity(n * 10);

    // Register inputs.
    let inputs: Vec<BReverse<F>> = x
        .iter()
        .map(|&val| {
            let idx = tape.new_input(val);
            BReverse::from_tape(val, idx)
        })
        .collect();

    let _guard = BtapeGuard::new(&mut tape);
    let output = f(&inputs);

    tape.set_output(output.index);
    let value = output.value;
    (tape, value)
}

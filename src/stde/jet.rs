use crate::bytecode_tape::BytecodeTape;
use crate::taylor::Taylor;
use crate::Float;

/// Propagate direction `v` through tape using second-order Taylor mode.
///
/// Constructs `Taylor<F, 3>` inputs where `input[i] = [x[i], v[i], 0]`,
/// runs `forward_tangent`, and extracts the output coefficients.
///
/// Returns `(f(x), nabla_f . v, v^T H v / 2)`.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`.
pub fn taylor_jet_2nd<F: Float>(tape: &BytecodeTape<F>, x: &[F], v: &[F]) -> (F, F, F) {
    let mut buf = Vec::new();
    taylor_jet_2nd_with_buf(tape, x, v, &mut buf)
}

/// Like [`taylor_jet_2nd`] but reuses a caller-provided buffer to avoid
/// reallocation across multiple calls.
///
/// # Panics
///
/// Panics if `x.len()` or `v.len()` does not match `tape.num_inputs()`.
pub fn taylor_jet_2nd_with_buf<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    v: &[F],
    buf: &mut Vec<Taylor<F, 3>>,
) -> (F, F, F) {
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");
    assert_eq!(v.len(), n, "v.len() must match tape.num_inputs()");

    let inputs: Vec<Taylor<F, 3>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Taylor::new([xi, vi, F::zero()]))
        .collect();

    tape.forward_tangent(&inputs, buf);

    let out = buf[tape.output_index()];
    (out.coeffs[0], out.coeffs[1], out.coeffs[2])
}

/// Evaluate multiple directions through the tape.
///
/// Returns `(value, first_order, second_order)` where:
/// - `value` = f(x)
/// - `first_order[s]` = nabla_f . v_s  (directional first derivative)
/// - `second_order[s]` = v_s^T H v_s / 2  (half directional second derivative)
///
/// # Panics
///
/// Panics if any direction's length does not match `tape.num_inputs()`.
pub fn directional_derivatives<F: Float>(
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> (F, Vec<F>, Vec<F>) {
    let mut buf = Vec::new();
    let mut first_order = Vec::with_capacity(directions.len());
    let mut second_order = Vec::with_capacity(directions.len());
    let mut value = F::zero();

    for v in directions {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        first_order.push(c1);
        second_order.push(c2);
    }

    (value, first_order, second_order)
}

use super::types::{EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::diffop::SparseSamplingDistribution;
use crate::taylor_dyn::{TaylorArenaLocal, TaylorDyn, TaylorDynGuard};
use crate::Float;

/// Sparse STDE: estimate `Lu(x)` by sampling sparse k-jets.
///
/// Each sample index corresponds to an entry in `dist`. For each sample,
/// one forward pushforward of a sparse k-jet is performed.
///
/// The per-sample estimator is: `sign(C_α) · Z · prefactor · coeffs[k]`
/// where `Z = dist.normalization()` (paper Eq 13).
///
/// Paper reference: Eq 13, Section 4.3.
///
/// # Sampling precondition (unbiasedness)
///
/// The returned mean is an **unbiased** estimate of `Lu(x)` only when
/// `sampled_indices` are drawn from `dist` with probability proportional to
/// `|C_α|` — i.e. via [`SparseSamplingDistribution::sample_index`], which
/// importance-samples on the cumulative `|C_α|` weights. The per-sample
/// factor `Z = Σ|C_α|` cancels the `|C_α|/Z` draw probability so that
/// `E[sample] = Σ_α C_α · raw_α = Lu(x)`.
///
/// Passing indices any other way biases the estimate. In particular a
/// uniform enumeration `(0..dist.len())` is unbiased **only when every
/// `|C_α|` is equal** (e.g. a Laplacian or `DiffOp::diagonal` operator); for
/// mixed-coefficient operators it converges to `(Z/M)·Σ_α sign(C_α)·raw_α`,
/// which does not equal `Lu(x)`.
///
/// [`SparseSamplingDistribution::sample_index`]: crate::diffop::SparseSamplingDistribution::sample_index
///
/// # Panics
///
/// Panics if `sampled_indices` is empty or any index is out of bounds.
pub fn stde_sparse<F: Float + TaylorArenaLocal>(
    tape: &BytecodeTape<F>,
    x: &[F],
    dist: &SparseSamplingDistribution<F>,
    sampled_indices: &[usize],
) -> EstimatorResult<F> {
    assert!(
        !sampled_indices.is_empty(),
        "sampled_indices must not be empty"
    );
    let n = tape.num_inputs();
    assert_eq!(x.len(), n, "x.len() must match tape.num_inputs()");

    let order = dist.jet_order() + 1;
    let z = dist.normalization();
    let _guard = TaylorDynGuard::<F>::new(order);

    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();
    let mut buf: Vec<TaylorDyn<F>> = Vec::new();

    for &idx in sampled_indices {
        let entry = dist.entry(idx);

        // coeffs[0] = x[i] for all inputs; coeffs[slot] = 1/slot! per
        // entry.input_coeffs for the active variables.
        let inputs = crate::taylor_dyn::seed_taylor_dyn_jets(x, order, entry.input_coeffs());

        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];

        // Extract: raw = coeffs[output_coeff_index] * extraction_prefactor
        let raw = out_coeffs[entry.output_coeff_index()] * entry.extraction_prefactor();
        // Sample = sign * Z * raw
        let sample = entry.sign() * z * raw;
        acc.update(sample);
    }

    acc.into_result(value)
}

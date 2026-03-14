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

        // Build TaylorDyn inputs: coeffs[0] = x[i] for all, then set
        // coeffs[slot] = 1/slot! for active variables per entry.input_coeffs
        let inputs: Vec<TaylorDyn<F>> = (0..n)
            .map(|i| {
                let mut coeffs = vec![F::zero(); order];
                coeffs[0] = x[i];
                for &(var, slot, inv_fact) in entry.input_coeffs() {
                    if var == i && slot < order {
                        coeffs[slot] = inv_fact;
                    }
                }
                TaylorDyn::from_coeffs(&coeffs)
            })
            .collect();

        tape.forward_tangent(&inputs, &mut buf);

        let out_coeffs = buf[tape.output_index()].coeffs();
        value = out_coeffs[0];

        // Extract: raw = coeffs[output_coeff_index] * extraction_prefactor
        let raw = out_coeffs[entry.output_coeff_index()] * entry.extraction_prefactor();
        // Sample = sign * Z * raw
        let sample = entry.sign() * z * raw;
        acc.update(sample);
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: sampled_indices.len(),
    }
}

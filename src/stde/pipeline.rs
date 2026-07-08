use super::estimator::Estimator;
use super::jet::taylor_jet_2nd_with_buf;
use super::types::{EstimatorResult, WelfordAccumulator};
use crate::bytecode_tape::BytecodeTape;
use crate::Float;

/// Estimate a quantity using the given [`Estimator`] and Welford's online algorithm.
///
/// Evaluates the tape at `x` for each direction, computes the estimator's sample
/// from the Taylor jet, and aggregates with running mean and variance.
///
/// Non-finite samples are skipped and excluded from `num_samples`; if every
/// sample is non-finite the estimate is NaN.
///
/// # Panics
///
/// Panics if `directions` is empty or any direction's length does not match
/// `tape.num_inputs()`.
pub fn estimate<F: Float>(
    estimator: &impl Estimator<F>,
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");

    let mut buf = Vec::new();
    let mut value = F::zero();
    let mut acc = WelfordAccumulator::new();

    for v in directions.iter() {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        acc.update(estimator.sample(c0, c1, c2));
    }

    let (estimate, sample_variance, standard_error) = acc.finalize();

    EstimatorResult {
        value,
        estimate,
        sample_variance,
        standard_error,
        num_samples: acc.contributing(),
    }
}

/// Estimate a quantity using importance-weighted samples (West's 1979 algorithm).
///
/// Each direction `directions[s]` has an associated weight `weights[s]`.
/// The weighted mean is `Σ(w_s * sample_s) / Σ(w_s)` and the variance uses
/// the reliability-weight Bessel correction: `M2 / (W - W2/W)` where
/// `W = Σw_s` and `W2 = Σw_s²`.
///
/// Non-finite samples are skipped and excluded from `num_samples` (a NaN
/// or Inf sample carries no usable magnitude); zero-weight directions
/// contribute nothing to the statistics but still count as samples.
///
/// # Panics
///
/// Panics if `directions` is empty, `weights.len() != directions.len()`,
/// any weight is negative or NaN (West's algorithm requires non-negative
/// weights), or any direction's length does not match `tape.num_inputs()`.
pub fn estimate_weighted<F: Float>(
    estimator: &impl Estimator<F>,
    tape: &BytecodeTape<F>,
    x: &[F],
    directions: &[&[F]],
    weights: &[F],
) -> EstimatorResult<F> {
    assert!(!directions.is_empty(), "directions must not be empty");
    assert_eq!(
        weights.len(),
        directions.len(),
        "weights.len() must match directions.len()"
    );

    let mut buf = Vec::new();
    let mut value = F::zero();

    // West's (1979) weighted online algorithm
    let mut w_sum = F::zero();
    let mut w_sum2 = F::zero();
    let mut mean = F::zero();
    let mut m2 = F::zero();

    let mut nonfinite_skips = 0usize;
    for (k, v) in directions.iter().enumerate() {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        // A negative (or NaN) weight is a caller error, not sample data:
        // West's update assumes non-negative weights (w_sum must stay
        // positive and monotone). The positive-form predicate rejects NaN.
        let w = weights[k];
        assert!(
            w >= F::zero(),
            "estimate_weighted: weights must be non-negative and non-NaN"
        );
        // Skip zero-weight directions before evaluating the sample: a
        // direction that contributes nothing shouldn't affect the estimate
        // even if its sample would be non-finite.
        if w == F::zero() {
            continue;
        }
        let s = estimator.sample(c0, c1, c2);
        // Non-finite sample: skip and count (data condition, not an error).
        if !s.is_finite() {
            nonfinite_skips += 1;
            continue;
        }

        w_sum = w_sum + w;
        w_sum2 = w_sum2 + w * w;
        let delta = s - mean;
        mean = mean + (w / w_sum) * delta;
        let delta2 = s - mean;
        m2 = m2 + w * delta * delta2;
    }

    let n = directions.len();
    let denom = if w_sum > F::zero() {
        w_sum - w_sum2 / w_sum
    } else {
        F::zero()
    };
    let (sample_variance, standard_error) = if n > 1 && denom > F::zero() {
        let var = (m2 / denom).max(F::zero());
        // Effective sample size for weighted estimates: n_eff = w_sum^2 / w_sum2
        let n_eff = w_sum * w_sum / w_sum2;
        (var, (var / n_eff).sqrt())
    } else {
        (F::zero(), F::zero())
    };

    EstimatorResult {
        value,
        // Non-finite skips exhausting every weighted sample leave the
        // estimator with no information — surface NaN rather than the
        // accumulator-neutral 0. (All-zero weights, with nothing skipped,
        // keep the documented neutral-0 estimate.)
        estimate: if w_sum > F::zero() || nonfinite_skips == 0 {
            mean
        } else {
            F::nan()
        },
        sample_variance,
        standard_error,
        num_samples: n - nonfinite_skips,
    }
}

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
        num_samples: directions.len(),
    }
}

/// Estimate a quantity using importance-weighted samples (West's 1979 algorithm).
///
/// Each direction `directions[s]` has an associated weight `weights[s]`.
/// The weighted mean is `Σ(w_s * sample_s) / Σ(w_s)` and the variance uses
/// the reliability-weight Bessel correction: `M2 / (W - W2/W)` where
/// `W = Σw_s` and `W2 = Σw_s²`.
///
/// # Panics
///
/// Panics if `directions` is empty, `weights.len() != directions.len()`,
/// or any direction's length does not match `tape.num_inputs()`.
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

    for (k, v) in directions.iter().enumerate() {
        let (c0, c1, c2) = taylor_jet_2nd_with_buf(tape, x, v, &mut buf);
        value = c0;
        let s = estimator.sample(c0, c1, c2);
        assert!(s.is_finite(), "weighted estimator sample must be finite");
        let w = weights[k];
        if w == F::zero() {
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
        estimate: mean,
        sample_variance,
        standard_error,
        num_samples: n,
    }
}

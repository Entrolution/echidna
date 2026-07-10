use crate::Float;

/// Result of a stochastic estimation with sample statistics.
///
/// Contains the function value, the estimated operator value, and
/// sample statistics (variance, standard error) that quantify
/// estimator quality.
#[derive(Clone, Debug)]
pub struct EstimatorResult<F> {
    /// Function value f(x).
    pub value: F,
    /// Estimated operator value (e.g. Laplacian).
    pub estimate: F,
    /// Sample variance of the per-direction estimates.
    /// Zero when `num_samples == 1` (undefined, clamped to zero).
    pub sample_variance: F,
    /// Standard error of the mean: `sqrt(sample_variance / num_samples)`.
    pub standard_error: F,
    /// Number of finite (contributing) direction samples; non-finite
    /// samples are skipped and do not enter the estimate.
    pub num_samples: usize,
}

/// Result of a divergence estimation.
///
/// Separate from [`EstimatorResult`] because the function output is a vector
/// (`values: Vec<F>`) rather than a scalar (`value: F`).
#[derive(Clone, Debug)]
pub struct DivergenceResult<F> {
    /// Function output vector f(x).
    pub values: Vec<F>,
    /// Estimated divergence (trace of the Jacobian).
    pub estimate: F,
    /// Sample variance of per-direction estimates.
    pub sample_variance: F,
    /// Standard error of the mean.
    pub standard_error: F,
    /// Number of finite (contributing) direction samples; non-finite
    /// samples are skipped and do not enter the estimate.
    pub num_samples: usize,
}

/// Welford's online algorithm for incremental mean and variance.
pub(super) struct WelfordAccumulator<F> {
    mean: F,
    m2: F,
    count: usize,
}

impl<F: Float> WelfordAccumulator<F> {
    pub(super) fn new() -> Self {
        Self {
            mean: F::zero(),
            m2: F::zero(),
            count: 0,
        }
    }

    /// Accumulate one sample. Non-finite samples are skipped rather than
    /// poisoning the running mean and variance: a NaN or Inf sample carries
    /// no usable magnitude information, and estimators report how many
    /// samples actually contributed via `num_samples`.
    pub(super) fn update(&mut self, sample: F) {
        if !sample.is_finite() {
            return;
        }
        self.count += 1;
        let k1 = F::from(self.count).unwrap();
        let delta = sample - self.mean;
        self.mean = self.mean + delta / k1;
        let delta2 = sample - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    /// Number of samples that contributed to the running statistics
    /// (excludes skipped non-finite samples).
    pub(super) fn contributing(&self) -> usize {
        self.count
    }

    /// Package the finalized statistics into an [`EstimatorResult`].
    ///
    /// For the plain estimators only — variants that transform the raw mean
    /// (variance rescaling, exact-trace offsets) call [`finalize`](Self::finalize)
    /// directly so the transform stays visible at the estimator.
    pub(super) fn into_result(self, value: F) -> EstimatorResult<F> {
        let (estimate, sample_variance, standard_error) = self.finalize();
        EstimatorResult {
            value,
            estimate,
            sample_variance,
            standard_error,
            num_samples: self.contributing(),
        }
    }
    /// Finalize into `(mean, sample_variance, standard_error)`.
    ///
    /// With zero contributing samples the estimator has no information and
    /// the mean is NaN (not the accumulator's neutral 0.0, which would
    /// read as a confident estimate of zero).
    pub(super) fn finalize(&self) -> (F, F, F) {
        if self.count == 0 {
            return (F::nan(), F::zero(), F::zero());
        }
        let nf = F::from(self.count).unwrap();
        if self.count > 1 {
            let var = (self.m2 / (nf - F::one())).max(F::zero());
            (self.mean, var, (var / nf).sqrt())
        } else {
            (self.mean, F::zero(), F::zero())
        }
    }
}

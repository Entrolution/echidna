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
    /// Number of direction samples used.
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
    /// Number of direction samples used.
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

    /// # Precondition
    ///
    /// `sample` must be finite. A NaN or Inf sample will poison the running
    /// mean and variance, producing NaN for all subsequent updates.
    pub(super) fn update(&mut self, sample: F) {
        debug_assert!(
            sample.is_finite(),
            "WelfordAccumulator::update: sample must be finite"
        );
        self.count += 1;
        let k1 = F::from(self.count).unwrap();
        let delta = sample - self.mean;
        self.mean = self.mean + delta / k1;
        let delta2 = sample - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    pub(super) fn finalize(&self) -> (F, F, F) {
        let nf = F::from(self.count).unwrap();
        if self.count > 1 {
            let var = (self.m2 / (nf - F::one())).max(F::zero());
            (self.mean, var, (var / nf).sqrt())
        } else {
            (self.mean, F::zero(), F::zero())
        }
    }
}

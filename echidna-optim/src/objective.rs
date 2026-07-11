use echidna::{BytecodeTape, Float};

/// Trait for optimization objectives.
///
/// Implementors provide function evaluation and gradient computation.
/// Methods take `&mut self` to allow caching, eval counting, and internal buffers.
pub trait Objective<F: num_traits::Float> {
    /// Number of input variables.
    fn dim(&self) -> usize;

    /// Evaluate the objective and its gradient at `x`.
    ///
    /// Returns `(f(x), ∇f(x))`.
    fn eval_grad(&mut self, x: &[F]) -> (F, Vec<F>);

    /// Evaluate the objective, gradient, and full Hessian at `x`.
    ///
    /// Returns `(f(x), ∇f(x), H(x))` where `H[i][j] = ∂²f/∂x_i∂x_j`.
    ///
    /// Default implementation panics. Only solvers that need the Hessian call this.
    fn eval_hessian(&mut self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let _ = x;
        unimplemented!("eval_hessian not implemented for this objective")
    }

    /// Compute the Hessian-vector product H(x)·v.
    ///
    /// Returns `(∇f(x), H(x)·v)`.
    ///
    /// Default implementation panics. Only solvers that need HVP call this.
    fn hvp(&mut self, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
        let _ = (x, v);
        unimplemented!("hvp not implemented for this objective")
    }
}

/// Adapter wrapping a [`BytecodeTape`] as an [`Objective`].
pub struct TapeObjective<F: Float> {
    tape: BytecodeTape<F>,
    func_evals: usize,
}

impl<F: Float> TapeObjective<F> {
    /// Create a new `TapeObjective` from a recorded tape.
    ///
    /// ```
    /// use echidna_optim::{lbfgs, LbfgsConfig, TapeObjective};
    ///
    /// let (tape, _) = echidna::record(|x| x[0] * x[0] + x[1] * x[1], &[1.0_f64, 1.0]);
    /// let mut objective = TapeObjective::new(tape);
    /// let result = lbfgs(&mut objective, &[1.0, 1.0], &LbfgsConfig::default());
    /// assert!(result.x.iter().all(|&xi| xi.abs() < 1e-6));
    /// ```
    pub fn new(tape: BytecodeTape<F>) -> Self {
        TapeObjective {
            tape,
            func_evals: 0,
        }
    }

    /// Number of function evaluations performed so far.
    pub fn func_evals(&self) -> usize {
        self.func_evals
    }
}

impl<F: Float> Objective<F> for TapeObjective<F> {
    fn dim(&self) -> usize {
        self.tape.num_inputs()
    }

    fn eval_grad(&mut self, x: &[F]) -> (F, Vec<F>) {
        self.func_evals += 1;
        let grad = self.tape.gradient(x);
        let value = self.tape.output_value();
        (value, grad)
    }

    fn eval_hessian(&mut self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        self.func_evals += 1;
        self.tape.hessian(x)
    }

    fn hvp(&mut self, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
        self.func_evals += 1;
        self.tape.hvp(x, v)
    }
}

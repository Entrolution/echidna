use crate::float::Float;
use crate::taylor::Taylor;

impl<F: Float> super::BytecodeTape<F> {
    /// Forward-reverse Taylor pass for gradient + higher-order directional adjoints.
    ///
    /// Builds Taylor inputs `x_i(t) = x_i + v_i * t` (with zero higher coefficients),
    /// runs `forward_tangent`, then `reverse_tangent` to get Taylor-valued adjoints.
    ///
    /// Returns `(output, adjoints)` where:
    /// - `output` is the Taylor expansion of `f` along direction `v`
    /// - `adjoints[i].coeff(0)` = `∂f/∂x_i` (gradient)
    /// - `adjoints[i].coeff(1)` = `Σ_j (∂²f/∂x_i∂x_j) v_j` (HVP)
    /// - `adjoints[i].derivative(k)` = k-th order directional adjoint
    ///
    /// For K=2, the HVP component is equivalent to [`hvp`](Self::hvp).
    /// For K≥3, yields additional higher-order information in the same pass.
    ///
    /// Like [`hvp`](Self::hvp), takes `&self` and does not call `forward(x)`
    /// before the Taylor pass. Custom ops will use primal values from recording time.
    pub fn taylor_grad<const K: usize>(
        &self,
        x: &[F],
        v: &[F],
    ) -> (Taylor<F, K>, Vec<Taylor<F, K>>) {
        let mut fwd_buf = Vec::new();
        let mut adj_buf = Vec::new();
        self.taylor_grad_with_buf(x, v, &mut fwd_buf, &mut adj_buf)
    }

    /// Like [`taylor_grad`](Self::taylor_grad) but reuses caller-provided buffers
    /// to avoid allocation on repeated calls.
    pub fn taylor_grad_with_buf<const K: usize>(
        &self,
        x: &[F],
        v: &[F],
        fwd_buf: &mut Vec<Taylor<F, K>>,
        adj_buf: &mut Vec<Taylor<F, K>>,
    ) -> (Taylor<F, K>, Vec<Taylor<F, K>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v.len(), n, "wrong number of directions");

        // Build Taylor inputs: x_i(t) = x_i + v_i * t
        let taylor_inputs: Vec<Taylor<F, K>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| {
                let mut coeffs = [F::zero(); K];
                coeffs[0] = xi;
                if K > 1 {
                    coeffs[1] = vi;
                }
                Taylor::new(coeffs)
            })
            .collect();

        self.forward_tangent(&taylor_inputs, fwd_buf);
        let output = fwd_buf[self.output_index as usize];
        self.reverse_tangent(fwd_buf, adj_buf);

        (output, adj_buf[..n].to_vec())
    }

    // ── ODE Taylor integration ──

    /// Compute the Taylor expansion of the ODE solution `y(t)` to order K.
    ///
    /// Given a tape representing the right-hand side `f: R^n → R^n` of the ODE
    /// `y' = f(y)`, and an initial condition `y(0) = y0`, computes the Taylor
    /// coefficients `y_0, y_1, ..., y_{K-1}` such that
    /// `y(t) ≈ y_0 + y_1·t + y_2·t² + ... + y_{K-1}·t^{K-1}`.
    ///
    /// The tape must have `num_outputs == num_inputs` (autonomous ODE: f maps R^n → R^n).
    ///
    /// Returns one `Taylor<F, K>` per state variable. Use [`Taylor::eval_at`] to
    /// evaluate at a step size `h`, or inspect coefficients for error estimation.
    pub fn ode_taylor_step<const K: usize>(&self, y0: &[F]) -> Vec<Taylor<F, K>> {
        let mut buf = Vec::new();
        self.ode_taylor_step_with_buf(y0, &mut buf)
    }

    /// Like [`ode_taylor_step`](Self::ode_taylor_step) but reuses a caller-provided
    /// buffer to avoid allocation on repeated calls.
    pub fn ode_taylor_step_with_buf<const K: usize>(
        &self,
        y0: &[F],
        buf: &mut Vec<Taylor<F, K>>,
    ) -> Vec<Taylor<F, K>> {
        let n = self.num_inputs as usize;
        assert_eq!(y0.len(), n, "y0 length must match num_inputs");
        assert_eq!(
            self.num_outputs(),
            n,
            "ODE tape must have num_outputs == num_inputs (f: R^n -> R^n)"
        );

        let out_indices = self.all_output_indices();

        let mut y_coeffs = vec![[F::zero(); K]; n];
        for i in 0..n {
            y_coeffs[i][0] = y0[i];
        }

        for k in 0..K - 1 {
            let inputs: Vec<Taylor<F, K>> = (0..n).map(|i| Taylor::new(y_coeffs[i])).collect();

            self.forward_tangent(&inputs, buf);

            let divisor = F::from(k + 1).unwrap();
            for i in 0..n {
                y_coeffs[i][k + 1] = buf[out_indices[i] as usize].coeff(k) / divisor;
            }
        }

        (0..n).map(|i| Taylor::new(y_coeffs[i])).collect()
    }
}

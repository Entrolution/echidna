use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::float::{Float, IsAllZero};
use crate::opcode::{self, OpCode, UNUSED};
use num_traits::Float as NumFloat;

impl<F: Float> super::BytecodeTape<F> {
    // ── Forward-over-reverse (second-order) ──

    /// Forward sweep with tangent-carrying numbers. Reads opcodes and constants
    /// from `self`, writing results into `buf`. Does not mutate the tape.
    ///
    /// Generic over `T: NumFloat` so it works with both `Dual<F>` and
    /// `DualVec<F, N>`.
    pub fn forward_tangent<T: NumFloat>(&self, inputs: &[T], buf: &mut Vec<T>) {
        self.forward_tangent_inner(inputs, buf, |i, a_t, b_t| {
            // First-order chain rule: evaluate on primals, convert partials to T.
            let [a_idx, cb_idx] = self.arg_indices[i];
            let a_primal = self.values[a_idx as usize];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let b_primal = b_idx_opt
                .map(|bi| self.values[bi as usize])
                .unwrap_or(F::zero());
            let result = self.custom_ops[cb_idx as usize].eval(a_primal, b_primal);
            let (da, db) = self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, result);
            let result_t = T::from(result).unwrap();
            let da_t = T::from(da).unwrap();
            let db_t = T::from(db).unwrap();
            let a_re_t = T::from(a_primal).unwrap();
            let b_re_t = T::from(b_primal).unwrap();
            result_t + da_t * (a_t - a_re_t) + db_t * (b_t - b_re_t)
        });
    }

    /// Forward sweep specialized for `Dual<F>`, calling [`CustomOp::eval_dual`]
    /// so that custom ops propagate tangent information for second-order derivatives.
    pub(super) fn forward_tangent_dual(&self, inputs: &[Dual<F>], buf: &mut Vec<Dual<F>>) {
        self.forward_tangent_inner(inputs, buf, |i, a_t, b_t| {
            let [_a_idx, cb_idx] = self.arg_indices[i];
            self.custom_ops[cb_idx as usize].eval_dual(a_t, b_t)
        });
    }

    /// Common forward-tangent loop. The `handle_custom` closure receives
    /// `(tape_index, a_value, b_value)` for custom op slots and returns
    /// the result to store.
    fn forward_tangent_inner<T: NumFloat>(
        &self,
        inputs: &[T],
        buf: &mut Vec<T>,
        handle_custom: impl Fn(usize, T, T) -> T,
    ) {
        assert_eq!(
            inputs.len(),
            self.num_inputs as usize,
            "wrong number of inputs"
        );

        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());

        let mut input_idx = 0usize;
        for i in 0..self.opcodes.len() {
            match self.opcodes[i] {
                OpCode::Input => {
                    buf[i] = inputs[input_idx];
                    input_idx += 1;
                }
                OpCode::Const => {
                    buf[i] = T::from(self.values[i]).unwrap();
                }
                OpCode::Custom => {
                    let [a_idx, _cb_idx] = self.arg_indices[i];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let a_t = buf[a_idx as usize];
                    let b_t = b_idx_opt.map(|bi| buf[bi as usize]).unwrap_or(T::zero());
                    buf[i] = handle_custom(i, a_t, b_t);
                }
                op => {
                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = buf[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    buf[i] = opcode::eval_forward(op, a, b);
                }
            }
        }
    }

    /// Reverse sweep with tangent-carrying adjoints. Uses values from
    /// [`forward_tangent`](Self::forward_tangent). Uses [`IsAllZero`] to
    /// safely skip zero adjoints without dropping tangent contributions.
    pub(super) fn reverse_tangent<T: NumFloat + IsAllZero>(
        &self,
        tangent_vals: &[T],
        buf: &mut Vec<T>,
    ) {
        self.reverse_tangent_inner(tangent_vals, buf, |i| {
            // First-order: convert primal-float partials to T.
            let [a_idx, cb_idx] = self.arg_indices[i];
            let a_primal = self.values[a_idx as usize];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let b_primal = b_idx_opt
                .map(|bi| self.values[bi as usize])
                .unwrap_or(F::zero());
            let r_primal = self.values[i];
            let (da, db) = self.custom_ops[cb_idx as usize].partials(a_primal, b_primal, r_primal);
            (T::from(da).unwrap(), T::from(db).unwrap())
        });
    }

    /// Reverse sweep specialized for `Dual<F>`, calling [`CustomOp::partials_dual`]
    /// so that custom op partials carry tangent information for second-order derivatives.
    pub(super) fn reverse_tangent_dual(&self, tangent_vals: &[Dual<F>], buf: &mut Vec<Dual<F>>) {
        self.reverse_tangent_inner(tangent_vals, buf, |i| {
            let [a_idx, cb_idx] = self.arg_indices[i];
            let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
            let a_dual = tangent_vals[a_idx as usize];
            let b_dual = b_idx_opt
                .map(|bi| tangent_vals[bi as usize])
                .unwrap_or(Dual::constant(F::zero()));
            let r_dual = tangent_vals[i];
            self.custom_ops[cb_idx as usize].partials_dual(a_dual, b_dual, r_dual)
        });
    }

    /// Common reverse-tangent loop. The `custom_partials` closure receives
    /// `tape_index` for custom op slots and returns `(da, db)` as T-valued partials.
    fn reverse_tangent_inner<T: NumFloat + IsAllZero>(
        &self,
        tangent_vals: &[T],
        buf: &mut Vec<T>,
        custom_partials: impl Fn(usize) -> (T, T),
    ) {
        let n = self.num_variables as usize;
        buf.clear();
        buf.resize(n, T::zero());
        buf[self.output_index as usize] = T::one();

        for i in (0..self.opcodes.len()).rev() {
            match self.opcodes[i] {
                OpCode::Input | OpCode::Const => continue,
                OpCode::Custom => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, _cb_idx] = self.arg_indices[i];
                    let b_idx_opt = self.custom_second_args.get(&(i as u32)).copied();
                    let (da_t, db_t) = custom_partials(i);
                    buf[a_idx as usize] = buf[a_idx as usize] + da_t * adj;
                    if let Some(bi) = b_idx_opt {
                        buf[bi as usize] = buf[bi as usize] + db_t * adj;
                    }
                }
                op => {
                    let adj = buf[i];
                    if adj.is_all_zero() {
                        continue;
                    }
                    buf[i] = T::zero();

                    let [a_idx, b_idx] = self.arg_indices[i];
                    let a = tangent_vals[a_idx as usize];
                    let b = if b_idx != UNUSED && op != OpCode::Powi {
                        tangent_vals[b_idx as usize]
                    } else if op == OpCode::Powi {
                        T::from(b_idx).unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    let r = tangent_vals[i];
                    let (da, db) = opcode::reverse_partials(op, a, b, r);

                    buf[a_idx as usize] = buf[a_idx as usize] + da * adj;
                    if b_idx != UNUSED && op != OpCode::Powi {
                        buf[b_idx as usize] = buf[b_idx as usize] + db * adj;
                    }
                }
            }
        }
    }

    /// Hessian-vector product via forward-over-reverse.
    ///
    /// Returns `(gradient, H·v)` where both are `Vec<F>` of length
    /// [`num_inputs`](Self::num_inputs). The tape is not mutated.
    pub fn hvp(&self, x: &[F], v: &[F]) -> (Vec<F>, Vec<F>) {
        let mut dual_vals = Vec::new();
        let mut adjoint_buf = Vec::new();
        self.hvp_with_buf(x, v, &mut dual_vals, &mut adjoint_buf)
    }

    /// Like [`hvp`](Self::hvp) but reuses caller-provided buffers to avoid
    /// allocation on repeated calls (e.g. inside [`hessian`](Self::hessian)).
    pub fn hvp_with_buf(
        &self,
        x: &[F],
        v: &[F],
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) -> (Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v.len(), n, "wrong number of directions");

        let dual_inputs: Vec<Dual<F>> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| Dual::new(xi, vi))
            .collect();

        self.forward_tangent_dual(&dual_inputs, dual_vals_buf);
        self.reverse_tangent_dual(dual_vals_buf, adjoint_buf);

        let gradient: Vec<F> = (0..n).map(|i| adjoint_buf[i].re).collect();
        let hvp: Vec<F> = (0..n).map(|i| adjoint_buf[i].eps).collect();
        (gradient, hvp)
    }

    /// Like [`hvp_with_buf`](Self::hvp_with_buf) but also reuses a caller-provided
    /// input buffer, eliminating all allocations on repeated calls.
    pub(super) fn hvp_with_all_bufs(
        &self,
        x: &[F],
        v: &[F],
        dual_input_buf: &mut Vec<Dual<F>>,
        dual_vals_buf: &mut Vec<Dual<F>>,
        adjoint_buf: &mut Vec<Dual<F>>,
    ) {
        // Reuse the input buffer instead of allocating
        dual_input_buf.clear();
        dual_input_buf.extend(x.iter().zip(v.iter()).map(|(&xi, &vi)| Dual::new(xi, vi)));

        self.forward_tangent_dual(dual_input_buf, dual_vals_buf);
        self.reverse_tangent_dual(dual_vals_buf, adjoint_buf);
    }

    /// Full Hessian matrix via `n` Hessian-vector products.
    ///
    /// Returns `(value, gradient, hessian)` where `hessian[i][j] = ∂²f/∂x_i∂x_j`.
    /// The tape is not mutated.
    pub fn hessian(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<Dual<F>> = Vec::with_capacity(n);
        let mut dual_vals_buf = Vec::new();
        let mut adjoint_buf = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        for j in 0..n {
            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf
                .extend((0..n).map(|i| Dual::new(x[i], if i == j { F::one() } else { F::zero() })));

            self.forward_tangent_dual(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent_dual(&dual_vals_buf, &mut adjoint_buf);

            if j == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for (row, adj) in hessian.iter_mut().zip(adjoint_buf.iter()) {
                row[j] = adj.eps;
            }
        }

        (value, gradient, hessian)
    }

    /// Full Hessian matrix via batched forward-over-reverse.
    ///
    /// Processes `ceil(n/N)` batches instead of `n` individual HVPs,
    /// computing N Hessian columns simultaneously.
    pub fn hessian_vec<const N: usize>(&self, x: &[F]) -> (F, Vec<F>, Vec<Vec<F>>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");

        let mut dual_input_buf: Vec<DualVec<F, N>> = Vec::with_capacity(n);
        let mut dual_vals_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut adjoint_buf: Vec<DualVec<F, N>> = Vec::new();
        let mut hessian = vec![vec![F::zero(); n]; n];
        let mut gradient = vec![F::zero(); n];
        let mut value = F::zero();

        let num_batches = n.div_ceil(N);
        for batch in 0..num_batches {
            let base = batch * N;

            // Reuse input buffer
            dual_input_buf.clear();
            dual_input_buf.extend((0..n).map(|i| {
                let eps = std::array::from_fn(|lane| {
                    let col = base + lane;
                    if col < n && i == col {
                        F::one()
                    } else {
                        F::zero()
                    }
                });
                DualVec::new(x[i], eps)
            }));

            self.forward_tangent(&dual_input_buf, &mut dual_vals_buf);
            self.reverse_tangent(&dual_vals_buf, &mut adjoint_buf);

            if batch == 0 {
                value = dual_vals_buf[self.output_index as usize].re;
                for i in 0..n {
                    gradient[i] = adjoint_buf[i].re;
                }
            }

            for lane in 0..N {
                let col = base + lane;
                if col >= n {
                    break;
                }
                for i in 0..n {
                    hessian[i][col] = adjoint_buf[i].eps[lane];
                }
            }
        }

        (value, gradient, hessian)
    }

    // ── Higher-order derivatives ──

    /// Third-order directional derivative: `∑_{jk} (∂³f/∂x_i∂x_j∂x_k) v1_j v2_k`.
    ///
    /// Given directions `v1` and `v2`, computes:
    /// - `gradient`: `∇f(x)`
    /// - `hvp`: `H(x) · v1` (Hessian-vector product)
    /// - `third`: `(∂/∂v2)(H · v1)` (third-order tensor contracted with v1 and v2)
    ///
    /// Uses `Dual<Dual<F>>` (nested dual numbers): inner tangent for `v1`,
    /// outer tangent for `v2`.
    pub fn third_order_hvvp(&self, x: &[F], v1: &[F], v2: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = self.num_inputs as usize;
        assert_eq!(x.len(), n, "wrong number of inputs");
        assert_eq!(v1.len(), n, "wrong v1 length");
        assert_eq!(v2.len(), n, "wrong v2 length");

        // Build Dual<Dual<F>> inputs:
        //   inner.re = x[i], inner.eps = v1[i]  (for HVP direction)
        //   outer.re = inner, outer.eps = v2[i]  (for third-order direction)
        //
        // outer = Dual { re: Dual(x[i], v1[i]), eps: Dual(v2[i], 0) }
        let dd_inputs: Vec<Dual<Dual<F>>> = (0..n)
            .map(|i| Dual {
                re: Dual::new(x[i], v1[i]),
                eps: Dual::new(v2[i], F::zero()),
            })
            .collect();

        let mut dd_vals: Vec<Dual<Dual<F>>> = Vec::new();
        let mut dd_adj: Vec<Dual<Dual<F>>> = Vec::new();

        self.forward_tangent(&dd_inputs, &mut dd_vals);
        self.reverse_tangent(&dd_vals, &mut dd_adj);

        let gradient: Vec<F> = (0..n).map(|i| dd_adj[i].re.re).collect();
        let hvp: Vec<F> = (0..n).map(|i| dd_adj[i].re.eps).collect();
        let third: Vec<F> = (0..n).map(|i| dd_adj[i].eps.eps).collect();

        (gradient, hvp, third)
    }
}

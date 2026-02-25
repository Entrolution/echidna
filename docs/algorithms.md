# Algorithms and Theory

This document describes the mathematical foundations and algorithms implemented in echidna. Each section covers the theory, key references, and the corresponding echidna module.

For usage examples, see the [crate-level documentation](https://docs.rs/echidna).

---

## 1. Forward-Mode AD

Forward-mode automatic differentiation is built on the algebra of dual numbers.
A dual number has the form `a + b*epsilon` where `epsilon^2 = 0`. Extending
every elemental operation `f` to dual numbers yields `f(a + b*epsilon) =
f(a) + f'(a)*b*epsilon`, so the derivative is carried alongside the value
through every operation. This is called tangent propagation: at each step the
pair `(f(x), f'(x) * x_dot)` is propagated forward through the computation.
In echidna, `Dual<F>` stores the value in `re` and the tangent in `eps`.
For a function `f: R^n -> R^m`, one forward pass with a tangent seed vector
`v` computes the Jacobian-vector product (JVP) `J * v`. Computing the full
Jacobian requires `n` forward passes (one per input), so forward mode is most
efficient when `n` is small relative to `m`.

The batched variant `DualVec<F, N>` packs `N` tangent directions into a single
forward pass, amortizing the cost of value computation across multiple seed
vectors. This is particularly effective when combined with SIMD, as tangent
components map naturally to SIMD lanes.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapters 2--3.

**Module:** `echidna::dual`, `echidna::dual_vec`, `echidna::api::jvp`

---

## 2. Reverse-Mode AD

Reverse-mode AD computes derivatives by recording a forward trace and then
propagating adjoint values backward. During the forward sweep, each operation
and its intermediate values are stored on a tape. The reverse sweep then
walks the tape in LIFO order, accumulating adjoint contributions via the chain
rule. For a scalar-valued function `f: R^n -> R`, one reverse sweep computes
the full gradient `nabla f` — the vector-Jacobian product (VJP) `w^T * J`
with `w = 1`. The Baur-Strassen result guarantees that the cost of one reverse
sweep is at most 5 times the cost of the forward evaluation, regardless of `n`.

echidna implements two reverse-mode backends. The Adept-style tape
(`Reverse<F>`, managed by `echidna::tape`) precomputes partial derivatives
during the forward sweep and stores them as scalar multipliers on a two-stack
tape — one stack for multipliers, one for operand indices. The reverse sweep
then consists entirely of multiply-accumulate operations with no opcode
dispatch, yielding tight inner loops. A constant sentinel value (`u32::MAX`)
marks operands that are literals or constants, avoiding tape entries for
non-differentiable quantities and preventing tape bloat. The bytecode reverse
backend (`BReverse<F>`) records opcodes for deferred evaluation through the
`BytecodeTape`. Both backends skip zero adjoints during the reverse sweep to
avoid unnecessary arithmetic in sparse dependency structures.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 4;
Hogan, R.J. (2014), "Fast reverse-mode automatic differentiation using
expression templates in C++", *ACM TOMS*.

**Module:** `echidna::reverse`, `echidna::tape`, `echidna::breverse`,
`echidna::api::grad`

---

## 3. Bytecode Tape (Graph-Mode AD)

The bytecode tape provides a reusable graph representation of a computation.
Rather than re-tracing the function each time (as operator-overloading modes
do), the tape is recorded once and can be evaluated repeatedly with different
inputs. The tape uses a Structure-of-Arrays (SoA) layout: separate contiguous
arrays for opcodes, operand indices, and constant values. This layout improves
cache utilization — the opcode stream is compact enough to fit in L1 cache for
moderate-sized tapes, and the separation allows independent vectorization of
forward and reverse sweeps. The tape currently defines 38 opcodes covering all
standard elemental operations (arithmetic, transcendental, power, comparison).

A key advantage of the graph representation is that it enables optimization
passes before evaluation. echidna implements four tape transformations: Common
Subexpression Elimination (CSE) merges duplicate operations sharing the same
opcode and operands; Dead Code Elimination (DCE) removes operations that do
not contribute to any output; algebraic simplification catches identity
patterns (`x + 0 -> x`, `x * 1 -> x`) and absorbing patterns (`x * 0 -> 0`,
`x - x -> 0`) at recording time; and constant folding evaluates operations
on known constants at recording time. These optimizations are composable and
can be applied in sequence. The tape also enables second-order derivative
computation (Hessians), sparsity pattern detection, cross-country elimination,
and GPU offloading — all of which require a static graph structure.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 5
(computational graph representation).

**Module:** `echidna::bytecode_tape`, `echidna::opcode`, `echidna::breverse`

---

## 4. Higher-Order Derivatives

Higher-order derivatives in echidna are computed by nesting AD modes. The
canonical construction is forward-over-reverse: wrapping a reverse-mode type
inside a forward-mode dual number, `Dual<BReverse<f64>>`, produces a
Hessian-vector product (HVP) `H * v` in one combined forward-plus-reverse
pass. The forward seed `v` is set on the dual component; the forward sweep
records operations on the tape; the reverse sweep then produces adjoints whose
dual parts contain the HVP. To assemble the full `n x n` Hessian matrix, one
performs `n` such HVP computations, one per coordinate direction. The Hessian
is symmetric, so only the upper triangle needs to be computed, reducing the
work by nearly half.

The batched variant `DualVec<BReverse<f64>, N>` processes `N` tangent
directions simultaneously through a single tape recording, yielding `N`
columns of the Hessian per forward-plus-reverse pass. This is more efficient
than `N` separate `Dual<BReverse<f64>>` passes because the tape recording
overhead is paid only once. For sparse Hessians, graph coloring (see Section 6)
further reduces the number of required directions.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 8.

**Module:** `echidna::api::hessian`, `echidna::api::hvp`,
`echidna::api::hessian_vec`

---

## 5. Taylor Mode AD

Taylor mode propagates truncated Taylor series through a computation. Given a
function `f` and an input expanded as `x(t) = x_0 + x_1*t + ... + x_K*t^K`,
the output `y(t) = f(x(t))` is also a Taylor series whose coefficients `y_k`
can be computed incrementally. For multiplication, the coefficients follow the
Cauchy product rule: `(a*b)_k = sum_{j=0}^{k} a_j * b_{k-j}`. For
transcendental functions (exp, log, sin, cos), echidna uses the logarithmic
derivative technique from Griewank Chapter 13, which expresses higher-order
coefficients through recurrence relations involving lower-order ones. This
avoids the need to differentiate the elemental functions symbolically.

echidna provides two Taylor types. `Taylor<F, K>` is const-generic and
stack-allocated: the degree `K` is a compile-time constant, enabling
monomorphization and eliminating heap allocation. `TaylorDyn<F>` is
arena-based (using `bumpalo`), implements `Copy`, and allows the degree to be
chosen at runtime. Both types implement the full `num_traits::Float` trait and
echidna's `Scalar` trait, so they flow through `BytecodeTape::forward_tangent`
without any tape modifications. Shared propagation logic lives in
`taylor_ops.rs`.

Two important applications build on Taylor mode. ODE Taylor integration
(`ode_taylor_step`) computes the Taylor expansion of the solution `y(t)` to an
ODE `y' = f(y)` via coefficient bootstrapping: `y_{k+1} = coeff_k(f(y(t))) /
(k+1)`. This requires `K-1` forward passes through the tape, building up the
Taylor inputs incrementally, and yields both the solution and error estimates
from the trailing coefficients. Reverse-over-Taylor (`taylor_grad`) combines
a forward Taylor sweep with a reverse adjoint sweep: the zeroth-order adjoint
gives the gradient, the first-order adjoint gives the HVP, and higher-order
adjoints give k-th order directional derivatives — all in a single pass.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 13.

**Module:** `echidna::taylor`, `echidna::taylor_dyn`, `echidna::taylor_ops`

---

## 6. Sparse Derivatives

When the Jacobian or Hessian has a known sparsity structure, computing only
the nonzero entries can be vastly cheaper than computing the full dense matrix.
echidna's sparse derivative pipeline has three stages: sparsity detection,
graph coloring, and compressed evaluation. Sparsity detection propagates
bitsets through the bytecode tape to determine which outputs depend on which
inputs, producing a `JacobianSparsityPattern` or `SparsityPattern`. Graph
coloring then assigns colors to columns (for Jacobians) or rows/columns (for
Hessians) such that columns sharing a nonzero in the same row receive
different colors. For Jacobians, this is a greedy distance-2 coloring; for
symmetric Hessians, it is a star bicoloring that exploits symmetry to halve
the required colors.

Compressed evaluation performs one forward or reverse sweep per color group.
Each sweep uses a seed vector that is the sum of coordinate basis vectors for
all columns in that color group. Because the columns in a group share no
nonzero row, their contributions can be separated by inspecting which row
positions are nonzero. The result is stored in CSR (Compressed Sparse Row)
format for efficient downstream use. The number of sweeps equals the chromatic
number of the coloring, which for structured problems (banded, block-diagonal)
is far smaller than `n`.

**References:** Gebremedhin, Manne & Pothen (2005), "What color is your
Jacobian? Graph coloring for computing derivatives"; Griewank & Walther,
*Evaluating Derivatives*, Chapter 7.

**Module:** `echidna::sparse`, `echidna::api::sparse_jacobian`,
`echidna::api::sparse_hessian`

---

## 7. Cross-Country Elimination

Cross-country elimination computes Jacobians by performing vertex elimination
on the linearized computational graph — a DAG where edges carry the local
partial derivatives from the forward evaluation. The idea is to eliminate
intermediate vertices one at a time: when vertex `v` is eliminated, every pair
of incoming and outgoing edges is replaced by a single fill-in edge whose
weight is the product of the two original edge weights (an application of the
chain rule). After all intermediate vertices are eliminated, the remaining
edges directly connect input vertices to output vertices, and their accumulated
weights are the Jacobian entries.

The order in which vertices are eliminated affects the total operation count
due to fill-in. echidna uses Markowitz ordering: at each step, eliminate the
vertex with the smallest product `|predecessors| x |successors|`. This greedy
heuristic minimizes local fill-in and works well in practice. Cross-country
elimination can be cheaper than either pure forward mode (cost proportional to
`n`) or pure reverse mode (cost proportional to `m`) when the function has a
balanced input/output ratio. For extreme aspect ratios, it degenerates to one
of the pure modes.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 10.

**Module:** `echidna::cross_country`,
`BytecodeTape::jacobian_cross_country`

---

## 8. Checkpointing

Reverse-mode AD requires storing the entire forward trace before the backward
sweep can begin. For computations with many steps (long time integrations,
deep recurrent networks), this memory requirement can be prohibitive.
Checkpointing trades memory for recomputation: instead of storing every
intermediate value, only selected checkpoints are stored, and the forward
computation between checkpoints is re-executed during the backward sweep.

echidna implements binomial checkpointing via the Revolve algorithm. Given `s`
checkpoint slots and `n` time steps, Revolve produces an optimal schedule that
minimizes total recomputation. The schedule is provably optimal for the
binomial model (where each segment costs the same to recompute). Three
extensions address practical limitations. Online checkpointing
(`grad_checkpointed_online`) handles the case where the total step count is
unknown in advance: it uses periodic thinning, maintaining a fixed-size
checkpoint buffer and discarding every other entry when full, doubling the
spacing. The overhead is O(log N) recomputations for N steps. Disk-backed
checkpointing (`grad_checkpointed_disk`) stores checkpoint states as raw
binary files when the total checkpoint memory exceeds available RAM, using
unsafe byte transmutation (safe for all `Float` types since they are `Copy +
Sized` with no pointers) and a `DiskCheckpointGuard` Drop guard for
panic-safe cleanup. Hint-based placement (`grad_checkpointed_with_hints`)
accepts user-specified required checkpoint positions and distributes the
remaining slots proportionally across the resulting sub-intervals using the
largest-remainder method, then runs Revolve on each sub-interval independently.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 12;
Griewank & Walther (2000), "Algorithm 799: Revolve".

**Module:** `echidna::checkpoint`

---

## 9. Stochastic Taylor Derivative Estimators (STDE)

Full second-order derivative tensors are expensive to compute and store. For
many applications — Laplacians in PDE solvers, trace estimation in
optimization, Hessian diagonal preconditioning — only aggregate quantities
are needed. Stochastic Taylor Derivative Estimators use random jet propagation
to estimate these quantities without materializing the full Hessian. The core
idea is the Hutchinson trace estimator: for a random vector `v` drawn from a
distribution with identity covariance, `E[v^T H v] = tr(H)`, where `H` is the
Hessian. By pushing `Taylor<F, 3>` inputs with a random tangent direction
through the bytecode tape, the second-order Taylor coefficient at the output
yields `v^T H v` for that direction. Averaging over `S` random directions
gives the Laplacian estimate `tr(H) ~ (1/S) * sum_s v_s^T H v_s`.

echidna provides several estimators built on this primitive. `laplacian`
computes the Hutchinson trace estimate using user-provided direction vectors
(Rademacher or Gaussian — no `rand` dependency). `hessian_diagonal` computes
exact diagonal entries by using coordinate basis vectors as tangent directions.
`directional_derivatives` returns batch second-order directional derivatives.
Variance reduction is available via `laplacian_with_control`, which uses the
Hessian diagonal as a control variate: since the diagonal is computed exactly,
it can be subtracted from the stochastic estimate and added back, reducing
variance to match Rademacher performance even when using Gaussian vectors.
`laplacian_with_stats` uses Welford's online algorithm to track sample
variance and standard error alongside the running estimate.

**Higher-order diagonal estimation.** The coordinate-basis approach
generalises beyond k=2. `diagonal_kth_order` pushes basis vectors through
order-(k+1) `TaylorDyn` jets to extract exact `[∂^k u/∂x_j^k]` for all
coordinates. The output coefficient at index k stores `∂^k u/∂x_j^k / k!`,
so the derivative is `k! * coeffs[k]`. The stochastic variant subsamples
coordinate indices for an unbiased estimate of the diagonal sum.

**Parabolic PDE σ-transform.** For parabolic PDEs with diffusion matrix σ,
the operator `½ tr(σσ^T · H)` can be rewritten as `½ Σ_i (σ·e_i)^T H (σ·e_i)`
where each term is a directional second derivative along σ's columns. This
avoids forming off-diagonal Hessian entries and reduces to d second-order
Taylor pushforwards (d = number of columns of σ).

**Sparse STDE.** For general differential operators `L = Σ C_α D^α`, the
coefficient tensor C(L) defines a discrete distribution over sparse k-jets.
Each term α is sampled with probability `|C_α| / Z` where `Z = Σ|C_α|`, and
the per-sample estimator `sign(C_α) · Z · D^α u` is unbiased for `Lu`. The
derivative `D^α u` is extracted from a single forward pushforward via the
jet extraction formula from Section 10. This implements the core contribution
of Shi et al. (2024): it reduces the cost of estimating arbitrary k-th order
operators from O(n^k) (exact computation) to O(S) (S random samples), where
each sample costs one forward pushforward.

**References:** Shi et al. (2024), "Stochastic Taylor Derivative
Estimators", NeurIPS 2024 Best Paper;
Hutchinson (1990), "A stochastic estimator of the trace of the influence
matrix for Laplacian smoothing splines".

**Module:** `echidna::stde`

---

## 10. Differential Operator Evaluation

The STDE module (Section 9) uses random tangent directions through low-order
Taylor jets to estimate aggregate second-order quantities like the Laplacian.
The `diffop` module generalises this idea to extract *exact* arbitrary mixed
partial derivatives from a single jet pushforward per group of active
variables.

The mathematical foundation is the multivariate Faà di Bruno formula applied
to a parameterised curve. Given a function `u: R^n -> R` and a target mixed
partial `∂^Q u / (∂x_{i₁}^{q₁} ... ∂x_{iT}^{qT})`, assign each active
variable `i_t` a distinct integer "slot" `j_t`. Construct Taylor inputs where
variable `i_t` has coefficient `1/j_t!` at index `j_t` (all other higher-order
coefficients zero), then push through the tape via `forward_tangent`. The
output jet coefficient at index `k = Σ j_t · q_t` equals the target derivative
divided by the prefactor `Π_t (q_t! · (j_t!)^{q_t})`. This prefactor absorbs
both the Taylor scaling (`1/k!` at the output) and the input scaling (`1/j_t!`
per variable).

The slot assignment must satisfy a uniqueness condition: the target partition
of `k` into parts drawn from the assigned slots must be the *only* such
partition. If another partition exists, it contaminates the output coefficient
and the extraction is ambiguous. echidna uses a prime window sliding strategy:
assign slots from consecutive primes `PRIMES[offset..offset+T]` and verify
uniqueness by exhaustive partition enumeration. If a collision is detected,
increment the offset and retry. For practical PDE operators (total order ≤ 6,
≤ 4 active variables), convergence is fast.

Multi-indices with different sets of active variables receive separate
pushforwards ("pushforward groups"). This is essential: if slots from
non-active variables are present in the Taylor input, their contributions
create partition collisions that cannot be resolved by slot reassignment alone.
Multi-indices sharing the same active variable set are batched into a single
group, since their slot assignments are compatible and multiple output
coefficients can be read from one pushforward.

The `JetPlan` type precomputes all slot assignments, jet orders, and extraction
prefactors. Once built, a plan is reused across evaluation points — only the
Taylor input values change, not the combinatorial structure. This
plan-once-evaluate-many design mirrors the `BytecodeTape` philosophy and is
well suited to PDE solvers that evaluate the same differential operator at
many grid points.

The `DiffOp` type represents a linear differential operator `L = Σ C_α D^α`
as an explicit list of `(coefficient, multi-index)` pairs. It provides exact
evaluation via `DiffOp::eval` (delegating to `JetPlan` internally) and can
build a `SparseSamplingDistribution` for stochastic estimation. Convenience
constructors cover common operators: `laplacian(n)`, `biharmonic(n)`,
`diagonal(n, k)`. Inhomogeneous operators (e.g., `u_t + ∇²u`) can be
decomposed into homogeneous groups via `split_by_order()`, since the sparse
sampling distribution requires all terms to share the same total order.

**References:** Shi et al. (2024), "Stochastic Taylor Derivative
Estimators", NeurIPS (Section 3, jet extraction formula);
Griewank & Walther, *Evaluating Derivatives*, Chapter 13 (Taylor propagation).

**Module:** `echidna::diffop`

---

## 11. Nonsmooth AD

Standard AD assumes that all elemental operations are smooth (continuously
differentiable). In practice, many functions contain piecewise-linear
operations — `abs`, `min`, `max` — and step functions — `signum`, `floor`,
`ceil`, `round`, `trunc` — that introduce kinks or discontinuities where the
classical derivative does not exist. echidna's nonsmooth AD module detects
these kinks during forward evaluation and provides access to the Clarke
generalized derivative, which is the appropriate generalized derivative concept
for Lipschitz-continuous functions.

`BytecodeTape::forward_nonsmooth` performs a standard forward evaluation while
recording a `KinkEntry` for each of the 8 nonsmooth operations. Each entry
stores the tape index, opcode, switching value, and the branch that was taken.
For `abs`/`min`/`max`/`signum`, the switching value is the operand value
(kink at zero). For `floor`/`ceil`/`round`/`trunc`, the switching value is
the distance to the nearest integer (`a - a.round()`), which is zero exactly
at discontinuity points. The resulting `NonsmoothInfo` provides
`active_kinks(tol)` to query kinks within a given tolerance,
`is_smooth(tol)` to check whether the evaluation point lies in a smooth
region, and `signature()` to characterize the combinatorial branch structure.

echidna uses a two-tier classification for nonsmooth operations.
`is_nonsmooth()` returns true for all 8 ops and is used for proximity
detection — knowing whether the evaluation point is near any kink.
`has_nontrivial_subdifferential()` returns true only for `abs`/`min`/`max`,
where forced branch choices produce distinct partial derivatives (e.g.,
`abs` has slope +1 vs -1). Step functions (`signum`/`floor`/`ceil`/`round`/
`trunc`) have zero derivative on both sides of the kink, so enumerating
forced branches would add 2^k cost with no new information.

When kinks with nontrivial subdifferentials are present,
`BytecodeTape::clarke_jacobian` enumerates all `2^k` sign combinations for
`k` active nontrivial kinks, computing a limiting Jacobian for each
combination by forcing the sign choices in the nonsmooth operations and
running a standard reverse sweep. Step-function kinks are filtered before
enumeration. The convex hull of these limiting Jacobians is the Clarke
generalized Jacobian. A configurable kink limit (default 20) prevents
combinatorial explosion by returning `ClarkeError::TooManyKinks` when
exceeded.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 14;
Clarke, F.H. (1983), *Optimization and Nonsmooth Analysis*.

**Module:** `echidna::nonsmooth`,
`BytecodeTape::forward_nonsmooth`,
`BytecodeTape::clarke_jacobian`

---

## 12. Laurent Series

Laurent series extend Taylor series to handle singularities. A Laurent series
has the form `sum_{k=p}^{K} a_k * t^k` where the pole order `p` may be
negative, allowing representation of functions with poles. `Laurent<F, K>` is
a const-generic, stack-allocated, `Copy` type that stores `K` coefficients
along with a tracked pole order. Arithmetic operations reuse the Cauchy
product and division rules from Taylor propagation (`taylor_ops`), with
additional bookkeeping for pole order: multiplication adds pole orders,
division subtracts them, and the result is always normalized so the leading
coefficient is nonzero.

The type handles degenerate cases deliberately: essential singularities (e.g.,
`exp` or `sin` applied to a value with a pole) produce NaN, since the Laurent
expansion does not converge in such cases. Division by zero also produces NaN.
The `value()` method returns positive or negative infinity for genuine poles,
matching the expected mathematical behavior. `Laurent<F, K>` implements the
full `num_traits::Float` trait and echidna's `Scalar` trait, so it flows
through `BytecodeTape::forward_tangent` for automated singularity detection
in recorded computations.

**References:** Standard complex analysis; the implementation follows Taylor
propagation conventions from Griewank & Walther, Chapter 13, extended with
pole tracking.

**Module:** `echidna::laurent`

---

## 13. Implicit Differentiation

Many problems involve quantities `z*` defined implicitly by an equation
`F(z*, x) = 0`, where `z*` is the solution and `x` is the parameter. The
Implicit Function Theorem (IFT) states that if `F_z` (the Jacobian of `F`
with respect to `z`) is nonsingular at `(z*, x)`, then the derivative of `z*`
with respect to `x` exists and can be computed without differentiating through
the solver. The tangent-mode formula is `dz*/dx = -(F_z)^{-1} * F_x`:
compute the partial Jacobians `F_z` and `F_x` via AD, then solve a linear
system. The adjoint-mode formula is `dL/dx = -lambda^T * F_x` where
`F_z^T * lambda = dL/dz`, which requires only one linear solve regardless
of the number of parameters `x`. The full Jacobian `dz*/dx` can be assembled
by iterating either the tangent formula (column by column) or the adjoint
formula (row by row).

Second-order implicit derivatives are computed via nested dual numbers
`Dual<Dual<F>>`. A single forward pass through `forward_tangent` with doubly-
nested dual inputs extracts the second-order correction from the `.eps.eps`
component, which encodes the Hessian-direction-direction product. The function
`implicit_hvp` solves the resulting linear system to produce the implicit HVP,
and `implicit_hessian` builds the full second-order tensor by iterating over
`n(n+1)/2` direction pairs with LU factor reuse.

Piggyback differentiation handles the case where `z*` is found by iterating
a fixed-point map `z_{k+1} = G(z_k, x)` rather than by a direct solve. The
tangent piggyback propagates `z_dot_{k+1} = G_z * z_dot_k + G_x * x_dot`
alongside the primal iteration using a dual-number forward pass at each step.
The adjoint piggyback iterates `lambda_{k+1} = G_z^T * lambda_k + z_bar` at
the converged `z*` via reverse sweeps. The interleaved variant runs both the
primal and adjoint iterations in a single loop, reducing the total iteration
count from `K_primal + K_adjoint` to `max(K_primal, K_adjoint)`.

For large systems where `F_z` is sparse, the `sparse-implicit` feature
exploits structural sparsity. `SparseImplicitContext` precomputes the full
sparsity pattern and graph coloring, then uses echidna's sparse Jacobian
infrastructure for compressed evaluation and faer's sparse LU for the solve.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 15;
Christianson, B. (1994), "Reverse accumulation and attractive fixed points".

**Module:** `echidna_optim::implicit`, `echidna_optim::piggyback`,
`echidna_optim::sparse_implicit`

---

## 14. GPU Acceleration

Batched tape evaluation is naturally parallel: the same sequence of operations
is applied independently to many input vectors. echidna exploits this by
offloading the bytecode tape evaluation to the GPU. The CPU-side code handles
sparsity detection and graph coloring (which are sequential, irregular
algorithms), then dispatches batched tangent sweeps to the GPU for the
numerically intensive phase. Custom operations are rejected at tape upload
time since they cannot be compiled to GPU code.

The wgpu backend (feature `gpu-wgpu`) provides cross-platform GPU access via
Metal, Vulkan, and DX12. It implements four WGSL compute shaders: `forward`
(batched forward evaluation), `reverse` (batched adjoint sweep),
`tangent_forward` (forward tangent for JVP and sparse Jacobian), and
`tangent_reverse` (forward-over-reverse for HVP and sparse Hessian). All 43
opcodes are implemented in each shader. The backend is limited to f32 due to
WGSL's lack of native f64 support. The CUDA backend (feature `gpu-cuda`)
targets NVIDIA GPUs and supports both f32 and f64. It uses a single templated
CUDA kernel file compiled via NVRTC at runtime, instantiated for both `float`
and `double`. The `WgpuContext` and `CudaContext` types provide matching API
surfaces: `forward_batch`, `gradient_batch`, `sparse_jacobian`, `hvp_batch`,
and `sparse_hessian`.

### GPU-Accelerated STDE

When the `stde` feature is enabled alongside a GPU backend, a fifth compute
shader `taylor_forward_2nd` provides batched second-order Taylor forward
propagation. Each thread processes one batch element, propagating a `(c0, c1,
c2)` triple through the tape where `c0` is the primal value, `c1` is the
directional first derivative, and `c2 = v^T H v / 2`. The memory layout is
interleaved per variable per thread: `[c0_i, c1_i, c2_i]` contiguous per
variable, optimizing per-thread locality. All 43 opcodes implement full
second-order Taylor propagation rules: arithmetic via Cauchy products,
transcendentals via logarithmic derivative recurrences, and coupled
recurrences for sin/cos, sinh/cosh, and tan/tanh.

The high-level functions in `stde_gpu` flatten directions into batch format,
dispatch the Taylor kernel, and perform CPU-side Welford aggregation:

- `laplacian_gpu` — Hutchinson trace estimator: pushes S random directions
  through the kernel, computes `mean(2 * c2)` as the Laplacian estimate.
- `hessian_diagonal_gpu` — exact diagonal: uses n standard basis vectors as
  directions, extracts `diag(H)[j] = 2 * c2[j]`.
- `laplacian_with_control_gpu` — subtracts the exact diagonal contribution
  (control variate) to reduce estimator variance.

Buffer constraint: the working buffer requires `3 * num_variables *
batch_size * 4` bytes. For WebGPU's default 128 MB storage buffer limit,
max `num_variables * batch_size ≤ ~10M`.

**References:** General GPU computing literature; the tape evaluation model
follows the graph-mode execution paradigm common in ML frameworks, adapted
for AD-specific operations. The Taylor propagation rules follow the same
recurrences as the CPU implementation in `taylor_ops.rs`.

**Module:** `echidna::gpu` (`echidna::gpu::wgpu_backend`,
`echidna::gpu::cuda_backend`, `echidna::gpu::stde_gpu`)

---

## 15. Composable Mode Nesting

echidna's AD modes are designed to compose via type-level nesting. The key
insight is that `Dual<F>`, `Taylor<F, K>`, and `DualVec<F, N>` are all generic
over `F: Float`, and `Reverse<F>` and `BReverse<F>` both implement `Float`.
This enables compositions like `Dual<BReverse<f64>>` (forward-over-reverse =
HVP), `Taylor<BReverse<f64>, K>` (Taylor-over-reverse = higher-order
adjoints), `DualVec<BReverse<f64>, N>` (batched tangent-over-reverse =
multiple HVP directions per tape pass), and `Dual<Dual<BReverse<f64>>>`
(triple nesting for second-order implicit derivatives). The `Float` and
`IsAllZero` trait implementations for `Reverse<F>` and `BReverse<F>` are what
make these compositions possible: they allow reverse-mode types to serve as
the scalar type inside forward-mode wrappers.

The convenience function `composed_hvp` provides a one-shot interface for
forward-over-reverse HVP without requiring the user to manually set up nested
types and tapes. Internally, it constructs `Dual<BReverse<f64>>` inputs,
evaluates the function, and extracts the HVP from the adjoint dual components.
Reverse-wrapping-forward (`BReverse<Dual<f64>>`) is also supported via
`BtapeThreadLocal` implementations for `Dual<f32>` and `Dual<f64>`, enabling
reverse-over-forward HVP and column-by-column Hessian computation.

**References:** Griewank & Walther, *Evaluating Derivatives*, Chapter 8
(higher-order techniques via mode composition).

**Module:** `echidna::api::composed_hvp`, `echidna::float`

---

## 16. Optimization Solvers

The `echidna-optim` crate provides derivative-based optimization algorithms
that use echidna's AD capabilities for gradient and Hessian computation. All
solvers accept functions via the `Objective` or `TapeObjective` traits, which
abstract over the function evaluation and derivative computation interface.

**L-BFGS** is a limited-memory quasi-Newton method that approximates the
inverse Hessian using the last `m` gradient-step pairs (typically `m = 5..20`).
The two-loop recursion computes the search direction `H_k * g_k` in O(m*n)
time without forming the matrix explicitly. Combined with Armijo backtracking
line search (sufficient decrease condition), L-BFGS requires only gradient
information and is the workhorse method for large-scale unconstrained
optimization.

**Newton's method** computes the exact Hessian at each iteration via
forward-over-reverse AD and solves the Newton equation `H * d = -g` using
Cholesky factorization. This yields quadratic convergence near the solution
but requires O(n^2) storage and O(n^3) factorization per step, making it
practical only for moderate dimensions.

**Trust-region** uses the Steihaug-Toint truncated conjugate gradient method
to approximately solve the trust-region subproblem `min_d m(d) s.t. ||d|| <=
Delta`, where `m(d)` is the quadratic model. The trust radius `Delta` is
adapted based on the ratio of actual to predicted reduction: expanded when the
model is accurate, contracted when it overpredicts. This method is more robust
than line search for non-convex problems and handles indefinite Hessians
gracefully (CG detects negative curvature and stops early).

**References:** Nocedal & Wright, *Numerical Optimization*, Chapters 6--7
(L-BFGS, trust region); standard optimization textbooks.

**Module:** `echidna_optim::solvers::lbfgs`,
`echidna_optim::solvers::newton`,
`echidna_optim::solvers::trust_region`

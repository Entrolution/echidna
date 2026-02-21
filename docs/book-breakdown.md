# Evaluating Derivatives — Comprehensive Chapter Breakdown

**Source**: *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation, 2nd Edition*
by Andreas Griewank and Andrea Walther (SIAM, 2008)

**Purpose**: Technical reference for building the **echidna** Rust library — a comprehensive implementation of algorithmic differentiation (AD) techniques.

---

## How to Read This Document

Each chapter section contains:
- **Summary**: What the chapter covers and why it matters
- **Key Concepts**: The core mathematical and algorithmic ideas
- **Implementation Relevance**: What this means for echidna's design
- **Dependencies**: What must be built before this chapter's features

The chapters build on each other. The dependency graph is roughly:

```
Ch 1-3 (Foundations) → Ch 4 (Forward Mode) → Ch 5 (Reverse Mode)
                                ↓                     ↓
                          Ch 6 (Complexity)     Ch 7 (Taping)
                                ↓                     ↓
                          Ch 8 (2nd Order) ← ← ← ← ←┘
                                ↓
                    ┌───────────┼───────────────┐
                    ↓           ↓               ↓
              Ch 9 (Sparse) Ch 10 (Jacobian)  Ch 11 (Hessians)
                                                    ↓
                                              Ch 12 (Checkpointing)
                                                    ↓
                                              Ch 13 (Taylor/Higher-Order)
                                                    ↓
                                    ┌───────────────┼───────────┐
                                    ↓                           ↓
                              Ch 14 (Nonsmooth)          Ch 15 (Implicit)
```

---

## Part I: Basics (Chapters 1–4)

### Chapter 1 — Introduction (pp. 1–20)

**Summary**: Motivates AD by contrasting it with finite differences and symbolic differentiation. Introduces the fundamental idea: any numerical program computes a composition of elementary operations, and chain rule application to this composition yields exact derivatives at machine precision.

**Key Concepts**:
- **Truncation error** in finite differences: step size h creates O(h^p) error, but reducing h increases roundoff — no sweet spot
- **Expression swell** in symbolic differentiation: intermediate expressions grow exponentially
- **AD's niche**: exact derivatives (to machine precision) at bounded cost relative to function evaluation
- **Two modes**: forward (tangent) and reverse (adjoint), with fundamentally different cost profiles
- **Lighthouse example**: a running example used throughout the book — position of a light spot on a wall as a function of lighthouse rotation angle. Returns in Chapters 13, 14, and 15 with increasingly complex variants

**Implementation Relevance**: Establishes the core value proposition. The library must:
- Support both forward and reverse mode from the start
- Provide derivative accuracy at machine precision (not approximations)
- Scale to real-world numerical programs (not just toy expressions)

**Dependencies**: None — this is foundational motivation.

---

### Chapter 2 — A Framework for Evaluating Functions (pp. 21–48)

**Summary**: Defines the formal computational model that underpins all AD techniques. Introduces evaluation procedures, computational graphs, and the elemental function framework.

**Key Concepts**:
- **Evaluation Procedure**: A sequence of elementary operations transforming inputs x ∈ ℝ^n to outputs y ∈ ℝ^m via intermediates v_i
  - Three-part structure: independent variables (v_{1-n}...v_0), intermediate variables (v_1...v_l), dependent variables (y_1...y_m)
  - Each v_i = φ_i(v_j)_{j ≺ i} where φ_i is an elemental function
- **Assumption (ED)**: Elemental Differentiability — all elemental functions φ_i are at least once Lipschitz-continuously differentiable on open domains containing all argument values
- **Computational Graph**: DAG with vertices for variables and edges for direct dependencies
  - Edge labels c_{ij} = ∂v_i/∂v_j are the local partial derivatives
  - The graph encodes the complete data dependency structure
- **Extended Jacobian**: The (n+l) × (n+l) matrix L with entries c_{ij}, strictly lower triangular
- **Jacobian as Schur Complement**: F'(x) = R + T(I − L)^{−1}B where L captures internal dependencies, B input effects, T output effects, R direct input→output effects
- **Elemental functions**: unary (sin, cos, exp, log, sqrt, abs, ...) and binary (+, −, ×, /, pow, ...)
- **Table 2.2**: Key notation reference used throughout the book

**Implementation Relevance**: This chapter defines echidna's core data model:
- **`EvalProcedure`**: The central representation of a differentiable computation
- **`ComputationalGraph`**: DAG with typed nodes (input, intermediate, output) and weighted edges
- **Elemental operations**: Need a trait/enum for all supported elementary functions with their local derivatives
- The three-part variable structure (inputs, intermediates, outputs) shapes the API

**Dependencies**: None — this is the foundational data model.

---

### Chapter 3 — Fundamentals of Forward and Reverse (pp. 49–72)

**Summary**: Derives the forward and reverse mode formulas from chain rule applied to computational graphs. Establishes the fundamental complexity results.

**Key Concepts**:
- **Forward Mode (Tangent Propagation)**:
  - v̇_i = Σ_{j ≺ i} c_{ij} · v̇_j (propagate input perturbation forward)
  - Computes one column of Jacobian per seed direction ẋ
  - Cost: O(n) × TIME(F) for full n×m Jacobian when seeding with identity
  - Natural for n ≤ m (few inputs, many outputs)
- **Reverse Mode (Adjoint Propagation)**:
  - v̄_j += v̄_i · c_{ij} for all i ≻ j (propagate output sensitivity backward)
  - Computes one row of Jacobian per adjoint seed ȳ
  - **Cheap Gradient Principle**: ∇f costs ≤ ω × TIME(F) where ω ≤ 4 (typically 2–3)
  - Natural for m ≤ n (few outputs, many inputs) — the common case in optimization
- **Jacobian-Vector Products**: Forward mode computes F'(x)·ẋ; reverse mode computes ȳ^T·F'(x)
- **Vertex elimination interpretation**: Forward = eliminate from left; Reverse = eliminate from right
- **Table 3.2–3.4**: Forward and reverse propagation rules for all elemental operations
- **Incremental vs nonincremental**: Two implementation styles for local partial accumulation

**Implementation Relevance**: Core differentiation engine:
- **Forward mode**: Dual numbers with tangent component, seed vector propagation
- **Reverse mode**: Tape recording + backward sweep, adjoint accumulation
- Both modes need the same elemental derivative rules (Tables 3.2–3.4)
- The ω ≤ 4 bound for gradients is the key selling point for optimization users

**Dependencies**: Chapter 2 (evaluation procedure and graph model).

---

### Chapter 4 — Memory Issues and Complexity Bounds (pp. 73–100)

**Summary**: Analyzes the computational and memory complexity of AD in detail. Introduces the WORK vector model for tracking costs and addresses the key memory challenge of reverse mode.

**Key Concepts**:
- **WORK vector**: 4-component (MULT, NLMULT, ADD, NLADD) tracking multiplications and additions (linear vs nonlinear separately)
- **Assumption (TA)**: Temporal Additivity — total work is sum of elemental work
- **Assumption (EB)**: Elemental Task Boundedness — each elemental operation has bounded work
- **Temporal complexity**: Forward mode adds at most WORK(F) per direction; reverse mode at most 4× WORK(F) for gradient
- **Spatial complexity of reverse mode**: Must store all intermediate values (prevalues) for backward sweep
  - Naive: O(l) storage where l = number of intermediates
  - This motivates checkpointing (Chapter 12) and taping strategies (Chapter 7)
- **FLOP counting**: Precise operation counts for forward/reverse (Tables 4.1–4.5)
- **Data flow analysis**: Liveness analysis for reducing memory — only store values needed in the backward sweep
- **Homogeneity observation**: Temporal complexity of tangent and adjoint propagation are roughly equal per direction; the asymmetry is in how many directions are needed

**Implementation Relevance**:
- Memory management is critical — reverse mode's O(l) storage is the main engineering challenge
- Need efficient tape/trace data structures from the start
- WORK vector model useful for cost estimation and optimization heuristics
- Data flow / liveness analysis can dramatically reduce memory for reverse mode

**Dependencies**: Chapters 2–3.

---

## Part II: Extensions and Strategies (Chapters 5–8)

### Chapter 5 — Forward and Reverse Revisited (pp. 101–124)

**Summary**: Extends forward and reverse mode with Jacobian matrix propagation, vectorized operations, and combined forward-reverse strategies.

**Key Concepts**:
- **Jacobian Propagation**: Propagate entire Jacobian matrices V̇ (n×p tangent matrix) or W̄ (q×n adjoint matrix) instead of individual vectors
  - Forward: computes F'(x) · S for seed matrix S
  - Reverse: computes W · F'(x) for weight matrix W
- **Strip-mining**: Process multiple tangent/adjoint directions simultaneously for cache efficiency
- **Combined forward-reverse**: For intermediate Jacobians, can mix modes
  - Forward first, then reverse on the accumulated result
  - Or reverse first, then forward
- **Jacobian-free methods**: Iterative solvers (CG, GMRES) that only need Jacobian-vector products — AD provides these without ever forming the full Jacobian
- **Compression**: When Jacobian has known sparsity structure, can recover it from fewer directional derivatives than full dimension (detailed in Chapter 9)
- **Hessian-vector products**: Brief introduction — tangent-of-adjoint gives ∇²f · ẋ (detailed in Chapter 8)

**Implementation Relevance**:
- API should support both single-vector and matrix propagation modes
- Strip-mining / SIMD-friendly data layouts for vectorized tangent propagation
- Integration point for iterative linear algebra (CG, GMRES) that consumes Jv products
- Foundation for the compressed Jacobian computation in Chapter 9

**Dependencies**: Chapters 2–4.

---

### Chapter 6 — Implementation and Software (pp. 125–148)

**Summary**: Covers practical implementation approaches: operator overloading vs source transformation, and the software engineering of AD tools.

**Key Concepts**:
- **Operator Overloading (OO)**:
  - Redefine arithmetic operators on a custom numeric type that records operations
  - Advantages: easy to implement, works with existing code via type substitution
  - Disadvantages: overhead per operation, limited compile-time optimization, control flow recorded at runtime
  - Rust analogy: implement `Add`, `Mul`, etc. traits on an AD type
- **Source Transformation (ST)**:
  - Generate new source code that computes both function and derivatives
  - Advantages: full program analysis, optimal code generation, no runtime overhead
  - Disadvantages: requires parser/compiler infrastructure, language-specific
- **Taping / Tracing** (OO approach):
  - Record a "tape" of operations during forward evaluation
  - Replay tape backward for reverse mode
  - Tape stores: operation codes, argument indices, prevalues, partial derivatives
- **Activity analysis**: Identify which variables actually depend on inputs (active) vs constants — avoid differentiating dead code
- **Checkpointing** preview: Store selected snapshots to enable re-evaluation of segments (Chapter 12)
- **AD tools landscape**: ADIFOR, ADIC, TAF, Tapenade, ADOL-C, CppAD, etc.
- **Figure 6.1**: Taxonomy of AD implementation approaches

**Implementation Relevance**: Defines echidna's implementation strategy:
- Rust's trait system maps naturally to operator overloading approach
- Need efficient tape data structure (operation log, value storage)
- Activity analysis as an optimization pass
- Consider both eager (OO) and lazy (expression graph) evaluation strategies
- ADOL-C's design (C++ OO with tape) is the closest precedent

**Dependencies**: Chapters 2–5.

---

### Chapter 7 — Adjoint Code for Explicit Evaluations (pp. 149–186)

**Summary**: Details the implementation of reverse mode, including taping strategies, incremental vs nonincremental approaches, and practical adjoint code generation.

**Key Concepts**:
- **Assumption (PC)**: Path Connectedness — at most two arguments per elemental function (simplifies to binary DAG)
- **Definition (MW)**: Matrix Width — max simultaneous live values during sweep; determines memory for Jacobian propagation
- **Incremental Adjoint**: v̄_j += v̄_i · ∂φ_i/∂v_j — accumulate contributions as encountered
- **Nonincremental Adjoint**: v̄_j = Σ_{i ≻ j} v̄_i · c_{ij} — compute all at once from successors
  - Nonincremental requires knowing the complete set of successors before computing
- **Taping detail**:
  - LIFO (stack) organization: values recorded during forward, consumed during reverse
  - Three tapes: real values, integer indices, operation codes
  - Tape compression: don't record linear operations' prevalues (they're reconstructable)
- **Preaccumulation**: Compute local Jacobians of straight-line code segments before recording, reducing tape size
- **Subroutine handling**: Record/replay at subroutine boundaries; option to preaccumulate entire subroutines
- **Statement-level vs subroutine-level**: Granularity of taping
- **Split vs joint reversal**: Split = reverse each subroutine call independently; Joint = one big reverse
  - Trade-off: joint saves memory but splits are more parallelizable

**Implementation Relevance**: Core reverse mode implementation:
- Tape design: three separate stacks (reals, indices, opcodes) for cache efficiency
- Preaccumulation optimization for straight-line code segments
- Support for both incremental and nonincremental adjoint styles
- Subroutine-level taping for composability
- Memory layout matters enormously for performance

**Dependencies**: Chapters 2–6.

---

### Chapter 8 — Second-Order Methods (pp. 187–218)

**Summary**: Computing second derivatives (Hessians) efficiently via combinations of forward and reverse mode.

**Key Concepts**:
- **Hessian-vector product** ∇²f(x) · ẋ via tangent-of-adjoint (forward-over-reverse):
  1. Forward sweep recording tape
  2. Reverse sweep computing gradient (adjoint)
  3. Apply forward mode to the reverse sweep itself
  - Cost: O(1) × TIME(F) per Hessian-vector product
- **Second-order adjoint**: The full formula for propagating second-order information
  - Involves v̈ (second tangent), v̄ (adjoint), and w̄ (second-order adjoint) variables
  - Equations 8.8–8.12: coupled system of first and second order
- **Hessian matrix computation**: Need n Hessian-vector products for full n×n Hessian, or exploit sparsity
- **Symmetry exploitation**: Hessian is symmetric, so only need upper/lower triangle
- **Living-on-the-edge**: Compute Hessian-vector products without forming intermediate Jacobians
- **Second-order forward mode**: Propagate (v, v̇, v̈, v̈̇) — four components per variable
  - Computes ẋ^T ∇²f ẋ (scalar second directional derivative)
  - Need p(p+1)/2 evaluations for full Hessian via distinct direction pairs
- **Tables 8.1–8.5**: Propagation rules for second-order tangent and adjoint for all elementals
- **Edge pushing**: Alternative Hessian computation by pushing second-order information along graph edges

**Implementation Relevance**:
- Forward-over-reverse (tangent-of-adjoint) is the primary Hessian-vector product method
- Need composable mode nesting: forward mode applied to reverse mode code
- Second-order elemental rules (Tables 8.1–8.5) extend the first-order tables
- Hessian-vector products are critical for Newton methods in optimization

**Dependencies**: Chapters 3–7.

---

## Part III: Sparse and Structured Jacobians (Chapters 9–11)

### Chapter 9 — Sparse Forward and Reverse (pp. 219–250)

**Summary**: Exploiting sparsity in Jacobians by propagating only the nonzero structure, avoiding unnecessary computation.

**Key Concepts**:
- **Index domains and ranges**: Track which input indices affect each intermediate (domain) and which outputs each intermediate affects (range)
  - Forward: propagate index domains d_i ⊆ {1,...,n}
  - Reverse: propagate index ranges r_j ⊆ {1,...,m}
- **Sparse tangent propagation**: Only compute/store nonzero entries of tangent vectors
  - Use compressed storage (sparse vectors) instead of dense
  - Skip computation when tangent is structurally zero
- **Sparse adjoint propagation**: Analogous with adjoint vectors
- **Structural analysis**: Determine Jacobian sparsity pattern without computing values
  - Forward sweep of index sets gives column structure
  - Reverse sweep of index sets gives row structure
- **Definition (LP)**: Local Procedure — a subset of the evaluation procedure with its own local Jacobian
- **Definition (SC)**: Jacobian Dimension and Scarcity — when the Jacobian has s << n×m nonzeros
- **Matrix Compression** (CPR seeding):
  - **Column compression**: If Jacobian columns can be grouped such that no two columns in a group share a nonzero row, seed with p < n directions and recover full Jacobian
  - **Row compression**: Analogous for rows using adjoint mode
  - **Combined**: CPR (Column-Partial-Row) — use both forward and reverse with compression
  - Equivalent to **graph coloring**: column compression ↔ distance-2 coloring of column intersection graph
  - NR (Newsam-Ramsdell) seeding: exploit known sparsity pattern for optimal seeding
- **Figures 9.1–9.4**: Visualization of sparse structures and compression

**Implementation Relevance**:
- Sparse Jacobian computation is essential for large-scale problems
- Need sparse vector/matrix data structures with efficient merge operations
- Index set propagation for structural analysis (precursor to numerical computation)
- Graph coloring algorithms for optimal seed generation
- CPR/NR seeding strategies as built-in optimization

**Dependencies**: Chapters 2–7.

---

### Chapter 10 — Jacobian Accumulation and Elimination (pp. 251–274)

**Summary**: Cross-country elimination — computing Jacobians by eliminating vertices from the computational graph in non-standard orders (neither pure forward nor pure reverse).

**Key Concepts**:
- **Cross-country elimination**: Eliminate vertices from computational graph in any order, not just left-to-right (forward) or right-to-left (reverse)
  - Each elimination of vertex v_i creates fill-in edges: for each predecessor j and successor k of v_i, add edge (k,j) with weight c_{kj} += c_{ki} · c_{ij}
  - Total Jacobian = product of all local eliminations
- **Vertex elimination**: Remove a vertex, connecting all its predecessors to all successors with appropriate weights
- **Edge elimination**: More fine-grained — eliminate individual edges
- **Face elimination**: Most fine-grained — eliminate individual "faces" (predecessor-vertex-successor triples)
- **Definition (BP)**: Biclique Property — structural condition for equivalence of elimination approaches
- **NP-hardness**: Optimal elimination order (minimum total multiplications) is NP-hard
  - Even for vertex elimination alone
- **Markowitz heuristic**: Greedily eliminate vertex with minimum (in-degree × out-degree) product
  - Practical and effective, inspired by sparse LU factorization
- **Simulated annealing** and other metaheuristics for better elimination orders
- **Figures 10.1–10.8**: Detailed examples of vertex, edge, and face elimination on computational graphs
- **Jacobian accumulation as matrix product**: The accumulated Jacobian is equivalent to a specific sequence of sparse matrix multiplications

**Implementation Relevance**:
- Cross-country elimination as an optimization for Jacobian computation
- Markowitz heuristic implementation for elimination ordering
- Graph transformation operations (vertex/edge/face elimination with fill-in)
- Useful for small-to-medium computational graphs where optimal ordering matters
- Less critical than forward/reverse for initial implementation

**Dependencies**: Chapters 2–3, Chapter 9.

---

### Chapter 11 — Hessians, Higher Derivatives, and Structured Methods (pp. 275–300)

**Summary**: Efficient computation of Hessians exploiting structure: partial separability, symmetric computational graphs, and connections to Chapter 8's second-order methods.

**Key Concepts**:
- **Definition (SG)**: Symmetric Computational Graph — graph where forward and reverse sweeps have identical structure
  - Rule 16: Nonincremental reverse mode naturally produces a symmetric graph
  - This symmetry is key for efficient Hessian computation
- **Hessian Product Form**: ∇²f = Ź^T D Ź where Ź is the extended tangent matrix and D is diagonal
  - Computed by one forward sweep (getting Ź and D) + matrix multiply
  - D contains second derivatives of elemental functions
- **Definition (PS)**: Partial Separability — f(x) = Σ f_i(U_i x) where each f_i depends on few variables (U_i selects a subset)
  - **Value separability**: Direct sum decomposition of function value
  - **Argument separability**: Each component depends on few input variables
  - Hessian of partially separable function is sum of low-rank terms
- **Rule 17**: Cheap gradient principle does NOT extend to cheap Jacobians
  - Full m×n Jacobian always requires min(m,n) sweeps, no constant-factor bound
- **Rule 18**: Reverse beats forward for Jacobians only when nonlinear matrix width >> nonlinear height
- **Coloring for Hessians**: Exploit symmetry to reduce number of Hessian-vector products
  - Star coloring, acyclic coloring for direct/indirect recovery
  - Fewer colors needed than for general Jacobians due to symmetry
- **Tables 11.1–11.3**: Complexity comparisons for various Hessian computation strategies

**Implementation Relevance**:
- Partial separability detection and exploitation — common in optimization (each constraint involves few variables)
- Hessian product form (Ź^T D Ź) as an efficient representation
- Star/acyclic coloring for compressed Hessian computation
- Symmetric graph construction for Hessian-specific optimizations

**Dependencies**: Chapters 2–3, 8–9.

---

## Part IV: Memory Management and Advanced Topics (Chapters 12–15)

### Chapter 12 — Reversal Schedules and Checkpointing (pp. 301–340)

**Summary**: The fundamental memory management challenge of reverse mode and its solution via checkpointing. One of the most practically important chapters.

**Key Concepts**:
- **The problem**: Reverse mode needs all intermediate values, requiring O(l) memory for l operations. For long computations (e.g., time-stepping), this is prohibitive.
- **Definition (RS)**: Reversal Schedules — four atomic motions:
  1. **Advancing**: Evaluate forward without recording (move forward in time)
  2. **Recording**: Evaluate forward while recording to tape (expensive in memory)
  3. **Returning**: Move read head back to a checkpoint (no computation)
  4. **Reversing**: Execute one step of the adjoint (consumes recorded data)
- **Checkpointing strategy**: Store snapshots at selected time steps; re-evaluate segments between checkpoints as needed during the reverse sweep
- **Binomial checkpointing** (Griewank's Algorithm):
  - With c checkpoints and r reversals needed, optimal reach is β(c,r) = C(c+r, c) (binomial coefficient)
  - This gives logarithmic growth: l steps need only O(log l) checkpoints for O(l log l) total operations
  - Optimal in the class of offline schedules
  - Rule 21: Relative reversal cost grows like l^(1/c) — polynomial in l for fixed c
- **Online checkpointing**: When total number of steps l is not known in advance
  - Revolve algorithm and its online variants
  - Slightly suboptimal but practical for adaptive computations
- **Rule 19**: Joint reversals save memory; split reversals save operations
- **Rule 20**: All taping during call tree reversals occurs in LIFO fashion
- **Parallel checkpointing**: Distribute checkpoint storage and recomputation across processors
- **Multi-level checkpointing**: Different storage tiers (registers, RAM, disk) with different costs
- **Figures 12.1–12.8**: Diagrams of various checkpointing schedules
- **Tables 12.1–12.3**: Optimal checkpoint placement and operation counts
- **Call tree reversal**: Checkpointing applied to recursive/hierarchical program structure, not just flat time-stepping

**Implementation Relevance**: Critical for practical reverse mode:
- **Checkpoint manager**: Allocate, store, and retrieve checkpoints
- **Revolve algorithm**: Implement the binomial checkpointing schedule
- **Online variant**: Support unknown-length computations
- **Multi-level storage**: Abstract over memory tiers
- **Integration with tape**: Checkpoints interact with the taping system (Chapter 7)
- This is where the "engineering" of reverse mode really lives

**Dependencies**: Chapters 2–7.

---

### Chapter 13 — Taylor and Tensor Coefficients (pp. 301–340)

**Summary**: Higher-order derivative computation via Taylor arithmetic — a unified framework that extends first-order AD to arbitrary derivative order.

**Key Concepts**:
- **Taylor Coefficient Functions (Definition TC)**: For v(t) = Σ v_k t^k, the coefficients v_k encode k-th derivative information: v_k = (1/k!) d^k v/dt^k |_{t=0}
- **Definition (AI)**: Approximants of Intermediates — truncated Taylor series to degree d
- **Taylor arithmetic**: Extend each elemental operation to operate on truncated Taylor series (polynomials of degree d)
  - Addition: coefficient-wise
  - Multiplication: convolution (v·w)_k = Σ_{j=0}^{k} v_j · w_{k-j}
  - Division, transcendentals: recurrence relations derived from defining equations
  - Tables 13.1–13.4: Complete rules for all elementals in Taylor arithmetic
- **Rule 22**: Forward differentiation on scalars is encapsulated by extension to truncated Taylor series (Taylor arithmetic)
- **Univariate Taylor Propagation (UTP)**: The forward sweep in Taylor arithmetic
  - Cost: O(d²) × TIME(F) for degree-d Taylor coefficients of all intermediates
- **Higher-order Jacobians**: Proposition 13.3 — ∂y_j/∂x_i follows a shifted structure (A_{j-i})
  - Corollary 13.1: Extension operator E_d and differentiation operator D commute
- **Higher-Order Adjoints**: All adjoint vectors F̄_j for j < d obtained from one reverse sweep in Taylor arithmetic
  - Cost: (1+d)² × TIME(F)
  - Rule 23: What works for first derivatives yields higher derivatives by extension to Taylor arithmetic
- **ODE Integration via Taylor Series**:
  - For ẋ(t) = F(x(t)), coefficients satisfy x_{k+1} = F_k(x_0,...,x_k)/(k+1)
  - Recursive computation: cost O(d²) for d coefficients, O(d³/3) for d coefficients plus their Jacobian
  - **Sensitivity equation**: X'(t) = F'(x(t))X(t), with X_k recursion (eq 13.23)
- **Coefficient Doubling** (Table 13.7): Use Newton's method to compute d Taylor coefficients in O(log₂(d+2)) sweeps instead of d sweeps
  - Based on Corollary 13.2: Linearity in Higher Coefficients — y_k is linear in x_j for j > k/2
- **DAE Support**: Differential-algebraic equations 0 = F_k(z_0,...,z_k,(k+1)z_{k+1})
  - Block Hessenberg Jacobian structure (eq 13.27)
  - Requires consistent initialization

**Implementation Relevance**:
- **Taylor number type**: Polynomial coefficient arrays with arithmetic operations
- **UTP engine**: Forward propagation of Taylor coefficients through all elementals
- **Higher-order adjoint**: Reverse sweep on Taylor arithmetic for gradients of higher-order coefficients
- **ODE solver integration**: Taylor series method for ODE integration (high-order, A-stable)
- **Coefficient doubling**: Newton-based acceleration for long Taylor expansions
- This is a major feature differentiator — most AD libraries only do first order

**Dependencies**: Chapters 2–7 (for the higher-order adjoint part, also Chapter 8).

---

### Chapter 14 — Differentiation without Differentiability (pp. 341–366)

**Summary**: Handling programs with branches, absolute values, min/max, and other nonsmooth operations. Extends AD to piecewise differentiable functions.

**Key Concepts**:
- **Definition (PD)**: Piecewise Differentiability — function is smooth on each piece of a finite partition of the domain, with pieces defined by "selection functions" (branch conditions)
- **Proposition 14.1** (Fischer's Result): On the interior of each piece's domain, derivatives are well-defined and computable by standard AD
- **Rule 24**: When program branches apply on open subdomains (generic case), AD yields correct derivatives
- **Four categories of nonsmooth elementals**:
  1. **Kinks**: |v|, min(u,v), max(u,v) — continuous but not differentiable at isolated points
  2. **Roots**: √u, arcsin(u), arccos(u) — bounded values but infinite derivatives at boundary
  3. **Steps**: sign(u), Heaviside function — discontinuous; encode branch selection
  4. **Poles**: 1/u, log(u), tan(u) — infinite values at boundary
- **Convention**: ±∞ × 0 = 0 = 0 × NaN (zero annihilates exceptional values)
- **Assumption (IC)**: Isolated Criticalities — critical points (where nonsmoothness occurs) are isolated on any line through the domain
- **Definition (SD)**: Stable Domain — subset where the same selection functions apply in a neighborhood
- **Proposition 14.2** / Rule 25: Functions given by evaluation procedures are almost everywhere (full-measure) either real analytic or stably undefined. The complement (where derivatives don't exist) has measure zero.
- **Laurent Numbers (Definition LN)**: Extended number system for tracking behavior near critical points
  - Leading exponent e_v, significant degree d_v, coefficient array
  - Special values: NaL (not-a-Laurent = NaN analogue), zero, log singularities
  - Tables 14.6–14.8: Arithmetic on Laurent numbers (addition, univariates, conditionals)
  - Proposition 14.3: Laurent arithmetic consistently propagates the Laurent model
- **Definition (RA)**: Regular Arc — parameterized curve along which function has well-defined Laurent expansion with finite determinacy degree d_*
- **Definition (FS)**: Finite Slopes — no roots or poles among critical elementals
- **Proposition 14.5**: Under finite slopes, the limiting Jacobian along a regular arc exists and belongs to Clarke's generalized gradient
- **Generalized Jacobians**: Clarke's generalized gradient ∂f = conv{g | g = lim ∇f(x_k), x_k → x}

**Implementation Relevance**:
- **Abs/min/max support**: Must handle kink functions correctly, propagating derivatives from the active branch
- **NaN/Inf propagation**: Follow the convention ±∞ × 0 = 0
- **Branch tracking**: Record which branch was taken during evaluation; derivatives valid on the interior of each branch region
- **Laurent arithmetic** (optional advanced feature): For analyzing behavior near critical points
- **Generalized gradients**: Support for nonsmooth optimization (e.g., L1 regularization, ReLU in neural networks)
- Practically: most users hit this via abs, min, max, ReLU, and conditional branches

**Dependencies**: Chapters 2–3, conceptually Chapter 13 for Laurent numbers.

---

### Chapter 15 — Implicit and Iterative Differentiation (pp. 367–398)

**Summary**: Computing derivatives through implicit function definitions and iterative solvers (Newton's method, fixed-point iterations) without differentiating every iteration.

**Key Concepts**:
- **Implicit Function Theorem application**: Given F(z,x) = 0 defining z = z(x) implicitly:
  - **Implicit tangent**: ż_* = −F_z(z_*,x)^{−1} · F_x(z_*,x) · ẋ (eq 15.8)
  - **Implicit adjoint**: w̄_* = F_z(z_*,x)^{−T} · f_z(z_*,x)^T · ȳ (eq 15.9)
  - Avoids differentiating the solver iterations — only needs converged solution and Jacobian of the residual
- **Assumption (JR)**: Jacobian Regularity — F_z(z_*,x) is nonsingular at the solution
- **Reduced Jacobian**: dy/dx expressed as a generalized Schur complement (eq 15.5)
  - Two bracketing options: forward (eq 15.6) vs reverse (eq 15.7)
- **Direct/Adjoint Sensitivity Equations** (eqs 15.10–15.11):
  - Direct: F_z · Ż + F_x = 0 → solve for Ż
  - Adjoint: F_z^T · W̄ + ... = 0 → solve for W̄
- **Derivative Quality Criteria** (Lemma 15.1): Forward and reverse consistency check — if both forward and reverse derivatives agree, solution accuracy is confirmed
- **Corrected Function Estimate** (Corollary 15.1): Using adjoint information to correct function values, doubling convergence order
- **Two-Phase Approach**: First converge the iteration, then differentiate:
  - Phase 1: Run iteration to convergence
  - Phase 2: Apply implicit differentiation at converged point
- **Piggyback Approach**: Propagate derivatives alongside iteration:
  - Delayed piggyback: start derivative propagation partway through iteration
  - Coupled iteration of z and ż (eq 15.25)
  - Proposition 15.1: Derivative convergence factor ≤ ρ_* (same as iteration)
- **Assumption (GC)**: Global Contractivity — ||G'_k|| ≤ ρ < 1 (contraction mapping)
- **Newton's method specifics**: G_k(z,x) = z − F_z(z,x)^{−1} F(z,x)
  - Quadratic convergence of function → linear convergence of derivatives via piggyback
- **Preconditioner Deactivation**: Simplified derivative recurrence when preconditioner is frozen (eq 15.29)
  - Tables 15.1–15.2: Direct and adjoint fixed point iteration algorithms
- **Rule 26**: Fixed point iterations can and should be adjoined by recording only single steps (not the entire iteration history)
- **Second-Order Adjoint Fixed Point** (Table 15.3): Simultaneous convergence of z, ż, w̄, and ŵ (second-order adjoint)
- **Extended Computational Graph** (Figure 15.7): Visualization of the coupled iteration structure
- **Convergence rates**: All derivative orders converge at ρ_*^k with polynomial lag factor k^d for d-th order

**Implementation Relevance**:
- **Implicit differentiation**: Core feature for differentiating through linear/nonlinear solvers
  - `implicit_tangent(F, z_star, x, x_dot)` and `implicit_adjoint(F, z_star, x, y_bar)` operations
- **Fixed-point iteration differentiation**: Piggyback approach for iterative methods
  - Record single iteration step, not entire history (Rule 26)
- **Integration with linear algebra**: Needs linear solves (F_z^{-1} · rhs) for implicit diff
- **Newton solver wrapper**: Differentiate through Newton's method transparently
- Critical for: PDE-constrained optimization, training equilibrium models, differentiating physics simulations with implicit time stepping

**Dependencies**: Chapters 2–7, Chapter 8 for second-order.

---

## Appendix: Cross-Cutting Concerns

### Complete Rule Index

| Rule | Statement | Chapter |
|------|-----------|---------|
| 1 | Problem functions should be evaluated as compositions of elementals | 2 |
| 2 | Evaluate derivatives along with function values | 2 |
| 3 | To propagate tangents, accumulate the chain rule from right to left (forward) | 3 |
| 4 | To propagate adjoints, accumulate the chain rule from left to right (reverse) | 3 |
| 5 | Gradient by reverse mode costs ≤ 4× function evaluation | 3 |
| 6 | Separate linear and nonlinear operations in cost analysis | 4 |
| 7 | If function can be evaluated, so can its Jacobian–vector product (forward) | 5 |
| 8 | If function can be evaluated, so can its transpose-Jacobian–vector product (reverse) | 5 |
| 9 | Exploit structure (sparsity, symmetry) in Jacobian computation | 5 |
| 10 | Operator overloading is simplest AD implementation approach | 6 |
| 11 | Source transformation gives best performance | 6 |
| 12 | Taping with compression reduces memory overhead | 7 |
| 13 | Preaccumulation of straight-line segments reduces tape size | 7 |
| 14 | Split reversals for parallelism, joint reversals for memory | 7 |
| 15 | Hessian-vector products via forward-over-reverse | 8 |
| 16 | Nonincremental reverse makes graph symmetric for Hessians | 11 |
| 17 | Cheap gradient does NOT yield cheap Jacobians | 11 |
| 18 | Reverse beats forward only when nonlinear width >> nonlinear height | 11 |
| 19 | Joint reversals save memory, split reversals save operations | 12 |
| 20 | All taping during call tree reversals occurs in LIFO fashion | 12 |
| 21 | Relative reversal cost grows like l^(1/c) | 12 |
| 22 | Forward differentiation encapsulated in Taylor arithmetic | 13 |
| 23 | First-derivative techniques extend to higher derivatives via Taylor arithmetic | 13 |
| 24 | Program branches on open subdomains → AD yields useful derivatives | 14 |
| 25 | Functions from eval procedures are a.e. real analytic or stably undefined | 14 |
| 26 | Fixed point iterations adjoined by recording only single steps | 15 |

### Complete Formal Definitions and Assumptions

**Assumptions:**
| Code | Name | Page | Chapter |
|------|------|------|---------|
| ED | Elemental Differentiability | 23 | 2 |
| TA | Temporal Additivity of Task | 74 | 4 |
| EB | Elemental Task Boundedness | 78 | 4 |
| PC | Path Connectedness | 149 | 7 |
| IC | Isolated Criticalities | 348 | 14 |
| JR | Jacobian Regularity | 371 | 15 |
| GC | Global Contractivity | 378 | 15 |

**Definitions:**
| Code | Name | Page | Chapter |
|------|------|------|---------|
| MW | Matrix Width, Maximal Domain, Range Size | 149 | 7 |
| BP | Biclique Property | 206 | 10 |
| LP | Local Procedure and Jacobian | 222 | 9 |
| SC | Jacobian Dimension and Scarcity | 227 | 9 |
| SG | Symmetric Computational Graph | 238 | 11 |
| PS | Partial Separability | 252 | 11 |
| RS | Reversal Schedules | 280 | 12 |
| TC | Taylor Coefficient Functions | 304 | 13 |
| AI | Approximants of Intermediates | 304 | 13 |
| PD | Piecewise Differentiability | 342 | 14 |
| SD | Stable Domain | 348 | 14 |
| LN | Laurent Number | 351 | 14 |
| RA | Regular Arc and Determinacy Degree | 360 | 14 |
| FS | Finite Slopes | 361 | 14 |

---

## Suggested Implementation Phases

Based on the dependency structure and practical utility, here is a suggested phasing for echidna development:

### Phase 1: Core Foundation
**Chapters 2–3** — Evaluation procedure model, computational graph, forward mode (dual numbers), reverse mode (basic tape)

Deliverables:
- Dual number type with operator overloading
- Basic tape-based reverse mode
- Elemental function library (all operations from Tables 3.2–3.4)
- Jacobian-vector and vector-Jacobian products

### Phase 2: Production Reverse Mode
**Chapters 4, 7, 12** — Memory management, efficient taping, checkpointing

Deliverables:
- Optimized tape data structures (three-stack design)
- Preaccumulation for straight-line segments
- Binomial checkpointing (Revolve algorithm)
- Online checkpointing for unknown-length computations
- Memory-bounded reverse mode for large-scale problems

### Phase 3: Second-Order Derivatives
**Chapters 5, 8** — Jacobian propagation, Hessian-vector products, forward-over-reverse

Deliverables:
- Matrix-mode tangent and adjoint propagation
- Forward-over-reverse Hessian-vector products
- Second-order elemental rules
- Composable mode nesting

### Phase 4: Sparse Computation
**Chapters 9, 11** — Sparse Jacobians, compressed computation, graph coloring

Deliverables:
- Sparsity pattern detection (index domain/range propagation)
- Graph coloring for seed generation (distance-2, star, acyclic)
- Compressed Jacobian and Hessian computation
- Partial separability detection and exploitation
- CPR/NR seeding strategies

### Phase 5: Advanced Jacobian Accumulation
**Chapter 10** — Cross-country elimination, Markowitz heuristic

Deliverables:
- Vertex/edge/face elimination on computational graphs
- Markowitz-based elimination ordering
- Optimal Jacobian accumulation for small graphs

### Phase 6: Higher-Order Derivatives
**Chapter 13** — Taylor arithmetic, univariate Taylor propagation, higher-order adjoints

Deliverables:
- Taylor number type (truncated polynomial arithmetic)
- UTP engine for all elementals
- Higher-order adjoint computation
- ODE integration via Taylor series
- Coefficient doubling (Newton acceleration)

### Phase 7: Nonsmooth Extensions
**Chapter 14** — Piecewise differentiable functions, abs/min/max, generalized gradients

Deliverables:
- Correct handling of abs, min, max, ReLU, conditional branches
- Branch tracking and active set identification
- Generalized gradient computation (Clarke subdifferential)
- Optional: Laurent number arithmetic for singularity analysis

### Phase 8: Implicit and Iterative Differentiation
**Chapter 15** — Implicit function theorem, fixed-point iteration differentiation

Deliverables:
- Implicit tangent and adjoint computation
- Piggyback differentiation for iterative solvers
- Newton solver with transparent differentiation
- Second-order implicit derivatives
- Fixed-point iteration adjoint (single-step recording per Rule 26)

---

## Key Design Decisions for echidna

These cross-cutting decisions are resolved in [design-principles.md](design-principles.md). Summary:

1. **Numeric type**: Generic over `Float` trait (`f32`/`f64`)
2. **Tape representation**: SoA bytecode tape (opcodes, arg_indices, values, adjoints as separate contiguous arrays)
3. **Implementation approach**: Operator overloading via Rust traits (eager mode), with optional deferred graph mode for advanced optimisations
4. **Memory allocation**: `bumpalo` arena for tapes, const-generic stack allocation for small tangent vectors
5. **Parallelism**: SIMD (intra-core via `wide`) + `rayon` (inter-core), coarse-grained parallelism only
6. **GPU**: `wgpu` (cross-platform) primary, `cudarc` (NVIDIA) optional, all behind feature flags with CPU fallback
7. **Linear algebra**: `nalgebra` for both fixed and dynamic sizes
8. **Error handling**: NaN propagation on hot paths (per Ch 14), `Result` on cold paths
9. **API**: Closure-based for common cases, builder pattern for advanced config, trait-based for library integration
10. **Composability**: Mode nesting via generic type composition (forward-over-reverse = `Dual<Tape<T>>`)

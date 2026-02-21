# echidna — Design Principles & Architecture

## Goals (ordered by priority)

1. **Speed** — optimised for throughput; minimise overhead per elemental operation
2. **Idiomatic Rust API** — leverage the type system, ownership model, and trait ecosystem; feel native, not like a C++ port
3. **Hardware acceleration with graceful fallback** — SIMD → rayon → GPU when available; always a correct CPU path

## Non-Goals (for now)

- Deep learning framework (Burn, Candle already cover that)
- `no_std` / embedded (may revisit later)
- Source transformation / proc-macro AD (compiler plugin is Enzyme's territory)

---

## Landscape Positioning

The Rust AD ecosystem has a clear gap echidna can fill:

| Library | Forward | Reverse | SIMD | GPU | Higher-Order | Sparse | Implicit |
|---------|---------|---------|------|-----|--------------|--------|----------|
| `ad-trait` | Yes | Yes | Forward only | No | No | No | No |
| `num-dual` | Yes | No | No | No | 2nd order | No | No |
| `burn` | No | Yes (tensor) | Backend | Yes | No | No | No |
| `std::autodiff` | Yes | Yes | No | No (yet) | No | No | No |
| **echidna** | Yes | Yes | Yes | Yes | Yes (Taylor) | Yes | Yes |

echidna aims to be the **comprehensive scientific AD library** — the Rust equivalent of what ADOL-C + ColPack + Tapenade covers in C/Fortran, but with modern hardware acceleration and a native Rust API.

---

## Core Architectural Decisions

### 1. Numeric Type: Generic over `Float` trait

```rust
pub trait Float: Copy + Send + Sync + 'static {
    // core ops, SIMD widening, etc.
}
impl Float for f32 { ... }
impl Float for f64 { ... }
```

Rationale: `f32` matters for GPU workloads and memory-bound problems. `f64` is default for scientific computing. Generics let users choose. Bound `Copy + Send + Sync` ensures SIMD and parallel safety.

### 2. Computation Model: Trace-then-optimise (tape with optional graph)

**Two-tier API:**

- **Eager mode** (operator overloading): `Dual<T>` for forward, tape-recording type for reverse. Lowest barrier to entry — write normal Rust, get derivatives.
- **Graph mode** (deferred): Build a computation graph, optimise it (CSE, dead-code elimination, preaccumulation), then execute. Enables cross-country elimination (Ch 10), sparsity analysis (Ch 9), and GPU kernel compilation.

Most users start with eager mode. Graph mode unlocks the advanced features from Chapters 9–13.

### 3. Tape Representation: SoA bytecode tape

Structure-of-Arrays layout for the tape, not an `enum` per operation:

```rust
struct Tape<T: Float> {
    opcodes:    Vec<OpCode>,       // u8 — elemental operation identifier
    arg_indices: Vec<[u32; 2]>,    // operand indices (max 2 per Ch 7 Assumption PC)
    values:     Vec<T>,            // prevalues for reverse sweep
    adjoints:   Vec<T>,            // adjoint accumulation buffer
}
```

Rationale:
- SoA keeps each field contiguous → better cache utilisation and SIMD-friendliness
- `OpCode` as `u8` keeps the opcode stream compact (fits in L1 for moderate tapes)
- Fixed 2-argument layout (Assumption PC: path connectedness) avoids variable-length encoding
- Separate `values` and `adjoints` arrays enable independent SIMD sweeps

### 4. SIMD Strategy: `wide` on stable, feature-gated `std::simd`

```toml
[features]
default = []
nightly-simd = []  # enables std::simd paths
```

- **Primary path**: `wide` crate for portable SIMD on stable Rust (x86 AVX2/SSE4.2, ARM NEON, WASM)
- **Optional nightly path**: `std::simd` behind feature gate for users already on nightly
- **Vectorised forward mode**: Bundle multiple tangent directions into SIMD lanes (like `ad-trait`'s `adf_f64x4`), process all directions in one pass
- **SIMD elemental kernels**: Hand-optimised SIMD implementations of hot elemental operations (exp, log, sin, cos, etc.) — these are the inner loop
- **Auto-vectorisation assist**: Structure loops to help LLVM auto-vectorise where explicit SIMD isn't used; avoid iterator chains that defeat vectorisation

### 5. Parallelism: rayon for coarse, SIMD for fine

Two levels of parallelism, matching hardware:

- **SIMD (intra-core)**: Multiple tangent directions or multiple data points processed per instruction
- **rayon (inter-core)**: Parallel over independent chunks of work
  - Forward mode: parallel over seed directions (each direction is independent)
  - Sparse Jacobian: parallel over coloring groups
  - Checkpointing: parallel recomputation of segments (Ch 12)
  - Graph coloring: parallel greedy coloring

Rule of thumb from the research: each rayon task should do ≥10μs of work. For AD, this means parallelise at the "direction" or "segment" level, not at the "elemental operation" level.

### 6. GPU Strategy: wgpu primary, cudarc optional

```toml
[features]
gpu = ["wgpu"]
cuda = ["cudarc"]
```

- **wgpu** for cross-platform GPU (Vulkan/Metal/DX12/WebGPU) — single codebase runs everywhere
- **cudarc** optional for NVIDIA-specific workloads where CUDA kernel libraries matter
- **GPU kernels**: Written in WGSL (for wgpu) or CUDA C (for cudarc), not Rust (rust-gpu is too experimental)
- **What goes on GPU**:
  - Batched forward mode (many inputs, same function) — embarrassingly parallel
  - Large Taylor coefficient propagation (Ch 13) — convolution-heavy, maps to GPU well
  - Sparse matrix operations for large Jacobians
  - NOT individual elemental operations (too fine-grained, transfer overhead dominates)
- **Graceful fallback**: GPU backends behind feature flags; all operations have CPU implementations; runtime detection of GPU availability

### 7. Memory Management: Arena + pool allocators

- **`bumpalo` arena** for the tape during a single differentiation pass: fast allocation (pointer bump), O(1) bulk deallocation after backprop completes
- **Pool allocator** for graph nodes in graph mode: typed pool of fixed-size nodes, reusable across passes
- **Stack allocation** via const generics for small fixed-size tangent vectors:
  ```rust
  struct TangentVec<T: Float, const N: usize> { data: [T; N] }
  ```
  When N is known at compile time (common in forward mode with few inputs), avoid heap entirely.

### 8. Linear Algebra Backend

- **Small/fixed**: `nalgebra` with compile-time dimensions — stack-allocated, inlined, fast
- **Large/dynamic**: `nalgebra` dynamic matrices, with optional BLAS linkage for large solves
- **Sparse**: Custom sparse CSR/CSC for Jacobian storage — existing crates (`sprs`, `nalgebra-sparse`) are adequate but may need thin wrappers for integration with the tape

Rationale: `nalgebra` has the best trait ecosystem compatibility in Rust numerics. It already integrates with `num-dual` and `ad-trait`, establishing it as the de facto standard.

### 9. Error Handling

- **Hot paths**: No `Result`, no panics. Follow Chapter 14's convention: NaN propagation for undefined derivatives, ±∞ × 0 = 0.
- **Cold paths** (graph construction, configuration): Return `Result<T, EchidnaError>` for recoverable errors.
- **Debug mode**: Optional runtime checks behind `#[cfg(debug_assertions)]` that validate tape integrity, detect NaN sources, and check gradient consistency (Lemma 15.1).

### 10. API Ergonomics

**Closure-based API for common cases:**

```rust
// Gradient of a scalar function
let grad = echidna::grad(|x: &[f64]| {
    x[0].sin() * x[1].exp()
});
let g = grad(&[1.0, 2.0]);

// Jacobian-vector product
let jvp = echidna::jvp(|x: &[f64]| {
    vec![x[0] + x[1], x[0] * x[1]]
}, &x, &dx);

// Hessian-vector product
let hvp = echidna::hvp(|x: &[f64]| {
    x[0].powi(3) + x[1].powi(2)
}, &x, &v);
```

**Builder API for advanced configuration:**

```rust
let result = echidna::reverse()
    .with_checkpointing(Revolve::new(num_checkpoints))
    .with_sparsity(SparsityPattern::detect())
    .gradient(f, &x);
```

**Trait-based for library integration:**

```rust
// Make existing code differentiable by making it generic
fn rosenbrock<T: echidna::Scalar>(x: &[T]) -> T {
    // standard Rust code, works with f64 and AD types
}
```

---

## Data Layout Principles

### Forward Mode

```
Input seeds     → [s₀, s₁, ..., sₚ₋₁]     (p directions, SIMD-packed)
Per-variable    → [v, v̇₀, v̇₁, ..., v̇ₚ₋₁]  (value + tangents, contiguous)
```

Pack multiple tangent directions into SIMD lanes. For p ≤ SIMD width, one pass suffices. For p > SIMD width, strip-mine in chunks.

### Reverse Mode

```
Forward sweep   → tape records [opcode, args, prevalue] per operation
Reverse sweep   → reads tape LIFO, accumulates into adjoint buffer
```

The adjoint buffer is a flat `Vec<T>` indexed by variable index. During the reverse sweep, it is written to in reverse topological order — good spatial locality on the adjoint array, though tape access is sequential (LIFO).

### Graph Mode

```
Nodes           → SoA: [opcode], [arg0], [arg1], [value_slot]
Edges           → implicit in arg indices (no separate edge list)
Topological     → nodes stored in topological order (same as evaluation order)
```

Topological ordering means forward sweep is a linear scan. Reverse sweep is a reverse linear scan. No pointer chasing.

---

## Feature Flag Matrix

| Feature | Flag | Dependencies | What it enables |
|---------|------|-------------|-----------------|
| Default | — | `wide`, `nalgebra`, `bumpalo` | Forward + reverse mode, CPU SIMD |
| `rayon` | `parallel` | `rayon` | Multi-core parallelism |
| `gpu` | `gpu` | `wgpu` | Cross-platform GPU acceleration |
| `cuda` | `cuda` | `cudarc` | NVIDIA CUDA acceleration |
| `nightly-simd` | `nightly-simd` | (nightly compiler) | `std::simd` paths |
| `sparse` | `sparse` | — | Sparse Jacobian/Hessian computation |
| `taylor` | `taylor` | — | Higher-order Taylor arithmetic |
| `implicit` | `implicit` | — | Implicit/iterative differentiation |
| `serde` | `serde` | `serde` | Serialisation of tapes/graphs |

Default feature set is minimal: forward mode, reverse mode, SIMD-accelerated elemental operations, arena-allocated tape. Everything else is opt-in.

---

## Performance Budget Targets

Based on the theoretical bounds from the book and the `ad-trait` benchmark baseline:

| Operation | Target overhead vs function eval |
|-----------|--------------------------------|
| Forward mode (1 direction) | ≤ 2× |
| Reverse mode (gradient) | ≤ 4× (the ω bound from Ch 3) |
| Forward mode (p directions, SIMD) | ≤ 2× + p/W (W = SIMD width) |
| Hessian-vector product (fwd-over-rev) | ≤ 10× |
| Taylor degree d | ≤ O(d²) × function eval |

These are theoretical bounds. The implementation target is to stay within 1.5× of these bounds in practice (i.e., overhead from taping, allocation, and dispatch should add ≤ 50% on top of the theoretical minimum).

---

## Dependency Summary

```toml
[dependencies]
wide = "0.7"           # portable SIMD on stable
nalgebra = "0.33"      # linear algebra
bumpalo = "3"          # arena allocator
num-traits = "0.2"     # numeric trait interop

[dev-dependencies]
criterion = "0.5"      # benchmarking
approx = "0.5"         # floating-point comparison in tests

[features]
default = []
parallel = ["rayon"]
gpu = ["wgpu"]
cuda = ["cudarc"]
nightly-simd = []
sparse = []
taylor = []
implicit = []
serde = ["dep:serde"]
```

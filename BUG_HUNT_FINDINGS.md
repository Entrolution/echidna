# Findings Manifest — Cycle 1

## Summary
- **Modules reviewed**: 9 (across 2 batches)
- **Specialist agents**: 27 (0 rate-limited, 0 re-run)
- **Total findings**: 16 (0 critical, 2 high, 8 medium, 6 low)
- **Cross-validated**: 2 findings confirmed by 2+ agents
- **Contradictions**: 0 (1 resolved during collation)
- **Retracted**: 1 (Rem sparsity — false positive)

## Findings

### High

| # | File:Line | Bug | Impact | Cross-validated | Test gap |
|---|-----------|-----|--------|-----------------|----------|
| 1 | `gpu/kernels/tape_eval.cu:631`, `gpu/shaders/tangent_reverse.wgsl:277` | CBRT HVP computes `-2at/(9r³)` instead of `-2at/(9r⁵)` — off by factor r² | All GPU Hessian-vector products through `cbrt` are wrong by a value-dependent multiplicative error | Yes (correctness-gpu + math-gpu) | No GPU HVP test for cbrt |
| 2 | `gpu/taylor_codegen.rs:1417,1451,1470,1484,1502,1542,1574,1597,1779,1793` | CUDA Taylor codegen assigns `unsigned long long` j_base to `unsigned int` intermediates, silently truncating | Memory corruption for large tapes where batch × vars × K > ~4 billion | No | No large-tape integration test |

### Medium

| # | File:Line | Bug | Impact | Cross-validated | Test gap |
|---|-----------|-----|--------|-----------------|----------|
| 3 | `opcode.rs:342` | Bytecode `atan2` uses `a*a+b*b` instead of `hypot(a,b)²` | Zero derivatives for inputs where \|a\| or \|b\| > ~1.34e154 (f64); other AD modes use hypot | No | No large-input atan2 test |
| 4 | `dual.rs:236`, `dual_vec.rs:253`, `num_traits_impls.rs:760`, `opcode.rs:417`, `taylor_ops.rs:250` | `1 - x*x` in asin/acos/atanh derivatives loses ~15 digits near \|x\|→1 | Should use `(1-x)*(1+x)` — affects all AD modes | No | No near-boundary derivative precision test |
| 5 | `traits/std_ops.rs:50-57`, `dual_vec_ops.rs:50` | Division derivative computes `inv*inv` which overflows for denominators near min_positive | Inf derivative even when the true result is representable | No | No small-denominator test |
| 6 | `bytecode_tape/tangent.rs:325`, `bytecode_tape/sparse.rs:72` | `debug_assert!` guards custom ops in Hessian computations — silent in release | Wrong Hessian/sparse-Hessian results in release builds when custom ops are present | No | Tests only run in debug mode |
| 7 | `gpu/taylor_codegen.rs:601` | WGSL Taylor `bid*nv*K` can overflow u32; chunking logic doesn't guard against this | Silent index wraparound for moderate-size problems on WebGPU | No | No u32-boundary test |
| 8 | `taylor_ops.rs:652-670` | `taylor_hypot` computes intermediate `a²+b²` without rescaling | All derivative coefficients silently zeroed for large inputs; underflow produces infinity for small inputs | No | No large/small-input Taylor hypot test |
| 9 | `stde/pipeline.rs:97` | `estimate_weighted` divides `w_sum2 / w_sum` when all weights are zero | Produces NaN internally, returns silently wrong zero estimate | Yes (numerical-stde + correctness-stde) | No all-zero-weight test |
| 10 | `stde/types.rs:78`, `stde/pipeline.rs:102` | Welford `m2` can go slightly negative from FP cancellation; `sqrt(negative)` → NaN | Standard error becomes NaN, poisoning EstimatorResult | No | No near-identical-samples test |

### Low

| # | File:Line | Bug | Impact | Cross-validated | Test gap |
|---|-----------|-----|--------|-----------------|----------|
| 11 | `bytecode_tape/thread_local.rs:48-50`, `tape.rs:257-259` | Thread-local borrow guard is per-thread not per-type — false reentrance panic | Prevents `BReverse<f64>` inside `BReverse<f32>` recording on same thread | No | No cross-type nesting test |
| 12 | `bytecode_tape/serde_support.rs:100-119` | Deserialization skips validation for Custom opcodes | Maliciously crafted tapes with Custom opcodes can cause confusing panics | No | No adversarial serde test |
| 13 | `stde/laplacian.rs:274` | Hardcoded `1e-12` epsilon for Gram-Schmidt too small for f32 | Hutch++ retains numerically meaningless vectors for f32 inputs, corrupting trace estimate | No | No f32 Hutch++ test |
| 14 | `stde/higher_order.rs:39` | No f32 precision guard on factorial for k≥13 — 13! exceeds f32 mantissa | Higher-order derivative extraction loses significant digits for f32 at order ≥13 | No | No f32 high-order test |
| 15 | `echidna-optim/src/solvers/lbfgs.rs:138-144` | L-BFGS rho = 1/sy only checks `sy > 0`, not near-zero guard | rho overflows to Inf for near-zero curvature, corrupting search direction | No | No near-zero curvature L-BFGS test |
| 16 | `echidna-optim/src/linalg.rs:28` | LU singularity threshold `sqrt(epsilon)` = 0.00035 for f32 rejects merely ill-conditioned matrices | f32 solvers fail on matrices that are solvable | No | No f32 ill-conditioned solve test |

### Retracted (false positives identified during collation)

| # | Original claim | Why retracted |
|---|---------------|---------------|
| R1 | `Rem` classified as `ZeroDerivative` in sparse.rs is wrong because it has nonzero partials (numerical-tape) | `ZeroDerivative` refers to Hessian sparsity (second derivatives). Rem's partials (1, -trunc(a/b)) are piecewise constant, so all second derivatives are zero a.e. Confirmed correct by math-tape and math-sparse agents. Jacobian computation uses reverse_partials directly and is unaffected. |

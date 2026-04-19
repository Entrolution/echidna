# Security Policy

## Supported Versions

| Crate           | Version   | Supported |
|-----------------|-----------|-----------|
| `echidna`       | >= 0.9.0  | Yes       |
| `echidna-optim` | >= 0.12.0 | Yes       |
| `echidna`       | < 0.9.0   | No        |
| `echidna-optim` | < 0.12.0  | No        |

Only the latest release of each crate receives security updates.
`echidna` 0.9.0 and `echidna-optim` 0.12.0 are a coordinated release
(`echidna-optim` 0.12.0 depends on `echidna = "0.9.0"`).

### Known issues in unsupported versions

Pre-0.9.0 `echidna` carries the following known numerical correctness
bugs — see the CHANGELOG for per-version detail:

- **0.8.2**: GPU Taylor `HYPOT` higher-order coefficients overflow
  to Inf/NaN at extreme magnitudes (`|a.v[0]| ~ 1e20` in f32); GPU
  Taylor `HYPOT` at function-domain boundaries (`hypot(Inf, finite)`,
  `hypot(0, 0)` with non-zero seed) silently diverges from CPU.
- **0.8.1**: atan derivative silently returns 0 for `|x| > 1.34e154`;
  powf derivative silently returns 0 when `x^b` underflows; WGSL
  forward/reverse/hvp batch dispatch produces corrupted results when
  `batch_size × num_variables > 2³²`; Taylor max/min returns NaN
  instead of valid value; Revolve checkpointing uses `O(num_steps)`
  memory instead of `O(num_checkpoints)`.
- **0.8.0**: GPU cbrt HVP second derivative is wrong; asin/acos/atanh
  lose precision near domain boundaries; CUDA Taylor codegen
  truncates 64-bit offsets.
- **< 0.8.0**: additional issues documented in the changelog.

Pre-0.12.0 `echidna-optim` silently collapses solver failure modes:
piggyback / implicit / sparse-implicit solve functions returned
`Option<T>` where any failure — primal divergence, tangent
divergence, max-iter exhaustion, factor failure, residual exceedance
— mapped to `None`, making recovery decisions platform-dependent.
0.12.0 migrated all public entry points to `Result<T, E>` with
typed error variants (`PiggybackError`, `ImplicitError`,
`SparseImplicitError`) carrying iteration, norm, and dimension
diagnostics. Additionally, `linalg::lu_factor` pre-0.12.0 accepted
non-finite pivots (NaN / ±Inf passed both the exact-zero and
tolerance checks and propagated through stored LU factors),
returning NaN-tainted results under an `Ok` / `Some` label.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **security@entrolution.com** with details of the vulnerability.
3. Include steps to reproduce, if possible.

We aim to acknowledge reports within 48 hours and provide a fix or mitigation within 7 days for critical issues.

## Security Practices

- NaN propagation for undefined derivatives (no panics on hot paths).
- All floating-point operations use standard Rust primitives.

### Unsafe Usage

echidna uses `unsafe` in the following scoped contexts:

| Location | Purpose |
|----------|---------|
| `tape.rs`, `bytecode_tape/thread_local.rs`, `taylor_dyn.rs` | Thread-local mutable pointer dereference for tape/arena access. Each is RAII-guarded: the pointer is valid for the lifetime of the enclosing scope guard. |
| `checkpoint.rs` | Byte-level transmutation (`&[F]` ↔ `&[u8]`) for disk-backed checkpoint serialisation. Relies on `F: Float` being `f32`/`f64` (IEEE 754, no padding). |
| `gpu/cuda_backend.rs` | FFI kernel launches via `cudarc`. Each call passes validated buffer sizes and grid dimensions to the CUDA driver. |
| `traits/simba_impls.rs` | `extract_unchecked` / `replace_unchecked` for simba's `SimdValue` trait. Scalar types have only one lane, so the index is always 0. |

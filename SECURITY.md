# Security Policy

## Supported Versions

| Crate           | Version    | Supported |
|-----------------|------------|-----------|
| `echidna`       | >= 0.15.0  | Yes       |
| `echidna-optim` | >= 0.15.0  | Yes       |
| `echidna`       | < 0.15.0   | No        |
| `echidna-optim` | < 0.15.0   | No        |

Only the latest patch of the current minor is supported; 0.14.0 shipped an
inaccurate security note about RUSTSEC-2026-0097's reachability (corrected
in 0.14.1's changelog).

Only the latest release of each crate receives security updates.
`echidna` 0.15.0 and `echidna-optim` 0.15.0 are a coordinated release
(`echidna-optim` 0.15.0 depends on `echidna = "0.15.0"`).

### Known issues in unsupported versions

**0.13.x** is unsupported because 0.14 supersedes it. 0.13.0 was a minor
bump over 0.12.0: the optional `nalgebra` and `simba` integrations moved to
new major versions (`nalgebra` 0.35, `simba` 0.10), an ABI-breaking change
for consumers of those features, with no echidna API or numerical behaviour
change.

**0.12.0** has no known numerical correctness bugs; it is unsupported only
because 0.13.0 supersedes it (dependency currency — `nalgebra` 0.35 /
`simba` 0.10).

Every pre-0.12.0 `echidna` release carries the numerical correctness
bugs fixed in 0.12.0 — out-of-domain derivatives (the restricted
logarithms, `atanh`, `acosh`) returning a finite partial instead of
`NaN`, with the CPU and GPU backends disagreeing; a sign-bit-dependent
`abs` subgradient at `0` (`±1` rather than the minimal-norm `0`); and a
broad set of GPU / series edge cases (`powf` at a zero or negative base,
`asinh` / `acosh` overflow at large arguments, `signum(-0.0)`, ties-away
`round`, `max` / `min` NaN tie-breaks, `sqrt` at a zero primal, `%` by a
zero divisor, and out-of-domain or higher-order-`hypot` `Taylor` /
`Laurent` jets). See the 0.12.0 CHANGELOG for the full list. Older
releases additionally carry version-specific issues:

- **0.11.0 / 0.10.0**: no version-specific numerical bugs beyond the
  pre-0.12.0 set above; 0.10.0 → 0.11.0 only added simba trait
  implementations for `DualVec` and changed no numerical behaviour.
- **0.9.0**: targets wgpu 28, which is no longer maintained upstream
  and received no patches after `wgpu 29.0.0` (2026-03-18).
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

`echidna-optim` 0.13.2 has no known bugs; it is unsupported only because the
coordinated 0.13.3 release supersedes it.

Pre-0.13.2 `echidna-optim` carries solver-robustness bugs fixed in
0.13.2: the trust-region solver accepted an out-of-range `eta` (silently
stalling to `MaxIterations`), the L-BFGS curvature filter admitted
near-orthogonal `(s, y)` pairs that corrupted the search direction, the
backtracking line search accepted a misconfigured `rho >= 1` (infinite
loop) or `alpha_min <= 0` (degenerate zero-step), `piggyback_tangent_solve`
could return a truncated (unconverged) JVP from a warm-started primal, and
the sparse implicit-differentiation solves could return `Ok` with NaN
entries. See the 0.12.0 CHANGELOG.

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

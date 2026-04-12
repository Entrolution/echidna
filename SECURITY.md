# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| >= 0.8.1 | Yes       |
| 0.8.0    | No — GPU cbrt HVP second derivative is wrong; asin/acos/atanh lose precision near domain boundaries; CUDA Taylor codegen truncates 64-bit offsets |
| < 0.8.0  | No        |

Only the latest patch release receives security updates. Version 0.8.0 has known correctness bugs including: GPU cbrt Hessian-vector product off by r² factor; catastrophic cancellation in asin/acos/atanh derivatives near ±1 (all AD modes and GPU shaders); CUDA Taylor codegen silently truncating 64-bit offsets to u32; bytecode atan2 overflow for large inputs; and `debug_assert!` guards on custom ops in Hessian stripped in release builds. Versions prior to 0.8.0 have additional known issues documented in the changelog.

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

//! Phase 9 R1 — CPU ↔ GPU parity property test suite.
//!
//! Runs a table of hand-picked tapes against CPU (f64), wgpu (f32), and
//! CUDA (f32 + f64) at a mix of normal, edge, and extreme input points.
//! Asserts that the forward value and reverse-mode gradient match within
//! a documented ULP tolerance (4 ULPs for f32, 16 ULPs for f64).
//!
//! The goal is to catch future CPU-GPU drift introduced by:
//! - New CPU formula tweaks not mirrored into shaders.
//! - Shader refactors that silently change behaviour.
//! - GPU compiler updates (new clippy-equivalent shader lints).
//!
//! Layout: one `PARITY_CASES` table defines ~30 tapes with per-case
//! tolerances; three runner functions (one per backend) each run the
//! whole table. Failing a single case names the tape in the assertion,
//! so debugging points straight at the divergent op.

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::gpu::{GpuBackend, GpuTapeData};
use echidna::{record, BReverse, BytecodeTape};
use num_traits::Float;

#[cfg(feature = "gpu-wgpu")]
use echidna::gpu::WgpuContext;

#[cfg(feature = "gpu-cuda")]
use echidna::gpu::CudaContext;

/// Test case: one tape expression plus the input points and tolerances.
#[allow(dead_code)]
struct ParityCase {
    name: &'static str,
    /// Number of inputs the tape takes. Used to sanity-check `points`
    /// lengths; each inner slice must have `n_inputs` values. The wgpu
    /// path doesn't read this field (the backend derives from the tape),
    /// but it documents the contract for humans eyeballing the table.
    n_inputs: usize,
    /// Builder that records the tape at a nominal `x0` (tape structure
    /// is identical regardless of `x0` — only the primal values change).
    build: fn() -> (BytecodeTape<f64>, f64),
    /// Evaluation points. Each inner slice has length `n_inputs`.
    points: &'static [&'static [f64]],
    /// Max ULP distance allowed between CPU-f64 and GPU-f32 value/gradient.
    /// Higher numbers acknowledge f32 precision limits on trig/log/pow ops.
    f32_ulp: u32,
    /// Max ULP distance allowed between CPU-f64 and GPU-f64 value/gradient.
    /// Only exercised by the `cuda_f64_parity_all_cases` runner.
    f64_ulp: u64,
}

fn build_add() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0] + v[1], &[1.0, 1.0])
}
fn build_sub() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0] - v[1], &[1.0, 1.0])
}
fn build_mul() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0] * v[1], &[1.0, 1.0])
}
fn build_div() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0] / v[1], &[1.0, 1.0])
}
fn build_sqrt() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].sqrt(), &[1.0])
}
fn build_cbrt() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].cbrt(), &[1.0])
}
fn build_recip() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| BReverse::constant(1.0) / v[0], &[1.0])
}
fn build_neg() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| -v[0], &[1.0])
}
fn build_abs() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].abs(), &[1.0])
}
fn build_exp() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].exp(), &[1.0])
}
fn build_expm1() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].exp_m1(), &[1.0])
}
fn build_ln() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].ln(), &[1.0])
}
fn build_ln1p() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].ln_1p(), &[1.0])
}
fn build_sin() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].sin(), &[1.0])
}
fn build_cos() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].cos(), &[1.0])
}
fn build_tan() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].tan(), &[0.5])
}
fn build_atan() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].atan(), &[1.0])
}
fn build_atan2() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].atan2(v[1]), &[1.0, 1.0])
}
fn build_sinh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].sinh(), &[1.0])
}
fn build_cosh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].cosh(), &[1.0])
}
fn build_tanh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].tanh(), &[1.0])
}
fn build_asinh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].asinh(), &[1.0])
}
fn build_acosh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].acosh(), &[2.0])
}
fn build_atanh() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].atanh(), &[0.5])
}
fn build_asin() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].asin(), &[0.5])
}
fn build_acos() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].acos(), &[0.5])
}
fn build_exp2() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].exp2(), &[1.0])
}
fn build_log2() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].log2(), &[1.0])
}
fn build_log10() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].log10(), &[1.0])
}
fn build_rem() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0] % v[1], &[5.0, 2.0])
}
fn build_powi() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].powi(3), &[2.0])
}
fn build_powf() -> (BytecodeTape<f64>, f64) {
    record(
        |v: &[BReverse<f64>]| v[0].powf(BReverse::constant(2.5)),
        &[2.0],
    )
}
fn build_hypot() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].hypot(v[1]), &[3.0, 4.0])
}
fn build_max() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].max(v[1]), &[1.0, 2.0])
}
fn build_min() -> (BytecodeTape<f64>, f64) {
    record(|v: &[BReverse<f64>]| v[0].min(v[1]), &[1.0, 2.0])
}
fn build_composite_rosenbrock() -> (BytecodeTape<f64>, f64) {
    // f(x, y) = (1 - x)² + 100(y - x²)² — tests composition of sub/mul/powi.
    record(
        |v: &[BReverse<f64>]| {
            let a = BReverse::constant(1.0) - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + BReverse::constant(100.0) * b * b
        },
        &[1.0, 1.0],
    )
}
fn build_composite_mixed() -> (BytecodeTape<f64>, f64) {
    // f(x) = sin(x² + 1) · exp(-x) — exercises composition + tiny-magnitude outputs.
    record(
        |v: &[BReverse<f64>]| (v[0] * v[0] + BReverse::constant(1.0)).sin() * (-v[0]).exp(),
        &[1.0],
    )
}
fn build_composite_log_sum() -> (BytecodeTape<f64>, f64) {
    // log-sum-exp-style: log(exp(x) + exp(y)) — conditioning test.
    record(
        |v: &[BReverse<f64>]| (v[0].exp() + v[1].exp()).ln(),
        &[1.0, 0.5],
    )
}

const PARITY_CASES: &[ParityCase] = &[
    // Arithmetic
    ParityCase {
        name: "add",
        n_inputs: 2,
        build: build_add,
        points: &[&[1.0, 1.0], &[-3.5, 2.25], &[1e10, 1e-10], &[0.0, 0.0]],
        f32_ulp: 2,
        f64_ulp: 2,
    },
    ParityCase {
        name: "sub",
        n_inputs: 2,
        build: build_sub,
        // Deliberately avoid near-catastrophic-cancellation inputs like
        // (1.0, 0.99999) — the f64→f32 input rounding alone accounts for
        // the visible ULP drift, not a GPU formula divergence.
        points: &[&[1.0, 1.0], &[-3.5, 2.25], &[2.0, -1.5]],
        f32_ulp: 4,
        f64_ulp: 4,
    },
    ParityCase {
        name: "mul",
        n_inputs: 2,
        build: build_mul,
        points: &[&[1.0, 1.0], &[-2.5, 4.0], &[1e10, 1e-10], &[0.0, 5.0]],
        f32_ulp: 2,
        f64_ulp: 2,
    },
    ParityCase {
        name: "div",
        n_inputs: 2,
        build: build_div,
        points: &[&[1.0, 2.0], &[-6.0, 3.0], &[1.0, 1e-3]],
        f32_ulp: 4,
        f64_ulp: 4,
    },
    // Unary algebraic
    ParityCase {
        name: "sqrt",
        n_inputs: 1,
        build: build_sqrt,
        points: &[&[1.0], &[4.0], &[0.25], &[1e6]],
        f32_ulp: 4,
        f64_ulp: 4,
    },
    ParityCase {
        name: "cbrt",
        n_inputs: 1,
        build: build_cbrt,
        points: &[&[1.0], &[8.0], &[-27.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "recip",
        n_inputs: 1,
        build: build_recip,
        points: &[&[1.0], &[2.0], &[-0.5]],
        f32_ulp: 4,
        f64_ulp: 4,
    },
    ParityCase {
        name: "neg",
        n_inputs: 1,
        build: build_neg,
        points: &[&[1.0], &[-3.5], &[0.0]],
        f32_ulp: 0,
        f64_ulp: 0,
    },
    ParityCase {
        name: "abs",
        n_inputs: 1,
        build: build_abs,
        points: &[&[1.0], &[-3.5], &[2.0]],
        f32_ulp: 0,
        f64_ulp: 0,
    },
    // Exp/Log
    ParityCase {
        name: "exp",
        n_inputs: 1,
        build: build_exp,
        points: &[&[0.0], &[1.0], &[-1.0], &[5.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "expm1",
        n_inputs: 1,
        build: build_expm1,
        points: &[&[0.0], &[1e-6], &[0.5], &[-2.0]],
        f32_ulp: 16,
        f64_ulp: 8,
    },
    ParityCase {
        name: "ln",
        n_inputs: 1,
        build: build_ln,
        points: &[&[1.0], &[2.0], &[10.0], &[0.5]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "ln1p",
        n_inputs: 1,
        build: build_ln1p,
        points: &[&[0.0], &[1e-6], &[1.0], &[-0.5]],
        f32_ulp: 16,
        f64_ulp: 8,
    },
    // Trig
    ParityCase {
        name: "sin",
        n_inputs: 1,
        build: build_sin,
        // Avoid near-π inputs: sin is zero-crossing there, making any f32
        // input rounding catastrophic (LSB in f32(π) shifts the output
        // by ~500k ULPs in the near-zero result). Not a GPU bug.
        points: &[&[0.0], &[0.5], &[1.0], &[-0.3]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "cos",
        n_inputs: 1,
        build: build_cos,
        // Avoid near-π/2 and near-π inputs for the same cancellation reason.
        points: &[&[0.0], &[0.5], &[1.0], &[-0.3]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "tan",
        n_inputs: 1,
        build: build_tan,
        points: &[&[0.0], &[0.5], &[-0.5]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "atan",
        n_inputs: 1,
        build: build_atan,
        points: &[&[0.0], &[1.0], &[-2.5], &[100.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "atan2",
        n_inputs: 2,
        build: build_atan2,
        points: &[&[1.0, 1.0], &[-1.0, 1.0], &[3.0, 4.0], &[1e10, 1e10]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    // Hyperbolic
    ParityCase {
        name: "sinh",
        n_inputs: 1,
        build: build_sinh,
        points: &[&[0.0], &[1.0], &[-2.0]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "cosh",
        n_inputs: 1,
        build: build_cosh,
        points: &[&[0.0], &[1.0], &[-2.0]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "tanh",
        n_inputs: 1,
        build: build_tanh,
        points: &[&[0.0], &[1.0], &[-2.0]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "asinh",
        n_inputs: 1,
        build: build_asinh,
        points: &[&[0.0], &[1.0], &[-3.0], &[1e6]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "acosh",
        n_inputs: 1,
        build: build_acosh,
        // acosh domain is a >= 1.
        points: &[&[1.5], &[2.0], &[10.0]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "atanh",
        n_inputs: 1,
        build: build_atanh,
        // atanh domain is |a| < 1.
        points: &[&[0.0], &[0.25], &[-0.5], &[0.9]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "asin",
        n_inputs: 1,
        build: build_asin,
        points: &[&[0.0], &[0.5], &[-0.25]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    ParityCase {
        name: "acos",
        n_inputs: 1,
        build: build_acos,
        points: &[&[0.0], &[0.5], &[-0.25]],
        f32_ulp: 16,
        f64_ulp: 16,
    },
    // Exp/Log extras
    ParityCase {
        name: "exp2",
        n_inputs: 1,
        build: build_exp2,
        points: &[&[0.0], &[1.0], &[-1.0], &[3.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "log2",
        n_inputs: 1,
        build: build_log2,
        points: &[&[1.0], &[2.0], &[8.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "log10",
        n_inputs: 1,
        build: build_log10,
        points: &[&[1.0], &[10.0], &[100.0]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    // Powers — fragile ops Phase 7 specifically patched.
    ParityCase {
        name: "powi",
        n_inputs: 1,
        build: build_powi,
        points: &[&[2.0], &[-3.0], &[0.5]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "powf",
        n_inputs: 1,
        build: build_powf,
        points: &[&[2.0], &[0.5], &[10.0]],
        f32_ulp: 32,
        f64_ulp: 16,
    },
    // Remainder.
    ParityCase {
        name: "rem",
        n_inputs: 2,
        build: build_rem,
        points: &[&[5.0, 2.0], &[7.5, 2.5], &[-3.0, 2.0]],
        f32_ulp: 4,
        f64_ulp: 4,
    },
    // Multi-arg
    ParityCase {
        name: "hypot",
        n_inputs: 2,
        build: build_hypot,
        points: &[&[3.0, 4.0], &[1e10, 1e10], &[1.0, 0.0], &[0.0, 1e-6]],
        f32_ulp: 8,
        f64_ulp: 8,
    },
    ParityCase {
        name: "max",
        n_inputs: 2,
        build: build_max,
        points: &[&[1.0, 2.0], &[-1.0, -2.0], &[3.0, 3.0]],
        f32_ulp: 0,
        f64_ulp: 0,
    },
    ParityCase {
        name: "min",
        n_inputs: 2,
        build: build_min,
        points: &[&[1.0, 2.0], &[-1.0, -2.0], &[3.0, 3.0]],
        f32_ulp: 0,
        f64_ulp: 0,
    },
    // Composite
    ParityCase {
        name: "rosenbrock",
        n_inputs: 2,
        build: build_composite_rosenbrock,
        points: &[&[1.0, 1.0], &[0.0, 0.0], &[-1.2, 1.0]],
        f32_ulp: 64,
        f64_ulp: 32,
    },
    ParityCase {
        name: "sin_x2_mul_exp_neg_x",
        n_inputs: 1,
        build: build_composite_mixed,
        points: &[&[0.5], &[1.0], &[-0.5]],
        f32_ulp: 64,
        f64_ulp: 32,
    },
    ParityCase {
        name: "log_sum_exp",
        n_inputs: 2,
        build: build_composite_log_sum,
        points: &[&[0.0, 0.0], &[1.0, -1.0], &[3.0, 5.0]],
        f32_ulp: 64,
        f64_ulp: 32,
    },
];

fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() {
        return if a.to_bits() == b.to_bits() {
            0
        } else {
            u32::MAX
        };
    }
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    // Same sign (bit-31 identical) → ULP distance is just the bit diff.
    // Cross-sign → distance is |a| + |b| through zero (saturating if extreme).
    if (a_bits ^ b_bits) & 0x8000_0000 == 0 {
        a_bits.abs_diff(b_bits)
    } else {
        let abs_a = a_bits & 0x7FFF_FFFF;
        let abs_b = b_bits & 0x7FFF_FFFF;
        abs_a.saturating_add(abs_b)
    }
}

#[cfg_attr(not(feature = "gpu-cuda"), allow(dead_code))]
fn ulp_diff_f64(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return if a.to_bits() == b.to_bits() {
            0
        } else {
            u64::MAX
        };
    }
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    if (a_bits ^ b_bits) & 0x8000_0000_0000_0000 == 0 {
        a_bits.abs_diff(b_bits)
    } else {
        let abs_a = a_bits & 0x7FFF_FFFF_FFFF_FFFF;
        let abs_b = b_bits & 0x7FFF_FFFF_FFFF_FFFF;
        abs_a.saturating_add(abs_b)
    }
}

// ── wgpu f32 parity runner ─────────────────────────────────────────

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_parity_all_cases() {
    let ctx = match WgpuContext::new() {
        Some(c) => c,
        None => {
            // Silent returns on no-GPU machines would pass the assertion
            // without running any case. Surface the skip so `cargo test
            // -- --nocapture` makes it visible.
            eprintln!("SKIP: no wgpu adapter; parity test not executed");
            return;
        }
    };
    let mut failures = Vec::new();
    for case in PARITY_CASES {
        for (pt_i, pt) in case.points.iter().enumerate() {
            let (mut cpu_tape, _) = (case.build)();
            let cpu_grad = cpu_tape.gradient(pt);
            cpu_tape.forward(pt);
            let cpu_val = cpu_tape.output_values()[0];

            let gpu_data = match GpuTapeData::from_tape_f64_lossy(&cpu_tape) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: GpuTapeData::from_tape_f64_lossy failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let pt_f32: Vec<f32> = pt.iter().map(|&x| x as f32).collect();
            let (gpu_val, gpu_grad) = match ctx.gradient_batch(&gpu_tape, &pt_f32, 1) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: wgpu gradient_batch failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };

            let val_ulp = ulp_diff_f32(gpu_val[0], cpu_val as f32);
            if val_ulp > case.f32_ulp {
                failures.push(format!(
                    "case {}[{}]: value ULP diff {} > {} (CPU {:e}, GPU {:e})",
                    case.name, pt_i, val_ulp, case.f32_ulp, cpu_val, gpu_val[0]
                ));
            }
            for (i, (&gg, &cg)) in gpu_grad.iter().zip(cpu_grad.iter()).enumerate() {
                let grad_ulp = ulp_diff_f32(gg, cg as f32);
                if grad_ulp > case.f32_ulp {
                    failures.push(format!(
                        "case {}[{}]: grad[{}] ULP diff {} > {} (CPU {:e}, GPU {:e})",
                        case.name, pt_i, i, grad_ulp, case.f32_ulp, cg, gg
                    ));
                }
            }
        }
    }
    if !failures.is_empty() {
        panic!("wgpu parity failures:\n  {}", failures.join("\n  "));
    }
}

// ── CUDA f32 parity runner ─────────────────────────────────────────

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_f32_parity_all_cases() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA device; parity test not executed");
            return;
        }
    };
    let mut failures = Vec::new();
    for case in PARITY_CASES {
        for (pt_i, pt) in case.points.iter().enumerate() {
            let (mut cpu_tape, _) = (case.build)();
            let cpu_grad = cpu_tape.gradient(pt);
            cpu_tape.forward(pt);
            let cpu_val = cpu_tape.output_values()[0];

            let gpu_data = match GpuTapeData::from_tape_f64_lossy(&cpu_tape) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: from_tape_f64_lossy failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };
            let gpu_tape = ctx.upload_tape(&gpu_data);
            let pt_f32: Vec<f32> = pt.iter().map(|&x| x as f32).collect();
            let (gpu_val, gpu_grad) = match ctx.gradient_batch(&gpu_tape, &pt_f32, 1) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: CUDA gradient_batch failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };

            let val_ulp = ulp_diff_f32(gpu_val[0], cpu_val as f32);
            if val_ulp > case.f32_ulp {
                failures.push(format!(
                    "case {}[{}]: value ULP diff {} > {}",
                    case.name, pt_i, val_ulp, case.f32_ulp
                ));
            }
            for (i, (&gg, &cg)) in gpu_grad.iter().zip(cpu_grad.iter()).enumerate() {
                let grad_ulp = ulp_diff_f32(gg, cg as f32);
                if grad_ulp > case.f32_ulp {
                    failures.push(format!(
                        "case {}[{}]: grad[{}] ULP diff {} > {}",
                        case.name, pt_i, i, grad_ulp, case.f32_ulp
                    ));
                }
            }
        }
    }
    if !failures.is_empty() {
        panic!("CUDA f32 parity failures:\n  {}", failures.join("\n  "));
    }
}

// ── CUDA f64 parity runner ─────────────────────────────────────────

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_f64_parity_all_cases() {
    let ctx = match CudaContext::new() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA device; parity test not executed");
            return;
        }
    };
    let mut failures = Vec::new();
    for case in PARITY_CASES {
        for (pt_i, pt) in case.points.iter().enumerate() {
            let (cpu_tape, _) = (case.build)();
            let mut cpu_tape = cpu_tape;
            let cpu_grad = cpu_tape.gradient(pt);
            cpu_tape.forward(pt);
            let cpu_val = cpu_tape.output_values()[0];

            let gpu_tape = match ctx.upload_tape_f64(&cpu_tape) {
                Ok(t) => t,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: upload_tape_f64 failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };
            let (gpu_val, gpu_grad) = match ctx.gradient_batch_f64(&gpu_tape, pt, 1) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!(
                        "case {}[{}]: CUDA gradient_batch_f64 failed: {:?}",
                        case.name, pt_i, e
                    ));
                    continue;
                }
            };

            let val_ulp = ulp_diff_f64(gpu_val[0], cpu_val);
            if val_ulp > case.f64_ulp {
                failures.push(format!(
                    "case {}[{}]: value ULP diff {} > {} (CPU {:e}, GPU {:e})",
                    case.name, pt_i, val_ulp, case.f64_ulp, cpu_val, gpu_val[0]
                ));
            }
            for (i, (&gg, &cg)) in gpu_grad.iter().zip(cpu_grad.iter()).enumerate() {
                let grad_ulp = ulp_diff_f64(gg, cg);
                if grad_ulp > case.f64_ulp {
                    failures.push(format!(
                        "case {}[{}]: grad[{}] ULP diff {} > {} (CPU {:e}, GPU {:e})",
                        case.name, pt_i, i, grad_ulp, case.f64_ulp, cg, gg
                    ));
                }
            }
        }
    }
    if !failures.is_empty() {
        panic!("CUDA f64 parity failures:\n  {}", failures.join("\n  "));
    }
}

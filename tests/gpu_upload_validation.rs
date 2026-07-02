//! Upload-boundary validation: hand-built `GpuTapeData` with out-of-range
//! indices must be rejected before any device buffer is created. The CUDA
//! kernels index raw device memory with these fields, so an unvalidated
//! upload would turn a corrupt index into an out-of-bounds device access.

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use echidna::gpu::{GpuBackend, GpuTapeData};

/// Minimal valid 3-slot tape — `f(x) = x * 3` as [Input, Const, Mul] —
/// with one corruption applied.
fn tape_data(corrupt: impl FnOnce(&mut GpuTapeData)) -> GpuTapeData {
    let mut d = GpuTapeData {
        opcodes: vec![0, 1, 4], // Input, Const, Mul
        arg0: vec![u32::MAX, u32::MAX, 0],
        arg1: vec![u32::MAX, u32::MAX, 1],
        constants: vec![0.0, 3.0, 0.0],
        num_ops: 3,
        num_inputs: 1,
        num_variables: 3,
        output_index: 2,
        output_indices: vec![2],
    };
    corrupt(&mut d);
    d
}

fn assert_upload_panics<B: GpuBackend>(ctx: &B, data: &GpuTapeData, what: &str) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ctx.upload_tape(data);
    }));
    let payload = result.expect_err(&format!("{what}: uploading invalid GpuTapeData must panic"));
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_default();
    assert!(
        msg.contains("invalid GpuTapeData"),
        "{what}: unexpected panic message: {msg}"
    );
}

fn check_upload_validation<B: GpuBackend>(ctx: &B) {
    // Baseline: untouched data uploads fine.
    let _ = ctx.upload_tape(&tape_data(|_| {}));

    // Operand index outside the values buffer.
    assert_upload_panics(ctx, &tape_data(|d| d.arg0[2] = 9999), "oob arg0");

    // Unary op with a stray second index: the reverse sweep accumulates
    // into `adjoints[arg1]` for every non-Powi op whose second slot is
    // not UNUSED, so this would be an out-of-bounds device write.
    assert_upload_panics(
        ctx,
        &tape_data(|d| {
            d.opcodes[2] = 24; // Sin
            d.arg1[2] = 9999;
        }),
        "unary stray arg1",
    );

    // Output index sentinel / out of range (the upload path synthesizes
    // `[output_index]` when `output_indices` is empty, so both matter).
    assert_upload_panics(
        ctx,
        &tape_data(|d| {
            d.output_index = u32::MAX;
            d.output_indices = Vec::new();
        }),
        "output_index sentinel",
    );
    assert_upload_panics(
        ctx,
        &tape_data(|d| d.output_indices = vec![3]),
        "oob output_indices",
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_upload_rejects_invalid_tape_data() {
    let ctx = match echidna::gpu::WgpuContext::new() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no wgpu adapter; upload-validation test not executed");
            return;
        }
    };
    check_upload_validation(&ctx);
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_upload_rejects_invalid_tape_data() {
    let ctx = match echidna::gpu::CudaContext::new() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA device; upload-validation test not executed");
            return;
        }
    };
    check_upload_validation(&ctx);
}

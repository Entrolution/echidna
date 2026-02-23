//! GPU acceleration for batched tape evaluation.
//!
//! Provides two backends:
//! - **wgpu** (`gpu-wgpu` feature): cross-platform (Metal, Vulkan, DX12), f32 only
//! - **CUDA** (`gpu-cuda` feature): NVIDIA only, f32 + f64
//!
//! # Context Contract
//!
//! Both `WgpuContext` and `CudaContext` follow the same method pattern:
//!
//! - `new() -> Option<Self>` — acquire a GPU device, `None` if unavailable
//! - `upload_tape(&self, data: &GpuTapeData) -> TapeBuffers` — upload tape to device
//! - `forward_batch(&self, tape, inputs, batch) -> Result<Vec<f32>, GpuError>` — batched forward
//! - `gradient_batch(&self, tape, inputs, batch) -> Result<(Vec<f32>, Vec<f32>), GpuError>` — batched gradient
//! - `sparse_jacobian(&self, tape, tape_cpu, x) -> Result<..., GpuError>` — GPU-accelerated sparse Jacobian
//! - `sparse_hessian(&self, tape, tape_cpu, x) -> Result<..., GpuError>` — GPU-accelerated sparse Hessian
//!
//! No trait is extracted yet — a `GpuBackend` trait can be added later once both
//! backends are stable and the shared surface is proven.

use crate::bytecode_tape::BytecodeTape;
use crate::opcode::OpCode;

#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

#[cfg(feature = "gpu-cuda")]
pub mod cuda_backend;

#[cfg(feature = "gpu-wgpu")]
pub use wgpu_backend::{WgpuContext, WgpuTapeBuffers};

#[cfg(feature = "gpu-cuda")]
pub use cuda_backend::{CudaContext, CudaTapeBuffers};

/// Error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU device found.
    NoDevice,
    /// Shader or kernel compilation failed.
    ShaderCompilation(String),
    /// GPU ran out of memory.
    OutOfMemory,
    /// Tape contains custom ops which cannot run on GPU.
    CustomOpsNotSupported,
    /// Backend-specific error.
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no suitable GPU device found"),
            GpuError::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
            GpuError::OutOfMemory => write!(f, "GPU out of memory"),
            GpuError::CustomOpsNotSupported => {
                write!(f, "tape contains custom ops which cannot run on GPU")
            }
            GpuError::Other(msg) => write!(f, "GPU error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Flattened tape representation for GPU upload.
///
/// All arrays are the same length (`num_ops`). The GPU shader walks index 0..num_ops
/// sequentially, executing each opcode on the per-thread values buffer.
///
/// Created via [`GpuTapeData::from_tape`] (f32) or [`GpuTapeData::from_tape_f64_lossy`] (f64→f32).
pub struct GpuTapeData {
    /// OpCode discriminants as u32 (one per tape entry).
    pub opcodes: Vec<u32>,
    /// First argument index for each operation.
    pub arg0: Vec<u32>,
    /// Second argument index for each operation.
    pub arg1: Vec<u32>,
    /// Initial values buffer (constants and zeros, f32).
    pub constants: Vec<f32>,
    /// Total number of tape entries.
    pub num_ops: u32,
    /// Number of input variables.
    pub num_inputs: u32,
    /// Total entries in the values buffer (inputs + constants + intermediates).
    pub num_variables: u32,
    /// Primary output index.
    pub output_index: u32,
    /// All output indices (for multi-output tapes).
    pub output_indices: Vec<u32>,
}

impl GpuTapeData {
    /// Convert a `BytecodeTape<f32>` to GPU-uploadable format.
    ///
    /// Returns `Err(CustomOpsNotSupported)` if the tape contains custom ops,
    /// since custom Rust closures cannot execute on GPU hardware.
    pub fn from_tape(tape: &BytecodeTape<f32>) -> Result<Self, GpuError> {
        if tape.has_custom_ops() {
            return Err(GpuError::CustomOpsNotSupported);
        }

        let opcodes_raw = tape.opcodes_slice();
        let args = tape.arg_indices_slice();
        let vals = tape.values_slice();
        let n = opcodes_raw.len();

        let opcodes: Vec<u32> = opcodes_raw.iter().map(|op| *op as u32).collect();
        let arg0: Vec<u32> = args.iter().map(|a| a[0]).collect();
        let arg1: Vec<u32> = args.iter().map(|a| a[1]).collect();
        let constants: Vec<f32> = vals.to_vec();

        Ok(GpuTapeData {
            opcodes,
            arg0,
            arg1,
            constants,
            num_ops: n as u32,
            num_inputs: tape.num_inputs() as u32,
            num_variables: tape.num_variables_count() as u32,
            output_index: tape.output_index() as u32,
            output_indices: tape.all_output_indices().to_vec(),
        })
    }

    /// Convert a `BytecodeTape<f64>` to GPU-uploadable f32 format.
    ///
    /// All f64 values are cast to f32, which loses precision. The method name
    /// makes this explicit — use the CUDA backend for native f64 support.
    ///
    /// Returns `Err(CustomOpsNotSupported)` if the tape contains custom ops.
    pub fn from_tape_f64_lossy(tape: &BytecodeTape<f64>) -> Result<Self, GpuError> {
        if tape.has_custom_ops() {
            return Err(GpuError::CustomOpsNotSupported);
        }

        let opcodes_raw = tape.opcodes_slice();
        let args = tape.arg_indices_slice();
        let vals = tape.values_slice();
        let n = opcodes_raw.len();

        let opcodes: Vec<u32> = opcodes_raw.iter().map(|op| *op as u32).collect();
        let arg0: Vec<u32> = args.iter().map(|a| a[0]).collect();
        let arg1: Vec<u32> = args.iter().map(|a| a[1]).collect();
        let constants: Vec<f32> = vals.iter().map(|&v| v as f32).collect();

        Ok(GpuTapeData {
            opcodes,
            arg0,
            arg1,
            constants,
            num_ops: n as u32,
            num_inputs: tape.num_inputs() as u32,
            num_variables: tape.num_variables_count() as u32,
            output_index: tape.output_index() as u32,
            output_indices: tape.all_output_indices().to_vec(),
        })
    }
}

/// Metadata for the tape, uploaded as a uniform buffer to GPU shaders.
///
/// Layout matches the WGSL `TapeMeta` struct (4 × u32 = 16 bytes).
#[cfg(feature = "gpu-wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TapeMeta {
    pub num_ops: u32,
    pub num_inputs: u32,
    pub num_variables: u32,
    pub num_outputs: u32,
    pub batch_size: u32,
    pub _pad: [u32; 3],
}

/// Map an [`OpCode`] to the integer constant used in WGSL/CUDA shaders.
///
/// The mapping matches the `OpCode` discriminant (`#[repr(u8)]`), cast to u32.
#[inline]
pub fn opcode_to_gpu(op: OpCode) -> u32 {
    op as u32
}

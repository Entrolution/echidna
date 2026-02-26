//! GPU acceleration for batched tape evaluation.
//!
//! Provides two backends:
//! - **wgpu** (`gpu-wgpu` feature): cross-platform (Metal, Vulkan, DX12), f32 only
//! - **CUDA** (`gpu-cuda` feature): NVIDIA only, f32 + f64
//!
//! # Context Contract
//!
//! Both [`WgpuContext`] and [`CudaContext`] implement the [`GpuBackend`] trait,
//! which defines the shared f32 operation set:
//!
//! - `new() -> Option<Self>` — acquire a GPU device (inherent, not in trait)
//! - [`upload_tape`](GpuBackend::upload_tape) — upload tape to device
//! - [`forward_batch`](GpuBackend::forward_batch) — batched forward evaluation
//! - [`gradient_batch`](GpuBackend::gradient_batch) — batched gradient (forward + reverse)
//! - [`sparse_jacobian`](GpuBackend::sparse_jacobian) — GPU-accelerated sparse Jacobian
//! - [`hvp_batch`](GpuBackend::hvp_batch) — batched Hessian-vector product
//! - [`sparse_hessian`](GpuBackend::sparse_hessian) — GPU-accelerated sparse Hessian
//! - `taylor_forward_2nd_batch` — batched second-order Taylor forward propagation (inherent, requires `stde`)
//!
//! CUDA additionally provides f64 methods as inherent methods on [`CudaContext`].
//!
//! # GPU-Accelerated STDE (requires `stde`)
//!
//! The [`stde_gpu`] module provides GPU-accelerated versions of the CPU STDE
//! functions. These use batched second-order Taylor forward propagation to
//! evaluate many directions in parallel:
//!
//! - [`stde_gpu::laplacian_gpu`] — Hutchinson trace estimator on GPU
//! - [`stde_gpu::hessian_diagonal_gpu`] — exact Hessian diagonal via basis pushforwards
//! - [`stde_gpu::laplacian_with_control_gpu`] — variance-reduced Laplacian with diagonal control variate
//!
//! The Taylor kernel propagates `(c0, c1, c2)` triples through the tape for
//! each batch element, where c2 = v^T H v / 2. All 44 opcodes are supported.

use crate::bytecode_tape::BytecodeTape;
use crate::opcode::OpCode;

#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

#[cfg(feature = "gpu-cuda")]
pub mod cuda_backend;

#[cfg(feature = "stde")]
pub mod stde_gpu;

#[cfg(feature = "gpu-wgpu")]
pub use wgpu_backend::{WgpuContext, WgpuTapeBuffers};

#[cfg(feature = "gpu-cuda")]
pub use cuda_backend::{CudaContext, CudaTapeBuffers};

/// Common interface for GPU backends (f32 operations).
///
/// Both [`WgpuContext`] and [`CudaContext`] implement this trait for the f32
/// operation set. CUDA additionally provides f64 methods as inherent methods
/// on [`CudaContext`] directly.
///
/// # Associated Type
///
/// [`TapeBuffers`](GpuBackend::TapeBuffers) is the backend-specific opaque
/// handle returned by [`upload_tape`](GpuBackend::upload_tape) and passed to
/// all dispatch methods. It holds GPU-resident buffers and is not cloneable.
///
/// # Implementing a New Backend
///
/// A backend must implement all six methods. Construction (`new()`) is not
/// part of the trait — backends may have different initialization requirements.
pub trait GpuBackend {
    /// Backend-specific uploaded tape handle.
    type TapeBuffers;

    /// Upload a tape to the GPU.
    ///
    /// The returned handle is used for all subsequent operations and holds
    /// GPU-resident buffers for the tape's opcodes, arguments, and constants.
    fn upload_tape(&self, data: &GpuTapeData) -> Self::TapeBuffers;

    /// Batched forward evaluation.
    ///
    /// `inputs` is `[f32; batch_size * num_inputs]` (row-major, one point per row).
    /// Returns output values `[f32; batch_size * num_outputs]`.
    fn forward_batch(
        &self,
        tape: &Self::TapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<Vec<f32>, GpuError>;

    /// Batched gradient (forward + reverse sweep).
    ///
    /// Returns `(outputs, gradients)` where outputs is
    /// `[f32; batch_size * num_outputs]` and gradients is
    /// `[f32; batch_size * num_inputs]`.
    fn gradient_batch(
        &self,
        tape: &Self::TapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError>;

    /// GPU-accelerated sparse Jacobian.
    ///
    /// CPU detects sparsity and computes coloring; GPU dispatches colored
    /// tangent sweeps. Returns `(output_values, pattern, jacobian_values)`.
    fn sparse_jacobian(
        &self,
        tape: &Self::TapeBuffers,
        tape_cpu: &mut BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(Vec<f32>, crate::sparse::JacobianSparsityPattern, Vec<f32>), GpuError>;

    /// Batched Hessian-vector product (forward-over-reverse).
    ///
    /// `tangent_dirs` is `[f32; batch_size * num_inputs]` — one direction per
    /// batch element. Returns `(gradients, hvps)` each
    /// `[f32; batch_size * num_inputs]`.
    fn hvp_batch(
        &self,
        tape: &Self::TapeBuffers,
        x: &[f32],
        tangent_dirs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError>;

    /// GPU-accelerated sparse Hessian.
    ///
    /// CPU detects Hessian sparsity and computes distance-2 coloring; GPU
    /// dispatches HVP sweeps. Returns `(value, gradient, pattern, hessian_values)`.
    fn sparse_hessian(
        &self,
        tape: &Self::TapeBuffers,
        tape_cpu: &mut BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(f32, Vec<f32>, crate::sparse::SparsityPattern, Vec<f32>), GpuError>;
}

/// Result of a batched second-order Taylor forward propagation.
///
/// Each field has `batch_size * num_outputs` elements (row-major: one row per batch element).
/// The Taylor convention is `c[k] = f^(k)(t₀) / k!`, so:
/// - `values[i]` = f(x) (primal value)
/// - `c1s[i]` = directional first derivative
/// - `c2s[i]` = directional second derivative / 2
pub struct TaylorBatchResult<F> {
    /// Primal output values `[batch_size * num_outputs]`.
    pub values: Vec<F>,
    /// First-order Taylor coefficients `[batch_size * num_outputs]`.
    pub c1s: Vec<F>,
    /// Second-order Taylor coefficients `[batch_size * num_outputs]`.
    pub c2s: Vec<F>,
}

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

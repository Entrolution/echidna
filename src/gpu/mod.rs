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
//! - [`taylor_forward_2nd_batch`](GpuBackend::taylor_forward_2nd_batch) — batched second-order Taylor forward propagation (requires `stde`)
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

#[cfg(feature = "stde")]
pub mod taylor_codegen;

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

    /// Batched second-order Taylor forward propagation.
    ///
    /// Each batch element pushes one direction through the tape, producing
    /// a Taylor jet with 3 coefficients (c0=value, c1=first derivative,
    /// c2=second derivative / 2).
    ///
    /// `primal_inputs` is `[f32; batch_size * num_inputs]` — primals for each element.
    /// `direction_seeds` is `[f32; batch_size * num_inputs]` — c1 seeds for each element.
    ///
    /// Returns `TaylorBatchResult` with `values`, `c1s`, `c2s` each of size
    /// `[f32; batch_size * num_outputs]`.
    /// Batched second-order Taylor forward propagation.
    ///
    /// Default implementation delegates to `taylor_forward_kth_batch(order=3)`.
    #[cfg(feature = "stde")]
    fn taylor_forward_2nd_batch(
        &self,
        tape: &Self::TapeBuffers,
        primal_inputs: &[f32],
        direction_seeds: &[f32],
        batch_size: u32,
    ) -> Result<TaylorBatchResult<f32>, GpuError> {
        let kth =
            self.taylor_forward_kth_batch(tape, primal_inputs, direction_seeds, batch_size, 3)?;
        let mut coeffs = kth.coefficients.into_iter();
        Ok(TaylorBatchResult {
            values: coeffs.next().unwrap(),
            c1s: coeffs.next().unwrap(),
            c2s: coeffs.next().unwrap(),
        })
    }

    /// Batched K-th order Taylor forward propagation.
    ///
    /// Supports `order` in 1..=5. Each batch element pushes one direction through
    /// the tape, producing K Taylor coefficients (c0, c1, ..., c_{K-1}).
    ///
    /// `primal_inputs` is `[f32; batch_size * num_inputs]` — primals for each element.
    /// `direction_seeds` is `[f32; batch_size * num_inputs]` — c1 seeds for each element.
    ///
    /// Returns `TaylorKthBatchResult` with `coefficients[k]` of size
    /// `[f32; batch_size * num_outputs]` for each k in 0..order.
    #[cfg(feature = "stde")]
    fn taylor_forward_kth_batch(
        &self,
        tape: &Self::TapeBuffers,
        primal_inputs: &[f32],
        direction_seeds: &[f32],
        batch_size: u32,
        order: usize,
    ) -> Result<TaylorKthBatchResult<f32>, GpuError>;
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

/// Result of a batched K-th order Taylor forward propagation.
///
/// `coefficients[k]` has `batch_size * num_outputs` elements for coefficient index k.
/// The Taylor convention is `c[k] = f^(k)(t₀) / k!`.
#[cfg(feature = "stde")]
pub struct TaylorKthBatchResult<F> {
    /// Taylor coefficients: `coefficients[k]` is the k-th order coefficient vector
    /// with `batch_size * num_outputs` elements.
    pub coefficients: Vec<Vec<F>>,
    /// The Taylor order (number of coefficients per output).
    pub order: usize,
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
    /// Build `GpuTapeData` from a tape's structural data and pre-converted constants.
    fn build_from_tape<F: crate::float::Float>(
        tape: &BytecodeTape<F>,
        constants: Vec<f32>,
    ) -> Self {
        let opcodes_raw = tape.opcodes_slice();
        let args = tape.arg_indices_slice();
        let n = opcodes_raw.len();

        GpuTapeData {
            opcodes: opcodes_raw.iter().map(|op| *op as u32).collect(),
            arg0: args.iter().map(|a| a[0]).collect(),
            arg1: args.iter().map(|a| a[1]).collect(),
            constants,
            // SAFETY(u32 cast): n is the number of tape opcodes. Exceeding u32::MAX (~4.3B)
            // would require ~17 GB of opcode storage alone, which is impractical.
            num_ops: n as u32,
            // SAFETY(u32 cast): num_inputs, num_variables, and output_index are bounded
            // by tape size (same order as num_ops), which cannot practically reach u32::MAX.
            num_inputs: tape.num_inputs() as u32,
            num_variables: tape.num_variables_count() as u32,
            output_index: tape.output_index() as u32,
            output_indices: tape.all_output_indices().to_vec(),
        }
    }

    /// Convert a `BytecodeTape<f32>` to GPU-uploadable format.
    ///
    /// Returns `Err(CustomOpsNotSupported)` if the tape contains custom ops,
    /// since custom Rust closures cannot execute on GPU hardware.
    pub fn from_tape(tape: &BytecodeTape<f32>) -> Result<Self, GpuError> {
        if tape.has_custom_ops() {
            return Err(GpuError::CustomOpsNotSupported);
        }
        Ok(Self::build_from_tape(tape, tape.values_slice().to_vec()))
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
        let constants = tape.values_slice().iter().map(|&v| v as f32).collect();
        Ok(Self::build_from_tape(tape, constants))
    }
}

/// Metadata for the tape, uploaded as a uniform buffer to GPU shaders.
///
/// Layout matches the WGSL `TapeMeta` struct (4 × u32 = 16 bytes).
#[cfg(feature = "gpu-wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TapeMeta {
    /// Number of opcodes in the tape.
    pub num_ops: u32,
    /// Number of input variables.
    pub num_inputs: u32,
    /// Number of intermediate variables (working slots).
    pub num_variables: u32,
    /// Number of outputs.
    pub num_outputs: u32,
    /// Number of evaluation points in the batch.
    pub batch_size: u32,
    /// Padding to 32-byte alignment.
    pub _pad: [u32; 3],
}

/// Map an [`OpCode`] to the integer constant used in WGSL/CUDA shaders.
///
/// The mapping matches the `OpCode` discriminant (`#[repr(u8)]`), cast to u32.
#[inline]
#[must_use]
pub fn opcode_to_gpu(op: OpCode) -> u32 {
    op as u32
}

/// Default maximum buffer size for WebGPU (128 MiB).
///
/// WebGPU's `maxBufferSize` limit is 256 MiB, but we use 128 MiB as a
/// conservative default to avoid hitting device-specific limits.
#[cfg(feature = "stde")]
pub const WGPU_MAX_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

/// Maximum workgroup dispatches per dimension in WebGPU (65535).
#[cfg(feature = "stde")]
const MAX_WORKGROUPS_PER_DIM: u64 = 65535;

/// Workgroup size used by the Taylor forward shader.
#[cfg(feature = "stde")]
const TAYLOR_WORKGROUP_SIZE: u64 = 256;

/// Chunked batched second-order Taylor forward propagation.
///
/// Splits a large batch into chunks that fit within GPU buffer size limits,
/// dispatches each chunk, and concatenates results. This avoids hitting WebGPU's
/// 128 MiB buffer limit or workgroup dispatch limits.
///
/// # Arguments
///
/// - `backend`: any `GpuBackend` implementation
/// - `tape`: uploaded tape buffers
/// - `primal_inputs`: `[f32; batch_size * num_inputs]` — primals for each element
/// - `direction_seeds`: `[f32; batch_size * num_inputs]` — c1 seeds for each element
/// - `batch_size`: total number of batch elements
/// - `num_inputs`: number of input variables per element
/// - `num_variables`: total tape variable slots (inputs + constants + intermediates)
/// - `max_buffer_bytes`: maximum GPU buffer size in bytes (use [`WGPU_MAX_BUFFER_BYTES`])
///
/// # Errors
///
/// Returns `GpuError::Other` if `max_buffer_bytes` is too small for even a single element.
#[cfg(feature = "stde")]
#[allow(clippy::too_many_arguments)]
pub fn taylor_forward_2nd_batch_chunked<B: GpuBackend>(
    backend: &B,
    tape: &B::TapeBuffers,
    primal_inputs: &[f32],
    direction_seeds: &[f32],
    batch_size: u32,
    num_inputs: u32,
    num_variables: u32,
    max_buffer_bytes: u64,
) -> Result<TaylorBatchResult<f32>, GpuError> {
    if batch_size == 0 {
        return Ok(TaylorBatchResult {
            values: vec![],
            c1s: vec![],
            c2s: vec![],
        });
    }

    // The largest buffer is the jets working buffer: batch_size * num_variables * 3 * 4 bytes
    let bytes_per_element = (num_variables as u64) * 3 * 4;
    if bytes_per_element == 0 {
        return Err(GpuError::Other("num_variables is zero".into()));
    }

    let mut chunk_size = max_buffer_bytes / bytes_per_element;
    if chunk_size == 0 {
        return Err(GpuError::Other(format!(
            "max_buffer_bytes ({max_buffer_bytes}) too small for a single element \
             ({bytes_per_element} bytes per element)"
        )));
    }

    // Also cap at workgroup dispatch limit: 65535 workgroups * 256 threads
    let dispatch_limit = MAX_WORKGROUPS_PER_DIM * TAYLOR_WORKGROUP_SIZE;
    chunk_size = chunk_size.min(dispatch_limit);

    // Cap chunk_size so that WGSL u32 index `bid * nv * K` cannot overflow.
    // K=3 for second-order Taylor jets.
    let nv_k = (num_variables as u64) * 3;
    if nv_k > 0 {
        chunk_size = chunk_size.min(u32::MAX as u64 / nv_k);
    }

    let chunk_size = chunk_size as u32;

    // If everything fits in one chunk, dispatch directly
    if batch_size <= chunk_size {
        return backend.taylor_forward_2nd_batch(tape, primal_inputs, direction_seeds, batch_size);
    }

    // Multi-chunk dispatch
    let ni = num_inputs as usize;
    let mut all_values = Vec::new();
    let mut all_c1s = Vec::new();
    let mut all_c2s = Vec::new();

    let mut offset = 0u32;
    while offset < batch_size {
        let this_chunk = chunk_size.min(batch_size - offset);
        let start = (offset as usize) * ni;
        let end = start + (this_chunk as usize) * ni;

        let chunk_result = backend.taylor_forward_2nd_batch(
            tape,
            &primal_inputs[start..end],
            &direction_seeds[start..end],
            this_chunk,
        )?;

        all_values.extend(chunk_result.values);
        all_c1s.extend(chunk_result.c1s);
        all_c2s.extend(chunk_result.c2s);

        offset += this_chunk;
    }

    Ok(TaylorBatchResult {
        values: all_values,
        c1s: all_c1s,
        c2s: all_c2s,
    })
}

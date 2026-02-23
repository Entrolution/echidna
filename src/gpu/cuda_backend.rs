//! CUDA GPU backend for echidna.
//!
//! Provides `CudaContext` — the NVIDIA-specific GPU backend that supports both
//! f32 and f64 precision. Requires the `gpu-cuda` feature and a CUDA toolkit.
//!
//! # Usage
//!
//! ```no_run
//! use echidna::gpu::{CudaContext, GpuTapeData};
//! use echidna::{record, Scalar};
//!
//! let ctx = CudaContext::new().expect("CUDA device required");
//! let (tape, _) = record(|v| v[0] * v[0] + v[1], &[1.0_f32, 2.0]);
//! let gpu_data = GpuTapeData::from_tape(&tape).unwrap();
//! let tape_bufs = ctx.upload_tape(&gpu_data);
//! let results = ctx.forward_batch(&tape_bufs, &[1.0f32, 2.0, 3.0, 4.0], 2).unwrap();
//! ```

use std::sync::Arc;

use cudarc::driver::{
    CudaContext as CudarContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
};
use cudarc::nvrtc::compile_ptx;

use super::{GpuError, GpuTapeData};

const KERNEL_SRC: &str = include_str!("kernels/tape_eval.cu");
const BLOCK_SIZE: u32 = 256;

/// Uploaded tape data on CUDA device.
pub struct CudaTapeBuffers {
    pub(crate) opcodes: CudaSlice<u32>,
    pub(crate) arg0: CudaSlice<u32>,
    pub(crate) arg1: CudaSlice<u32>,
    pub(crate) constants_f32: CudaSlice<f32>,
    pub(crate) output_indices: CudaSlice<u32>,
    pub(crate) num_ops: u32,
    pub(crate) num_inputs: u32,
    pub(crate) num_variables: u32,
    pub(crate) num_outputs: u32,
}

/// Uploaded tape data with f64 constants for native f64 operations.
pub struct CudaTapeBuffersF64 {
    pub(crate) opcodes: CudaSlice<u32>,
    pub(crate) arg0: CudaSlice<u32>,
    pub(crate) arg1: CudaSlice<u32>,
    pub(crate) constants_f64: CudaSlice<f64>,
    pub(crate) output_indices: CudaSlice<u32>,
    pub(crate) num_ops: u32,
    pub(crate) num_inputs: u32,
    pub(crate) num_variables: u32,
    pub(crate) num_outputs: u32,
}

/// CUDA GPU context for echidna.
///
/// Compiles and caches PTX kernels on first creation. Supports both f32 and f64.
pub struct CudaContext {
    ctx: Arc<CudarContext>,
    stream: Arc<CudaStream>,
    // f32 kernels
    forward_f32: CudaFunction,
    reverse_f32: CudaFunction,
    tangent_fwd_f32: CudaFunction,
    tangent_rev_f32: CudaFunction,
    // f64 kernels
    forward_f64: CudaFunction,
    reverse_f64: CudaFunction,
    tangent_fwd_f64: CudaFunction,
    tangent_rev_f64: CudaFunction,
}

impl CudaContext {
    /// Create a new CUDA context on device 0.
    ///
    /// Returns `None` if no CUDA device is available or compilation fails.
    pub fn new() -> Option<Self> {
        let ctx = CudarContext::new(0).ok()?;
        let stream = ctx.default_stream();

        // Compile f32 kernels
        let src_f32 = format!("#define FLOAT_TYPE float\n{}", KERNEL_SRC);
        let ptx_f32 = compile_ptx(&src_f32).ok()?;
        let module_f32 = ctx.load_module(ptx_f32).ok()?;

        let forward_f32 = module_f32.load_function("forward_eval").ok()?;
        let reverse_f32 = module_f32.load_function("reverse_sweep").ok()?;
        let tangent_fwd_f32 = module_f32.load_function("tangent_forward").ok()?;
        let tangent_rev_f32 = module_f32.load_function("tangent_reverse").ok()?;

        // Compile f64 kernels
        let src_f64 = format!("#define FLOAT_TYPE double\n{}", KERNEL_SRC);
        let ptx_f64 = compile_ptx(&src_f64).ok()?;
        let module_f64 = ctx.load_module(ptx_f64).ok()?;

        let forward_f64 = module_f64.load_function("forward_eval").ok()?;
        let reverse_f64 = module_f64.load_function("reverse_sweep").ok()?;
        let tangent_fwd_f64 = module_f64.load_function("tangent_forward").ok()?;
        let tangent_rev_f64 = module_f64.load_function("tangent_reverse").ok()?;

        Some(CudaContext {
            ctx,
            stream,
            forward_f32,
            reverse_f32,
            tangent_fwd_f32,
            tangent_rev_f32,
            forward_f64,
            reverse_f64,
            tangent_fwd_f64,
            tangent_rev_f64,
        })
    }

    fn grid_dim(batch_size: u32) -> (u32, u32, u32) {
        ((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
    }

    fn block_dim() -> (u32, u32, u32) {
        (BLOCK_SIZE, 1, 1)
    }

    /// Upload an f32 tape to the GPU.
    pub fn upload_tape(&self, data: &GpuTapeData) -> CudaTapeBuffers {
        let s = &self.stream;
        CudaTapeBuffers {
            opcodes: s.clone_htod(&data.opcodes).unwrap(),
            arg0: s.clone_htod(&data.arg0).unwrap(),
            arg1: s.clone_htod(&data.arg1).unwrap(),
            constants_f32: s.clone_htod(&data.constants).unwrap(),
            output_indices: s.clone_htod(&data.output_indices).unwrap(),
            num_ops: data.num_ops,
            num_inputs: data.num_inputs,
            num_variables: data.num_variables,
            num_outputs: data.output_indices.len() as u32,
        }
    }

    /// Upload an f64 tape to the GPU for native f64 operations.
    pub fn upload_tape_f64(
        &self,
        tape: &crate::BytecodeTape<f64>,
    ) -> Result<CudaTapeBuffersF64, GpuError> {
        if tape.has_custom_ops() {
            return Err(GpuError::CustomOpsNotSupported);
        }
        let s = &self.stream;
        let opcodes: Vec<u32> = tape.opcodes_slice().iter().map(|op| *op as u32).collect();
        let args = tape.arg_indices_slice();
        let arg0: Vec<u32> = args.iter().map(|a| a[0]).collect();
        let arg1: Vec<u32> = args.iter().map(|a| a[1]).collect();
        let constants: Vec<f64> = tape.values_slice().to_vec();
        let output_indices = tape.all_output_indices().to_vec();

        Ok(CudaTapeBuffersF64 {
            opcodes: s.clone_htod(&opcodes).unwrap(),
            arg0: s.clone_htod(&arg0).unwrap(),
            arg1: s.clone_htod(&arg1).unwrap(),
            constants_f64: s.clone_htod(&constants).unwrap(),
            output_indices: s.clone_htod(&output_indices).unwrap(),
            num_ops: tape.opcodes_slice().len() as u32,
            num_inputs: tape.num_inputs() as u32,
            num_variables: tape.num_variables_count() as u32,
            num_outputs: output_indices.len() as u32,
        })
    }

    // ── f32 operations ──

    /// Batched forward evaluation (f32).
    pub fn forward_batch(
        &self,
        tape: &CudaTapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<Vec<f32>, GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;
        let no = tape.num_outputs;

        assert_eq!(inputs.len(), (batch_size * ni) as usize);

        let d_inputs = s
            .clone_htod(inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_values = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_outputs = s
            .alloc_zeros::<f32>((batch_size * no) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.forward_f32);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f32);
        builder.arg(&d_inputs);
        builder.arg(&mut d_values);
        builder.arg(&mut d_outputs);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&no);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let results = s
            .clone_dtoh(&d_outputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok(results)
    }

    /// Batched gradient (forward + reverse) (f32).
    pub fn gradient_batch(
        &self,
        tape: &CudaTapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;
        let no = tape.num_outputs;

        assert_eq!(inputs.len(), (batch_size * ni) as usize);

        let d_inputs = s
            .clone_htod(inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_values = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_outputs = s
            .alloc_zeros::<f32>((batch_size * no) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        // Forward pass
        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.forward_f32);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f32);
        builder.arg(&d_inputs);
        builder.arg(&mut d_values);
        builder.arg(&mut d_outputs);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&no);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        // Reverse pass
        let mut d_adjoints = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_grads = s
            .alloc_zeros::<f32>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let mut builder = s.launch_builder(&self.reverse_f32);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&d_values);
        builder.arg(&mut d_adjoints);
        builder.arg(&mut d_grads);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let output_vals = s
            .clone_dtoh(&d_outputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let grads = s
            .clone_dtoh(&d_grads)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok((output_vals, grads))
    }

    /// Sparse Jacobian using forward-mode tangent sweeps (f32).
    pub fn sparse_jacobian(
        &self,
        tape: &CudaTapeBuffers,
        tape_cpu: &mut crate::BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(Vec<f32>, crate::sparse::SparsityPattern, Vec<f32>), GpuError> {
        let ni = tape.num_inputs as usize;
        let no = tape.num_outputs as usize;

        let jac_pattern = crate::sparse::JacobianSparsityPattern::detect(tape_cpu);
        let pattern = &jac_pattern.pattern;
        let (colors, num_colors) = crate::sparse::column_coloring(pattern);

        if num_colors == 0 {
            tape_cpu.forward(x);
            let vals = tape_cpu.output_values();
            return Ok((vals, pattern.clone(), vec![]));
        }

        // Build tangent seeds for each color
        let batch = num_colors as u32;
        let mut seeds = Vec::with_capacity(batch as usize * ni);
        for c in 0..num_colors {
            for i in 0..ni {
                seeds.push(if colors[i] == c as u32 {
                    1.0f32
                } else {
                    0.0f32
                });
            }
        }

        // Dispatch tangent forward kernel
        let s = &self.stream;
        let nv = tape.num_variables;

        let d_primals_in = {
            let mut replicated = Vec::with_capacity(batch as usize * ni);
            for _ in 0..batch {
                replicated.extend_from_slice(x);
            }
            s.clone_htod(&replicated)
                .map_err(|e| GpuError::Other(format!("{e}")))?
        };
        let d_seeds = s
            .clone_htod(&seeds)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_primals = s
            .alloc_zeros::<f32>((batch * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tangents = s
            .alloc_zeros::<f32>((batch * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tangent_out = s
            .alloc_zeros::<f32>((batch * tape.num_outputs) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.tangent_fwd_f32);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f32);
        builder.arg(&d_primals_in);
        builder.arg(&d_seeds);
        builder.arg(&mut d_primals);
        builder.arg(&mut d_tangents);
        builder.arg(&mut d_tangent_out);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&tape.num_inputs);
        builder.arg(&nv);
        builder.arg(&tape.num_outputs);
        builder.arg(&batch);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let tangent_outs = s
            .clone_dtoh(&d_tangent_out)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        // Get output values from CPU
        tape_cpu.forward(x);
        let output_values = tape_cpu.output_values();

        // Extract Jacobian entries
        let nnz = pattern.nnz();
        let mut jac_values = vec![0.0f32; nnz];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let c = colors[col as usize] as usize;
            jac_values[k] = tangent_outs[c * no + row as usize];
        }

        Ok((output_values, pattern.clone(), jac_values))
    }

    /// HVP batch (forward-over-reverse) (f32).
    pub fn hvp_batch(
        &self,
        tape: &CudaTapeBuffers,
        x: &[f32],
        tangent_dirs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;

        assert_eq!(x.len(), ni as usize);
        assert_eq!(tangent_dirs.len(), (batch_size * ni) as usize);

        let mut primal_inputs = Vec::with_capacity((batch_size * ni) as usize);
        for _ in 0..batch_size {
            primal_inputs.extend_from_slice(x);
        }

        let d_primal_in = s
            .clone_htod(&primal_inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let d_seeds = s
            .clone_htod(tangent_dirs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_primals = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tans = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_adj_re = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_adj_eps = s
            .alloc_zeros::<f32>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_grads = s
            .alloc_zeros::<f32>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_hvps = s
            .alloc_zeros::<f32>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.tangent_rev_f32);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f32);
        builder.arg(&d_primal_in);
        builder.arg(&d_seeds);
        builder.arg(&mut d_primals);
        builder.arg(&mut d_tans);
        builder.arg(&mut d_adj_re);
        builder.arg(&mut d_adj_eps);
        builder.arg(&mut d_grads);
        builder.arg(&mut d_hvps);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let grads = s
            .clone_dtoh(&d_grads)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let hvps = s
            .clone_dtoh(&d_hvps)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok((grads, hvps))
    }

    /// Sparse Hessian using GPU HVP sweeps (f32).
    pub fn sparse_hessian(
        &self,
        tape: &CudaTapeBuffers,
        tape_cpu: &mut crate::BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(f32, Vec<f32>, crate::sparse::SparsityPattern, Vec<f32>), GpuError> {
        let ni = tape.num_inputs as usize;

        let pattern = tape_cpu.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        if num_colors == 0 {
            tape_cpu.forward(x);
            let val = tape_cpu.output_value();
            let grad = tape_cpu.gradient(x);
            return Ok((val, grad, pattern, vec![]));
        }

        let batch = num_colors as u32;
        let mut tangent_dirs = Vec::with_capacity(batch as usize * ni);
        for c in 0..num_colors {
            for i in 0..ni {
                tangent_dirs.push(if colors[i] == c as u32 {
                    1.0f32
                } else {
                    0.0f32
                });
            }
        }

        let (grads, hvps) = self.hvp_batch(tape, x, &tangent_dirs, batch)?;

        let gradient: Vec<f32> = grads[..ni].to_vec();

        let nnz = pattern.nnz();
        let mut hess_values = vec![0.0f32; nnz];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let c = colors[col as usize] as usize;
            hess_values[k] = hvps[c * ni + row as usize];
        }

        tape_cpu.forward(x);
        let value = tape_cpu.output_value();

        Ok((value, gradient, pattern, hess_values))
    }

    // ── f64 operations ──

    /// Batched forward evaluation (f64, CUDA only).
    pub fn forward_batch_f64(
        &self,
        tape: &CudaTapeBuffersF64,
        inputs: &[f64],
        batch_size: u32,
    ) -> Result<Vec<f64>, GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;
        let no = tape.num_outputs;

        assert_eq!(inputs.len(), (batch_size * ni) as usize);

        let d_inputs = s
            .clone_htod(inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_values = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_outputs = s
            .alloc_zeros::<f64>((batch_size * no) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.forward_f64);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f64);
        builder.arg(&d_inputs);
        builder.arg(&mut d_values);
        builder.arg(&mut d_outputs);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&no);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let results = s
            .clone_dtoh(&d_outputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok(results)
    }

    /// Batched gradient (f64, CUDA only).
    pub fn gradient_batch_f64(
        &self,
        tape: &CudaTapeBuffersF64,
        inputs: &[f64],
        batch_size: u32,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;
        let no = tape.num_outputs;

        assert_eq!(inputs.len(), (batch_size * ni) as usize);

        let d_inputs = s
            .clone_htod(inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_values = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_outputs = s
            .alloc_zeros::<f64>((batch_size * no) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        // Forward
        let mut builder = s.launch_builder(&self.forward_f64);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f64);
        builder.arg(&d_inputs);
        builder.arg(&mut d_values);
        builder.arg(&mut d_outputs);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&no);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        // Reverse
        let mut d_adjoints = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_grads = s
            .alloc_zeros::<f64>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let mut builder = s.launch_builder(&self.reverse_f64);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&d_values);
        builder.arg(&mut d_adjoints);
        builder.arg(&mut d_grads);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let output_vals = s
            .clone_dtoh(&d_outputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let grads = s
            .clone_dtoh(&d_grads)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok((output_vals, grads))
    }

    /// Sparse Jacobian (f64, CUDA only).
    pub fn sparse_jacobian_f64(
        &self,
        tape: &CudaTapeBuffersF64,
        tape_cpu: &mut crate::BytecodeTape<f64>,
        x: &[f64],
    ) -> Result<(Vec<f64>, crate::sparse::SparsityPattern, Vec<f64>), GpuError> {
        let ni = tape.num_inputs as usize;
        let no = tape.num_outputs as usize;

        let jac_pattern = crate::sparse::JacobianSparsityPattern::detect(tape_cpu);
        let pattern = &jac_pattern.pattern;
        let (colors, num_colors) = crate::sparse::column_coloring(pattern);

        if num_colors == 0 {
            tape_cpu.forward(x);
            let vals = tape_cpu.output_values();
            return Ok((vals, pattern.clone(), vec![]));
        }

        let batch = num_colors as u32;
        let mut seeds = Vec::with_capacity(batch as usize * ni);
        for c in 0..num_colors {
            for i in 0..ni {
                seeds.push(if colors[i] == c as u32 {
                    1.0f64
                } else {
                    0.0f64
                });
            }
        }

        let s = &self.stream;
        let nv = tape.num_variables;

        let d_primals_in = {
            let mut replicated = Vec::with_capacity(batch as usize * ni);
            for _ in 0..batch {
                replicated.extend_from_slice(x);
            }
            s.clone_htod(&replicated)
                .map_err(|e| GpuError::Other(format!("{e}")))?
        };
        let d_seeds = s
            .clone_htod(&seeds)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_primals = s
            .alloc_zeros::<f64>((batch * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tangents = s
            .alloc_zeros::<f64>((batch * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tangent_out = s
            .alloc_zeros::<f64>((batch * tape.num_outputs) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.tangent_fwd_f64);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f64);
        builder.arg(&d_primals_in);
        builder.arg(&d_seeds);
        builder.arg(&mut d_primals);
        builder.arg(&mut d_tangents);
        builder.arg(&mut d_tangent_out);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&tape.num_inputs);
        builder.arg(&nv);
        builder.arg(&tape.num_outputs);
        builder.arg(&batch);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let tangent_outs = s
            .clone_dtoh(&d_tangent_out)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        tape_cpu.forward(x);
        let output_values = tape_cpu.output_values();

        let nnz = pattern.nnz();
        let mut jac_values = vec![0.0f64; nnz];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let c = colors[col as usize] as usize;
            jac_values[k] = tangent_outs[c * no + row as usize];
        }

        Ok((output_values, pattern.clone(), jac_values))
    }

    /// Sparse Hessian (f64, CUDA only).
    pub fn sparse_hessian_f64(
        &self,
        tape: &CudaTapeBuffersF64,
        tape_cpu: &mut crate::BytecodeTape<f64>,
        x: &[f64],
    ) -> Result<(f64, Vec<f64>, crate::sparse::SparsityPattern, Vec<f64>), GpuError> {
        let ni = tape.num_inputs as usize;

        let pattern = tape_cpu.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        if num_colors == 0 {
            tape_cpu.forward(x);
            let val = tape_cpu.output_value();
            let grad = tape_cpu.gradient(x);
            return Ok((val, grad, pattern, vec![]));
        }

        let batch = num_colors as u32;
        let mut tangent_dirs = Vec::with_capacity(batch as usize * ni);
        for c in 0..num_colors {
            for i in 0..ni {
                tangent_dirs.push(if colors[i] == c as u32 {
                    1.0f64
                } else {
                    0.0f64
                });
            }
        }

        let (grads, hvps) = self.hvp_batch_f64(tape, x, &tangent_dirs, batch)?;

        let gradient: Vec<f64> = grads[..ni].to_vec();

        let nnz = pattern.nnz();
        let mut hess_values = vec![0.0f64; nnz];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let c = colors[col as usize] as usize;
            hess_values[k] = hvps[c * ni + row as usize];
        }

        tape_cpu.forward(x);
        let value = tape_cpu.output_value();

        Ok((value, gradient, pattern, hess_values))
    }

    /// HVP batch (f64, CUDA only).
    pub fn hvp_batch_f64(
        &self,
        tape: &CudaTapeBuffersF64,
        x: &[f64],
        tangent_dirs: &[f64],
        batch_size: u32,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        let s = &self.stream;
        let ni = tape.num_inputs;
        let nv = tape.num_variables;

        assert_eq!(x.len(), ni as usize);
        assert_eq!(tangent_dirs.len(), (batch_size * ni) as usize);

        let mut primal_inputs = Vec::with_capacity((batch_size * ni) as usize);
        for _ in 0..batch_size {
            primal_inputs.extend_from_slice(x);
        }

        let d_primal_in = s
            .clone_htod(&primal_inputs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let d_seeds = s
            .clone_htod(tangent_dirs)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_primals = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_tans = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_adj_re = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_adj_eps = s
            .alloc_zeros::<f64>((batch_size * nv) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_grads = s
            .alloc_zeros::<f64>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let mut d_hvps = s
            .alloc_zeros::<f64>((batch_size * ni) as usize)
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let cfg = LaunchConfig {
            grid_dim: Self::grid_dim(batch_size),
            block_dim: Self::block_dim(),
            shared_mem_bytes: 0,
        };

        let mut builder = s.launch_builder(&self.tangent_rev_f64);
        builder.arg(&tape.opcodes);
        builder.arg(&tape.arg0);
        builder.arg(&tape.arg1);
        builder.arg(&tape.constants_f64);
        builder.arg(&d_primal_in);
        builder.arg(&d_seeds);
        builder.arg(&mut d_primals);
        builder.arg(&mut d_tans);
        builder.arg(&mut d_adj_re);
        builder.arg(&mut d_adj_eps);
        builder.arg(&mut d_grads);
        builder.arg(&mut d_hvps);
        builder.arg(&tape.output_indices);
        builder.arg(&tape.num_ops);
        builder.arg(&ni);
        builder.arg(&nv);
        builder.arg(&batch_size);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::Other(format!("{e}")))?;

        s.synchronize()
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let grads = s
            .clone_dtoh(&d_grads)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        let hvps = s
            .clone_dtoh(&d_hvps)
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        Ok((grads, hvps))
    }
}

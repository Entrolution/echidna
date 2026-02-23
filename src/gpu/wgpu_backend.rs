//! wgpu compute backend for GPU-accelerated tape evaluation.
//!
//! Cross-platform (Metal, Vulkan, DX12). f32 only — WGSL does not support f64.

use super::{GpuError, GpuTapeData, TapeMeta};

/// GPU buffers holding an uploaded tape (wgpu backend).
pub struct WgpuTapeBuffers {
    pub(crate) opcodes_buf: wgpu::Buffer,
    pub(crate) arg0_buf: wgpu::Buffer,
    pub(crate) arg1_buf: wgpu::Buffer,
    pub(crate) constants_buf: wgpu::Buffer,
    pub(crate) output_indices_buf: wgpu::Buffer,
    pub(crate) num_ops: u32,
    pub(crate) num_inputs: u32,
    pub(crate) num_variables: u32,
    pub(crate) num_outputs: u32,
}

/// wgpu compute context — holds the device, queue, and compiled pipelines.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    forward_pipeline: wgpu::ComputePipeline,
    reverse_pipeline: wgpu::ComputePipeline,
    tangent_fwd_pipeline: wgpu::ComputePipeline,
    tangent_rev_pipeline: wgpu::ComputePipeline,
    tape_bind_group_layout: wgpu::BindGroupLayout,
    forward_io_bind_group_layout: wgpu::BindGroupLayout,
    reverse_io_bind_group_layout: wgpu::BindGroupLayout,
    tangent_fwd_io_bind_group_layout: wgpu::BindGroupLayout,
    tangent_rev_io_bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuContext {
    /// Acquire a GPU device. Returns `None` if no suitable adapter is found.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        // Bind group layout 0: tape data (read-only storage + uniform)
        let tape_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("echidna_tape_bgl"),
                entries: &[
                    // binding 0: opcodes
                    bgl_storage_ro(0),
                    // binding 1: arg0
                    bgl_storage_ro(1),
                    // binding 2: arg1
                    bgl_storage_ro(2),
                    // binding 3: constants
                    bgl_storage_ro(3),
                    // binding 4: TapeMeta uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 5: output_indices
                    bgl_storage_ro(5),
                ],
            });

        // Bind group layout 1a: forward I/O buffers
        let forward_io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("echidna_fwd_io_bgl"),
                entries: &[
                    // binding 0: inputs [B * num_inputs] (read-only)
                    bgl_storage_ro(0),
                    // binding 1: values [B * num_variables] (read-write)
                    bgl_storage_rw(1),
                    // binding 2: outputs [B * num_outputs] (read-write)
                    bgl_storage_rw(2),
                ],
            });

        // Bind group layout 1b: reverse I/O buffers
        let reverse_io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("echidna_rev_io_bgl"),
                entries: &[
                    // binding 0: values [B * num_variables] (read-only, from forward)
                    bgl_storage_ro(0),
                    // binding 1: adjoints [B * num_variables] (read-write)
                    bgl_storage_rw(1),
                    // binding 2: grad_out [B * num_inputs] (read-write)
                    bgl_storage_rw(2),
                ],
            });

        // Forward pipeline
        let fwd_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("echidna_forward_pl"),
            bind_group_layouts: &[&tape_bind_group_layout, &forward_io_bind_group_layout],
            immediate_size: 0,
        });

        let fwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("echidna_forward_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/forward.wgsl").into()),
        });

        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("echidna_forward_pipeline"),
            layout: Some(&fwd_layout),
            module: &fwd_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Reverse pipeline
        let rev_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("echidna_reverse_pl"),
            bind_group_layouts: &[&tape_bind_group_layout, &reverse_io_bind_group_layout],
            immediate_size: 0,
        });

        let rev_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("echidna_reverse_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reverse.wgsl").into()),
        });

        let reverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("echidna_reverse_pipeline"),
            layout: Some(&rev_layout),
            module: &rev_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Tangent forward pipeline
        let tangent_fwd_io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("echidna_tfwd_io_bgl"),
                entries: &[
                    bgl_storage_ro(0), // primal_inputs
                    bgl_storage_ro(1), // tangent_seeds
                    bgl_storage_rw(2), // primals working
                    bgl_storage_rw(3), // tangents working
                    bgl_storage_rw(4), // tangent_outputs
                ],
            });

        let tfwd_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("echidna_tangent_fwd_pl"),
            bind_group_layouts: &[&tape_bind_group_layout, &tangent_fwd_io_bind_group_layout],
            immediate_size: 0,
        });

        let tfwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("echidna_tangent_fwd_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tangent_forward.wgsl").into()),
        });

        let tangent_fwd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("echidna_tangent_fwd_pipeline"),
                layout: Some(&tfwd_layout),
                module: &tfwd_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Tangent reverse pipeline (forward-over-reverse for HVP)
        let tangent_rev_io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("echidna_trev_io_bgl"),
                entries: &[
                    bgl_storage_ro(0), // primal_inputs
                    bgl_storage_ro(1), // tangent_seeds
                    bgl_storage_rw(2), // primals working
                    bgl_storage_rw(3), // tangents working
                    bgl_storage_rw(4), // adj_re
                    bgl_storage_rw(5), // adj_eps
                    bgl_storage_rw(6), // grad_out
                    bgl_storage_rw(7), // hvp_out
                ],
            });

        let trev_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("echidna_tangent_rev_pl"),
            bind_group_layouts: &[&tape_bind_group_layout, &tangent_rev_io_bind_group_layout],
            immediate_size: 0,
        });

        let trev_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("echidna_tangent_rev_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tangent_reverse.wgsl").into()),
        });

        let tangent_rev_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("echidna_tangent_rev_pipeline"),
                layout: Some(&trev_layout),
                module: &trev_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Some(WgpuContext {
            device,
            queue,
            forward_pipeline,
            reverse_pipeline,
            tangent_fwd_pipeline,
            tangent_rev_pipeline,
            tape_bind_group_layout,
            forward_io_bind_group_layout,
            reverse_io_bind_group_layout,
            tangent_fwd_io_bind_group_layout,
            tangent_rev_io_bind_group_layout,
        })
    }

    /// Upload a tape to GPU buffers.
    pub fn upload_tape(&self, data: &GpuTapeData) -> WgpuTapeBuffers {
        use wgpu::util::DeviceExt;

        let opcodes_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opcodes"),
                contents: bytemuck::cast_slice(&data.opcodes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let arg0_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("arg0"),
                contents: bytemuck::cast_slice(&data.arg0),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let arg1_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("arg1"),
                contents: bytemuck::cast_slice(&data.arg1),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let constants_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("constants"),
                contents: bytemuck::cast_slice(&data.constants),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let num_outputs = if data.output_indices.is_empty() {
            1u32
        } else {
            data.output_indices.len() as u32
        };
        let output_indices = if data.output_indices.is_empty() {
            vec![data.output_index]
        } else {
            data.output_indices.clone()
        };

        let output_indices_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("output_indices"),
                    contents: bytemuck::cast_slice(&output_indices),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        WgpuTapeBuffers {
            opcodes_buf,
            arg0_buf,
            arg1_buf,
            constants_buf,
            output_indices_buf,
            num_ops: data.num_ops,
            num_inputs: data.num_inputs,
            num_variables: data.num_variables,
            num_outputs,
        }
    }

    /// Evaluate the tape at `batch_size` input points in parallel on the GPU.
    ///
    /// `inputs` is a flat `[f32; batch_size * num_inputs]` array in row-major order:
    /// `[x0_0, x0_1, ..., x0_n, x1_0, x1_1, ..., x1_n, ...]`.
    ///
    /// Returns `[f32; batch_size * num_outputs]` — one output per batch element
    /// (or `num_outputs` per element for multi-output tapes).
    pub fn forward_batch(
        &self,
        tape: &WgpuTapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<Vec<f32>, GpuError> {
        use wgpu::util::DeviceExt;

        let num_inputs = tape.num_inputs;
        let num_variables = tape.num_variables;
        let num_outputs = tape.num_outputs;

        assert_eq!(
            inputs.len(),
            (batch_size * num_inputs) as usize,
            "inputs length must be batch_size * num_inputs"
        );

        // Create per-dispatch meta uniform with batch_size
        let meta = TapeMeta {
            num_ops: tape.num_ops,
            num_inputs,
            num_variables,
            num_outputs,
            batch_size,
            _pad: [0; 3],
        };
        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tape_meta"),
                contents: bytemuck::bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Input buffer (read-only from shader)
        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Values buffer: B * num_variables (working memory per thread)
        let values_size = (batch_size * num_variables) as u64 * 4;
        let values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values"),
            size: values_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Output buffer: B * num_outputs
        let output_size = (batch_size * num_outputs) as u64 * 4;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for readback
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Tape bind group (group 0)
        let tape_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tape_bg"),
            layout: &self.tape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tape.opcodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tape.arg0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tape.arg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tape.constants_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tape.output_indices_buf.as_entire_binding(),
                },
            ],
        });

        // I/O bind group (group 1)
        let io_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("io_bg"),
            layout: &self.forward_io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Encode and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &tape_bg, &[]);
            pass.set_bind_group(1, &io_bg, &[]);
            pass.dispatch_workgroups(batch_size.div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        let sub_idx = self.queue.submit(std::iter::once(encoder.finish()));

        // Readback
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });

        rx.recv()
            .map_err(|e| GpuError::Other(format!("channel recv failed: {e}")))?
            .map_err(|e| GpuError::Other(format!("buffer map failed: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }

    /// Compute gradients at `batch_size` input points in parallel on the GPU.
    ///
    /// Runs forward evaluation followed by a reverse adjoint sweep.
    ///
    /// Returns `(outputs, gradients)`:
    /// - `outputs`: `[f32; batch_size * num_outputs]`
    /// - `gradients`: `[f32; batch_size * num_inputs]` in row-major order
    pub fn gradient_batch(
        &self,
        tape: &WgpuTapeBuffers,
        inputs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        use wgpu::util::DeviceExt;

        let num_inputs = tape.num_inputs;
        let num_variables = tape.num_variables;
        let num_outputs = tape.num_outputs;

        assert_eq!(
            inputs.len(),
            (batch_size * num_inputs) as usize,
            "inputs length must be batch_size * num_inputs"
        );

        // Create per-dispatch meta uniform
        let meta = TapeMeta {
            num_ops: tape.num_ops,
            num_inputs,
            num_variables,
            num_outputs,
            batch_size,
            _pad: [0; 3],
        };
        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tape_meta"),
                contents: bytemuck::bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Input buffer
        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Values buffer: B * num_variables (shared between forward and reverse)
        let values_size = (batch_size * num_variables) as u64 * 4;
        let values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values"),
            size: values_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Output buffer: B * num_outputs
        let output_count = (batch_size * num_outputs) as u64;
        let output_size = output_count * 4;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Adjoint buffer: B * num_variables
        let adjoints_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adjoints"),
            size: values_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Gradient output buffer: B * num_inputs
        let grad_count = (batch_size * num_inputs) as u64;
        let grad_size = grad_count * 4;
        let grad_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_out"),
            size: grad_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let output_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grad_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_staging"),
            size: grad_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Tape bind group (shared between forward and reverse)
        let tape_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tape_bg"),
            layout: &self.tape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tape.opcodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tape.arg0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tape.arg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tape.constants_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tape.output_indices_buf.as_entire_binding(),
                },
            ],
        });

        // Forward I/O bind group
        let fwd_io_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fwd_io_bg"),
            layout: &self.forward_io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Reverse I/O bind group (values is read-only here, from forward pass)
        let rev_io_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rev_io_bg"),
            layout: &self.reverse_io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adjoints_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = batch_size.div_ceil(256);

        // Encode: forward pass → reverse pass → copy results
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gradient_enc"),
            });

        // Forward pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &tape_bg, &[]);
            pass.set_bind_group(1, &fwd_io_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Reverse pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reverse_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reverse_pipeline);
            pass.set_bind_group(0, &tape_bg, &[]);
            pass.set_bind_group(1, &rev_io_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &output_staging, 0, output_size);
        encoder.copy_buffer_to_buffer(&grad_buf, 0, &grad_staging, 0, grad_size);
        let sub_idx = self.queue.submit(std::iter::once(encoder.finish()));

        // Readback both buffers
        let out_slice = output_staging.slice(..);
        let grad_slice = grad_staging.slice(..);

        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        out_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });
        grad_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });

        rx1.recv()
            .map_err(|e| GpuError::Other(format!("channel recv failed: {e}")))?
            .map_err(|e| GpuError::Other(format!("output map failed: {e}")))?;
        rx2.recv()
            .map_err(|e| GpuError::Other(format!("channel recv failed: {e}")))?
            .map_err(|e| GpuError::Other(format!("grad map failed: {e}")))?;

        let out_data = out_slice.get_mapped_range();
        let outputs: Vec<f32> = bytemuck::cast_slice(&out_data).to_vec();
        drop(out_data);
        output_staging.unmap();

        let grad_data = grad_slice.get_mapped_range();
        let grads: Vec<f32> = bytemuck::cast_slice(&grad_data).to_vec();
        drop(grad_data);
        grad_staging.unmap();

        Ok((outputs, grads))
    }
    /// Compute a sparse Jacobian using forward-mode tangent sweeps on GPU.
    ///
    /// CPU performs sparsity detection and graph coloring; GPU dispatches all
    /// colored tangent sweeps in parallel.
    ///
    /// `tape_cpu` is needed for sparsity detection (which uses the tape structure).
    /// `x` is the evaluation point.
    ///
    /// Returns `(output_values, sparsity_pattern, jacobian_values)`:
    /// - `output_values`: function values at x
    /// - `sparsity_pattern`: the Jacobian sparsity pattern
    /// - `jacobian_values`: non-zero Jacobian entries matching the pattern
    pub fn sparse_jacobian(
        &self,
        tape: &WgpuTapeBuffers,
        tape_cpu: &mut crate::BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(Vec<f32>, crate::sparse::JacobianSparsityPattern, Vec<f32>), GpuError> {
        use wgpu::util::DeviceExt;

        let num_inputs = tape.num_inputs as usize;
        let num_outputs = tape.num_outputs as usize;
        let num_variables = tape.num_variables;

        // CPU: detect sparsity and compute coloring
        let pattern = tape_cpu.detect_jacobian_sparsity();
        let (colors, num_colors) = crate::sparse::column_coloring(&pattern);

        if num_colors == 0 {
            // All-zero Jacobian
            tape_cpu.forward(x);
            let vals = tape_cpu.output_values();
            let vals_f32: Vec<f32> = vals.to_vec();
            return Ok((vals_f32, pattern, vec![]));
        }

        // Build tangent seed vectors: one per color
        // Each color c gets a seed where input[i].tangent = 1 if colors[i] == c, else 0
        let batch = num_colors;
        let mut primal_inputs = Vec::with_capacity(batch as usize * num_inputs);
        let mut tangent_seeds = Vec::with_capacity(batch as usize * num_inputs);

        for c in 0..num_colors {
            for i in 0..num_inputs {
                primal_inputs.push(x[i]);
                tangent_seeds.push(if colors[i] == c { 1.0f32 } else { 0.0f32 });
            }
        }

        // Create per-dispatch meta
        let meta = TapeMeta {
            num_ops: tape.num_ops,
            num_inputs: tape.num_inputs,
            num_variables,
            num_outputs: tape.num_outputs,
            batch_size: batch,
            _pad: [0; 3],
        };
        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tape_meta"),
                contents: bytemuck::bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let primal_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("primal_inputs"),
                contents: bytemuck::cast_slice(&primal_inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let seed_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tangent_seeds"),
                contents: bytemuck::cast_slice(&tangent_seeds),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_size = (batch * num_variables) as u64 * 4;
        let primals_work = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("primals_work"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tangents_work = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tangents_work"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_size = (batch * tape.num_outputs) as u64 * 4;
        let tangent_out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tangent_outputs"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tape_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tape_bg"),
            layout: &self.tape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tape.opcodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tape.arg0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tape.arg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tape.constants_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tape.output_indices_buf.as_entire_binding(),
                },
            ],
        });

        let io_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tfwd_io_bg"),
            layout: &self.tangent_fwd_io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: primal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: primals_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tangents_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tangent_out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sparse_jac_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tangent_fwd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tangent_fwd_pipeline);
            pass.set_bind_group(0, &tape_bg, &[]);
            pass.set_bind_group(1, &io_bg, &[]);
            pass.dispatch_workgroups(batch.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&tangent_out_buf, 0, &staging, 0, out_size);
        let sub_idx = self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });
        rx.recv()
            .map_err(|e| GpuError::Other(format!("recv: {e}")))?
            .map_err(|e| GpuError::Other(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let tangent_results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // CPU: extract Jacobian entries from compressed tangent results
        // tangent_results[c * num_outputs + o] = sum over {i : colors[i]==c} J[o][i] * 1
        // Since coloring is valid, each entry J[o][i] appears in exactly one color.
        let nnz = pattern.nnz();
        let mut jac_values = vec![0.0f32; nnz];

        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let o = row as usize; // output index
            let i = col as usize; // input index
            let c = colors[i] as usize;
            jac_values[k] = tangent_results[c * num_outputs + o];
        }

        // Get output values from CPU (could also read from GPU primals, but simpler)
        tape_cpu.forward(x);
        let output_values: Vec<f32> = tape_cpu.output_values();

        Ok((output_values, pattern, jac_values))
    }

    /// Batched Hessian-vector product via forward-over-reverse on GPU.
    ///
    /// Dispatches `batch_size` HVP computations in parallel, each with the same
    /// primal inputs `x` but different tangent directions.
    ///
    /// `tangent_dirs` is `[f32; batch_size * num_inputs]` — one direction per element.
    ///
    /// Returns `(gradients, hvps)` each of shape `[f32; batch_size * num_inputs]`.
    pub fn hvp_batch(
        &self,
        tape: &WgpuTapeBuffers,
        x: &[f32],
        tangent_dirs: &[f32],
        batch_size: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        use wgpu::util::DeviceExt;

        let ni = tape.num_inputs;
        let nv = tape.num_variables;

        assert_eq!(x.len(), ni as usize);
        assert_eq!(tangent_dirs.len(), (batch_size * ni) as usize);

        // Build primal inputs: same x replicated for each batch element
        let mut primal_inputs = Vec::with_capacity((batch_size * ni) as usize);
        for _ in 0..batch_size {
            primal_inputs.extend_from_slice(x);
        }

        let meta = TapeMeta {
            num_ops: tape.num_ops,
            num_inputs: ni,
            num_variables: nv,
            num_outputs: tape.num_outputs,
            batch_size,
            _pad: [0; 3],
        };

        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("meta"),
                contents: bytemuck::bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let primal_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("primals_in"),
                contents: bytemuck::cast_slice(&primal_inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let seed_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("seeds"),
                contents: bytemuck::cast_slice(tangent_dirs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_size = (batch_size * nv) as u64 * 4;
        let grad_size = (batch_size * ni) as u64 * 4;

        let primals_work = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pw"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tangents_work = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tw"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let adj_re_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ar"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let adj_eps_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ae"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let grad_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("go"),
            size: grad_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hvp_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ho"),
            size: grad_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let grad_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gs"),
            size: grad_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hvp_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hs"),
            size: grad_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tape_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tape_bg"),
            layout: &self.tape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tape.opcodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tape.arg0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tape.arg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tape.constants_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tape.output_indices_buf.as_entire_binding(),
                },
            ],
        });

        let io_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trev_io"),
            layout: &self.tangent_rev_io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: primal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: primals_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tangents_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adj_re_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adj_eps_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: hvp_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hvp_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trev_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tangent_rev_pipeline);
            pass.set_bind_group(0, &tape_bg, &[]);
            pass.set_bind_group(1, &io_bg, &[]);
            pass.dispatch_workgroups(batch_size.div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&grad_buf, 0, &grad_staging, 0, grad_size);
        encoder.copy_buffer_to_buffer(&hvp_buf, 0, &hvp_staging, 0, grad_size);
        let sub_idx = self.queue.submit(std::iter::once(encoder.finish()));

        let gs = grad_staging.slice(..);
        let hs = hvp_staging.slice(..);
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        gs.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });
        hs.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });

        rx1.recv()
            .map_err(|e| GpuError::Other(format!("{e}")))?
            .map_err(|e| GpuError::Other(format!("{e}")))?;
        rx2.recv()
            .map_err(|e| GpuError::Other(format!("{e}")))?
            .map_err(|e| GpuError::Other(format!("{e}")))?;

        let gd = gs.get_mapped_range();
        let grads: Vec<f32> = bytemuck::cast_slice(&gd).to_vec();
        drop(gd);
        grad_staging.unmap();

        let hd = hs.get_mapped_range();
        let hvps: Vec<f32> = bytemuck::cast_slice(&hd).to_vec();
        drop(hd);
        hvp_staging.unmap();

        Ok((grads, hvps))
    }

    /// Compute a sparse Hessian using forward-over-reverse HVP sweeps on GPU.
    ///
    /// CPU performs Hessian sparsity detection and distance-2 coloring; GPU
    /// dispatches all colored HVP sweeps in parallel.
    ///
    /// Returns `(value, gradient, sparsity_pattern, hessian_values)`.
    pub fn sparse_hessian(
        &self,
        tape: &WgpuTapeBuffers,
        tape_cpu: &mut crate::BytecodeTape<f32>,
        x: &[f32],
    ) -> Result<(f32, Vec<f32>, crate::sparse::SparsityPattern, Vec<f32>), GpuError> {
        let ni = tape.num_inputs as usize;

        // CPU: detect sparsity and compute distance-2 coloring
        let pattern = tape_cpu.detect_sparsity();
        let (colors, num_colors) = crate::sparse::greedy_coloring(&pattern);

        if num_colors == 0 {
            tape_cpu.forward(x);
            let val = tape_cpu.output_value();
            let grad = tape_cpu.gradient(x);
            return Ok((val, grad, pattern, vec![]));
        }

        // Build tangent seeds: one per color
        let batch = num_colors;
        let mut tangent_dirs = Vec::with_capacity(batch as usize * ni);
        for c in 0..num_colors {
            for &color in &colors[..ni] {
                tangent_dirs.push(if color == c { 1.0f32 } else { 0.0f32 });
            }
        }

        let (grads, hvps) = self.hvp_batch(tape, x, &tangent_dirs, batch)?;

        // Extract gradient from first HVP (all share the same gradient)
        let gradient: Vec<f32> = grads[..ni].to_vec();

        // Extract Hessian entries from compressed HVP results
        let nnz = pattern.nnz();
        let mut hess_values = vec![0.0f32; nnz];
        for (k, (&row, &col)) in pattern.rows.iter().zip(pattern.cols.iter()).enumerate() {
            let i = row as usize;
            let j = col as usize;
            // H[i][j] = hvp[color_of_j][i] (since seed_j = 1 for color_of_j)
            let c = colors[j] as usize;
            hess_values[k] = hvps[c * ni + i];
        }

        // Get function value from CPU
        tape_cpu.forward(x);
        let value = tape_cpu.output_value();

        Ok((value, gradient, pattern, hess_values))
    }
}

// ── Helpers ──

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

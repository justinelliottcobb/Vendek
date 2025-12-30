use std::sync::Arc;

use bytemuck;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::Camera;
use crate::world::{FrameUniforms, HoneycombCell, HoneycombWorld, RaymarchParams, VendekPhase};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Parameters that can be adjusted at runtime
#[derive(Clone, Copy)]
pub struct RuntimeParams {
    pub membrane_thickness: f32,
    pub membrane_glow: f32,
    pub step_size: f32,
    pub density: f32,
    pub max_steps: u32,
    pub enable_coupling: bool,
    pub palette: u32,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        Self {
            membrane_thickness: MEMBRANE_THICKNESS,
            membrane_glow: MEMBRANE_GLOW,
            step_size: STEP_SIZE,
            density: 1.0,
            max_steps: MAX_STEPS,
            enable_coupling: true,
            palette: 0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn read_js_params() -> RuntimeParams {
    let window = web_sys::window().unwrap();
    let params = js_sys::Reflect::get(&window, &"vendekParams".into()).ok();

    if let Some(params) = params {
        if params.is_object() {
            let get_f32 = |key: &str, default: f32| -> f32 {
                js_sys::Reflect::get(&params, &key.into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(default)
            };

            return RuntimeParams {
                membrane_thickness: get_f32("membraneThickness", MEMBRANE_THICKNESS),
                membrane_glow: get_f32("membraneGlow", MEMBRANE_GLOW),
                step_size: get_f32("stepSize", STEP_SIZE),
                density: get_f32("density", 1.0),
                max_steps: get_f32("maxSteps", MAX_STEPS as f32) as u32,
                enable_coupling: get_f32("enableCoupling", 1.0) > 0.5,
                palette: get_f32("palette", 0.0) as u32,
            };
        }
    }

    RuntimeParams::default()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_js_params() -> RuntimeParams {
    RuntimeParams::default()
}

// Constants for initial visualization
const VOLUME_MIN: Vec3 = Vec3::new(-12.0, -12.0, -12.0);
const VOLUME_MAX: Vec3 = Vec3::new(12.0, 12.0, 12.0);
const MAX_STEPS: u32 = 128;
const STEP_SIZE: f32 = 0.15;
const MEMBRANE_THICKNESS: f32 = 0.4;
const MEMBRANE_GLOW: f32 = 0.5;

pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // Compute pipeline resources
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_0: wgpu::BindGroup,
    compute_bind_group_1: wgpu::BindGroup,
    compute_bind_group_layout_1: wgpu::BindGroupLayout,

    // Render pipeline resources
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    render_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    frame_uniform_buffer: wgpu::Buffer,
    raymarch_params_buffer: wgpu::Buffer,

    // Storage texture for compute output
    storage_texture: wgpu::Texture,
    storage_texture_view: wgpu::TextureView,

    // Sampler for display shader
    sampler: wgpu::Sampler,
}

impl GpuState {
    pub async fn new(window: Arc<Window>, world: &HoneycombWorld) -> Self {
        let size = window.inner_size();
        let mut width = size.width.max(1);
        let mut height = size.height.max(1);

        // On WASM, window.inner_size() can return incorrect values
        // Fall back to querying the window dimensions directly
        #[cfg(target_arch = "wasm32")]
        {
            let web_window = web_sys::window().unwrap();
            let fallback_width = web_window.inner_width().unwrap().as_f64().unwrap() as u32;
            let fallback_height = web_window.inner_height().unwrap().as_f64().unwrap() as u32;

            web_sys::console::log_1(&format!(
                "GPU init - winit size: {}x{}, web_sys size: {}x{}",
                width, height, fallback_width, fallback_height
            ).into());

            // Use web_sys dimensions if winit reports tiny values
            if width < 100 || height < 100 {
                width = fallback_width.max(100);
                height = fallback_height.max(100);
                web_sys::console::log_1(&format!(
                    "Using fallback dimensions: {}x{}", width, height
                ).into());
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        log::info!("GPU init - size: {}x{}", width, height);

        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window).unwrap();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create storage texture for compute output
        let (storage_texture, storage_texture_view) =
            Self::create_storage_texture(&device, width, height);

        // Create sampler for display
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffers
        let frame_uniforms = FrameUniforms {
            view_proj: glam::Mat4::IDENTITY,
            inv_view_proj: glam::Mat4::IDENTITY,
            camera_position: Vec3::ZERO,
            time: 0.0,
            resolution: [width as f32, height as f32],
            near: 0.1,
            far: 100.0,
        };

        let frame_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Frame Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[frame_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let raymarch_params = RaymarchParams {
            volume_min: VOLUME_MIN,
            _pad0: 0.0,
            volume_max: VOLUME_MAX,
            _pad1: 0.0,
            max_steps: MAX_STEPS,
            step_size: STEP_SIZE,
            membrane_thickness: MEMBRANE_THICKNESS,
            membrane_glow: MEMBRANE_GLOW,
            density_multiplier: 1.0,
            enable_coupling: 1.0,
            palette: 0,
            _pad2: 0,
        };

        let raymarch_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raymarch Params Buffer"),
            contents: bytemuck::cast_slice(&[raymarch_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create storage buffers for world data
        let phases_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Phases Buffer"),
            contents: bytemuck::cast_slice(&world.phases),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cells_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cells Buffer"),
            contents: bytemuck::cast_slice(&world.cells),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Load shaders
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Honeycomb Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/honeycomb.wgsl").into()),
        });

        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        // Create bind group layouts for compute pipeline
        let compute_bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout 0"),
                entries: &[
                    // Frame uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<FrameUniforms>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // Raymarch params
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<RaymarchParams>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // Phases storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<VendekPhase>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // Cells storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<HoneycombCell>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout 1"),
                entries: &[
                    // Output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // Create compute bind groups
        let compute_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 0"),
            layout: &compute_bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: raymarch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: phases_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cells_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group 1"),
            layout: &compute_bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&storage_texture_view),
            }],
        });

        // Create compute pipeline
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout_0, &compute_bind_group_layout_1],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create render bind group layout
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create render bind group - use a separate texture view for sampling
        let sample_texture_view =
            storage_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sample_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size: winit::dpi::PhysicalSize::new(width, height),
            compute_pipeline,
            compute_bind_group_0,
            compute_bind_group_1,
            compute_bind_group_layout_1,
            render_pipeline,
            render_bind_group,
            render_bind_group_layout,
            frame_uniform_buffer,
            raymarch_params_buffer,
            storage_texture,
            storage_texture_view,
            sampler,
        }
    }

    fn create_storage_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Storage Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let mut width = new_size.width;
        let mut height = new_size.height;

        // On WASM, resize can be called with tiny values
        #[cfg(target_arch = "wasm32")]
        {
            if width < 100 || height < 100 {
                let web_window = web_sys::window().unwrap();
                width = web_window.inner_width().unwrap().as_f64().unwrap() as u32;
                height = web_window.inner_height().unwrap().as_f64().unwrap() as u32;
            }
            web_sys::console::log_1(&format!(
                "Resize called: input {}x{}, using {}x{}",
                new_size.width, new_size.height, width, height
            ).into());
        }

        if width > 0 && height > 0 {
            self.size = winit::dpi::PhysicalSize::new(width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            // Recreate storage texture
            let (storage_texture, storage_texture_view) =
                Self::create_storage_texture(&self.device, width, height);
            self.storage_texture = storage_texture;
            self.storage_texture_view = storage_texture_view;

            // Recreate compute bind group 1
            self.compute_bind_group_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group 1"),
                layout: &self.compute_bind_group_layout_1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.storage_texture_view),
                }],
            });

            // Recreate render bind group
            let sample_texture_view = self
                .storage_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group"),
                layout: &self.render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&sample_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
        }
    }

    pub fn render(&mut self, camera: &Camera, time: f32) -> Result<(), wgpu::SurfaceError> {
        // Read runtime parameters from JavaScript
        let runtime_params = read_js_params();

        // Update frame uniforms
        let aspect = self.size.width as f32 / self.size.height as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let frame_uniforms = FrameUniforms {
            view_proj,
            inv_view_proj,
            camera_position: camera.position(),
            time,
            resolution: [self.size.width as f32, self.size.height as f32],
            near: camera.near,
            far: camera.far,
        };

        self.queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::cast_slice(&[frame_uniforms]),
        );

        // Update raymarch params with runtime values
        let raymarch_params = RaymarchParams {
            volume_min: VOLUME_MIN,
            _pad0: 0.0,
            volume_max: VOLUME_MAX,
            _pad1: 0.0,
            max_steps: runtime_params.max_steps,
            step_size: runtime_params.step_size,
            membrane_thickness: runtime_params.membrane_thickness,
            membrane_glow: runtime_params.membrane_glow,
            density_multiplier: runtime_params.density,
            enable_coupling: if runtime_params.enable_coupling { 1.0 } else { 0.0 },
            palette: runtime_params.palette,
            _pad2: 0,
        };

        self.queue.write_buffer(
            &self.raymarch_params_buffer,
            0,
            bytemuck::cast_slice(&[raymarch_params]),
        );

        // Get output texture
        let output = self.surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group_0, &[]);
            compute_pass.set_bind_group(1, &self.compute_bind_group_1, &[]);

            let workgroups_x = (self.size.width + 7) / 8;
            let workgroups_y = (self.size.height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

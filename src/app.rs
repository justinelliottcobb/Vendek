use std::sync::Arc;

use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera;
use crate::gpu::GpuState;
use crate::input::InputState;
use crate::world::HoneycombWorld;

// World generation constants
const CELL_COUNT: usize = 128;
const PHASE_COUNT: usize = 12;
const WORLD_SEED: u64 = 42;

struct AppState {
    window: Arc<Window>,
    gpu: GpuState,
    camera: Camera,
    input: InputState,
    #[allow(dead_code)]
    world: HoneycombWorld,
    time: f32,
    last_frame: web_time::Instant,
}

enum AppPhase {
    Uninitialized,
    Initializing { window: Arc<Window> },
    Running(AppState),
}

struct App {
    phase: AppPhase,
}

impl App {
    fn new() -> Self {
        Self {
            phase: AppPhase::Uninitialized,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Only initialize if we haven't started yet
        if !matches!(self.phase, AppPhase::Uninitialized) {
            return;
        }

        let window_attributes = Window::default_attributes().with_title("Vendek - Far Side Explorer");

        #[cfg(not(target_arch = "wasm32"))]
        let window_attributes =
            window_attributes.with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;

            let canvas = window.canvas().unwrap();

            // Get target container
            let web_window = web_sys::window().unwrap();
            let document = web_window.document().unwrap();

            // Set canvas size BEFORE attaching to DOM
            let width = web_window.inner_width().unwrap().as_f64().unwrap() as u32;
            let height = web_window.inner_height().unwrap().as_f64().unwrap() as u32;
            let width = width.max(100);
            let height = height.max(100);

            canvas.set_width(width);
            canvas.set_height(height);

            // Set explicit style dimensions too
            let style = canvas.style();
            let _ = style.set_property("width", &format!("{}px", width));
            let _ = style.set_property("height", &format!("{}px", height));

            if let Some(container) = document.get_element_by_id("canvas-container") {
                // Append canvas to container
                container.append_child(&canvas).unwrap();
            } else {
                // Append to body
                document.body().unwrap().append_child(&canvas).unwrap();
            }

            let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
        }

        // Start async GPU initialization
        let window_clone = window.clone();

        #[cfg(target_arch = "wasm32")]
        {
            self.phase = AppPhase::Initializing { window: window.clone() };

            // Use a static to communicate back to the app
            // This is a workaround for WASM's async limitations with winit
            wasm_bindgen_futures::spawn_local(async move {
                let world = HoneycombWorld::generate(WORLD_SEED, CELL_COUNT, PHASE_COUNT);
                let gpu = GpuState::new(window_clone.clone(), &world).await;

                // Store in thread-local for retrieval
                PENDING_STATE.with(|cell| {
                    *cell.borrow_mut() = Some(PendingState {
                        window: window_clone,
                        gpu,
                        world,
                    });
                });
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let world = HoneycombWorld::generate(WORLD_SEED, CELL_COUNT, PHASE_COUNT);
            let gpu = pollster::block_on(GpuState::new(window_clone, &world));

            self.phase = AppPhase::Running(AppState {
                window,
                gpu,
                camera: Camera::new(),
                input: InputState::new(),
                world,
                time: 0.0,
                last_frame: web_time::Instant::now(),
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Check for pending WASM initialization
        #[cfg(target_arch = "wasm32")]
        if matches!(self.phase, AppPhase::Initializing { .. }) {
            PENDING_STATE.with(|cell| {
                if let Some(pending) = cell.borrow_mut().take() {
                    self.phase = AppPhase::Running(AppState {
                        window: pending.window,
                        gpu: pending.gpu,
                        camera: Camera::new(),
                        input: InputState::new(),
                        world: pending.world,
                        time: 0.0,
                        last_frame: web_time::Instant::now(),
                    });
                }
            });
        }

        let state = match &mut self.phase {
            AppPhase::Running(s) => s,
            _ => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                state.gpu.resize(physical_size);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.input.handle_key(code, event.state);

                    // Close on Escape
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                    }
                }
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                state.input.handle_mouse_button(button, btn_state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                let old_pos = state.input.mouse_position;
                state.input.handle_mouse_move(new_pos);

                // Handle camera controls
                if state.input.is_mouse_held(MouseButton::Left) {
                    let delta = new_pos - old_pos;
                    state.camera.orbit(delta);
                } else if state.input.is_mouse_held(MouseButton::Right) {
                    let delta = new_pos - old_pos;
                    state.camera.pan(delta);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                state.input.handle_scroll(scroll);
                state.camera.zoom(scroll);
            }

            WindowEvent::RedrawRequested => {
                // Calculate delta time
                let now = web_time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.time += dt;

                // Update camera
                state.camera.update(dt);

                // Render
                match state.gpu.render(&state.camera, state.time) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        state.gpu.resize(state.gpu.size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of memory");
                        event_loop.exit();
                    }
                    Err(e) => {
                        log::warn!("Surface error: {:?}", e);
                    }
                }

                // Clear frame input state
                state.input.end_frame();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        match &self.phase {
            AppPhase::Running(state) => {
                state.window.request_redraw();
            }
            AppPhase::Initializing { window } => {
                window.request_redraw();
            }
            _ => {}
        }
    }
}

#[cfg(target_arch = "wasm32")]
struct PendingState {
    window: Arc<Window>,
    gpu: GpuState,
    world: HoneycombWorld,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_STATE: std::cell::RefCell<Option<PendingState>> = const { std::cell::RefCell::new(None) };
}

pub async fn run() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}

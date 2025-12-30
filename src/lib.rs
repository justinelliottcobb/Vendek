#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod app;
mod camera;
mod gpu;
mod input;
mod world;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("Failed to init logger");
    app::run().await;
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_main() {
    env_logger::init();
    pollster::block_on(app::run());
}

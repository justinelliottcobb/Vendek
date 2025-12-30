use glam::Vec2;
use std::collections::HashSet;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

pub struct InputState {
    pub keys_held: HashSet<KeyCode>,
    pub mouse_buttons: HashSet<MouseButton>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub scroll_delta: f32,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            keys_held: HashSet::new(),
            mouse_buttons: HashSet::new(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: 0.0,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.keys_held.insert(code);
            }
            ElementState::Released => {
                self.keys_held.remove(&code);
            }
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.mouse_buttons.insert(button);
            }
            ElementState::Released => {
                self.mouse_buttons.remove(&button);
            }
        }
    }

    pub fn handle_mouse_move(&mut self, position: Vec2) {
        self.mouse_delta = position - self.mouse_position;
        self.mouse_position = position;
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.scroll_delta = delta;
    }

    pub fn end_frame(&mut self) {
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
    }

    #[allow(dead_code)]
    pub fn is_key_held(&self, code: KeyCode) -> bool {
        self.keys_held.contains(&code)
    }

    pub fn is_mouse_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

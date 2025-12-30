use glam::{Mat4, Vec2, Vec3};

pub struct Camera {
    pub focus: Vec3,
    pub distance: f32,
    pub yaw: f32,   // radians
    pub pitch: f32, // radians
    pub fov: f32,   // radians
    pub near: f32,
    pub far: f32,

    // Smooth interpolation targets
    target_focus: Vec3,
    target_distance: f32,
    target_yaw: f32,
    target_pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            focus: Vec3::ZERO,
            distance: 25.0,
            yaw: 0.0,
            pitch: 0.3,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            target_focus: Vec3::ZERO,
            target_distance: 25.0,
            target_yaw: 0.0,
            target_pitch: 0.3,
        }
    }

    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.focus + Vec3::new(x, y, z)
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.focus, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, self.near, self.far)
    }

    pub fn orbit(&mut self, delta: Vec2) {
        self.target_yaw += delta.x * 0.01;
        self.target_pitch = (self.target_pitch + delta.y * 0.01).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.target_distance = (self.target_distance * (1.0 - delta * 0.1)).clamp(5.0, 50.0);
    }

    pub fn pan(&mut self, delta: Vec2) {
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let up = Vec3::Y;
        self.target_focus += right * delta.x * 0.02 + up * delta.y * 0.02;
    }

    pub fn update(&mut self, dt: f32) {
        let smoothing = 1.0 - (-10.0 * dt).exp();
        self.focus = self.focus.lerp(self.target_focus, smoothing);
        self.distance = self.distance + (self.target_distance - self.distance) * smoothing;
        self.yaw = self.yaw + (self.target_yaw - self.yaw) * smoothing;
        self.pitch = self.pitch + (self.target_pitch - self.pitch) * smoothing;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

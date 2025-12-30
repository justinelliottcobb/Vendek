use glam::{Mat4, Vec3, Vec4};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VendekPhase {
    /// RGB color + density for raymarching
    pub color_density: Vec4,
    /// xyz = anisotropic scattering coefficients, w = mean free path
    pub scattering: Vec4,
    /// x = oscillation freq, y = amplitude, z = damping, w = coupling strength
    pub membrane_params: Vec4,
    /// Unique phase identifier
    pub phase_id: u32,
    pub _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HoneycombCell {
    /// Voronoi seed position in world space
    pub position: Vec3,
    /// Index into the phases array
    pub phase_index: u32,
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FrameUniforms {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub camera_position: Vec3,
    pub time: f32,
    pub resolution: [f32; 2],
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RaymarchParams {
    pub volume_min: Vec3,
    pub _pad0: f32,
    pub volume_max: Vec3,
    pub _pad1: f32,
    pub max_steps: u32,
    pub step_size: f32,
    pub membrane_thickness: f32,
    pub membrane_glow: f32,
    pub density_multiplier: f32,
    pub enable_coupling: f32,  // 1.0 = enabled, 0.0 = disabled
    pub palette: u32,
    pub _pad2: u32,
}

/// Spatial grid for accelerating Voronoi lookups
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GridCell {
    /// Indices of Voronoi cells in this grid cell (up to 8, -1 = empty)
    pub cell_indices: [i32; 8],
    /// Number of valid indices
    pub count: u32,
    pub _pad: [u32; 3],
}

pub struct SpatialGrid {
    pub cells: Vec<GridCell>,
    pub grid_size: u32,  // cells per dimension
}

impl SpatialGrid {
    pub fn build(voronoi_cells: &[HoneycombCell], volume_min: Vec3, volume_max: Vec3, grid_size: u32) -> Self {
        let volume_extent = volume_max - volume_min;
        let cell_size = volume_extent / grid_size as f32;
        let total_cells = (grid_size * grid_size * grid_size) as usize;

        let mut grid_cells = vec![GridCell {
            cell_indices: [-1; 8],
            count: 0,
            _pad: [0; 3],
        }; total_cells];

        // For each Voronoi cell, add it to nearby grid cells
        for (voronoi_idx, voronoi_cell) in voronoi_cells.iter().enumerate() {
            let pos = voronoi_cell.position;

            // Find grid cell containing this Voronoi center
            let grid_pos = ((pos - volume_min) / cell_size).floor();
            let gx = (grid_pos.x as i32).clamp(0, grid_size as i32 - 1) as u32;
            let gy = (grid_pos.y as i32).clamp(0, grid_size as i32 - 1) as u32;
            let gz = (grid_pos.z as i32).clamp(0, grid_size as i32 - 1) as u32;

            // Add to this cell and neighbors (3x3x3 neighborhood)
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = gx as i32 + dx;
                        let ny = gy as i32 + dy;
                        let nz = gz as i32 + dz;

                        if nx >= 0 && nx < grid_size as i32 &&
                           ny >= 0 && ny < grid_size as i32 &&
                           nz >= 0 && nz < grid_size as i32 {
                            let idx = (nz as u32 * grid_size * grid_size + ny as u32 * grid_size + nx as u32) as usize;
                            let grid_cell = &mut grid_cells[idx];
                            if (grid_cell.count as usize) < 8 {
                                grid_cell.cell_indices[grid_cell.count as usize] = voronoi_idx as i32;
                                grid_cell.count += 1;
                            }
                        }
                    }
                }
            }
        }

        Self {
            cells: grid_cells,
            grid_size,
        }
    }
}

pub struct HoneycombWorld {
    pub phases: Vec<VendekPhase>,
    pub cells: Vec<HoneycombCell>,
    // pub spatial_grid: SpatialGrid, // TODO: re-enable for performance
}

impl HoneycombWorld {
    pub fn generate(seed: u64, cell_count: usize, phase_count: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate distinct vendek phases with varied visual properties
        let phases: Vec<VendekPhase> = (0..phase_count)
            .map(|i| {
                let hue = (i as f32) / (phase_count as f32);
                let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);

                VendekPhase {
                    color_density: Vec4::new(r, g, b, rng.gen_range(0.02..0.08)),
                    scattering: Vec4::new(
                        rng.gen_range(0.1..1.0),
                        rng.gen_range(0.1..1.0),
                        rng.gen_range(0.1..1.0),
                        rng.gen_range(0.5..2.0),
                    ),
                    membrane_params: Vec4::new(
                        rng.gen_range(0.5..5.0),  // frequency
                        rng.gen_range(0.01..0.1), // amplitude
                        rng.gen_range(0.1..0.5),  // damping
                        rng.gen_range(0.1..1.0),  // coupling
                    ),
                    phase_id: i as u32,
                    _pad: [0; 3],
                }
            })
            .collect();

        // Generate Voronoi seeds
        let cells: Vec<HoneycombCell> = (0..cell_count)
            .map(|_| HoneycombCell {
                position: Vec3::new(
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                ),
                phase_index: rng.gen_range(0..phase_count as u32),
            })
            .collect();

        Self { phases, cells }
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

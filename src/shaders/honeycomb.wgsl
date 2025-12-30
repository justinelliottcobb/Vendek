struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    near: f32,
    far: f32,
}

struct RaymarchParams {
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    max_steps: u32,
    step_size: f32,
    membrane_thickness: f32,
    membrane_glow: f32,
    density_multiplier: f32,
    enable_coupling: f32,
    palette: u32,
    _pad2: u32,
}

// Apply color palette transformation
fn apply_palette(base_color: vec3<f32>, phase_id: u32, palette: u32) -> vec3<f32> {
    let hue = f32(phase_id % 12u) / 12.0;

    switch palette {
        // 0: Rainbow (original)
        case 0u: {
            return base_color;
        }
        // 1: Ocean
        case 1u: {
            let ocean_hue = 0.5 + hue * 0.15; // Blues and teals
            return hsv_to_rgb(ocean_hue, 0.6, 0.8 + hue * 0.2);
        }
        // 2: Fire
        case 2u: {
            let fire_hue = hue * 0.12; // Reds to yellows
            return hsv_to_rgb(fire_hue, 0.9, 0.9);
        }
        // 3: Forest
        case 3u: {
            let forest_hue = 0.25 + hue * 0.15; // Greens and browns
            return hsv_to_rgb(forest_hue, 0.5 + hue * 0.3, 0.4 + hue * 0.4);
        }
        // 4: Neon
        case 4u: {
            let neon_hue = hue;
            return hsv_to_rgb(neon_hue, 1.0, 1.0);
        }
        // 5: Pastel
        case 5u: {
            return hsv_to_rgb(hue, 0.3, 0.95);
        }
        // 6: Monochrome
        case 6u: {
            let brightness = 0.3 + hue * 0.5;
            return vec3(brightness);
        }
        default: {
            return base_color;
        }
    }
}

// HSV to RGB conversion
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(((h * 6.0) % 2.0) - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    let h6 = h * 6.0;
    if h6 < 1.0 {
        rgb = vec3(c, x, 0.0);
    } else if h6 < 2.0 {
        rgb = vec3(x, c, 0.0);
    } else if h6 < 3.0 {
        rgb = vec3(0.0, c, x);
    } else if h6 < 4.0 {
        rgb = vec3(0.0, x, c);
    } else if h6 < 5.0 {
        rgb = vec3(x, 0.0, c);
    } else {
        rgb = vec3(c, 0.0, x);
    }
    return rgb + m;
}

struct VendekPhase {
    color_density: vec4<f32>,
    scattering: vec4<f32>,
    membrane_params: vec4<f32>,
    phase_id: u32,
    _pad: array<u32, 3>,
}

struct HoneycombCell {
    position: vec3<f32>,
    phase_index: u32,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> params: RaymarchParams;
@group(0) @binding(2) var<storage, read> phases: array<VendekPhase>;
@group(0) @binding(3) var<storage, read> cells: array<HoneycombCell>;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Ray-box intersection
fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t1 = (params.volume_min - ray_origin) * inv_dir;
    let t2 = (params.volume_max - ray_origin) * inv_dir;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return vec2(max(tmin, 0.0), tmax);
}

// Calculate fade factor for soft volume boundaries
fn boundary_fade(pos: vec3<f32>) -> f32 {
    let fade_distance = 2.0; // Distance from edge to start fading
    let normalized = (pos - params.volume_min) / (params.volume_max - params.volume_min);

    // Distance from each face (0 at edge, 0.5 at center)
    let dist_from_edge = min(normalized, 1.0 - normalized);

    // Minimum distance to any face
    let min_dist = min(min(dist_from_edge.x, dist_from_edge.y), dist_from_edge.z);

    // Convert to world units and apply fade
    let world_dist = min_dist * (params.volume_max.x - params.volume_min.x);
    return smoothstep(0.0, fade_distance, world_dist);
}

// Find closest Voronoi cell and distance to second-closest (for membrane detection)
fn voronoi_cell(pos: vec3<f32>) -> vec3<f32> {
    // Returns: (closest_cell_index, dist_to_closest, dist_to_second_closest)
    var min_dist = 1e10;
    var second_dist = 1e10;
    var closest_idx = 0u;

    let cell_count = arrayLength(&cells);
    for (var i = 0u; i < cell_count; i++) {
        let cell_pos = cells[i].position;
        let d = distance(pos, cell_pos);
        if d < min_dist {
            second_dist = min_dist;
            min_dist = d;
            closest_idx = i;
        } else if d < second_dist {
            second_dist = d;
        }
    }

    return vec3(f32(closest_idx), min_dist, second_dist);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    // Reconstruct ray from pixel coordinates
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;

    let clip_near = vec4(ndc.x, -ndc.y, 0.0, 1.0);
    let clip_far = vec4(ndc.x, -ndc.y, 1.0, 1.0);
    var world_near = frame.inv_view_proj * clip_near;
    var world_far = frame.inv_view_proj * clip_far;
    world_near /= world_near.w;
    world_far /= world_far.w;

    let ray_origin = world_near.xyz;
    let ray_dir = normalize(world_far.xyz - world_near.xyz);

    // Find intersection with volume bounds
    let t_range = intersect_box(ray_origin, ray_dir);

    if t_range.x >= t_range.y {
        // Outside volume - dark background
        textureStore(output, vec2<i32>(gid.xy), vec4(0.02, 0.02, 0.03, 1.0));
        return;
    }

    // Raymarch through the volume
    var accumulated_color = vec3(0.0);
    var accumulated_alpha = 0.0;

    let t_start = t_range.x;
    let t_end = t_range.y;
    var t = t_start;

    for (var step = 0u; step < params.max_steps; step++) {
        if t >= t_end || accumulated_alpha > 0.98 {
            break;
        }

        let pos = ray_origin + ray_dir * t;

        // Soft boundary fade
        let edge_fade = boundary_fade(pos);
        if edge_fade < 0.01 {
            t += params.step_size;
            continue;
        }

        let vor = voronoi_cell(pos);
        let cell_idx = u32(vor.x);
        let dist_closest = vor.y;
        let dist_second = vor.z;

        // Get phase for this cell with slow time-based transitions
        let base_phase_idx = cells[cell_idx].phase_index;
        let phase_count = arrayLength(&phases);

        // Slow phase drift based on cell position and time
        let cell_pos = cells[cell_idx].position;
        let drift_speed = 0.05; // Very slow transition
        let phase_drift = sin(frame.time * drift_speed + cell_pos.x * 0.3 + cell_pos.y * 0.2 + cell_pos.z * 0.1);

        // Blend between current phase and next phase
        let next_phase_idx = (base_phase_idx + 1u) % phase_count;
        let blend_factor = phase_drift * 0.5 + 0.5; // 0 to 1

        let phase_a = phases[base_phase_idx];
        let phase_b = phases[next_phase_idx];

        // Create interpolated phase
        var phase: VendekPhase;
        phase.color_density = mix(phase_a.color_density, phase_b.color_density, blend_factor * 0.3);
        phase.membrane_params = mix(phase_a.membrane_params, phase_b.membrane_params, blend_factor * 0.2);

        // Membrane detection: how close are we to a cell boundary?
        let membrane_dist = (dist_second - dist_closest) * 0.5;
        let membrane_factor = smoothstep(0.0, params.membrane_thickness, membrane_dist);

        // Base cell color with density, modulated by edge fade and density multiplier
        // Apply palette transformation
        var sample_color = apply_palette(phase.color_density.rgb, base_phase_idx, params.palette);
        var sample_alpha = phase.color_density.a * params.step_size * edge_fade * params.density_multiplier;

        // Add membrane glow at boundaries
        if membrane_factor < 1.0 {
            let phase_freq = phase.membrane_params.x;
            var oscillation: f32;
            var membrane_color: vec3<f32>;

            // Expensive coupling calculation (can be disabled for performance)
            if params.enable_coupling > 0.5 {
                let phase_coupling = phase.membrane_params.w;

                // Find the second-closest cell to get its phase
                var second_closest_idx = 0u;
                var second_min_dist = 1e10;
                let cell_count = arrayLength(&cells);
                for (var i = 0u; i < cell_count; i++) {
                    let d = distance(pos, cells[i].position);
                    if d > dist_closest + 0.01 && d < second_min_dist {
                        second_min_dist = d;
                        second_closest_idx = i;
                    }
                }
                let second_phase_idx = cells[second_closest_idx].phase_index;
                let second_phase = phases[second_phase_idx];
                let second_freq = second_phase.membrane_params.x;

                // Coupled oscillation - interference between two adjacent cell frequencies
                let base_phase = phase_freq * frame.time + dist_closest * 2.0;
                let coupled_phase = second_freq * frame.time + dist_second * 2.0;
                let interference = sin(base_phase) * 0.5 + sin(coupled_phase) * phase_coupling * 0.5;
                oscillation = interference * 0.5 + 0.5;

                // Membrane color blends the two adjacent phases
                let blend_color = mix(phase.color_density.rgb, second_phase.color_density.rgb, 0.5);
                membrane_color = mix(blend_color, vec3(1.0), 0.6) * params.membrane_glow;
            } else {
                // Simple oscillation without coupling (faster)
                let base_phase = phase_freq * frame.time + dist_closest * 2.0;
                oscillation = sin(base_phase) * 0.5 + 0.5;
                membrane_color = mix(phase.color_density.rgb, vec3(1.0), 0.7) * params.membrane_glow;
            }

            let membrane_intensity = (1.0 - membrane_factor) * (0.3 + 0.7 * oscillation);
            sample_color = mix(sample_color, membrane_color, membrane_intensity);
            sample_alpha += membrane_intensity * 0.15;
        }

        // Front-to-back compositing
        let contrib = sample_color * sample_alpha * (1.0 - accumulated_alpha);
        accumulated_color += contrib;
        accumulated_alpha += sample_alpha * (1.0 - accumulated_alpha);

        t += params.step_size;
    }

    // Blend with background
    let bg_color = vec3(0.02, 0.02, 0.03);
    var final_color = accumulated_color + bg_color * (1.0 - accumulated_alpha);

    // Depth fog - fade distant parts toward background
    let avg_depth = (t_start + t) * 0.5; // Approximate average depth
    let fog_density = 0.015;
    let fog_factor = 1.0 - exp(-fog_density * avg_depth);
    let fog_color = vec3(0.05, 0.05, 0.08); // Slightly blue-tinted fog
    final_color = mix(final_color, fog_color, fog_factor * 0.5);

    textureStore(output, vec2<i32>(gid.xy), vec4(final_color, 1.0));
}

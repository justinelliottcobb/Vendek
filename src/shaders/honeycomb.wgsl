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

// Find the two closest Voronoi cells and their distances
fn voronoi_query(p: vec3<f32>) -> vec4<u32> {
    var min_dist = 1e10;
    var second_dist = 1e10;
    var closest: u32 = 0u;
    var second: u32 = 0u;

    let cell_count = arrayLength(&cells);
    for (var i = 0u; i < cell_count; i++) {
        let d = distance(p, cells[i].position);
        if d < min_dist {
            second_dist = min_dist;
            second = closest;
            min_dist = d;
            closest = i;
        } else if d < second_dist {
            second_dist = d;
            second = i;
        }
    }

    return vec4<u32>(
        closest,
        second,
        bitcast<u32>(min_dist),
        bitcast<u32>(second_dist)
    );
}

// How close are we to the boundary between two cells?
fn membrane_factor(dist1: f32, dist2: f32) -> f32 {
    let ratio = dist1 / max(dist2, 0.001);
    return exp(-abs(1.0 - ratio) / params.membrane_thickness);
}

// Interference pattern from two adjacent phases
fn membrane_oscillation(p: vec3<f32>, phase1: VendekPhase, phase2: VendekPhase) -> f32 {
    let freq1 = phase1.membrane_params.x;
    let freq2 = phase2.membrane_params.x;
    let amp1 = phase1.membrane_params.y;
    let amp2 = phase2.membrane_params.y;

    let t = frame.time;
    let spatial_freq = 10.0;
    let wave1 = sin(freq1 * t + dot(p, vec3(1.0, 0.5, 0.3)) * spatial_freq) * amp1;
    let wave2 = sin(freq2 * t + dot(p, vec3(0.3, 1.0, 0.5)) * spatial_freq) * amp2;

    return wave1 + wave2;
}

// Sample the honeycomb at a point
fn sample_honeycomb(p: vec3<f32>) -> vec4<f32> {
    let query = voronoi_query(p);
    let closest = query.x;
    let second = query.y;
    let dist1 = bitcast<f32>(query.z);
    let dist2 = bitcast<f32>(query.w);

    let phase = phases[cells[closest].phase_index];
    let neighbor_phase = phases[cells[second].phase_index];

    var color = phase.color_density.rgb;
    var density = phase.color_density.a;

    // Membrane contribution
    let membrane = membrane_factor(dist1, dist2);
    if membrane > 0.01 {
        let oscillation = membrane_oscillation(p, phase, neighbor_phase);
        let membrane_color = mix(phase.color_density.rgb, neighbor_phase.color_density.rgb, 0.5);
        let glow = membrane_color * (1.0 + oscillation * 2.0) * params.membrane_glow;

        color = mix(color, glow, membrane);
        density = mix(density, density * 2.0, membrane);
    }

    return vec4(color, density);
}

// Ray-box intersection
fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t1 = (params.volume_min - ray_origin) * inv_dir;
    let t2 = (params.volume_max - ray_origin) * inv_dir;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return vec2(max(tmin, 0.0), tmax);
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
        textureStore(output, vec2<i32>(gid.xy), vec4(0.02, 0.02, 0.03, 1.0));
        return;
    }

    // Raymarch through the honeycomb
    var accumulated_color = vec3(0.0);
    var accumulated_alpha = 0.0;
    var t = t_range.x;

    for (var i = 0u; i < params.max_steps; i++) {
        if t >= t_range.y || accumulated_alpha > 0.99 {
            break;
        }

        let p = ray_origin + ray_dir * t;
        let sample = sample_honeycomb(p);

        // Beer-Lambert absorption
        let step_alpha = 1.0 - exp(-sample.a * params.step_size);
        let weight = step_alpha * (1.0 - accumulated_alpha);

        accumulated_color += sample.rgb * weight;
        accumulated_alpha += weight;

        t += params.step_size;
    }

    let bg = vec3(0.02, 0.02, 0.03);
    let final_color = accumulated_color + bg * (1.0 - accumulated_alpha);

    textureStore(output, vec2<i32>(gid.xy), vec4(final_color, 1.0));
}

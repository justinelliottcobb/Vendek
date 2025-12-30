struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2(0.0, 1.0),
        vec2(2.0, 1.0),
        vec2(0.0, -1.0),
    );

    var out: VertexOutput;
    out.position = vec4(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var render_texture: texture_2d<f32>;
@group(0) @binding(1) var render_sampler: sampler;

// Simple bloom by sampling neighbors and adding bright contribution
fn bloom_sample(uv: vec2<f32>, tex_size: vec2<f32>) -> vec3<f32> {
    let pixel_size = 1.0 / tex_size;
    var bloom = vec3(0.0);

    // Sample in a cross pattern at multiple distances
    let offsets = array<vec2<f32>, 8>(
        vec2(-2.0, 0.0), vec2(2.0, 0.0), vec2(0.0, -2.0), vec2(0.0, 2.0),
        vec2(-4.0, 0.0), vec2(4.0, 0.0), vec2(0.0, -4.0), vec2(0.0, 4.0)
    );
    let weights = array<f32, 8>(0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08);

    for (var i = 0; i < 8; i++) {
        let sample_uv = uv + offsets[i] * pixel_size;
        let sample_color = textureSample(render_texture, render_sampler, sample_uv).rgb;

        // Only bloom bright pixels (threshold)
        let brightness = max(max(sample_color.r, sample_color.g), sample_color.b);
        let bloom_threshold = 0.5;
        let bloom_contribution = max(brightness - bloom_threshold, 0.0) / (1.0 - bloom_threshold);

        bloom += sample_color * bloom_contribution * weights[i];
    }

    return bloom;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = textureSample(render_texture, render_sampler, in.uv).rgb;
    let tex_size = vec2<f32>(textureDimensions(render_texture));

    // Add bloom
    let bloom = bloom_sample(in.uv, tex_size);
    let bloom_intensity = 0.4;
    var final_color = base_color + bloom * bloom_intensity;

    // Subtle tone mapping to prevent over-saturation
    final_color = final_color / (1.0 + final_color * 0.2);

    return vec4(final_color, 1.0);
}

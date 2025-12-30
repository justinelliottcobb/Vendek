# Vendek

A volumetric "Honeycomb" renderer inspired by Greg Egan's *Schild's Ladder*. The visualization runs both natively and in-browser via WebGPU.

## Overview

The Honeycomb is a Voronoi-structured volume where each cell contains a distinct "vendek phase" — a Planck-scale physics configuration with unique visual properties. Membranes between cells oscillate and glow with interference patterns. The entire scene is raymarched volumetrically using a compute shader.

This is the foundation for simulating "far side" environments from *Schild's Ladder*: regions of space filled with novel physics configurations that ships must adapt to traverse.

## Requirements

- Rust (stable)
- For native: A GPU with Vulkan, Metal, or DX12 support
- For web: Chrome 113+ or another WebGPU-enabled browser
- For web builds: `wasm-bindgen-cli` (`cargo install wasm-bindgen-cli`)

## Building and Running

### Native

```bash
cargo run
```

For release builds:

```bash
cargo run --release
```

### Web

```bash
./build-web.sh
cargo run --bin serve
```

Then open http://localhost:3000 in a WebGPU-enabled browser.

## Controls

| Input | Action |
|-------|--------|
| Left mouse drag | Orbit camera around focus point |
| Right mouse drag | Pan focus point |
| Scroll wheel | Zoom in/out |
| Escape | Exit (native only) |

## Project Structure

```
vendek/
├── Cargo.toml
├── rust-toolchain.toml
├── index.html
├── build-web.sh
├── web/
│   └── bootstrap.js
└── src/
    ├── lib.rs              # Entry point (shared native/web)
    ├── main.rs             # Native entry point
    ├── app.rs              # Application loop with winit
    ├── gpu.rs              # wgpu setup, pipelines, rendering
    ├── world.rs            # HoneycombWorld, VendekPhase, GPU types
    ├── camera.rs           # Orbital camera with smooth interpolation
    ├── input.rs            # Platform-agnostic input handling
    └── shaders/
        ├── honeycomb.wgsl  # Compute shader for volumetric raymarching
        └── display.wgsl    # Fullscreen quad display shader
```

## Technical Details

- **Rendering**: Volumetric raymarching via compute shader
- **World structure**: 64 Voronoi cells with 8 distinct vendek phases
- **Membrane effects**: Interference patterns based on phase oscillation frequencies
- **Camera**: Orbital with smooth interpolation

## Dependencies

- `wgpu` - WebGPU implementation
- `winit` - Cross-platform windowing
- `glam` - Linear algebra
- `bytemuck` - GPU buffer casting
- `rand` / `rand_chacha` - Deterministic world generation

## Future Extensions

- Ship rendering (self-luminous point representing the Sarumpaet)
- Probe visualization (particles that launch and return)
- Visibility system (fog-of-war based on probe data)
- Phase transitions (ship adapting when crossing membranes)
- The Bright (continuous mixing rather than discrete cells)
- Sprites (information-carrying excitations)

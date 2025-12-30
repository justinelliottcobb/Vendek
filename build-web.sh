#!/bin/bash
set -e

echo "Building for WebGPU..."
RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build \
    --target wasm32-unknown-unknown \
    --release \
    --lib

echo "Running wasm-bindgen..."
wasm-bindgen \
    --out-dir pkg \
    --target web \
    target/wasm32-unknown-unknown/release/vendek.wasm

echo "Build complete!"
echo ""
echo "To run locally:"
echo "  python3 -m http.server 8080"
echo "  Open http://localhost:8080 in Chrome 113+"

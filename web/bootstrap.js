import init from '../pkg/vendek.js';

async function run() {
    try {
        await init();
    } catch (e) {
        console.error("Failed to initialize:", e);
        document.body.innerHTML = `
            <div style="color: white; padding: 20px; font-family: sans-serif;">
                <h1>WebGPU not available</h1>
                <p>This application requires WebGPU support.</p>
                <p>Please use Chrome 113+ or another WebGPU-enabled browser.</p>
                <p style="margin-top: 20px; color: #888;">Error: ${e}</p>
            </div>
        `;
    }
}

run();

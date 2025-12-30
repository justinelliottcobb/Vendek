use axum::{
    http::{HeaderName, HeaderValue},
    Router,
};
use std::net::SocketAddr;
use tower_http::{services::ServeDir, set_header::SetResponseHeaderLayer};

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    // Serve static files from the project root
    // Required headers for SharedArrayBuffer (needed by some WASM features)
    let serve_dir = ServeDir::new(".")
        .append_index_html_on_directories(true);

    let app = Router::new()
        .fallback_service(serve_dir)
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("cross-origin-opener-policy"),
            HeaderValue::from_static("same-origin"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("cross-origin-embedder-policy"),
            HeaderValue::from_static("require-corp"),
        ));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Serving at http://localhost:{}", port);
    println!("Press Ctrl+C to stop");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

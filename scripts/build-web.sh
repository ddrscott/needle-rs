#!/bin/sh
# Build the browser demo's two engines into web/: `pkg` (WebAssembly SIMD,
# fused multiply-add computed exactly in software) and `pkg-relaxed` (the
# hardware's fused multiply-add through relaxed SIMD, which the page uses only
# after checking the browser really fuses it). Needs wasm-pack.
set -eu
cd "$(dirname "$0")/.."
build() {
    RUSTFLAGS="-C target-feature=$2" wasm-pack build crates/needle-wasm --release --target web \
        --out-dir "../../web/$1" --out-name needle_wasm -- --target-dir "target/web-$1"
    rm -f "web/$1/.gitignore" "web/$1/package.json" "web/$1/README.md"
}
build pkg +simd128
build pkg-relaxed +simd128,+relaxed-simd
ls -la web/pkg web/pkg-relaxed

#!/bin/sh
# Publish the browser demo to https://askscottpierce.com/needle-rs: build the
# engines, stage web/ at site/needle-rs/ (file path = URL path), deploy the
# assets-only Worker in wrangler.jsonc. Needs wasm-pack and `wrangler login`.
set -eu
cd "$(dirname "$0")/.."
./scripts/build-web.sh
rm -rf site
mkdir -p site/needle-rs
cp -R web/. site/needle-rs/
npx -y wrangler@latest deploy

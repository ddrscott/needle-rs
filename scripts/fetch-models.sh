#!/bin/sh
# Download the published Needle 3 weights and tokenizer into models/.
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
base=https://huggingface.co/Cactus-Compute/needle3/resolve/main
fetch() { [ -f "models/$2" ] || curl -fL --progress-bar -o "models/$2" "$base/$1"; }
fetch needle3.cact needle3.cact
fetch checkpoints/needle3.safetensors needle3.safetensors
fetch tokenizer/tokenizer.model tokenizer.model
ls -la models

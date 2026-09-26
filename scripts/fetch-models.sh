#!/bin/sh
# Download the published Needle 3 weights and tokenizer into models/.
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
# Pinned to the revision the parity and speed numbers were measured with.
base=https://huggingface.co/Cactus-Compute/needle3/resolve/b274efcb211a9eef48c9a88da4b43bd569696a39
fetch() { [ -f "models/$2" ] || curl -fL --progress-bar -o "models/$2" "$base/$1"; }
fetch needle3.cact needle3.cact
fetch checkpoints/needle3.safetensors needle3.safetensors
fetch tokenizer/tokenizer.model tokenizer.model
ls -la models

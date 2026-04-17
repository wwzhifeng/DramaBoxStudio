#!/bin/bash
# IC-LoRA Voice Cloning Inference
# Usage: bash scripts/inference.sh --voice ref.wav --prompt "A woman speaks..." --output out.wav --lora path/to/lora.safetensors

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "$SCRIPT_DIR/src/inference.py" \
  --checkpoint ltx-2.3-22b-dev-audio-only.safetensors \
  --full-checkpoint ltx-2.3-22b-dev.safetensors \
  --cfg-scale 2.5 --stg-scale 1.5 --rescale-scale 0 --modality-scale 1 \
  --duration-multiplier 1.1 \
  --seed 42 \
  "$@"

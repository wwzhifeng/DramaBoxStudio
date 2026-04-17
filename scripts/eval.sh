#!/bin/bash
# Run evaluation set
# Usage: bash scripts/eval.sh --lora path/to/lora.safetensors --eval expressive --output eval_output/

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LORA=""
EVAL_TYPE="short"
OUTPUT_DIR="eval_output"
GPUS=8

while [[ $# -gt 0 ]]; do
  case $1 in
    --lora) LORA="$2"; shift 2;;
    --eval) EVAL_TYPE="$2"; shift 2;;
    --output) OUTPUT_DIR="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

case $EVAL_TYPE in
  short) CONFIG="$SCRIPT_DIR/evals/eval_short.txt";;
  long) CONFIG="$SCRIPT_DIR/evals/eval_long.txt";;
  stress) CONFIG="$SCRIPT_DIR/evals/eval_stress.txt";;
  expressive) CONFIG="$SCRIPT_DIR/evals/eval_expressive.txt";;
  multilang) CONFIG="$SCRIPT_DIR/evals/eval_multilang.txt";;
  *) echo "Unknown eval type: $EVAL_TYPE (use: short, long, stress, expressive, multilang)"; exit 1;;
esac

mkdir -p "$OUTPUT_DIR"/{generated,refs}

gpu=0
mapfile -t LINES < <(grep -v '^#' "$CONFIG" | grep '|')

for line in "${LINES[@]}"; do
  name=$(echo "$line" | cut -d'|' -f1 | xargs)
  ref=$(echo "$line" | cut -d'|' -f2 | xargs)
  prompt=$(echo "$line" | cut -d'|' -f3-)
  prompt="${prompt## }"
  [ -z "$name" ] && continue

  [[ "$ref" == *.mp3 ]] || [[ "$ref" == *.flac ]] && \
    ffmpeg -y -i "$ref" "$OUTPUT_DIR/refs/${name}.wav" -loglevel quiet 2>/dev/null || \
    cp "$ref" "$OUTPUT_DIR/refs/${name}.wav" 2>/dev/null

  LORA_FLAG=""
  [ -n "$LORA" ] && LORA_FLAG="--lora $LORA"

  CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/src/inference.py" \
    --checkpoint ltx-2.3-22b-dev-audio-only.safetensors \
    --full-checkpoint ltx-2.3-22b-dev.safetensors \
    $LORA_FLAG --voice-sample "$ref" --prompt "$prompt" \
    --output "$OUTPUT_DIR/generated/${name}.wav" \
    --cfg-scale 2.5 --stg-scale 1.5 --rescale-scale 0 --modality-scale 1 \
    --duration-multiplier 1.1 --seed 42 2>&1 | grep "Output" &

  gpu=$(( (gpu + 1) % GPUS ))
  if [ $gpu -eq 0 ]; then wait; fi
done
wait
echo "Eval done: $(ls "$OUTPUT_DIR/generated/"*.wav 2>/dev/null | wc -l) samples in $OUTPUT_DIR/"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="mvtec_loco"

MODEL_SIZE="${MODEL_SIZE:-medium}"
WEIGHTS="${WEIGHTS:-models/teacher_medium.pth}"
IMAGENET_TRAIN_PATH="${IMAGENET_TRAIN_PATH:-../../datasets/ImageNet2012/train}"
MVTEC_LOCO_PATH="${MVTEC_LOCO_PATH:-../../datasets/mvtec_loco_anomaly_detection}"

if [[ ! -d "$MVTEC_LOCO_PATH" ]]; then
  echo "ERROR: MVTEC_LOCO_PATH not found: $MVTEC_LOCO_PATH" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: weights file not found: $WEIGHTS" >&2
  exit 1
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUT="$OUTPUT_DIR"
else
  base="output"
  mkdir -p "$base"
  n=1
  while [[ -e "$base/$n" ]]; do
    n=$((n + 1))
  done
  OUT="$base/$n"
fi

echo "Using --output_dir: $OUT"
echo "Scanning subdatasets under: $MVTEC_LOCO_PATH"

mapfile -t subdatasets < <(find "$MVTEC_LOCO_PATH" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
if [[ ${#subdatasets[@]} -eq 0 ]]; then
  echo "ERROR: no subdataset dirs found under: $MVTEC_LOCO_PATH" >&2
  exit 1
fi

for subdataset in "${subdatasets[@]}"; do
  echo "=== Training: dataset=$DATASET subdataset=$subdataset model_size=$MODEL_SIZE ==="
  "$PYTHON_BIN" efficientad.py \
    --dataset "$DATASET" \
    --subdataset "$subdataset" \
    --model_size "$MODEL_SIZE" \
    --weights "$WEIGHTS" \
    --imagenet_train_path "$IMAGENET_TRAIN_PATH" \
    --mvtec_loco_path "$MVTEC_LOCO_PATH" \
    --output_dir "$OUT"
done

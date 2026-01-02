#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

MVTEC_LOCO_PATH="${MVTEC_LOCO_PATH:-../../datasets/mvtec_loco_anomaly_detection}"

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUT="$OUTPUT_DIR"
else
  OUT="output/1"
  if [[ -d output ]]; then
    mapfile -t out_nums < <(find output -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n)
    if [[ ${#out_nums[@]} -gt 0 ]]; then
      last_idx=$(( ${#out_nums[@]} - 1 ))
      OUT="output/${out_nums[$last_idx]}"
    fi
  fi
fi

ANOMALY_MAPS_DIR="${ANOMALY_MAPS_DIR:-$OUT/anomaly_maps/mvtec_loco}"
METRICS_DIR="${METRICS_DIR:-$OUT/metrics/mvtec_loco}"

if [[ ! -d "$MVTEC_LOCO_PATH" ]]; then
  echo "ERROR: MVTEC_LOCO_PATH not found: $MVTEC_LOCO_PATH" >&2
  exit 1
fi

if [[ ! -d "$ANOMALY_MAPS_DIR" ]]; then
  echo "ERROR: ANOMALY_MAPS_DIR not found: $ANOMALY_MAPS_DIR" >&2
  exit 1
fi

if [[ ! -f "mvtec_loco_ad_evaluation/evaluate_experiment.py" ]]; then
  echo "ERROR: missing mvtec_loco_ad_evaluation/evaluate_experiment.py" >&2
  exit 1
fi

echo "Using --dataset_base_dir: $MVTEC_LOCO_PATH"
echo "Using --anomaly_maps_dir: $ANOMALY_MAPS_DIR"
echo "Using --output_dir base: $METRICS_DIR"

mapfile -t subdatasets < <(find "$MVTEC_LOCO_PATH" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
if [[ ${#subdatasets[@]} -eq 0 ]]; then
  echo "ERROR: no subdataset dirs found under: $MVTEC_LOCO_PATH" >&2
  exit 1
fi

for subdataset in "${subdatasets[@]}"; do
  test_maps_dir="$ANOMALY_MAPS_DIR/$subdataset/test"
  if [[ ! -d "$test_maps_dir" ]]; then
    echo "SKIP: anomaly maps not found: $test_maps_dir" >&2
    continue
  fi

  echo "=== Evaluating: object_name=$subdataset ==="
  "$PYTHON_BIN" mvtec_loco_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir "$MVTEC_LOCO_PATH" \
    --anomaly_maps_dir "$ANOMALY_MAPS_DIR" \
    --output_dir "$METRICS_DIR/$subdataset" \
    --object_name "$subdataset"
done

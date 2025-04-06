#!/bin/bash

# USAGE:
# ./run_infer.sh \
#   ./sample_images/source.jpg \
#   ./sample_images/target.jpg \
#   ./checkpoints/checkpoint_epoch_10.pth \
#   ./outputs/final_result.jpg

SOURCE_IMG="$1"
TARGET_IMG="$2"
CHECKPOINT="$3"
OUTPUT_PATH="$4"

# Optional: Make sure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Check files exist
if [[ ! -f "$SOURCE_IMG" ]]; then
  echo "[✗] Source image not found: $SOURCE_IMG"
  exit 1
fi

if [[ ! -f "$TARGET_IMG" ]]; then
  echo "[✗] Target image not found: $TARGET_IMG"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[✗] Checkpoint file not found: $CHECKPOINT"
  exit 1
fi

# Run inference
echo "[🚀] Running inference..."
python infer.py \
  --source "$SOURCE_IMG" \
  --target "$TARGET_IMG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_PATH"

echo "[✓] Done."

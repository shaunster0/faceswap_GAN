#!/bin/bash

# USAGE:
#./update_and_train.sh \
#  ./new_images \
#  ./archive/lfw-funneled/lfw_funneled \
#  ./config/config.yaml \
#  ./checkpoints

# ─── User Inputs ─────────────────────────────────────────────
NEW_IMAGES_DIR="$1"                      # New images to add (e.g., ./new_images)
EXISTING_DATASET_DIR="$2"               # Dataset base (e.g., ./archive/lfw-funneled/lfw_funneled)
CONFIG_FILE="$3"                        # Path to config.yaml
CHECKPOINT_DIR="$4"                     # Directory where checkpoints are saved
# ─────────────────────────────────────────────────────────────

echo "[+] Copying new images from '$NEW_IMAGES_DIR' into dataset at '$EXISTING_DATASET_DIR'..."

# Recursively copy new images into correct subdirectories
find "$NEW_IMAGES_DIR" -type f -name "*.jpg" | while read -r image_path; do
    person=$(basename "$(dirname "$image_path")")
    mkdir -p "$EXISTING_DATASET_DIR/$person"
    cp "$image_path" "$EXISTING_DATASET_DIR/$person/"
done

echo "[✓] Image sync complete."

# ─── Find latest checkpoint (if any) ─────────────────────────
CHECKPOINT_PATH=""
if [ -d "$CHECKPOINT_DIR" ]; then
    latest_checkpoint=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -n 1)
    if [ -f "$latest_checkpoint" ]; then
        echo "[+] Resuming from latest checkpoint: $latest_checkpoint"
        CHECKPOINT_PATH="--checkpoint $latest_checkpoint"
    else
        echo "[!] No checkpoints found. Starting from scratch."
    fi
else
    echo "[!] Checkpoint directory '$CHECKPOINT_DIR' not found. Creating..."
    mkdir -p "$CHECKPOINT_DIR"
fi

# ─── Launch training ─────────────────────────────────────────
echo "[🚀] Launching training..."
python train.py \
  --config "$CONFIG_FILE" \
  --data_dir "$EXISTING_DATASET_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  $CHECKPOINT_PATH


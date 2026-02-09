#!/usr/bin/env bash
# Launch frame-flicker IC LoRA training on 8 GPUs via DDP.
# Run with no arguments: bash train.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"
ACCEL_CFG="$HERE/configs/accelerate_ddp.yaml"
TRAIN_CFG="$HERE/configs/frame_flicker_ic_lora.yaml"

# Nuke any default accelerate config that could interfere
rm -f ~/.cache/huggingface/accelerate/default_config.yaml

echo "======================================================================"
echo "  Frame Flicker IC LoRA Training - 8x GPU DDP"
echo "  Accelerate: $ACCEL_CFG"
echo "  Train cfg:  $TRAIN_CFG"
echo "  Trainer:    $LTX_TRAINER"
echo "======================================================================"

cd "$LTX_TRAINER"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run accelerate launch \
    --config_file "$ACCEL_CFG" \
    scripts/train.py "$TRAIN_CFG"

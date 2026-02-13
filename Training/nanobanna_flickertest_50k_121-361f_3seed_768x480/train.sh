#!/usr/bin/env bash
# Launch nanobanna flicker IC LoRA training on 8 GPUs via DDP.
# 50K samples, variable frame lengths (121/241/361), 3 seeds, 768x480 resolution
# Run with no arguments: bash train.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"
ACCEL_CFG="$HERE/configs/accelerate_ddp.yaml"
TRAIN_CFG="$HERE/configs/nanobanna_flicker_ic_lora.yaml"

# Nuke any default accelerate config that could interfere
rm -f ~/.cache/huggingface/accelerate/default_config.yaml

echo "======================================================================"
echo "  Nanobanna Flicker IC LoRA Training - 8x GPU DDP"
echo "  50K samples, 121-361 frames, 3 seeds, 768x480"
echo "  Accelerate: $ACCEL_CFG"
echo "  Train cfg:  $TRAIN_CFG"
echo "  Trainer:    $LTX_TRAINER"
echo "======================================================================"

cd "$LTX_TRAINER"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run accelerate launch \
    --config_file "$ACCEL_CFG" \
    scripts/train.py "$TRAIN_CFG"

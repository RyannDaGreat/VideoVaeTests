#!/usr/bin/env bash
# Run inference on the latest checkpoint using 8 test videos across 8 GPUs in parallel.
# Run with no arguments: bash validate.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"
OUTPUTS_DIR="$HERE/outputs"
INFERENCE_DIR="$HERE/inference_outputs"
TEST_JSON="$HERE/datasets/test_set.json"

# Find the latest checkpoint
LATEST=$(ls -1v "$OUTPUTS_DIR/checkpoints"/lora_weights_step_*.safetensors 2>/dev/null | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoints found in $OUTPUTS_DIR/checkpoints/"
    exit 1
fi
STEP_NAME=$(basename "$LATEST" .safetensors | sed 's/lora_weights_//')
STEP_DIR="$INFERENCE_DIR/$STEP_NAME"
mkdir -p "$STEP_DIR"

echo "======================================================================"
echo "  Validation - $STEP_NAME (8 GPUs parallel)"
echo "  LoRA:  $LATEST"
echo "  Output: $STEP_DIR"
echo "======================================================================"

cd "$LTX_TRAINER"

run() {
    local GPU="$1"
    local VID_ID="$2"
    local REF="$3"
    local CAPTION="$4"
    local OUTPUT="$STEP_DIR/${VID_ID}.mp4"

    if [ -f "$OUTPUT" ]; then
        echo "[GPU $GPU] $VID_ID already exists, skipping"
        return
    fi

    echo "[GPU $GPU] Generating $VID_ID..."
    uv run python scripts/inference.py \
        --checkpoint /models/LTX2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized \
        --lora-path "$LATEST" \
        --reference-video "$REF" \
        --prompt "$CAPTION" \
        --height 320 --width 512 --num-frames 121 \
        --skip-audio \
        --include-reference-in-output \
        --device "cuda:$GPU" \
        --output "$OUTPUT" \
    && echo "[GPU $GPU] Done: $VID_ID" \
    || echo "[GPU $GPU] FAILED: $VID_ID"
}

# Extract first 8 test videos into arrays
eval "$(python3 -c "
import json
with open('$TEST_JSON') as f:
    tests = json.load(f)[:8]
for i, t in enumerate(tests):
    vid = t['video_id']
    ref = t['reference_path']
    # Escape single quotes in caption for bash
    cap = t['caption'].replace(\"'\", \"'\\\\''\")
    print(f\"VID_IDS[{i}]='{vid}'\")
    print(f\"REFS[{i}]='{ref}'\")
    print(f\"CAPS[{i}]='{cap}'\")
print(f'NUM_VIDEOS={len(tests)}')
")"

declare -a VID_IDS REFS CAPS

# Launch all 8 in parallel, one per GPU
for i in $(seq 0 $((NUM_VIDEOS - 1))); do
    run "$i" "${VID_IDS[$i]}" "${REFS[$i]}" "${CAPS[$i]}" &
done

wait
echo ""
echo "Done. Videos in: $STEP_DIR"

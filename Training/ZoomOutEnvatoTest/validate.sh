#!/usr/bin/env bash
# Run inference on the latest checkpoint using first 4 test videos.
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
echo "  Validation - $STEP_NAME"
echo "  LoRA:  $LATEST"
echo "  Output: $STEP_DIR"
echo "======================================================================"

# Run first 4 test videos sequentially on GPU 0
cd "$LTX_TRAINER"
while IFS= read -r line; do
    VID_ID=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['video_id'])")
    REF=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['reference_path'])")
    CAPTION=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['caption'])")
    OUTPUT="$STEP_DIR/${VID_ID}.mp4"

    echo ""
    echo "Generating $VID_ID..."
    uv run python scripts/inference.py \
        --checkpoint /models/LTX2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized \
        --lora-path "$LATEST" \
        --reference-video "$REF" \
        --prompt "$CAPTION" \
        --height 320 --width 512 --num-frames 121 \
        --skip-audio \
        --include-reference-in-output \
        --device cuda:0 \
        --output "$OUTPUT" \
    || echo "WARNING: failed for $VID_ID"
done < <(python3 -c "import json; [print(json.dumps(t)) for t in json.load(open('$TEST_JSON'))[:4]]")

echo ""
echo "Done. Videos in: $STEP_DIR"

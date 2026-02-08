#!/usr/bin/env bash
# Periodic inference loop: monitors for new checkpoints, generates test videos.
# Splits work across GPU 6 (first half) and GPU 7 (second half).
# Run with no arguments: bash inference.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"
OUTPUTS_DIR="$HERE/outputs"
INFERENCE_DIR="$HERE/inference_outputs"
TEST_JSON="$HERE/datasets/test_set.json"
CHECKPOINT_PATH="$REPO_ROOT/LTX2/models/ltx-2-19b-dev.safetensors"
TEXT_ENCODER="$REPO_ROOT/LTX2/models/gemma-3-12b-it-qat-q4_0-unquantized"

NUM_TEST_VIDEOS=10  # Use first 10 from test set
HEIGHT=320
WIDTH=512
NUM_FRAMES=121
FRAME_RATE=24.0
INFERENCE_STEPS=30
GUIDANCE_SCALE=4.0

mkdir -p "$INFERENCE_DIR"

if [ ! -f "$TEST_JSON" ]; then
    echo "ERROR: $TEST_JSON not found. Run prepare_dataset.py first."
    exit 1
fi

# Track which checkpoints we've already processed
PROCESSED_FILE="$INFERENCE_DIR/.processed_checkpoints"
touch "$PROCESSED_FILE"

echo "======================================================================"
echo "  Inference Monitor - GPUs 6,7"
echo "  Watching: $OUTPUTS_DIR for new checkpoints"
echo "  Test set: $TEST_JSON ($NUM_TEST_VIDEOS videos)"
echo "======================================================================"

run_inference_for_checkpoint() {
    local LORA_PATH="$1"
    local STEP_NAME="$2"
    local STEP_DIR="$INFERENCE_DIR/$STEP_NAME"
    mkdir -p "$STEP_DIR"

    echo ""
    echo "--- Generating videos for $STEP_NAME ---"
    echo "LoRA: $LORA_PATH"

    # Read test videos and split across GPUs
    python3 -c "
import json, subprocess, sys, os

with open('$TEST_JSON') as f:
    tests = json.load(f)[:$NUM_TEST_VIDEOS]

half = len(tests) // 2
gpu6_tests = tests[:half]
gpu7_tests = tests[half:]

procs = []
for gpu_id, gpu_tests, label in [(6, gpu6_tests, 'gpu6'), (7, gpu7_tests, 'gpu7')]:
    for i, t in enumerate(gpu_tests):
        vid_id = t['video_id']
        output = '$STEP_DIR/${label}_vid%02d_%s.mp4' % (i+1, vid_id)
        if os.path.exists(output.replace('\${label}', label).replace('\${i+1}', str(i+1)).replace('\${vid_id}', vid_id)):
            continue
        cmd = [
            sys.executable,
            '$LTX_TRAINER/scripts/inference.py',
            '--checkpoint', '$CHECKPOINT_PATH',
            '--text-encoder-path', '$TEXT_ENCODER',
            '--lora-path', '$LORA_PATH',
            '--reference-video', t['reference_path'],
            '--prompt', t['caption'][:500],
            '--height', '$HEIGHT',
            '--width', '$WIDTH',
            '--num-frames', '$NUM_FRAMES',
            '--frame-rate', '$FRAME_RATE',
            '--num-inference-steps', '$INFERENCE_STEPS',
            '--guidance-scale', '$GUIDANCE_SCALE',
            '--skip-audio',
            '--include-reference-in-output',
            '--device', f'cuda:{gpu_id}',
            '--output', output,
        ]
        print(f'[{label}] Generating {vid_id}...')
        result = subprocess.run(cmd, cwd='$LTX_TRAINER')
        if result.returncode != 0:
            print(f'WARNING: inference failed for {vid_id}')
" || echo "WARNING: some inference jobs failed"

    echo "--- Done: $STEP_NAME ---"
}

echo "Entering monitoring loop..."
while true; do
    # Find checkpoint directories (contain lora_weights.safetensors)
    FOUND_NEW=false
    for CKPT_DIR in $(find "$OUTPUTS_DIR" -name "lora_weights.safetensors" -type f 2>/dev/null | sort); do
        CKPT_PARENT="$(dirname "$CKPT_DIR")"
        STEP_NAME="$(basename "$CKPT_PARENT")"
        LORA_FILE="$CKPT_DIR"

        # Skip if already processed
        if grep -qF "$STEP_NAME" "$PROCESSED_FILE" 2>/dev/null; then
            continue
        fi

        echo ""
        echo "======================================================================"
        echo "  NEW CHECKPOINT: $STEP_NAME"
        echo "======================================================================"

        run_inference_for_checkpoint "$LORA_FILE" "$STEP_NAME"

        echo "$STEP_NAME" >> "$PROCESSED_FILE"
        FOUND_NEW=true
    done

    if [ "$FOUND_NEW" = false ]; then
        echo "[$(date '+%H:%M:%S')] No new checkpoints. Sleeping 60s..."
    fi
    sleep 60
done

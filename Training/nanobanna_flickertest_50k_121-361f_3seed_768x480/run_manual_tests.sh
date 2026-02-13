#!/usr/bin/env bash
# Run manual tests from manual_tests/tests.json using the latest checkpoint.
# Each test is processed in parallel across GPUs (round-robin).
# Run with no arguments: bash run_manual_tests.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"
OUTPUTS_DIR="$HERE/outputs"
MANUAL_DIR="$HERE/manual_tests"
TESTS_JSON="$MANUAL_DIR/tests.json"

NUM_GPUS=8

# Find the latest checkpoint
LATEST=$(ls -1v "$OUTPUTS_DIR/checkpoints"/lora_weights_step_*.safetensors 2>/dev/null | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoints found in $OUTPUTS_DIR/checkpoints/"
    exit 1
fi
STEP_NAME=$(basename "$LATEST" .safetensors | sed 's/lora_weights_//')

echo "======================================================================"
echo "  Manual Tests - $STEP_NAME"
echo "  LoRA:  $LATEST"
echo "  Tests: $TESTS_JSON"
echo "======================================================================"

# First, generate reference videos from tests.json using transform logic
# Then run inference on each test
cd "$LTX_TRAINER"

# Parse tests and build reference videos + run inference
eval "$(python3 -c "
import json, sys, os
sys.path.insert(0, '$HERE/datasets')
sys.path.insert(0, os.path.expanduser('~/CleanCode'))
import rp
from pathlib import Path

MANUAL_DIR = Path('$MANUAL_DIR')

with open('$TESTS_JSON') as f:
    tests = json.load(f)

i = 0
for t in tests:
    name = t['name']
    input_video = str(MANUAL_DIR / t['input_video'])
    first_frame_path = str(MANUAL_DIR / t['first_frame']) if t.get('first_frame') else ''
    output_video = str(MANUAL_DIR / t['output_video']).replace('.mp4', '_${STEP_NAME}.mp4')
    caption = t['caption'].replace(\"'\", \"'\\\\\\''\")
    num_frames = t.get('num_frames', 121)
    keyframes = t.get('keyframes', [])
    num_steps = t.get('num_diffusion_steps', 30)
    seed = t.get('seed', 42)

    # Generate flickery reference video
    ref_dir = MANUAL_DIR / 'generated_references'
    ref_dir.mkdir(exist_ok=True)
    ref_path = ref_dir / f'{name}_ref.mp4'

    if not ref_path.exists():
        print(f'# Generating reference video for {name}...', file=sys.stderr)
        video = rp.load_video(input_video, use_cache=False)

        # Load first frame as the keyframe image
        keyframe_img = rp.load_image(str(MANUAL_DIR / t['first_frame']), use_cache=False)
        keyframe_img = rp.as_byte_image(rp.as_rgb_image(keyframe_img, copy=False), copy=False)

        # Resize to match video width
        keyframe_img, video = rp.resize_videos_to_hold([keyframe_img], video, width=768)
        height = rp.get_video_height(keyframe_img)
        video = rp.crop_images(video, height=height, origin='center')

        # Subsample video to target frame count
        import numpy as np
        indices = np.linspace(0, len(video)-1, num_frames, dtype=int)
        video = video[indices]

        # Create flickery reference: repeat the single keyframe image at all keyframe positions
        nn_indices = rp.quantize_to_nearest_values(range(num_frames), keyframes)
        nn_frames = rp.np.stack([keyframe_img[0]] * num_frames)

        # Pulse mask
        indicator_size = 0.2
        mask_width = round(indicator_size * rp.get_video_width(keyframe_img))
        out_height = round((1 + indicator_size) * height)
        out_width = round((1 + indicator_size) * rp.get_video_width(keyframe_img))

        pulse_mask = rp.np.zeros((num_frames, out_height, mask_width, 3), dtype=rp.np.uint8)
        for ki in keyframes:
            if ki < num_frames:
                pulse_mask[ki] = 255

        out_shape = (num_frames, out_height, out_width, 3)
        ref_out = rp.np.zeros(out_shape, dtype=rp.np.uint8)
        ref_out[:, -height:, -rp.get_video_width(keyframe_img):] = nn_frames
        ref_out[:, :out_height, :mask_width] = pulse_mask
        ref_out = rp.resize_videos(ref_out, size=(height, rp.get_video_width(keyframe_img)))[0]

        rp.save_video_mp4(ref_out, str(ref_path), framerate=25)
        print(f'# Saved reference: {ref_path}', file=sys.stderr)

    print(f\"NAMES[{i}]='{name}'\")
    print(f\"REFS[{i}]='{ref_path}'\")
    print(f\"OUTPUTS[{i}]='{output_video}'\")
    print(f\"CAPS[{i}]='{caption}'\")
    print(f\"NFRAMES[{i}]='{num_frames}'\")
    print(f\"NSTEPS[{i}]='{num_steps}'\")
    print(f\"SEEDS[{i}]='{seed}'\")
    print(f\"FIRST_FRAMES[{i}]='{first_frame_path}'\")
    i += 1

print(f'NUM_TESTS={i}')
")"

declare -a NAMES REFS OUTPUTS CAPS NFRAMES NSTEPS SEEDS FIRST_FRAMES

echo "Running $NUM_TESTS manual tests..."

run() {
    local GPU="$1"
    local NAME="$2"
    local REF="$3"
    local OUTPUT="$4"
    local CAPTION="$5"
    local NFRAMES="$6"
    local NSTEPS="$7"
    local SEED="$8"

    if [ -f "$OUTPUT" ]; then
        echo "[GPU $GPU] $NAME already exists, skipping"
        return
    fi

    echo "[GPU $GPU] Running test: $NAME (${NFRAMES}f, ${NSTEPS} steps, seed $SEED)..."
    uv run python scripts/inference.py \
        --checkpoint /models/LTX2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized \
        --lora-path "$LATEST" \
        --reference-video "$REF" \
        --prompt "$CAPTION" \
        --height 480 --width 768 --num-frames "$NFRAMES" \
        --inference-steps "$NSTEPS" \
        --seed "$SEED" \
        --skip-audio \
        --include-reference-in-output \
        --device "cuda:$GPU" \
        --output "$OUTPUT" \
    && echo "[GPU $GPU] Done: $NAME" \
    || echo "[GPU $GPU] FAILED: $NAME"
}

# Launch in waves of NUM_GPUS
for ((start=0; start<NUM_TESTS; start+=NUM_GPUS)); do
    for ((j=0; j<NUM_GPUS && start+j<NUM_TESTS; j++)); do
        i=$((start + j))
        run "$j" "${NAMES[$i]}" "${REFS[$i]}" "${OUTPUTS[$i]}" "${CAPS[$i]}" "${NFRAMES[$i]}" "${NSTEPS[$i]}" "${SEEDS[$i]}" &
    done
    wait
done

echo ""
echo "Done. Outputs in: $MANUAL_DIR/test_outputs/"

#!/usr/bin/env bash
# Generate frame-flicker IC LoRA dataset from Envato videos.
# Reuses the same train/test split as ZoomOutEnvatoTest.
# Run with no arguments: bash datasets/make_dataset.bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
REPO_ROOT="$(cd "$PROJECT" && git rev-parse --show-toplevel)"
LTX_TRAINER="$REPO_ROOT/LTX2/src/packages/ltx-trainer"

# Source data
ENVATO_RAW="/root/CleanCode/Datasets/Envato/downloads/raw_videos"
ZOOM_OUT="$REPO_ROOT/Training/ZoomOutEnvatoTest/datasets"

# Output dirs
VIDEOS_DIR="$HERE/videos"
REF_DIR="$HERE/reference_videos"
TEST_VIDEOS_DIR="$HERE/test_videos"
TEST_REF_DIR="$HERE/test_reference_videos"

TRANSFORM="$HERE/transform.py"
WORKERS=32
NUM_FRAMES=121

mkdir -p "$VIDEOS_DIR" "$REF_DIR" "$TEST_VIDEOS_DIR" "$TEST_REF_DIR"

echo "======================================================================"
echo "  Frame Flicker Dataset Generator"
echo "  Source: $ZOOM_OUT (reusing train/test split)"
echo "  Workers: $WORKERS"
echo "======================================================================"

# --------------------------------------------------------------------------
# Phase 1: Generate ref/tgt pairs using transform.py
# --------------------------------------------------------------------------
generate_pairs() {
    local SPLIT_JSON="$1"
    local VID_DIR="$2"
    local RDIR="$3"
    local LABEL="$4"

    # Extract video IDs and captions, derive Envato source paths
    local TASKS
    TASKS=$(python3 -c "
import json, sys

with open('$SPLIT_JSON') as f:
    data = json.load(f)

# Handle both formats: dataset.json has 'media_path', test_set.json has 'video_id'
for entry in data:
    if 'video_id' in entry:
        vid_id = entry['video_id']
    else:
        # media_path is like 'videos/DFL7MW3.mp4'
        vid_id = entry['media_path'].split('/')[-1].replace('.mp4', '')

    # Derive Envato source path: ID -> first2/next2/ID.mp4
    src_rel = vid_id[:2] + '/' + vid_id[2:4] + '/' + vid_id + '.mp4'
    src_path = '$ENVATO_RAW/' + src_rel

    print(vid_id + '\t' + src_path)
")

    local TOTAL
    TOTAL=$(echo "$TASKS" | wc -l)
    local DONE=0
    local SKIPPED=0

    echo ""
    echo "--- $LABEL: $TOTAL videos, $WORKERS parallel workers ---"

    # Run in parallel with job control
    local RUNNING=0
    while IFS=$'\t' read -r VID_ID SRC_PATH; do
        local TGT="$VID_DIR/${VID_ID}.mp4"
        local REF="$RDIR/${VID_ID}.mp4"

        # Skip if both already exist
        if [ -f "$TGT" ] && [ -f "$REF" ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Skip if source doesn't exist
        if [ ! -f "$SRC_PATH" ]; then
            echo "  SKIP $VID_ID: source not found"
            continue
        fi

        python3 "$TRANSFORM" "$SRC_PATH" \
            --ref_out_path "$REF" \
            --tgt_out_path "$TGT" \
            --num_frames "$NUM_FRAMES" &

        RUNNING=$((RUNNING + 1))
        if [ "$RUNNING" -ge "$WORKERS" ]; then
            wait -n
            RUNNING=$((RUNNING - 1))
        fi

        DONE=$((DONE + 1))
        if [ $((DONE % 100)) -eq 0 ]; then
            echo "  [$LABEL] $DONE/$TOTAL processed..."
        fi
    done <<< "$TASKS"

    wait
    echo "  [$LABEL] Done. Processed: $DONE, Skipped (existing): $SKIPPED"
}

# Train split
generate_pairs "$ZOOM_OUT/dataset.json" "$VIDEOS_DIR" "$REF_DIR" "train"

# Test split
generate_pairs "$ZOOM_OUT/test_set.json" "$TEST_VIDEOS_DIR" "$TEST_REF_DIR" "test"

# --------------------------------------------------------------------------
# Phase 2: Generate dataset JSON files
# --------------------------------------------------------------------------
echo ""
echo "--- Generating dataset JSON files ---"

python3 -c "
import json, os

# Train set: read from zoom-out dataset.json for captions, use our local video paths
with open('$ZOOM_OUT/dataset.json') as f:
    zoom_data = json.load(f)

train_entries = []
for entry in zoom_data:
    vid_id = entry['media_path'].split('/')[-1].replace('.mp4', '')
    tgt = '$VIDEOS_DIR/' + vid_id + '.mp4'
    ref = '$REF_DIR/' + vid_id + '.mp4'
    if os.path.exists(tgt) and os.path.exists(ref):
        train_entries.append({
            'caption': entry['caption'],
            'media_path': 'videos/' + vid_id + '.mp4',
            'reference_path': 'reference_videos/' + vid_id + '.mp4',
        })

with open('$HERE/dataset.json', 'w') as f:
    json.dump(train_entries, f, indent=2)
print(f'Train: {len(train_entries)} entries -> $HERE/dataset.json')

# Test set
with open('$ZOOM_OUT/test_set.json') as f:
    zoom_test = json.load(f)

test_entries = []
for entry in zoom_test:
    vid_id = entry['video_id']
    tgt = '$TEST_VIDEOS_DIR/' + vid_id + '.mp4'
    ref = '$TEST_REF_DIR/' + vid_id + '.mp4'
    if os.path.exists(tgt) and os.path.exists(ref):
        test_entries.append({
            'video_id': vid_id,
            'caption': entry['caption'],
            'video_path': tgt,
            'reference_path': ref,
        })

with open('$HERE/test_set.json', 'w') as f:
    json.dump(test_entries, f, indent=2)
print(f'Test:  {len(test_entries)} entries -> $HERE/test_set.json')
"

# --------------------------------------------------------------------------
# Phase 3: Precompute latents with LTX process_dataset.py
# --------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  Phase 3: Precomputing latents (LTX process_dataset.py)"
echo "======================================================================"

cd "$LTX_TRAINER"
uv run python scripts/process_dataset.py \
    "$HERE/dataset.json" \
    --resolution-buckets "512x320x121" \
    --model-path /models/LTX2/ltx-2-19b-dev.safetensors \
    --text-encoder-path /models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized \
    --reference-column reference_path

echo ""
echo "======================================================================"
echo "  Done! Dataset ready for training."
echo "======================================================================"

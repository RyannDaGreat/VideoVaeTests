#!/usr/bin/env python3
"""Prepare zoom-out IC LoRA dataset from Envato videos.

Filters the Envato CSV to videos that exist on disk, generates synchronized
target + reference video pairs (same resolution, same frames), creates the
dataset metadata, and calls the LTX trainer's process_dataset.py to precompute
latents.

Run with no arguments:
    python prepare_dataset.py
"""

import csv
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# === PATH CONFIGURATION (all at top, easy to refactor) ===
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=str(HERE),
).decode().strip())

# External dataset (only global path - lives outside repo)
ENVATO_CSV = Path("/root/CleanCode/Datasets/Envato/captioned_envato_3869336.csv")
ENVATO_VIDEOS = Path("/root/CleanCode/Datasets/Envato/downloads/raw_videos")

# Repo-relative paths
LTX_TRAINER = REPO_ROOT / "LTX2" / "src" / "packages" / "ltx-trainer"
MODEL_PATH = str(REPO_ROOT / "LTX2" / "models" / "ltx-2-19b-dev.safetensors")
TEXT_ENCODER_PATH = str(REPO_ROOT / "LTX2" / "models" / "gemma-3-12b-it-qat-q4_0-unquantized")

# Output paths (relative to this script)
DATASETS_DIR = HERE / "datasets"
VIDEOS_DIR = DATASETS_DIR / "videos"
REF_VIDEOS_DIR = DATASETS_DIR / "reference_videos"
TEST_VIDEOS_DIR = DATASETS_DIR / "test_videos"
TEST_REF_VIDEOS_DIR = DATASETS_DIR / "test_reference_videos"
DATASET_JSON = DATASETS_DIR / "dataset.json"

# === TRAINING PARAMETERS ===
NUM_TEST = 50
MIN_DURATION_SEC = 5.0  # Need 121 frames at 25fps = 4.84s, pad a bit
MAX_FRAMES = 121        # 8*15+1, LTX-2 temporal constraint
TARGET_FPS = 25         # LTX-2 default frame rate
RESOLUTION_BUCKET = "512x320x121"  # WxHxF for process_dataset.py
SEED = 42
PARALLEL_WORKERS = 16


def banner(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")


def create_video_pair(src_path, target_path, ref_path):
    """Create a synchronized target + reference video pair.

    Both videos will have IDENTICAL resolution, frame count, and fps.
    - Target: first MAX_FRAMES frames, native resolution, 25fps
    - Reference: same frames, but center-cropped 50% each dim and resized back

    This guarantees frame-perfect synchronization between the pair.
    """
    # Target: take first 121 frames at 25fps, keep native resolution
    cmd_target = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-frames:v", str(MAX_FRAMES),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        str(target_path),
    ]
    # Reference: same first 121 frames, center-crop 50% then resize back
    cmd_ref = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", "crop=iw/2:ih/2,scale=iw*2:ih*2:flags=lanczos",
        "-frames:v", str(MAX_FRAMES),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        str(ref_path),
    ]
    for cmd, label in [(cmd_target, "target"), (cmd_ref, "reference")]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg {label} failed for {src_path}: {result.stderr[-300:]}")


def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return 0
    return float(result.stdout.strip())


def main():
    t0 = time.time()

    banner("Phase 1: Loading Envato CSV")
    print(f"CSV: {ENVATO_CSV}")
    rows = []
    with open(ENVATO_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows):,} rows from CSV")

    banner("Phase 2: Filtering to existing videos")
    valid = []
    missing = 0
    for row in tqdm(rows, desc="Checking video files"):
        video_rel = row.get("videos", "")
        if not video_rel:
            missing += 1
            continue
        video_path = ENVATO_VIDEOS / video_rel
        if video_path.exists():
            row["_full_path"] = str(video_path)
            row["_video_id"] = Path(video_rel).stem
            valid.append(row)
        else:
            missing += 1
    print(f"Found {len(valid):,} existing videos, {missing:,} missing")

    banner("Phase 3: Filtering short videos (parallel ffprobe)")

    def check_dur(row):
        return row, get_video_duration(row["_full_path"])

    usable = []
    too_short = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(check_dur, row) for row in valid]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Checking durations"):
            row, dur = f.result()
            if dur >= MIN_DURATION_SEC:
                usable.append(row)
            else:
                too_short += 1
    print(f"Usable: {len(usable):,}, too short (<{MIN_DURATION_SEC}s): {too_short:,}")

    if len(usable) < NUM_TEST + 10:
        raise RuntimeError(f"Not enough usable videos ({len(usable)}) for train/test split")

    banner("Phase 4: Train/test split")
    random.seed(SEED)
    random.shuffle(usable)
    test_set = usable[:NUM_TEST]
    train_set = usable[NUM_TEST:]
    print(f"Test: {len(test_set)}, Train: {len(train_set)}")

    banner("Phase 5: Creating directories")
    for d in [DATASETS_DIR, VIDEOS_DIR, REF_VIDEOS_DIR, TEST_VIDEOS_DIR, TEST_REF_VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("Directories created")

    banner("Phase 6: Creating synchronized target + reference video pairs")
    print(f"  Both target and reference get identical: {MAX_FRAMES} frames @ {TARGET_FPS}fps")
    print(f"  Reference = center-crop 50% each dim, resize back to original")
    print(f"  Parallel workers: {PARALLEL_WORKERS}")

    def process_split(split, videos_dir, ref_dir, label):
        tasks = []
        for row in split:
            vid_id = row["_video_id"]
            src = Path(row["_full_path"])
            target = videos_dir / f"{vid_id}.mp4"
            ref = ref_dir / f"{vid_id}.mp4"
            # Skip if BOTH already exist
            if target.exists() and ref.exists():
                continue
            tasks.append((src, target, ref))

        if not tasks:
            print(f"  {label}: all {len(split)} pairs already exist, skipping")
            return

        print(f"  {label}: creating {len(tasks)} pairs ({len(split) - len(tasks)} already exist)")
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
            futures = {pool.submit(create_video_pair, s, t, r): t for s, t, r in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Pairs {label}"):
                target = futures[f]
                exc = f.exception()
                if exc:
                    print(f"  WARNING: failed {target.name}: {exc}")

    process_split(train_set, VIDEOS_DIR, REF_VIDEOS_DIR, "train")
    process_split(test_set, TEST_VIDEOS_DIR, TEST_REF_VIDEOS_DIR, "test")

    banner("Phase 7: Writing dataset JSON")
    # IMPORTANT: paths must be relative to the dataset JSON file location
    # because process_captions.py uses media_path to name output .pt files
    dataset_entries = []
    for row in train_set:
        vid_id = row["_video_id"]
        target = VIDEOS_DIR / f"{vid_id}.mp4"
        ref = REF_VIDEOS_DIR / f"{vid_id}.mp4"
        if target.exists() and ref.exists():
            dataset_entries.append({
                "caption": row.get("caption", ""),
                "media_path": f"videos/{vid_id}.mp4",
                "reference_path": f"reference_videos/{vid_id}.mp4",
            })
    with open(DATASET_JSON, "w") as f:
        json.dump(dataset_entries, f, indent=2)
    print(f"Wrote {len(dataset_entries)} entries to {DATASET_JSON}")

    # Test set metadata uses absolute paths (for inference script convenience)
    test_json = DATASETS_DIR / "test_set.json"
    test_entries = []
    for row in test_set:
        vid_id = row["_video_id"]
        target = TEST_VIDEOS_DIR / f"{vid_id}.mp4"
        ref = TEST_REF_VIDEOS_DIR / f"{vid_id}.mp4"
        if target.exists() and ref.exists():
            test_entries.append({
                "video_id": vid_id,
                "caption": row.get("caption", ""),
                "video_path": str(target),
                "reference_path": str(ref),
            })
    with open(test_json, "w") as f:
        json.dump(test_entries, f, indent=2)
    print(f"Wrote {len(test_entries)} test entries to {test_json}")

    banner("Phase 8: Running LTX process_dataset.py to precompute latents")
    precompute_dir = str(DATASETS_DIR / ".precomputed")
    # Must use 'uv run' from the ltx-trainer directory for proper environment
    cmd = [
        "uv", "run", "python", "scripts/process_dataset.py",
        str(DATASET_JSON),
        "--resolution-buckets", RESOLUTION_BUCKET,
        "--model-path", MODEL_PATH,
        "--text-encoder-path", TEXT_ENCODER_PATH,
        "--reference-column", "reference_path",
        "--output-dir", precompute_dir,
    ]
    print(f"Running from {LTX_TRAINER}:")
    print(f"  {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, cwd=str(LTX_TRAINER))
    if result.returncode != 0:
        raise RuntimeError(f"process_dataset.py failed with exit code {result.returncode}")

    elapsed = time.time() - t0
    banner(f"DONE! Total time: {elapsed/60:.1f} minutes")
    print(f"  Train videos: {len(dataset_entries)}")
    print(f"  Test videos:  {len(test_entries)}")
    print(f"  Dataset JSON: {DATASET_JSON}")
    print(f"  Precomputed:  {precompute_dir}")


if __name__ == "__main__":
    main()

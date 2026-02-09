#!/usr/bin/env python3
"""Generate frame-flicker IC LoRA dataset from Envato videos.

Reuses the same train/test split as ZoomOutEnvatoTest. Runs transform.py on
each video in parallel, generates dataset JSONs, then calls LTX process_dataset.py
to precompute latents.

Run with no arguments: python datasets/make_dataset.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

# === PATHS ===
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO_ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=str(PROJECT),
).decode().strip())

ENVATO_RAW = Path("/root/CleanCode/Datasets/Envato/downloads/raw_videos")
ZOOM_OUT = REPO_ROOT / "Training" / "ZoomOutEnvatoTest" / "datasets"
LTX_TRAINER = REPO_ROOT / "LTX2" / "src" / "packages" / "ltx-trainer"
TRANSFORM = HERE / "transform.py"

VIDEOS_DIR = HERE / "videos"
REF_DIR = HERE / "reference_videos"
TEST_VIDEOS_DIR = HERE / "test_videos"
TEST_REF_DIR = HERE / "test_reference_videos"

WORKERS = 32
NUM_FRAMES = 121


def vid_id_from_entry(entry):
    """Extract video ID from either dataset.json or test_set.json format."""
    if "video_id" in entry:
        return entry["video_id"]
    return Path(entry["media_path"]).stem


def envato_source_path(vid_id):
    """Derive Envato source path from video ID: ABC... -> AB/C.../ABC....mp4"""
    return ENVATO_RAW / vid_id[:2] / vid_id[2:4] / f"{vid_id}.mp4"


def generate_pairs(split_json, vid_dir, ref_dir, label):
    """Run transform.py on all videos in a split, WORKERS at a time."""
    with open(split_json) as f:
        entries = json.load(f)
    if MAX_ITEMS:
        entries = entries[:MAX_ITEMS]

    # Build task list, skip existing
    tasks = []
    skipped = 0
    for entry in entries:
        vid_id = vid_id_from_entry(entry)
        src = envato_source_path(vid_id)
        tgt = vid_dir / f"{vid_id}.mp4"
        ref = ref_dir / f"{vid_id}.mp4"

        if tgt.exists() and ref.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"  SKIP {vid_id}: source not found")
            continue
        tasks.append((vid_id, src, tgt, ref))

    print(f"\n--- {label}: {len(tasks)} to process, {skipped} already exist ---")
    if not tasks:
        return

    # Run WORKERS subprocesses at a time
    active = []
    for vid_id, src, tgt, ref in tqdm(tasks, desc=label):
        cmd = [
            sys.executable, str(TRANSFORM), str(src),
            "--ref_out_path", str(ref),
            "--tgt_out_path", str(tgt),
            "--num_frames", str(NUM_FRAMES),
        ]
        active.append((vid_id, subprocess.Popen(cmd)))

        if len(active) >= WORKERS:
            # Wait for the first one to finish
            vid, proc = active.pop(0)
            proc.wait()
            if proc.returncode != 0:
                print(f"  FAILED: {vid}")

    # Drain remaining
    for vid, proc in active:
        proc.wait()
        if proc.returncode != 0:
            print(f"  FAILED: {vid}")


def generate_jsons():
    """Generate dataset.json and test_set.json for LTX trainer."""
    print("\n--- Generating dataset JSON files ---")

    # Train set
    with open(ZOOM_OUT / "dataset.json") as f:
        zoom_data = json.load(f)
    if MAX_ITEMS:
        zoom_data = zoom_data[:MAX_ITEMS]

    train_entries = []
    for entry in zoom_data:
        vid_id = Path(entry["media_path"]).stem
        if (VIDEOS_DIR / f"{vid_id}.mp4").exists() and (REF_DIR / f"{vid_id}.mp4").exists():
            train_entries.append({
                "caption": entry["caption"],
                "media_path": f"videos/{vid_id}.mp4",
                "reference_path": f"reference_videos/{vid_id}.mp4",
            })

    with open(HERE / "dataset.json", "w") as f:
        json.dump(train_entries, f, indent=2)
    print(f"Train: {len(train_entries)} entries")

    # Test set
    with open(ZOOM_OUT / "test_set.json") as f:
        zoom_test = json.load(f)

    test_entries = []
    for entry in zoom_test:
        vid_id = entry["video_id"]
        tgt = TEST_VIDEOS_DIR / f"{vid_id}.mp4"
        ref = TEST_REF_DIR / f"{vid_id}.mp4"
        if tgt.exists() and ref.exists():
            test_entries.append({
                "video_id": vid_id,
                "caption": entry["caption"],
                "video_path": str(tgt),
                "reference_path": str(ref),
            })

    with open(HERE / "test_set.json", "w") as f:
        json.dump(test_entries, f, indent=2)
    print(f"Test:  {len(test_entries)} entries")


NUM_GPUS = 8


def precompute_latents():
    """Call LTX process_dataset.py on 8 GPUs in parallel, one chunk per GPU."""
    with open(HERE / "dataset.json") as f:
        dataset = json.load(f)

    print("\n" + "=" * 70)
    print(f"  Precomputing latents ({len(dataset)} videos across {NUM_GPUS} GPUs)")
    print("=" * 70 + "\n")

    # Split dataset into NUM_GPUS chunks, write temporary JSONs in datasets/ dir
    # (must be same dir as dataset.json so relative media_path values resolve correctly)
    chunk_size = (len(dataset) + NUM_GPUS - 1) // NUM_GPUS
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    procs = []
    chunk_files = []
    for gpu_id, chunk in enumerate(chunks):
        chunk_json = HERE / f".chunk_{gpu_id}.json"
        chunk_files.append(chunk_json)
        with open(chunk_json, "w") as f:
            json.dump(chunk, f)

        cmd = [
            "uv", "run", "python", "scripts/process_dataset.py",
            str(chunk_json),
            "--resolution-buckets", "512x320x121",
            "--model-path", "/models/LTX2/ltx-2-19b-dev.safetensors",
            "--text-encoder-path", "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized",
            "--reference-column", "reference_path",
            "--output-dir", str(HERE / ".precomputed"),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        print(f"  [GPU {gpu_id}] {len(chunk)} videos")
        procs.append((gpu_id, subprocess.Popen(cmd, cwd=str(LTX_TRAINER), env=env)))

    # Wait for all
    failed = []
    for gpu_id, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            failed.append(gpu_id)

    # Cleanup chunk files
    for f in chunk_files:
        f.unlink(missing_ok=True)

    if failed:
        raise RuntimeError(f"process_dataset.py failed on GPUs: {failed}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Limit to 24 datapoints for testing")
    args = parser.parse_args()

    global MAX_ITEMS
    MAX_ITEMS = 24 if args.test else None

    t0 = time.time()

    for d in [VIDEOS_DIR, REF_DIR, TEST_VIDEOS_DIR, TEST_REF_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    mode = f"TEST MODE ({MAX_ITEMS} items)" if MAX_ITEMS else "FULL"
    print("=" * 70)
    print(f"  Frame Flicker Dataset Generator [{mode}]")
    print(f"  Source: {ZOOM_OUT} (reusing train/test split)")
    print(f"  Workers: {WORKERS}")
    print("=" * 70)

    generate_pairs(ZOOM_OUT / "dataset.json", VIDEOS_DIR, REF_DIR, "train")
    generate_pairs(ZOOM_OUT / "test_set.json", TEST_VIDEOS_DIR, TEST_REF_DIR, "test")
    generate_jsons()
    precompute_latents()

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed / 60:.1f} minutes")


MAX_ITEMS = None

if __name__ == "__main__":
    main()

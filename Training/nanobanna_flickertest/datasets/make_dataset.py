#!/usr/bin/env python3
"""Generate frame-flicker IC LoRA dataset from Pexels paired data.

Processes Yash's Pexels paired data samples, generates reference/target video pairs,
creates dataset JSONs, and precomputes latents using LTX trainer.

Run with no arguments: python datasets/make_dataset.py
Test mode (24 samples): python datasets/make_dataset.py --test

To change from 10K to 50K samples: change NUM_SAMPLES = 10 to NUM_SAMPLES = 50 below.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

# === CONFIGURATION ===
# Change this to 50 to process full 50K dataset (just change the "10" to "50")
NUM_SAMPLES = 10  # in thousands (10 = 10,000 samples, 50 = 50,000 samples)

# === PATHS ===
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO_ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=str(PROJECT),
).decode().strip())

PEXELS_PAIRED_DATA = Path("/root/CleanCode/Sandbox/RP_Dumps/YashDump/jan10_last_50K_pexels_v2/paired_data")
LTX_TRAINER = REPO_ROOT / "LTX2" / "src" / "packages" / "ltx-trainer"
TRANSFORM = HERE / "transform.py"

VIDEOS_DIR = HERE / "videos"
REF_DIR = HERE / "reference_videos"
TEST_VIDEOS_DIR = HERE / "test_videos"
TEST_REF_DIR = HERE / "test_reference_videos"
DEBUG_DIR = PROJECT / "debug_videos"  # Debug videos in training folder root

WORKERS = 32
NUM_FRAMES = 121
NUM_TEST_SAMPLES = 50
NUM_DEBUG_VIDEOS = 20  # Generate 20 debug videos for inspection


def get_sample_directories():
    """Get all Pexels sample directories, randomly sample NUM_SAMPLES*1000."""
    all_samples = sorted([d for d in PEXELS_PAIRED_DATA.iterdir() if d.is_dir()])
    print(f"Found {len(all_samples)} total Pexels samples")

    if MAX_ITEMS:
        # Test mode: just take first MAX_ITEMS
        return all_samples[:MAX_ITEMS], []

    # Production mode: randomly sample NUM_SAMPLES*1000
    import random
    random.seed(42)  # Reproducible

    target_count = NUM_SAMPLES * 1000
    if len(all_samples) < target_count:
        print(f"WARNING: Only {len(all_samples)} samples available, using all")
        selected = all_samples
    else:
        selected = random.sample(all_samples, target_count)

    # Split: NUM_TEST_SAMPLES for test, rest for train
    test_samples = selected[:NUM_TEST_SAMPLES]
    train_samples = selected[NUM_TEST_SAMPLES:]

    print(f"Selected {len(train_samples)} train + {len(test_samples)} test = {len(selected)} total")
    return train_samples, test_samples


def generate_pairs(samples, vid_dir, ref_dir, label):
    """Run transform.py on all samples, WORKERS at a time."""
    # Build task list, skip existing
    tasks = []
    skipped = 0
    for sample_dir in samples:
        sample_id = sample_dir.name
        tgt = vid_dir / f"{sample_id}.mp4"
        ref = ref_dir / f"{sample_id}.mp4"

        if tgt.exists() and ref.exists():
            skipped += 1
            continue

        # Check that required files exist
        if not (sample_dir / "after.png").exists():
            print(f"  SKIP {sample_id}: missing after.png")
            continue
        if not (sample_dir / "raw_video.mp4").exists():
            print(f"  SKIP {sample_id}: missing raw_video.mp4")
            continue
        if not (sample_dir / "metadata.json").exists():
            print(f"  SKIP {sample_id}: missing metadata.json")
            continue

        tasks.append((sample_id, sample_dir, tgt, ref))

    print(f"\n--- {label}: {len(tasks)} to process, {skipped} already exist ---")
    if not tasks:
        return

    # Run WORKERS subprocesses at a time
    active = []
    for sample_id, sample_dir, tgt, ref in tqdm(tasks, desc=label):
        cmd = [
            sys.executable, str(TRANSFORM), str(sample_dir),
            "--ref_out_path", str(ref),
            "--tgt_out_path", str(tgt),
            "--num_frames", str(NUM_FRAMES),
        ]
        active.append((sample_id, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))

        if len(active) >= WORKERS:
            # Wait for the first one to finish
            sample_id, proc = active.pop(0)
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode()
                print(f"  FAILED: {sample_id} - {stderr[:200]}")

    # Drain remaining
    for sample_id, proc in active:
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            print(f"  FAILED: {sample_id} - {stderr[:200]}")


def generate_jsons(train_samples, test_samples):
    """Generate dataset.json and test_set.json for LTX trainer."""
    print("\n--- Generating dataset JSON files ---")

    # Train set
    train_entries = []
    for sample_dir in train_samples:
        sample_id = sample_dir.name
        tgt = VIDEOS_DIR / f"{sample_id}.mp4"
        ref = REF_DIR / f"{sample_id}.mp4"
        if tgt.exists() and ref.exists():
            # Load caption from metadata.json - video_caption[0]
            meta_path = sample_dir / "metadata.json"
            caption = "A video"  # Fallback
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    video_captions = meta.get('video_caption', ["A video"])
                    caption = video_captions[0] if video_captions and video_captions[0] is not None else "A video"
                except Exception as e:
                    print(f"  WARNING: Could not read caption from {sample_id}: {e}")
                    caption = "A video"  # Ensure fallback on exception

            train_entries.append({
                "caption": caption,
                "media_path": f"videos/{sample_id}.mp4",
                "reference_path": f"reference_videos/{sample_id}.mp4",
            })

    with open(HERE / "dataset.json", "w") as f:
        json.dump(train_entries, f, indent=2)
    print(f"Train: {len(train_entries)} entries")

    # Test set
    test_entries = []
    for sample_dir in test_samples:
        sample_id = sample_dir.name
        tgt = TEST_VIDEOS_DIR / f"{sample_id}.mp4"
        ref = TEST_REF_DIR / f"{sample_id}.mp4"
        if tgt.exists() and ref.exists():
            # Load caption from metadata.json - video_caption[0]
            meta_path = sample_dir / "metadata.json"
            caption = "A video"  # Fallback
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    video_captions = meta.get('video_caption', ["A video"])
                    caption = video_captions[0] if video_captions and video_captions[0] is not None else "A video"
                except Exception as e:
                    print(f"  WARNING: Could not read caption from {sample_id}: {e}")
                    caption = "A video"  # Ensure fallback on exception

            test_entries.append({
                "video_id": sample_id,
                "caption": caption,
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

    # Split dataset into NUM_GPUS chunks, write temporary JSONs
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


def generate_debug_videos(train_samples):
    """Generate side-by-side debug videos for inspection."""
    import random
    import sys

    print("\n--- Generating debug videos for inspection ---")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Select random samples for debug videos
    debug_samples = random.sample(train_samples, min(NUM_DEBUG_VIDEOS, len(train_samples)))

    sys.path.insert(0, str(REPO_ROOT))
    import rp

    for sample_dir in tqdm(debug_samples, desc="Debug videos"):
        sample_id = sample_dir.name
        ref_path = VIDEOS_DIR / f"{sample_id}.mp4"
        tgt_path = VIDEOS_DIR / f"{sample_id}.mp4"

        # Check if both exist
        if not ref_path.exists() or not tgt_path.exists():
            continue

        try:
            # Load videos
            ref_video = rp.load_video(str(REF_DIR / f"{sample_id}.mp4"), use_cache=False)
            tgt_video = rp.load_video(str(VIDEOS_DIR / f"{sample_id}.mp4"), use_cache=False)

            # Create side-by-side with labels
            debug_video = rp.tiled_videos(
                rp.labeled_videos(
                    [ref_video, tgt_video],
                    ["Reference (flickery)", "Target (smooth)"],
                    size=20,
                ),
                border_thickness=2,
                border_color="white",
            )

            # Add frame numbers
            debug_video = rp.labeled_images(
                debug_video,
                [f"Frame {i+1}/{NUM_FRAMES}" for i in range(NUM_FRAMES)],
                size=20,
            )

            # Save
            output_path = DEBUG_DIR / f"{sample_id}_debug.mp4"
            rp.save_video_mp4(debug_video, str(output_path), framerate=25)

        except Exception as e:
            print(f"  FAILED debug video for {sample_id}: {e}")

    print(f"✓ Generated {len(list(DEBUG_DIR.glob('*_debug.mp4')))} debug videos in {DEBUG_DIR}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Limit to 24 datapoints for testing")
    args = parser.parse_args()

    global MAX_ITEMS
    MAX_ITEMS = 24 if args.test else None

    t0 = time.time()

    for d in [VIDEOS_DIR, REF_DIR, TEST_VIDEOS_DIR, TEST_REF_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    mode_str = f"TEST MODE ({MAX_ITEMS} items)" if MAX_ITEMS else f"FULL ({NUM_SAMPLES}K samples)"
    print("=" * 70)
    print(f"  Frame Flicker Dataset Generator - Pexels Paired Data [{mode_str}]")
    print(f"  Source: {PEXELS_PAIRED_DATA}")
    print(f"  Workers: {WORKERS}")
    print(f"  Output: {HERE}")
    print("=" * 70)

    train_samples, test_samples = get_sample_directories()

    if train_samples:
        generate_pairs(train_samples, VIDEOS_DIR, REF_DIR, "train")
    if test_samples:
        generate_pairs(test_samples, TEST_VIDEOS_DIR, TEST_REF_DIR, "test")

    generate_jsons(train_samples, test_samples)

    # Generate debug videos before latent precomputation (faster feedback)
    if train_samples and not MAX_ITEMS:  # Only for full dataset, not test mode
        generate_debug_videos(train_samples)

    precompute_latents()

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed / 60:.1f} minutes")


MAX_ITEMS = None

if __name__ == "__main__":
    main()

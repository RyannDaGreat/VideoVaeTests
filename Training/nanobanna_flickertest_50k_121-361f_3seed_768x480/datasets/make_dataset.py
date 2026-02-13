#!/usr/bin/env python3
"""Generate frame-flicker IC LoRA dataset from Pexels paired data.

NEW IN THIS VERSION (50k_121-361f_3seed_768x480):
- Variable frame lengths: 121, 241, or 361 frames per video (based on source video length)
- 3 seeds per sample: 3 different temporal crops per video (seeds 0, 1, 2)
- Resolution: 768x480 (up from 512x320)
- Full 50K dataset (all available samples)
- Resolution buckets for latent precomputation: 768x480x121;768x480x241;768x480x361

Processes Yash's Pexels paired data samples, generates reference/target video pairs,
creates dataset JSONs, and precomputes latents using LTX trainer.

Run with no arguments: python datasets/make_dataset.py
Test mode (24 samples): python datasets/make_dataset.py --test
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

# =============================================================================
# CONFIGURATION - Change NUM_SAMPLES to control dataset size
# =============================================================================
NUM_SAMPLES = 50  # in thousands (50 = 50,000 samples = ALL samples)
# =============================================================================

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
DEBUG_DIR = HERE / "debug_videos"  # Debug videos inside datasets/

WORKERS = 32
NUM_TEST_SAMPLES = 1000
NUM_DEBUG_VIDEOS = 100  # Generate 100 debug videos for inspection
NUM_SEEDS = 3  # 3 temporal crops per sample (seeds 0, 1, 2)

# Frame length buckets (must satisfy F % 8 == 1)
FRAME_BUCKETS = [361, 241, 121]  # Ordered largest-first for optimal bucket selection
SPEEDUP_FACTOR = 2  # Pexels 50fps -> LTX 25fps


def get_optimal_frame_count(raw_num_frames):
    """Determine the best frame bucket for a video based on its source frame count.

    With 2x speedup, need 2*F source frames for F output frames:
      361 frames -> need 722 source frames
      241 frames -> need 482 source frames
      121 frames -> need 242 source frames

    Returns the largest bucket the video can support, or None if too short.
    """
    for bucket in FRAME_BUCKETS:
        required_raw = bucket * SPEEDUP_FACTOR
        if raw_num_frames >= required_raw:
            return bucket
    return None  # Video too short for any bucket


def scan_samples_for_frame_counts(sample_dirs):
    """First pass: scan all sample directories to determine optimal frame counts.

    Returns:
        dict: sample_id -> optimal_frame_count (121, 241, or 361)
        Samples that are too short or missing files are excluded.
    """
    import rp

    print("\n--- First pass: Scanning videos for optimal frame counts ---")
    frame_map = {}
    skipped_short = 0
    skipped_missing = 0
    bucket_counts = {b: 0 for b in FRAME_BUCKETS}

    for sample_dir in tqdm(sample_dirs, desc="Scanning frame counts"):
        sample_id = sample_dir.name

        # Check required files exist
        raw_video_path = sample_dir / "raw_video.mp4"
        if not raw_video_path.exists():
            skipped_missing += 1
            continue
        if not (sample_dir / "after.png").exists():
            skipped_missing += 1
            continue
        if not (sample_dir / "metadata.json").exists():
            skipped_missing += 1
            continue

        try:
            raw_num_frames = rp.get_video_file_num_frames(str(raw_video_path))
            optimal = get_optimal_frame_count(raw_num_frames)
            if optimal is None:
                skipped_short += 1
                continue
            frame_map[sample_id] = optimal
            bucket_counts[optimal] += 1
        except Exception as e:
            print(f"  WARNING: Could not read frame count for {sample_id}: {e}")
            skipped_missing += 1

    print(f"\n  Frame count scan results:")
    print(f"    Total scanned: {len(sample_dirs)}")
    print(f"    Usable samples: {len(frame_map)}")
    print(f"    Skipped (too short): {skipped_short}")
    print(f"    Skipped (missing files): {skipped_missing}")
    print(f"    Bucket distribution:")
    for bucket in FRAME_BUCKETS:
        print(f"      {bucket} frames: {bucket_counts[bucket]} samples")

    return frame_map


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


def generate_pairs(samples, vid_dir, ref_dir, label, frame_map):
    """Run transform.py on all samples with 3 seeds each, WORKERS at a time.

    For each sample, generates 3 variants:
      {sample_id}_s0.mp4, {sample_id}_s1.mp4, {sample_id}_s2.mp4
    Each variant uses a different seed for the temporal crop.
    """
    # Build task list, skip existing
    tasks = []
    skipped = 0
    skipped_no_bucket = 0
    for sample_dir in samples:
        sample_id = sample_dir.name

        # Get optimal frame count from pre-scanned map
        num_frames = frame_map.get(sample_id)
        if num_frames is None:
            skipped_no_bucket += 1
            continue

        for seed in range(NUM_SEEDS):
            tgt = vid_dir / f"{sample_id}_s{seed}.mp4"
            ref = ref_dir / f"{sample_id}_s{seed}.mp4"

            if tgt.exists() and ref.exists():
                skipped += 1
                continue

            tasks.append((sample_id, sample_dir, tgt, ref, num_frames, seed))

    print(f"\n--- {label}: {len(tasks)} to process, {skipped} already exist, {skipped_no_bucket} no bucket ---")
    if not tasks:
        return

    # Run WORKERS subprocesses at a time
    active = []
    for sample_id, sample_dir, tgt, ref, num_frames, seed in tqdm(tasks, desc=label):
        cmd = [
            sys.executable, str(TRANSFORM), str(sample_dir),
            "--ref_out_path", str(ref),
            "--tgt_out_path", str(tgt),
            "--num_frames", str(num_frames),
            "--seed", str(seed),
        ]
        active.append((sample_id, seed, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))

        if len(active) >= WORKERS:
            # Wait for the first one to finish
            sample_id, seed, proc = active.pop(0)
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode()
                print(f"  FAILED: {sample_id}_s{seed} - {stderr[:200]}")

    # Drain remaining
    for sample_id, seed, proc in active:
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            print(f"  FAILED: {sample_id}_s{seed} - {stderr[:200]}")


def generate_jsons(train_samples, test_samples, frame_map):
    """Generate dataset.json and test_set.json for LTX trainer.

    Each sample produces 3 entries (one per seed), with naming: {sample_id}_s{seed}.mp4
    """
    print("\n--- Generating dataset JSON files ---")

    # Train set
    train_entries = []
    for sample_dir in train_samples:
        sample_id = sample_dir.name
        num_frames = frame_map.get(sample_id)
        if num_frames is None:
            continue

        # Load caption from metadata.json - video_caption[0] with None fallback
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

        for seed in range(NUM_SEEDS):
            tgt = VIDEOS_DIR / f"{sample_id}_s{seed}.mp4"
            ref = REF_DIR / f"{sample_id}_s{seed}.mp4"
            if tgt.exists() and ref.exists():
                train_entries.append({
                    "caption": caption,
                    "media_path": f"videos/{sample_id}_s{seed}.mp4",
                    "reference_path": f"reference_videos/{sample_id}_s{seed}.mp4",
                })

    with open(HERE / "dataset.json", "w") as f:
        json.dump(train_entries, f, indent=2)
    print(f"Train: {len(train_entries)} entries")

    # Test set
    test_entries = []
    for sample_dir in test_samples:
        sample_id = sample_dir.name
        num_frames = frame_map.get(sample_id)
        if num_frames is None:
            continue

        # Load caption from metadata.json - video_caption[0] with None fallback
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

        for seed in range(NUM_SEEDS):
            tgt = TEST_VIDEOS_DIR / f"{sample_id}_s{seed}.mp4"
            ref = TEST_REF_DIR / f"{sample_id}_s{seed}.mp4"
            if tgt.exists() and ref.exists():
                test_entries.append({
                    "video_id": f"{sample_id}_s{seed}",
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
    print(f"  Resolution buckets: 768x480x121;768x480x241;768x480x361")
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
            "--resolution-buckets", "768x480x121;768x480x241;768x480x361",
            "--model-path", "/models/LTX2/ltx-2-19b-dev.safetensors",
            "--text-encoder-path", "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized",
            "--reference-column", "reference_path",
            "--output-dir", str(HERE / ".precomputed"),
            "--vae-tiling",  # CRITICAL: Prevents OOM on 361-frame videos at 768x480
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
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


def generate_debug_videos(train_samples, frame_map):
    """Generate side-by-side debug videos for inspection."""
    import random
    import sys

    print("\n--- Generating debug videos for inspection ---")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Select random samples that have valid frame counts
    valid_samples = [s for s in train_samples if s.name in frame_map]
    debug_samples = random.sample(valid_samples, min(NUM_DEBUG_VIDEOS, len(valid_samples)))

    sys.path.insert(0, str(REPO_ROOT))
    import rp

    for sample_dir in tqdm(debug_samples, desc="Debug videos"):
        sample_id = sample_dir.name
        num_frames = frame_map[sample_id]

        # Use seed 0 variant for debug videos
        ref_path = REF_DIR / f"{sample_id}_s0.mp4"
        tgt_path = VIDEOS_DIR / f"{sample_id}_s0.mp4"

        # Check if both exist
        if not ref_path.exists() or not tgt_path.exists():
            continue

        try:
            # Load videos
            ref_video = rp.load_video(str(ref_path), use_cache=False)
            tgt_video = rp.load_video(str(tgt_path), use_cache=False)

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
                [f"Frame {i+1}/{num_frames}" for i in range(num_frames)],
                size=20,
            )

            # Save
            output_path = DEBUG_DIR / f"{sample_id}_s0_{num_frames}f_debug.mp4"
            rp.save_video_mp4(debug_video, str(output_path), framerate=25)

        except Exception as e:
            print(f"  FAILED debug video for {sample_id}: {e}")

    print(f"Generated {len(list(DEBUG_DIR.glob('*_debug.mp4')))} debug videos in {DEBUG_DIR}")


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
    print(f"  Variable frame lengths: {FRAME_BUCKETS}")
    print(f"  Seeds per sample: {NUM_SEEDS}")
    print(f"  Resolution: 768x480")
    print(f"  Source: {PEXELS_PAIRED_DATA}")
    print(f"  Workers: {WORKERS}")
    print(f"  Output: {HERE}")
    print("=" * 70)

    train_samples, test_samples = get_sample_directories()

    # First pass: scan all videos for optimal frame counts
    all_samples_to_scan = train_samples + test_samples
    frame_map = scan_samples_for_frame_counts(all_samples_to_scan)

    # Second pass: generate video pairs with 3 seeds each
    if train_samples:
        generate_pairs(train_samples, VIDEOS_DIR, REF_DIR, "train", frame_map)
    if test_samples:
        generate_pairs(test_samples, TEST_VIDEOS_DIR, TEST_REF_DIR, "test", frame_map)

    generate_jsons(train_samples, test_samples, frame_map)

    # Generate debug videos before latent precomputation (faster feedback)
    if train_samples and not MAX_ITEMS:  # Only for full dataset, not test mode
        generate_debug_videos(train_samples, frame_map)

        # Generate HTML debug viewer page
        from generate_debug_page import generate_debug_page
        generate_debug_page()

    precompute_latents()

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed / 60:.1f} minutes")


MAX_ITEMS = None

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run manual tests: generate references, precompute latents, launch inference.

Usage:
    python manual_tests/run_tests.py                         # Full pipeline
    python manual_tests/run_tests.py generate_references     # Only generate reference videos
    python manual_tests/run_tests.py precompute_latents      # Only precompute VAE latents
    python manual_tests/run_tests.py run --checkpoint PATH   # Use specific checkpoint
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import fire

sys.path.insert(0, "/root/CleanCode")
import rp

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO_ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=str(PROJECT),
).decode().strip())
LTX_TRAINER = REPO_ROOT / "LTX2" / "src" / "packages" / "ltx-trainer"
OUTPUTS_DIR = PROJECT / "outputs"
TESTS_JSON = HERE / "tests.json"
REF_DIR = HERE / "generated_references"
LATENT_DIR = HERE / "precomputed_latents"
NUM_GPUS = 8

MODEL_PATH = "/models/LTX2/ltx-2-19b-dev.safetensors"
TEXT_ENCODER_PATH = "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized"


def _load_tests():
    """Load test list from tests.json. Supports both formats:
    - New: {"name": "suite_name", "tests": [...]}
    - Old: [...] (bare list)
    """
    with open(TESTS_JSON) as f:
        config = json.load(f)
    if isinstance(config, list):
        return config
    return config.get("tests", [])


def _find_latest_checkpoint():
    ckpts = sorted((OUTPUTS_DIR / "checkpoints").glob("lora_weights_step_*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {OUTPUTS_DIR / 'checkpoints'}")
    return ckpts[-1]


def generate_references():
    """Generate flickery reference videos (input video + pulse mask) for all tests."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    tests = _load_tests()
    print(f"Generating references for {len(tests)} tests...")

    for t in tests:
        name = t["name"]
        ref_path = REF_DIR / f"{name}_ref.mp4"
        if ref_path.exists():
            print(f"  SKIP {name}: already exists")
            continue

        num_frames = t.get("num_frames", 121)
        keyframes = t.get("keyframes", [])
        indicator_size = 0.2

        # Load input video, take first num_frames frames (loop if video is shorter)
        video = rp.load_video(str(HERE / t["input_video"]), use_cache=False)
        if len(video) < num_frames:
            reps = (num_frames // len(video)) + 1
            video = np.tile(video, (reps, 1, 1, 1))[:num_frames]
            print(f"    Looped {len(video)//reps} frames -> {num_frames} frames")
        else:
            print(f"    Taking first {num_frames} of {len(video)} frames...")
            video = video[:num_frames]
        video = rp.resize_images(video, size=(480, 768), show_progress=True)
        height, vid_width = 480, 768

        # Nearest-neighbor keyframe replacement: hold each keyframe still until the next one
        # This is the flickery format the model was trained on
        nn_indices = rp.quantize_to_nearest_values(range(num_frames), keyframes)
        nn_frames = video[nn_indices]
        print(f"    Compositing reference ({num_frames} frames, {len(keyframes)} keyframes, nearest-neighbor)...")

        mask_width = round(indicator_size * vid_width)
        out_height = round((1 + indicator_size) * height)
        out_width = round((1 + indicator_size) * vid_width)

        pulse_mask = np.zeros((num_frames, out_height, mask_width, 3), dtype=np.uint8)
        for ki in keyframes:
            if 0 <= ki < num_frames:
                pulse_mask[ki] = 255

        ref_out = np.zeros((num_frames, out_height, out_width, 3), dtype=np.uint8)
        ref_out[:, -height:, -vid_width:] = nn_frames  # Nearest-neighbor keyframe frames (flickery)
        ref_out[:, :out_height, :mask_width] = pulse_mask  # Pulse indicator in top-left

        # Resize composite back to target resolution
        ref_out = rp.resize_images(ref_out, size=(height, vid_width), show_progress=True)

        # Save lossless to avoid H.264 artifacts on mostly-static content
        print(f"    Saving {ref_path.name} ({len(ref_out)} frames, crf=0)...")
        rp.save_video_mp4(ref_out, str(ref_path), framerate=25, crf=0)
        print(f"  Generated: {ref_path}")

        # Also generate composited first-frame image (same canvas format as reference)
        # so the I2V conditioning matches the training distribution
        first_frame_path = REF_DIR / f"{name}_first_frame.png"
        keyframe_img = rp.load_image(str(HERE / t["first_frame"]), use_cache=False)
        keyframe_img = rp.as_byte_image(rp.as_rgb_image(keyframe_img, copy=False), copy=False)
        keyframe_img = rp.cv_resize_image(keyframe_img, (height, vid_width))
        # Same 1.2x canvas: image in bottom-right, white pulse bar in top-left (frame 0 is a keyframe)
        first_out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        first_out[-height:, -vid_width:] = keyframe_img
        if 0 in keyframes:
            first_out[:out_height, :mask_width] = 255  # White pulse only if frame 0 is a keyframe
        first_out = rp.cv_resize_image(first_out, (height, vid_width))
        rp.save_image(first_out, str(first_frame_path))
        print(f"  Generated: {first_frame_path}")

    print("Done generating references.")


def precompute_latents():
    """Pre-compute VAE latents for reference videos using tiled encoding (avoids OOM).

    Runs inside the LTX trainer uv environment via subprocess since process_videos.py
    has dependencies (pillow_heif etc) only available there.
    """
    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    tests = _load_tests()

    to_compute = []
    for t in tests:
        name = t["name"]
        ref_path = REF_DIR / f"{name}_ref.mp4"
        latent_path = LATENT_DIR / f"{name}_ref.pt"
        if latent_path.exists():
            print(f"  SKIP {name}: latent already exists")
            continue
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference video missing: {ref_path}. Run generate_references first.")
        to_compute.append((name, str(ref_path), str(latent_path)))

    if not to_compute:
        print("All latents already computed.")
        return

    # Build inline script that encodes all pending references
    entries = ", ".join(f'("{n}", "{r}", "{l}")' for n, r, l in to_compute)
    script = f'''
import sys, torch
sys.path.insert(0, "scripts")
from einops import rearrange
from process_videos import encode_video
from ltx_trainer.model_loader import load_video_vae_encoder
from ltx_trainer.video_utils import read_video

MODEL = "{MODEL_PATH}"
vae = load_video_vae_encoder(MODEL, device="cuda", dtype=torch.bfloat16)
print("VAE encoder loaded.", flush=True)

for name, ref_path, latent_path in [{entries}]:
    video, fps = read_video(ref_path)
    print(f"  Encoding {{name}} ({{video.shape[0]}} frames)...", flush=True)

    # read_video returns [T, C, H, W] float32 in [0, 1] - convert to [-1, 1] BCTHW
    ref = rearrange(video, "T C H W -> 1 C T H W") * 2.0 - 1.0

    # Trim to valid frame count: (F-1) % 8 == 0
    valid_t = (ref.shape[2] - 1) // 8 * 8 + 1
    ref = ref[:, :, :valid_t]

    with torch.inference_mode():
        result = encode_video(vae, ref, dtype=torch.bfloat16, use_tiling=True)

    torch.save({{
        "latents": result["latents"][0].cpu().contiguous(),
        "num_frames": result["num_frames"],
        "height": result["height"],
        "width": result["width"],
        "fps": fps,
    }}, latent_path)
    print(f"  Saved: {{latent_path}} (shape={{result['latents'][0].shape}})", flush=True)

print("Latent precomputation done.")
'''
    print("Precomputing reference latents (tiled VAE encoding)...", flush=True)
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=str(LTX_TRAINER),
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    if result.returncode != 0:
        raise RuntimeError("Latent precomputation failed")


def run(checkpoint: str = None, skip_existing: bool = False):
    """Generate references, precompute latents, then run inference on all tests."""
    generate_references()
    precompute_latents()

    if checkpoint is None:
        checkpoint = str(_find_latest_checkpoint())
    step_name = Path(checkpoint).stem.replace("lora_weights_", "")

    # Load config - top-level "name" field determines the output subfolder
    with open(TESTS_JSON) as f:
        config = json.load(f)
    suite_name = config.get("name", "unnamed")
    tests = config.get("tests", config if isinstance(config, list) else [])

    # Create unique output subfolder (avoids overwriting previous runs)
    suite_dir = Path(rp.get_unique_copy_path(str(HERE / "test_outputs" / suite_name)))
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config as a record of this run
    with open(suite_dir / "tests.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  {suite_name} - {step_name} ({len(tests)} tests)")
    print(f"  LoRA: {checkpoint}")
    print(f"  Output: {suite_dir}")
    print(f"{'=' * 70}\n")

    active = []
    for i, t in enumerate(tests):
        gpu_id = i % NUM_GPUS
        name = t["name"]
        latent_path = LATENT_DIR / f"{name}_ref.pt"
        output_path = suite_dir / f"{name}_{step_name}.mp4"

        if skip_existing and output_path.exists():
            print(f"  SKIP {name}: already exists")
            continue

        cmd = [
            "uv", "run", "python", "scripts/inference.py",
            "--checkpoint", MODEL_PATH,
            "--text-encoder-path", TEXT_ENCODER_PATH,
            "--lora-path", checkpoint,
            # IC-LoRA reference: pre-computed latent (avoids OOM on long videos)
            "--reference-latent", str(latent_path),
            # I2V first-frame conditioning: composited in same canvas format as reference
            "--condition-image", str(REF_DIR / f"{name}_first_frame.png"),
            "--prompt", t["caption"],
            "--height", "480", "--width", "768",
            "--num-frames", str(t.get("num_frames", 121)),
            "--num-inference-steps", str(t.get("num_diffusion_steps", 30)),
            "--seed", str(t.get("seed", 42)),
            "--skip-audio",
            "--device", f"cuda:{gpu_id}",
            "--output", str(output_path),
        ]

        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        print(f"  [GPU {gpu_id}] {name}")
        active.append((name, subprocess.Popen(cmd, cwd=str(LTX_TRAINER), env=env)))

        if len(active) >= NUM_GPUS:
            for n, p in active:
                p.wait()
                print(f"  {'Done' if p.returncode == 0 else 'FAILED'}: {n}")
            active = []

    for n, p in active:
        p.wait()
        print(f"  {'Done' if p.returncode == 0 else 'FAILED'}: {n}")

    print(f"\nOutputs in: {suite_dir}")


if __name__ == "__main__":
    fire.Fire({
        "run": run,
        "generate_references": generate_references,
        "precompute_latents": precompute_latents,
    })

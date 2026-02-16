#!/usr/bin/env python3
"""Run manual tests: generate references + launch parallel inference.

Reference Frame Layout
======================
Each reference frame is composed of three regions:

    ┌────────────┬──────────────────────┐
    │            │      PADDING         │
    │ PULSE_MASK │  (black, fills gap)  │
    │            ├──────────────────────┤
    │  (fixed    │      CONTENT         │
    │   width)   │  (video frame,       │
    │            │   aspect preserved)  │
    └────────────┴──────────────────────┘

- PULSE_MASK: fixed-width left strip, white on keyframe frames, black otherwise.
- CONTENT:   actual video frames (NN-filled from keyframes). Always maintains
             the original source aspect ratio — never stretched.
- PADDING:   black space above content to fill remaining height.

The condition image (--condition-image) is frame 0 extracted from the reference
video, so it matches the reference exactly.

Usage:
    python manual_tests/run_tests.py run
    python manual_tests/run_tests.py generate_references
    python manual_tests/run_tests.py run --checkpoint PATH
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import _jsonnet
import cv2
import numpy as np
import fire

sys.path.insert(0, "/root/CleanCode")
import rp

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO_ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=str(PROJECT),
).decode().strip())
LTX_TRAINER = REPO_ROOT / "LTX2" / "src" / "packages" / "ltx-trainer"
DOWNLOAD_MODELS = REPO_ROOT / "LTX2" / "models" / "download_models.py"
CHECKPOINTS_DIR = PROJECT / "outputs" / "checkpoints"
TESTS_JSONNET = HERE / "tests.jsonnet"
OUTPUTS_DIR = HERE / "test_outputs"
NUM_GPUS = 8

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_WIDTH = 768
BASE_HEIGHT = 480
PULSE_MASK_PX = 128  # fixed pulse mask width in final output (pixels)

# ── Model paths (localized) ─────────────────────────────────────────────────
MODEL_CHECKPOINT = "/models/LTX2/ltx-2-19b-dev.safetensors"
TEXT_ENCODER = "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized"
SPATIAL_UPSAMPLER = "/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA = "/models/LTX2/ltx-2-19b-distilled-lora-384.safetensors"


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure Functions
# ═══════════════════════════════════════════════════════════════════════════════

def nn_fill_frames(video, keyframes, num_frames):
    """
    Nearest-neighbor fill: each frame gets the nearest keyframe's video content.

    Pure function — no side effects.

    >>> frames = np.arange(5).reshape(5, 1, 1, 1)
    >>> kf = [0, 2, 4]
    >>> result = nn_fill_frames(frames, kf, 5)
    >>> list(result.reshape(-1))
    [0, 2, 2, 4, 4]
    """
    kf_sorted = sorted(ki for ki in keyframes if 0 <= ki < num_frames)
    filled = np.empty_like(video[:num_frames])
    for i in range(num_frames):
        idx = np.searchsorted(kf_sorted, i)
        left = kf_sorted[max(0, idx - 1)]
        right = kf_sorted[min(idx, len(kf_sorted) - 1)]
        nearest = left if (i - left) <= (right - i) else right
        filled[i] = video[nearest]
    return filled


def compute_content_size(target_width, target_height, mask_px, source_aspect):
    """
    Compute content dimensions that fit within the available space while
    preserving the source aspect ratio.

    Available space = (target_width - mask_px) wide, target_height tall.
    Content fills as much of that as possible without stretching.

    Pure function — no side effects.

    >>> compute_content_size(768, 480, 128, 16/9)
    (640, 360)
    >>> compute_content_size(1152, 736, 128, 16/9)
    (1024, 576)
    """
    avail_w = target_width - mask_px
    avail_h = target_height

    # Fit content within available space preserving aspect ratio
    if avail_w / avail_h > source_aspect:
        # Height-limited: content fills full height
        content_h = avail_h
        content_w = round(avail_h * source_aspect)
    else:
        # Width-limited: content fills full width
        content_w = avail_w
        content_h = round(avail_w / source_aspect)

    return content_w, content_h


def build_reference_frame(content, mask_value, target_width, target_height, mask_px):
    """
    Composite a single reference frame from content + pulse mask.

    Pure function — no side effects.

    Args:
        content:  HWC uint8 array — the video content (aspect-preserved).
        mask_value: 0 or 255 — pulse mask intensity for this frame.
        target_width: total output width.
        target_height: total output height.
        mask_px: pulse mask width in pixels.

    Returns:
        HWC uint8 array of shape (target_height, target_width, 3).
    """
    frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    ch, cw = content.shape[:2]

    # Content goes bottom-right of the non-mask area
    x_offset = mask_px + (target_width - mask_px - cw)  # right-align
    y_offset = target_height - ch  # bottom-align (padding is above)
    frame[y_offset:y_offset + ch, x_offset:x_offset + cw] = content

    # Pulse mask on the left
    frame[:, :mask_px] = mask_value

    return frame


def make_test_name(num_frames, num_keyframes, width, height, seed, num_steps, index):
    """
    Generate a procedural test name encoding all parameters.

    Pure function — no side effects.

    >>> make_test_name(121, 64, 768, 480, 42, 30, 0)
    '121f_64kf_768x480_30st_s42_i0'
    """
    return f"{num_frames}f_{num_keyframes}kf_{width}x{height}_{num_steps}st_s{seed}_i{index}"


def resolve_keyframes(keyframes, num_frames, seed):
    """
    Resolve keyframe specification to a concrete sorted list of frame indices.

    Supports two formats:
    - List of ints: returned as-is (already explicit keyframes)
    - String "random N": generate N randomly distributed keyframes seeded by `seed`,
      always including frame 0.

    Pure function — no side effects (uses isolated RNG).

    >>> resolve_keyframes([0, 5, 10], 121, 42)
    [0, 5, 10]
    >>> kf = resolve_keyframes("random 8", 121, 42)
    >>> len(kf)
    8
    >>> kf[0]
    0
    >>> kf == sorted(set(kf))
    True
    """
    if isinstance(keyframes, list):
        return keyframes
    if isinstance(keyframes, str) and keyframes.startswith("random "):
        n = int(keyframes.split()[1])
        rng = random.Random(seed)
        n = min(n, num_frames)
        # Always include frame 0; pick n-1 more from 1..num_frames-1
        candidates = list(range(1, num_frames))
        chosen = rng.sample(candidates, min(n - 1, len(candidates)))
        return sorted([0] + chosen)
    raise ValueError(f"Unknown keyframes format: {keyframes!r}")


def make_batch_dirname(batch_title, step_name):
    """
    Generate a batch output directory name.

    Pure function — no side effects.

    >>> make_batch_dirname("kf_sweep_1152x736", "step_06300")
    'kf_sweep_1152x736_step_06300'
    """
    return f"{batch_title}_{step_name}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Video I/O Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_video_lossless(path, frames, fps=25):
    """Save an NHWC uint8 video with lossless H.264 (CRF 0, yuv444p)."""
    import av
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0"}
        for frame_arr in frames:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _load_tests():
    """Load tests from .jsonnet, resolve keyframe shorthands, return list of dicts."""
    raw_json = _jsonnet.evaluate_file(str(TESTS_JSONNET))
    tests = json.loads(raw_json)
    for t in tests:
        t["keyframes"] = resolve_keyframes(
            t["keyframes"],
            t.get("num_frames", 121),
            t.get("seed", 42),
        )
    return tests


def _find_latest_checkpoint():
    ckpts = sorted(CHECKPOINTS_DIR.glob("lora_weights_step_*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CHECKPOINTS_DIR}")
    return ckpts[-1]


def generate_references(batch_dir=None):
    """
    Generate flickery reference videos for all tests in tests.json.

    Each reference video is NN-filled from keyframe positions with a fixed-width
    pulse mask on the left. Content always preserves source aspect ratio.
    """
    tests = _load_tests()
    ref_dir = Path(batch_dir) / "references" if batch_dir else HERE / "generated_references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating references for {len(tests)} tests...")

    for t in tests:
        name = t["name"]
        ref_path = ref_dir / f"{name}_ref.mp4"
        cond_path = ref_dir / f"{name}_condition.png"
        if ref_path.exists():
            print(f"  SKIP {name}: already exists")
            continue

        num_frames = t.get("num_frames", 121)
        keyframes = t.get("keyframes", [])
        target_w = t.get("width", BASE_WIDTH)
        target_h = t.get("height", BASE_HEIGHT)

        # Load source video and first-frame condition image
        video = rp.load_video(str(HERE / t["input_video"]), use_cache=False)
        keyframe_img = rp.load_image(str(HERE / t["first_frame"]), use_cache=False)
        keyframe_img = rp.as_byte_image(rp.as_rgb_image(keyframe_img, copy=False), copy=False)

        # Take first N frames
        print(f"    Taking first {num_frames} of {len(video)} frames...")
        video = np.array(video[:num_frames])

        # Compute content size preserving source aspect ratio
        source_h, source_w = video[0].shape[:2]
        source_aspect = source_w / source_h
        content_w, content_h = compute_content_size(target_w, target_h, PULSE_MASK_PX, source_aspect)
        print(f"    Target: {target_w}x{target_h}, content: {content_w}x{content_h}, mask: {PULSE_MASK_PX}px")

        # Resize video and keyframe_img to content size (aspect-preserving)
        [keyframe_img], video = rp.resize_videos_to_hold([keyframe_img], video, width=content_w)
        keyframe_img = rp.crop_images([keyframe_img], height=content_h, origin="center")[0]
        video = rp.crop_images(video, height=content_h, origin="center")

        # NN-fill from keyframes
        print(f"    NN-filling {num_frames} frames from {len(keyframes)} keyframes...")
        nn_frames = nn_fill_frames(video, keyframes, num_frames)
        if t.get("ref_first_frame", True):
            nn_frames[0] = keyframe_img

        # Build composite frames
        print(f"    Compositing reference frames...")
        ref_out = np.stack([
            build_reference_frame(
                content=nn_frames[i],
                mask_value=255 if i in set(keyframes) else 0,
                target_width=target_w,
                target_height=target_h,
                mask_px=PULSE_MASK_PX,
            )
            for i in range(num_frames)
        ])
        print(f"    ref_out: {ref_out.shape}, saving {ref_path.name} (lossless)...")

        save_video_lossless(ref_path, ref_out)
        print(f"  Generated: {ref_path}")

        # Save frame 0 as condition image
        cv2.imwrite(str(cond_path), cv2.cvtColor(ref_out[0], cv2.COLOR_RGB2BGR))
        print(f"  Condition: {cond_path}")

    print("Done generating references.")
    return ref_dir


def run(checkpoint: str = None, skip_existing: bool = False):
    """Generate references, then run inference on all tests. Saves to archival folder."""
    # Ensure models are localized
    print("Ensuring models are downloaded and localized...")
    subprocess.run([sys.executable, str(DOWNLOAD_MODELS)], check=True)

    if checkpoint is None:
        checkpoint = str(_find_latest_checkpoint())
    step_name = Path(checkpoint).stem.replace("lora_weights_", "")

    tests = _load_tests()

    # Determine batch title from JSON or derive from first test
    batch_title = tests[0].get("batch_title", "manual_tests") if tests else "manual_tests"

    # Create archival output directory
    batch_dirname = make_batch_dirname(batch_title, step_name)
    batch_dir = OUTPUTS_DIR / batch_dirname
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved tests as vanilla JSON into the batch folder for archival
    with open(batch_dir / "tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    # Generate references into the batch folder
    ref_dir = generate_references(batch_dir=batch_dir)

    print(f"\n{'=' * 70}")
    print(f"  Manual Tests - {step_name} ({len(tests)} tests)")
    print(f"  LoRA: {checkpoint}")
    print(f"  Output: {batch_dir}")
    print(f"{'=' * 70}\n")

    active = []
    for i, t in enumerate(tests):
        gpu_id = i % NUM_GPUS
        name = t["name"]
        ref_path = ref_dir / f"{name}_ref.mp4"
        cond_path = ref_dir / f"{name}_condition.png"
        output_path = batch_dir / f"{name}_{step_name}.mp4"

        if skip_existing and output_path.exists():
            print(f"  SKIP {name}: already exists")
            continue

        latent_path = batch_dir / f"{name}_{step_name}_latent.pt"
        cmd = [
            "uv", "run", "python", "scripts/inference.py",
            "--checkpoint", MODEL_CHECKPOINT,
            "--text-encoder-path", TEXT_ENCODER,
            "--lora-path", checkpoint,
            "--reference-video", str(ref_path),
            "--condition-image", str(cond_path),
            "--prompt", t["caption"],
            "--height", str(t.get("height", BASE_HEIGHT)),
            "--width", str(t.get("width", BASE_WIDTH)),
            "--num-frames", str(t.get("num_frames", 121)),
            "--num-inference-steps", str(t.get("num_diffusion_steps", 30)),
            "--seed", str(t.get("seed", 42)),
            "--skip-audio",
            "--include-reference-in-output",
            "--device", f"cuda:{gpu_id}",
            "--output", str(output_path),
            "--save-latent", str(latent_path),
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

    # Stage 2 upscaling (if any tests have it enabled)
    stage2_tests = [(i, t) for i, t in enumerate(tests) if t.get("stage_2", {}).get("enabled", False)]
    if stage2_tests:
        print(f"\n{'=' * 70}")
        print(f"  Stage 2 Upscaling - {len(stage2_tests)} tests")
        print(f"{'=' * 70}\n")

        active = []
        for i, t in stage2_tests:
            gpu_id = i % NUM_GPUS
            name = t["name"]
            latent_path = batch_dir / f"{name}_{step_name}_latent.pt"
            stage2_path = batch_dir / f"{name}_{step_name}_stage2.mp4"

            if not latent_path.exists():
                print(f"  SKIP {name}: stage 1 latent missing ({latent_path})")
                continue
            if skip_existing and stage2_path.exists():
                print(f"  SKIP {name}: stage 2 already exists")
                continue

            cmd = [
                "uv", "run", "python", str(HERE / "stage2_upscale.py"),
                "upscale",
                "--latent-path", str(latent_path),
                "--output", str(stage2_path),
                "--prompt", t["caption"],
                "--device", f"cuda:{gpu_id}",
                "--seed", str(t.get("seed", 42)),
                "--frame-rate", str(t.get("frame_rate", 25.0)),
                "--lora-path", checkpoint,
            ]

            env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            print(f"  [GPU {gpu_id}] Stage 2: {name}")
            active.append((name, subprocess.Popen(cmd, cwd=str(LTX_TRAINER), env=env)))

            if len(active) >= NUM_GPUS:
                for n, p in active:
                    p.wait()
                    print(f"  {'Done' if p.returncode == 0 else 'FAILED'}: {n} (stage 2)")
                active = []

        for n, p in active:
            p.wait()
            print(f"  {'Done' if p.returncode == 0 else 'FAILED'}: {n} (stage 2)")

    print(f"\nOutputs in: {batch_dir}")


if __name__ == "__main__":
    fire.Fire({"run": run, "generate_references": generate_references})

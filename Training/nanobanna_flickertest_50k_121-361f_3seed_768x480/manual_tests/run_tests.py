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
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _jsonnet
import cv2
import numpy as np
import fire
import torch

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

    >>> content = np.full((4, 6, 3), 128, dtype=np.uint8)
    >>> frame = build_reference_frame(content, 255, 10, 6, 2)
    >>> frame.shape
    (6, 10, 3)
    >>> frame[0, 0, 0]  # pulse mask = 255
    255
    >>> frame[2, 4, 0]  # right-aligned content = 128
    128
    >>> frame[0, 3, 0]  # top-left padding = 0
    0
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


def resolve_seed(seed):
    """
    Resolve seed specification to a concrete integer.

    Supports two formats:
    - int: returned as-is
    - String "random": generate a random seed (from system entropy)

    Pure function — no side effects.

    >>> resolve_seed(42)
    42
    >>> isinstance(resolve_seed("random"), int)
    True
    """
    if isinstance(seed, int):
        return seed
    if isinstance(seed, str) and seed.strip().lower() == "random":
        return random.SystemRandom().randint(0, 2**31 - 1)
    raise ValueError(f"Unknown seed format: {seed!r}")


def resolve_keyframes(keyframes, num_frames, seed):
    """
    Resolve keyframe specification to a concrete sorted list of frame indices.

    Supports three formats:
    - List of ints: returned as-is (already explicit keyframes)
    - String "random N": N randomly distributed keyframes seeded by `seed`,
      always including frame 0.
    - String "uniform N": N uniformly spaced keyframes from frame 0 to frame num_frames-1.

    Pure function — no side effects (uses isolated RNG for "random").

    >>> resolve_keyframes([0, 5, 10], 121, 42)
    [0, 5, 10]
    >>> kf = resolve_keyframes("random 8", 121, 42)
    >>> len(kf)
    8
    >>> kf[0]
    0
    >>> kf == sorted(set(kf))
    True
    >>> resolve_keyframes("uniform 5", 121, 42)
    [0, 30, 60, 90, 120]
    >>> resolve_keyframes("uniform 3", 121, 42)
    [0, 60, 120]
    >>> resolve_keyframes("uniform 1", 121, 42)
    [0]
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
    if isinstance(keyframes, str) and keyframes.startswith("uniform "):
        n = int(keyframes.split()[1])
        if n <= 1:
            return [0]
        # Evenly spaced from 0 to num_frames-1, inclusive on both ends
        return [round(i * (num_frames - 1) / (n - 1)) for i in range(n)]
    raise ValueError(f"Unknown keyframes format: {keyframes!r}")


def make_batch_dirname(batch_title, step_name):
    """
    Generate a batch output directory name.

    Pure function — no side effects.

    >>> make_batch_dirname("kf_sweep_1152x736", "step_06300")
    'kf_sweep_1152x736_step_06300'
    """
    return f"{batch_title}_{step_name}"


def format_duration(seconds):
    """
    Format a duration in seconds to a human-readable string.

    Pure function — no side effects.

    >>> format_duration(0.5)
    '500ms'
    >>> format_duration(65.3)
    '1m 5s'
    >>> format_duration(3661)
    '1h 1m 1s'
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def collect_run_metadata(checkpoint_path, batch_dir):
    """
    Collect machine, environment, and git metadata for archival.

    Pure function — reads system state but has no side effects.
    """
    # Timestamps
    now = rp.get_current_date()
    timestamps = {
        "utc": rp.format_date(now, "utc"),
        "est": rp.format_date(now, "est"),
        "pst": rp.format_date(now, "pst"),
        "epoch": time.time(),
    }

    # Machine
    env = os.environ
    machine = {
        "hostname": env.get("BD_HOSTNAME", platform.node()),
        "ip": env.get("BD_IP", env.get("EC2_LOCAL_IPV4", "")),
        "instance_name": env.get("CLARA_NAME", ""),
        "cluster": env.get("NETFLIX_CLUSTER", ""),
        "availability_zone": env.get("EC2_AVAILABILITY_ZONE", ""),
        "docker_image": env.get("BD_IMAGE_NAME", ""),
        "docker_image_version": env.get("BD_IMAGE_VERSION", ""),
        "docker_buildtime": env.get("BD_IMAGE_BUILDTIME", ""),
    }

    # GPU info
    try:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        gpu_mem = [f"{torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GiB" for i in range(torch.cuda.device_count())]
    except Exception:
        gpu_names, gpu_mem = [], []
    gpus = {
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "models": gpu_names,
        "vram": gpu_mem,
    }

    # Environment
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "cuda_home": env.get("CUDA_HOME", ""),
        "conda_env": env.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": env.get("CONDA_PREFIX", ""),
        "uv_venv": str(Path(sys.executable).parent.parent),
    }

    # Resources
    resources = {
        "num_gpu": env.get("TITUS_NUM_GPU", ""),
        "num_cpu": env.get("TITUS_NUM_CPU", ""),
        "mem_mb": env.get("TITUS_NUM_MEM", ""),
    }

    # Git
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
    except Exception:
        git_commit = ""
    try:
        uncommitted = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
        ).decode().strip().splitlines()
    except Exception:
        uncommitted = []

    git = {
        "commit": git_commit,
        "uncommitted_files": uncommitted,
    }

    # Paths
    paths = {
        "checkpoint": str(checkpoint_path),
        "batch_dir": str(batch_dir),
        "repo_root": str(REPO_ROOT),
        "ltx_trainer": str(LTX_TRAINER),
    }

    return {
        "timestamps": timestamps,
        "machine": machine,
        "gpus": gpus,
        "environment": environment,
        "resources": resources,
        "git": git,
        "paths": paths,
        "timings": {},  # filled in during run
    }


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


def save_hconcat_video(paths, output_path, framerate=25, video_bitrate=100_000_000):
    """
    Load videos, normalize sizes/lengths, horizontally concatenate, and save.

    Trims all videos to the shortest length, resizes to the largest dimensions
    (nearest-neighbor), concatenates them side-by-side, and saves as h264 mp4.

    >>> # save_hconcat_video(["a.mp4", "b.mp4"], output_path="out.mp4")
    """
    videos = rp.load_videos(paths)
    videos = rp.trim_videos_to_min_length(videos)
    videos = rp.resize_videos_to_max_size(videos, interp="nearest")
    video = rp.horizontally_concatenated_videos(videos)
    return rp.save_video_mp4(video, output_path, framerate=framerate, video_bitrate=video_bitrate)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _find_varying_keys(tests):
    """
    Find which keys have different values across a batch of test configs.

    Returns a list of keys whose values are not identical across all tests.
    Skips 'name' and 'seed' (always expected to differ).

    Pure function — no side effects.

    >>> _find_varying_keys([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    ['b']
    >>> _find_varying_keys([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    []
    """
    if len(tests) <= 1:
        return []
    skip = {"name", "seed", "batch_title", "batch_name", "caption", "input_video", "first_frame", "stage_2"}
    all_keys = set()
    for t in tests:
        all_keys.update(t.keys())
    varying = []
    for k in sorted(all_keys - skip):
        values = [repr(t.get(k)) for t in tests]
        if len(set(values)) > 1:
            varying.append(k)
    return varying


# Short labels for varying keys in filenames
_KEY_LABELS = {
    "keyframes": "kf",
    "guidance_scale": "cfg",
    "num_diffusion_steps": "st",
    "num_frames": "f",
    "width": "w",
    "height": "h",
    "i2v_guidance_scale": "i2v",
    "cfg_drop_image": "cdi",
    "ref_first_frame": "rff",
}


def _truncate_middle(s, max_len=30):
    """
    Truncate a string from the middle using a Unicode ellipsis, preserving both ends.

    If the string fits within max_len, it's returned unchanged.

    Pure function — no side effects.

    >>> _truncate_middle("short", 30)
    'short'
    >>> _truncate_middle("abcdefghijklmnopqrstuvwxyz0123456789", 20)
    'abcdefghi…0123456789'
    >>> _truncate_middle("hello world", 5)
    'he…ld'
    >>> _truncate_middle("ab", 1)
    '…'
    """
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    # Split available space evenly: left gets half (rounded down), right gets the rest
    left = (max_len - 1) // 2
    right = max_len - 1 - left
    return s[:left] + "…" + s[-right:]


def _format_value_for_name(key, value):
    """
    Format a config value for inclusion in a filename.

    Pure function — no side effects.

    >>> _format_value_for_name("keyframes", [0, 5, 10, 20])
    '4'
    >>> _format_value_for_name("keyframes", "random 32")
    '32'
    >>> _format_value_for_name("guidance_scale", 4.0)
    '4'
    >>> _format_value_for_name("guidance_scale", 1.25)
    '1.25'
    >>> _format_value_for_name("guidance_scale", 1.50)
    '1.5'
    >>> _format_value_for_name("guidance_scale", 1.10)
    '1.1'
    >>> _format_value_for_name("guidance_scale", 0.125)
    '.12'
    >>> _format_value_for_name("cfg_drop_image", True)
    '1'
    >>> _format_value_for_name("cfg_drop_image", 0.5)
    '.5'
    """
    if key == "keyframes":
        if isinstance(value, list):
            return str(len(value))
        if isinstance(value, str) and value.startswith("random "):
            return value.split()[1]
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}".rstrip("0").lstrip("0")
    result = str(value)
    return _truncate_middle(result, 30)


def _generate_test_names(tests):
    """
    Generate descriptive names for tests based on which fields vary across the batch.

    Format: <batch_name>_<index>_<varying_key1><value1>_<varying_key2><value2>_...
    Only fields that differ across the batch are included.

    Pure function — no side effects (mutates test dicts' "name" field in place).

    >>> tests = [
    ...     {"batch_name": "fish", "guidance_scale": 2, "keyframes": [0, 5]},
    ...     {"batch_name": "fish", "guidance_scale": 4, "keyframes": [0, 5]},
    ... ]
    >>> _generate_test_names(tests)
    >>> [t["name"] for t in tests]
    ['fish_0_cfg2', 'fish_1_cfg4']
    """
    varying = _find_varying_keys(tests)
    for i, t in enumerate(tests):
        parts = [t.get("batch_name", "test"), str(i)]
        for k in varying:
            label = _KEY_LABELS.get(k, k)
            val = _format_value_for_name(k, t.get(k))
            parts.append(f"{label}{val}")
        t["name"] = "_".join(parts)


def _load_tests():
    """
    Load tests from .jsonnet and resolve all string shorthands to concrete values.

    Resolves (in order):
    - seed: int or "random" → concrete int
    - keyframes: list or "random N" → concrete sorted list (uses resolved seed)
    - name: auto-generated from batch_name + index + varying fields
    """
    raw_json = _jsonnet.evaluate_file(str(TESTS_JSONNET))
    tests = json.loads(raw_json)
    for t in tests:
        t["seed"] = resolve_seed(t.get("seed", 42))
        t["keyframes"] = resolve_keyframes(
            t["keyframes"],
            t.get("num_frames", 121),
            t["seed"],
        )
    _generate_test_names(tests)
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

        # Take first N frames. If source is shorter, pad by repeating the last frame.
        # Padded frames get pulse_mask=0 (off), enabling video extension beyond source length.
        source_frame_count = len(video)
        if source_frame_count < num_frames:
            print(f"    VIDEO EXTENSION: source has {source_frame_count} frames, padding to {num_frames} by repeating last frame")
            last_frame = video[-1]
            padding = [last_frame] * (num_frames - source_frame_count)
            video = list(video) + padding
        else:
            print(f"    Taking first {num_frames} of {source_frame_count} frames...")
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
        # Padded frames (beyond original source length) get pulse_mask=0 always,
        # even if they're in the keyframes list. This signals "no keyframe data here"
        # and enables video extension: the model generates new content for these frames.
        print(f"    Compositing reference frames...")
        keyframe_set = set(keyframes)
        ref_out = np.stack([
            build_reference_frame(
                content=nn_frames[i],
                mask_value=255 if (i in keyframe_set and i < source_frame_count) else 0,
                target_width=target_w,
                target_height=target_h,
                mask_px=PULSE_MASK_PX,
            )
            for i in range(num_frames)
        ])
        print(f"    ref_out: {ref_out.shape}, saving {ref_path.name} (lossless)...")

        save_video_lossless(ref_path, ref_out)
        print(f"  Generated: {ref_path}")

        # Save condition image (always from clean first frame, independent of ref_first_frame)
        cond_frame = build_reference_frame(
            content=keyframe_img,
            mask_value=255,
            target_width=target_w,
            target_height=target_h,
            mask_px=PULSE_MASK_PX,
        )
        cv2.imwrite(str(cond_path), cv2.cvtColor(cond_frame, cv2.COLOR_RGB2BGR))
        print(f"  Condition: {cond_path}")

    print("Done generating references.")
    return ref_dir


def run(checkpoint: str = None, skip_existing: bool = False):
    """Generate references, then run inference on all tests. Saves to archival folder."""
    run_start = time.time()

    # Ensure models are localized
    print("Ensuring models are downloaded and localized...")
    subprocess.run([sys.executable, str(DOWNLOAD_MODELS)], check=True)

    if checkpoint is None:
        checkpoint = str(_find_latest_checkpoint())
    step_name = Path(checkpoint).stem.replace("lora_weights_", "")

    tests = _load_tests()

    # Determine batch title from JSON or derive from first test
    batch_title = tests[0].get("batch_title", "manual_tests") if tests else "manual_tests"

    # Create archival output directory (unique path to avoid collisions)
    batch_dirname = make_batch_dirname(batch_title, step_name)
    batch_dir = Path(rp.get_unique_copy_path(str(OUTPUTS_DIR / batch_dirname)))
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved tests as vanilla JSON + source jsonnet for archival
    with open(batch_dir / "tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    shutil.copy2(str(TESTS_JSONNET), str(batch_dir / "tests.jsonnet"))

    # Copy relevant raw inputs for self-contained archival
    raw_dir = batch_dir / "raw_inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_files = set()
    for t in tests:
        raw_files.add(t.get("input_video", ""))
        raw_files.add(t.get("first_frame", ""))
    for f in raw_files:
        src = HERE / f
        if src.exists():
            shutil.copy2(str(src), str(raw_dir / src.name))

    # Generate references into the batch folder
    ref_dir = generate_references(batch_dir=batch_dir)

    print(f"\n{'=' * 70}")
    print(f"  Manual Tests - {step_name} ({len(tests)} tests)")
    print(f"  LoRA: {checkpoint}")
    print(f"  Output: {batch_dir}")
    print(f"{'=' * 70}\n")

    stage1_start = time.time()
    latent_dir = batch_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def _build_stage1_cmd(t, gpu_id):
        """Build the stage 1 inference command for a test."""
        name = t["name"]
        return [
            "uv", "run", "python", "scripts/inference.py",
            "--checkpoint", MODEL_CHECKPOINT,
            "--text-encoder-path", TEXT_ENCODER,
            "--lora-path", checkpoint,
            "--reference-video", str(ref_dir / f"{name}_ref.mp4"),
            "--condition-image", str(ref_dir / f"{name}_condition.png"),
            "--prompt", t["caption"],
            "--height", str(t.get("height", BASE_HEIGHT)),
            "--width", str(t.get("width", BASE_WIDTH)),
            "--num-frames", str(t.get("num_frames", 121)),
            "--num-inference-steps", str(t.get("num_diffusion_steps", 30)),
            "--guidance-scale", str(t.get("guidance_scale", 4.0)),
            "--cfg-drop-image", str(t.get("cfg_drop_image", 0.0)),
            "--seed", str(t.get("seed", 42)),
            "--skip-audio",
            "--include-reference-in-output",
            "--device", f"cuda:{gpu_id}",
            "--output", str(batch_dir / f"{name}_{step_name}.mp4"),
            "--save-latent", str(latent_dir / f"{name}_{step_name}.pt"),
        ]

    def _build_stage2_cmd(t, gpu_id):
        """Build the stage 2 upscaling command for a test. Returns None if not applicable."""
        if not t.get("stage_2", {}).get("enabled", False):
            return None
        name = t["name"]
        lp = latent_dir / f"{name}_{step_name}.pt"
        if not lp.exists():
            return None
        src_path = str(HERE / t["input_video"])
        cap = cv2.VideoCapture(src_path)
        sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return [
            "uv", "run", "python", str(HERE / "stage2_upscale.py"), "upscale",
            "--latent-path", str(lp),
            "--output", str(batch_dir / f"{name}_{step_name}_stage2.mp4"),
            "--prompt", t["caption"],
            "--first-frame", str(HERE / t["first_frame"]),
            "--pulse-mask-px", str(PULSE_MASK_PX),
            "--source-aspect", str(sw / sh),
            "--device", f"cuda:{gpu_id}",
            "--seed", str(t.get("seed", 42)),
            "--frame-rate", str(t.get("frame_rate", 25.0)),
            "--lora-path", checkpoint,
        ]

    # Build per-GPU job queues: each GPU gets its own list of (stage1, stage2) pairs
    # Jobs run independently per GPU — no cross-GPU waiting.
    gpu_queues = [[] for _ in range(NUM_GPUS)]
    for i, t in enumerate(tests):
        gpu_id = i % NUM_GPUS
        gpu_queues[gpu_id].append(t)

    def _gpu_worker(gpu_id, queue):
        """Run all jobs for one GPU sequentially: stage1 → stage2 → stage1 → stage2 → ..."""
        for t in queue:
            name = t["name"]
            output_path = batch_dir / f"{name}_{step_name}.mp4"
            stage2_path = batch_dir / f"{name}_{step_name}_stage2.mp4"

            # Stage 1
            if skip_existing and output_path.exists():
                print(f"  [GPU {gpu_id}] SKIP stage 1: {name}")
            else:
                print(f"  [GPU {gpu_id}] Stage 1: {name}")
                p = subprocess.Popen(_build_stage1_cmd(t, gpu_id), cwd=str(LTX_TRAINER), env=env)
                p.wait()
                print(f"  [GPU {gpu_id}] {'Done' if p.returncode == 0 else 'FAILED'}: {name} (stage 1)")

            # Stage 2 (immediately after this test's stage 1)
            if skip_existing and stage2_path.exists():
                print(f"  [GPU {gpu_id}] SKIP stage 2: {name}")
            else:
                s2_cmd = _build_stage2_cmd(t, gpu_id)
                if s2_cmd:
                    print(f"  [GPU {gpu_id}] Stage 2: {name}")
                    p = subprocess.Popen(s2_cmd, cwd=str(LTX_TRAINER), env=env)
                    p.wait()
                    print(f"  [GPU {gpu_id}] {'Done' if p.returncode == 0 else 'FAILED'}: {name} (stage 2)")

    # Launch all GPU workers in parallel (each is a thread running its queue)
    import threading
    threads = []
    for gpu_id in range(NUM_GPUS):
        if gpu_queues[gpu_id]:
            t = threading.Thread(target=_gpu_worker, args=(gpu_id, gpu_queues[gpu_id]))
            t.start()
            threads.append(t)
            print(f"  GPU {gpu_id}: {len(gpu_queues[gpu_id])} tests queued")

    for t in threads:
        t.join()

    stage1_duration = time.time() - stage1_start
    print(f"\n  Total (all GPUs): {format_duration(stage1_duration)} ({stage1_duration*1000:.0f}ms)")

    # Save stage 2 comparison videos (original raw input side-by-side with stage 2 output)
    any_comparisons = any(t.get("save_stage2_comparison_video", False) for t in tests)
    if any_comparisons:
        make_comparison_videos(str(batch_dir))

    # Save metadata
    run_duration = time.time() - run_start
    metadata = collect_run_metadata(checkpoint, batch_dir)
    metadata["timings"] = {
        "pipeline_seconds": stage1_duration,
        "pipeline_human": format_duration(stage1_duration),
        "pipeline_ms": stage1_duration * 1000,
    }
    metadata["timings"]["total_seconds"] = run_duration
    metadata["timings"]["total_human"] = format_duration(run_duration)
    metadata["timings"]["total_ms"] = run_duration * 1000

    with open(batch_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {batch_dir / 'metadata.json'}")

    print(f"Total run time: {format_duration(run_duration)}")
    print(f"Outputs in: {batch_dir}")


def make_comparison_videos(batch_dir: str):
    """
    Generate side-by-side comparison videos for all stage 2 outputs in a batch directory.

    Reads tests.json from the batch dir to find raw input paths, then creates
    [raw input | stage 2 output] comparison videos in comparison_videos/ subfolder.

    Usage:
        python run_tests.py make_comparison_videos manual_tests/test_outputs/kf_sweep_1152x736_100step_step_09300
    """
    batch_dir = Path(batch_dir)
    tests_json = batch_dir / "tests.json"
    if not tests_json.exists():
        raise FileNotFoundError(f"No tests.json in {batch_dir}")

    with open(tests_json) as f:
        tests = json.load(f)

    comp_dir = batch_dir / "comparison_videos"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Find the step name from existing mp4s
    mp4s = list(batch_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 files in {batch_dir}")

    stage2_files = sorted(batch_dir.glob("*stage2*.mp4"))
    if not stage2_files:
        raise FileNotFoundError(f"No stage2 mp4 files in {batch_dir}")

    print(f"Found {len(stage2_files)} stage 2 outputs in {batch_dir}")

    for s2 in stage2_files:
        # Find matching test by name prefix
        s2_name = s2.stem  # e.g. "boat_0_cdi1_kf40_step_10250_stage2"
        matching = [t for t in tests if s2_name.startswith(t["name"])]
        if not matching:
            print(f"  SKIP {s2.name}: no matching test in tests.json")
            continue
        t = matching[0]
        raw_input = str(HERE / t["input_video"])
        if not Path(raw_input).exists():
            # Try raw_inputs subfolder in batch dir
            raw_input = str(batch_dir / "raw_inputs" / Path(t["input_video"]).name)
        if not Path(raw_input).exists():
            print(f"  SKIP {s2.name}: raw input not found")
            continue

        comp_path = comp_dir / s2.name.replace("_stage2.mp4", "_comparison.mp4")
        print(f"  {s2.name} -> {comp_path.name}")
        save_hconcat_video([raw_input, str(s2)], str(comp_path))
        print(f"    Done")

    print(f"\nComparison videos saved to {comp_dir}")


if __name__ == "__main__":
    fire.Fire({
        "run": run,
        "generate_references": generate_references,
        "make_comparison_videos": make_comparison_videos,
    })

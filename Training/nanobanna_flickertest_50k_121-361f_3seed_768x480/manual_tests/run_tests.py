#!/usr/bin/env python3
"""Run manual tests: generate references + launch parallel inference.

Usage:
    python manual_tests/run_tests.py                         # Generate refs + run inference
    python manual_tests/run_tests.py generate_references     # Only generate reference videos
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
DOWNLOAD_MODELS = REPO_ROOT / "LTX2" / "models" / "download_models.py"
OUTPUTS_DIR = PROJECT / "outputs"
TESTS_JSON = HERE / "tests.json"
REF_DIR = HERE / "generated_references"
NUM_GPUS = 8


def _load_tests():
    with open(TESTS_JSON) as f:
        return json.load(f)


def _find_latest_checkpoint():
    ckpts = sorted((OUTPUTS_DIR / "checkpoints").glob("lora_weights_step_*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {OUTPUTS_DIR / 'checkpoints'}")
    return ckpts[-1]


def generate_references():
    """
    Generate flickery reference videos for all tests in tests.json.

    Each reference video is built by NN-filling from keyframe positions:
      - Keyframe indices get the actual video frame from the input.
      - Non-keyframe indices get the nearest keyframe's video frame.
      - A pulse mask strip (left edge) is white on keyframe frames, black otherwise.

    Per-test JSON options:
      - ref_first_frame (bool, default True): Replace frame 0 of the reference
        with the I2V condition image (first_frame) instead of the video frame.
    """
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

        video = rp.load_video(str(HERE / t["input_video"]), use_cache=False)
        keyframe_img = rp.load_image(str(HERE / t["first_frame"]), use_cache=False)
        keyframe_img = rp.as_byte_image(rp.as_rgb_image(keyframe_img, copy=False), copy=False)

        # Take first num_frames frames
        print(f"    Taking first {num_frames} of {len(video)} frames...")
        video = np.array(video[:num_frames])

        # Resize both to match width=768, preserving aspect ratio
        print(f"    Resizing to width=768...")
        [keyframe_img], video = rp.resize_videos_to_hold([keyframe_img], video, width=768)
        height = rp.get_image_height(keyframe_img)
        vid_width = rp.get_image_width(keyframe_img)
        video = rp.crop_images(video, height=height, origin="center")
        print(f"    Result: width={vid_width}, height={height}, frames={num_frames}")

        print(f"    Compositing reference + pulse mask...")
        # NN-fill: each frame gets the nearest keyframe's video content.
        # Frame 0 is always the I2V condition image (keyframe_img).
        kf_sorted = sorted(ki for ki in keyframes if 0 <= ki < num_frames)
        nn_frames = np.empty_like(video)
        for i in range(num_frames):
            idx = np.searchsorted(kf_sorted, i)
            left = kf_sorted[max(0, idx - 1)]
            right = kf_sorted[min(idx, len(kf_sorted) - 1)]
            nearest = left if (i - left) <= (right - i) else right
            nn_frames[i] = video[nearest]
        if t.get("ref_first_frame", True):
            nn_frames[0] = keyframe_img

        mask_width = round(indicator_size * vid_width)
        out_height = round((1 + indicator_size) * height)
        out_width = round((1 + indicator_size) * vid_width)

        pulse_mask = np.zeros((num_frames, out_height, mask_width, 3), dtype=np.uint8)
        for ki in keyframes:
            if 0 <= ki < num_frames:
                pulse_mask[ki] = 255

        ref_out = np.zeros((num_frames, out_height, out_width, 3), dtype=np.uint8)
        ref_out[:, -height:, -vid_width:] = nn_frames
        ref_out[:, :out_height, :mask_width] = pulse_mask

        # Resize each frame with cv2 (rp.resize_videos mangles NHWC channels)
        import cv2
        ref_out = np.stack([
            cv2.resize(f, (vid_width, height), interpolation=cv2.INTER_AREA)
            for f in ref_out
        ])
        print(f"    ref_out: {ref_out.shape}, saving {ref_path.name} (lossless)...")

        # Save with PyAV + CRF 0 to avoid H.264 artifacts on mostly-static content
        import av
        with av.open(str(ref_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=25)
            stream.width = vid_width
            stream.height = height
            stream.pix_fmt = "yuv444p"
            stream.options = {"crf": "0"}
            for frame_arr in ref_out:
                frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        print(f"  Generated: {ref_path}")

        # Save frame 0 as the condition image (matches reference exactly)
        cond_path = REF_DIR / f"{name}_condition.png"
        import cv2
        cv2.imwrite(str(cond_path), cv2.cvtColor(ref_out[0], cv2.COLOR_RGB2BGR))
        print(f"  Condition: {cond_path}")

    print("Done generating references.")


def _ensure_models():
    """Download and localize models to /models/LTX2/ if not already present."""
    print("Ensuring models are downloaded and localized...")
    subprocess.run([sys.executable, str(DOWNLOAD_MODELS)], check=True)


def run(checkpoint: str = None, skip_existing: bool = False):
    """Generate references then run inference on all tests."""
    _ensure_models()
    generate_references()

    if checkpoint is None:
        checkpoint = str(_find_latest_checkpoint())
    step_name = Path(checkpoint).stem.replace("lora_weights_", "")

    tests = _load_tests()
    print(f"\n{'=' * 70}")
    print(f"  Manual Tests - {step_name} ({len(tests)} tests)")
    print(f"  LoRA: {checkpoint}")
    print(f"{'=' * 70}\n")

    active = []
    for i, t in enumerate(tests):
        gpu_id = i % NUM_GPUS
        name = t["name"]
        ref_path = REF_DIR / f"{name}_ref.mp4"
        output_path = HERE / t["output_video"].replace(".mp4", f"_{step_name}.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and output_path.exists():
            print(f"  SKIP {name}: already exists")
            continue

        cmd = [
            "uv", "run", "python", "scripts/inference.py",
            "--checkpoint", "/models/LTX2/ltx-2-19b-dev.safetensors",
            "--text-encoder-path", "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized",
            "--lora-path", checkpoint,
            "--reference-video", str(ref_path),
            "--condition-image", str(REF_DIR / f"{name}_condition.png"),
            "--prompt", t["caption"],
            "--height", "480", "--width", "768",
            "--num-frames", str(t.get("num_frames", 121)),
            "--num-inference-steps", str(t.get("num_diffusion_steps", 30)),
            "--seed", str(t.get("seed", 42)),
            "--skip-audio",
            "--include-reference-in-output",
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

    print(f"\nOutputs in: {HERE / 'test_outputs'}")


if __name__ == "__main__":
    fire.Fire({"run": run, "generate_references": generate_references})

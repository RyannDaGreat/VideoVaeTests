#!/usr/bin/env python3
"""Stage 2 upscaling from saved raw latent (no re-encoding round trip).

Takes a raw patchified latent saved by inference.py --save-latent, crops out the
pulse mask and padding in latent space, upsamples 2x with the spatial upsampler,
conditions frame 0 with the original high-res first frame, refines with distilled
LoRA (3 steps), and decodes to video.

Subprocess isolation: text encoding, VAE encoding, and VAE decoding run as
subcommands of this same script (via Fire) in separate processes, avoiding
Gemma/VAE memory leaks. See CLAUDE.md: recursive Fire > nested Python strings.

Commands:
    upscale        Full pipeline: crop → upsample → condition → denoise → decode
    encode_text    (subprocess) Encode text prompt, save embeddings to .pt
    encode_image   (subprocess) VAE-encode an image at target resolution to .pt
    decode_latent  (subprocess) Decode a spatial latent .pt to video .mp4
"""

import subprocess as sp
import sys
import tempfile
from pathlib import Path

import fire
import torch

sys.path.insert(0, "/root/CleanCode")

# ── Model paths ─────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "/models/LTX2/ltx-2-19b-dev.safetensors"
GEMMA_ROOT = "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized"
SPATIAL_UPSAMPLER = "/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA = "/models/LTX2/ltx-2-19b-distilled-lora-resized_dynamic_fro095_avg_rank_242_bf16.safetensors"
DETAILER_LORA = "/models/LTX2/ltx-2-19b-ic-lora-detailer.safetensors"

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

SELF = str(Path(__file__).resolve())
LTX_TRAINER_DIR = str(Path(__file__).resolve().parents[3] / "LTX2" / "src" / "packages" / "ltx-trainer")

# VAE spatial compression factor
VAE_SPATIAL = 32


def _mem(label=""):
    """Print GPU memory usage. Pure diagnostic, no side effects on state."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0)
        free, _ = torch.cuda.mem_get_info(0)
        print(f"    [MEM {label}] Allocated: {alloc/1e9:.1f} GiB, Free: {free/1e9:.1f} GiB")


def _cleanup():
    """Aggressively free GPU memory. No side effects beyond cache clearing."""
    import gc
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_subprocess(cmd):
    """
    Run a subcommand of this script in a child process.

    Pure function — child process exits, fully releasing GPU memory.
    """
    full_cmd = ["uv", "run", "python", SELF] + cmd
    result = sp.run(full_cmd, cwd=LTX_TRAINER_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {' '.join(cmd[:2])}")


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess subcommands (invoked by upscale via recursive calls to this script)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_text(prompt: str, output: str, device: str = "cuda:0"):
    """
    Encode a text prompt and save embeddings to a .pt file.

    Runs as a subprocess to avoid Gemma's 63.9 GiB GPU memory leak.
    """
    from ltx_core.text_encoders.gemma import encode_text as _encode_text
    from ltx_pipelines.utils import ModelLedger

    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device(device),
        checkpoint_path=MODEL_CHECKPOINT, gemma_root_path=GEMMA_ROOT,
    )
    te = ledger.text_encoder()
    (v_ctx, a_ctx), _ = _encode_text(te, prompts=[prompt, ""])
    torch.save({"v": v_ctx.cpu(), "a": a_ctx.cpu()}, output)
    print(f"Text encoding saved to {output}")


def encode_image(image_path: str, output: str, width: int, height: int, device: str = "cuda:0"):
    """
    Resize an image to (width, height), VAE-encode it, and save latent to .pt.

    Runs as a subprocess to avoid VAE encoder memory leaks.
    """
    from PIL import Image
    from torchvision import transforms
    from ltx_pipelines.utils import ModelLedger

    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = (img_tensor * 2.0 - 1.0).to(device, dtype=torch.float32)
    img_5d = img_tensor.unsqueeze(2)  # (1, 3, 1, H, W)

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device(device), checkpoint_path=MODEL_CHECKPOINT)
    encoder = ledger.video_encoder()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latent = encoder(img_5d)
    torch.save(latent.cpu(), output)
    print(f"Image encoded: {image_path} → {latent.shape} saved to {output}")


def decode_latent(
    latent_path: str,
    output: str,
    device: str = "cuda:0",
    seed: int = 42,
    frame_rate: float = 25.0,
):
    """
    Decode a spatial latent .pt file to an .mp4 video.

    Runs as a subprocess to avoid VAE decoder memory leaks.
    """
    import rp
    from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video
    from ltx_pipelines.utils import ModelLedger

    torch_device = torch.device(device)
    latent = torch.load(latent_path, weights_only=True).to(torch_device)
    print(f"Loaded latent: {latent.shape}")

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch_device, checkpoint_path=MODEL_CHECKPOINT)
    decoder = ledger.video_decoder()
    print(f"Decoder loaded, GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GiB")

    tiling_config = TilingConfig.default()
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    with torch.inference_mode():
        chunks = list(vae_decode_video(latent, decoder, tiling_config, generator))
        video = torch.cat(chunks, dim=0)
    video_np = rp.as_numpy_array(video.cpu())
    rp.save_video_mp4(video_np, output, framerate=frame_rate, video_bitrate=20000000)
    print(f"Decode + save complete: {output}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pure functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latent_crop(width, height, pulse_mask_px, source_aspect):
    """
    Compute latent-space crop coordinates to remove pulse mask and padding.

    Returns (top, left) in latent units — the number of latent rows/cols to skip.
    The content region is latent[:, :, :, top:, left:].

    Pure function — no side effects.

    >>> compute_latent_crop(1152, 736, 128, 1920/1080)
    (5, 4)
    >>> compute_latent_crop(768, 480, 128, 1920/1080)
    (2, 4)
    """
    avail_w = width - pulse_mask_px
    avail_h = height
    if avail_w / avail_h > source_aspect:
        content_h = avail_h
        content_w = round(avail_h * source_aspect)
    else:
        content_w = avail_w
        content_h = round(avail_w / source_aspect)

    top_pad = height - content_h  # padding above content (bottom-aligned)
    left_pad = pulse_mask_px + (avail_w - content_w)  # mask + any extra (right-aligned)

    lt_top = top_pad // VAE_SPATIAL
    lt_left = left_pad // VAE_SPATIAL
    return lt_top, lt_left


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def upscale(
    latent_path: str,
    output: str,
    prompt: str,
    first_frame: str,
    pulse_mask_px: int,
    source_aspect: float,
    device: str = "cuda:0",
    seed: int = 42,
    frame_rate: float = 25.0,
    lora_path: str = None,
):
    """
    Stage 2 upscaling from a saved raw patchified latent.

    Flow:
      1. Encode text in subprocess (Gemma leaks 63.9 GiB otherwise)
      2. Load + unpatchify latent → crop pulse mask + padding in latent space
      3. Upsample cropped latent 2x with spatial upsampler
      4. Encode condition image (original high-res first frame at 2x content dims)
      5. Denoise 3 distilled steps with condition image at frame 0
      6. Save denoised latent (so decode can be retried without redoing diffusion)
      7. Decode to video in subprocess (VAE decoder also leaks)

    LoRAs for stage 2: distilled + detailer (+ optionally trained LoRA).
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.upsampler import upsample_video
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils import ModelLedger
    from ltx_pipelines.utils.helpers import (
        cleanup_memory,
        denoise_audio_video,
        euler_denoising_loop,
        simple_denoising_func,
    )
    from ltx_pipelines.utils.types import PipelineComponents

    torch_device = torch.device(device)
    dtype = torch.bfloat16
    patchifier = VideoLatentPatchifier(patch_size=1)

    # ── Phase 1: Encode text in subprocess ──────────────────────────────
    print("\n[1/5] Encoding text (subprocess)...")
    embeddings_path = tempfile.mktemp(suffix=".pt")
    _run_subprocess(["encode_text", "--prompt", prompt, "--output", embeddings_path, "--device", device])
    embeddings = torch.load(embeddings_path, weights_only=True)
    v_ctx = embeddings["v"].to(torch_device)
    a_ctx = embeddings["a"].to(torch_device)
    Path(embeddings_path).unlink(missing_ok=True)
    _mem("after text encode subprocess")

    # ── Phase 2: Load latent, unpatchify, crop, upsample 2x ────────────
    print("\n[2/5] Loading latent → crop → upsample 2x...")
    data = torch.load(latent_path, weights_only=True)
    patchified_latent = data["latent"]
    s1_height = data["height"]
    s1_width = data["width"]
    num_frames = data["num_frames"]

    # Unpatchify from (B, N, C) to spatial (B, C, LT, LH, LW)
    lt = (num_frames - 1) // 8 + 1
    lh = s1_height // VAE_SPATIAL
    lw = s1_width // VAE_SPATIAL
    latent_shape = VideoLatentShape(batch=1, channels=128, frames=lt, height=lh, width=lw)
    spatial_latent = patchifier.unpatchify(patchified_latent, output_shape=latent_shape)
    spatial_latent = spatial_latent.to(device=torch_device, dtype=dtype)
    print(f"  Full latent: {spatial_latent.shape} ({s1_width}x{s1_height})")

    # Crop pulse mask + padding in latent space
    crop_top, crop_left = compute_latent_crop(s1_width, s1_height, pulse_mask_px, source_aspect)
    cropped = spatial_latent[:, :, :, crop_top:, crop_left:]
    content_lh, content_lw = cropped.shape[3], cropped.shape[4]
    content_h = content_lh * VAE_SPATIAL
    content_w = content_lw * VAE_SPATIAL
    print(f"  Cropped latent: {cropped.shape} (content {content_w}x{content_h}, removed top={crop_top} left={crop_left})")
    del spatial_latent

    # Target = 2x content
    target_h = content_h * 2
    target_w = content_w * 2
    print(f"  Stage 2 target: {target_w}x{target_h}")

    # Upsample
    up_ledger = ModelLedger(
        dtype=dtype, device=torch_device,
        checkpoint_path=MODEL_CHECKPOINT,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
    )
    video_encoder = up_ledger.video_encoder()
    upsampler_model = up_ledger.spatial_upsampler()
    with torch.inference_mode():
        upscaled = upsample_video(latent=cropped, video_encoder=video_encoder, upsampler=upsampler_model)
    print(f"  Upscaled latent: {upscaled.shape}")
    del video_encoder, upsampler_model, up_ledger, cropped
    _cleanup()
    _mem("after upsample")

    # ── Phase 3: Encode condition image in subprocess ───────────────────
    print(f"\n[3/5] Encoding condition image at {target_w}x{target_h} (subprocess)...")
    cond_latent_path = tempfile.mktemp(suffix=".pt")
    _run_subprocess([
        "encode_image",
        "--image-path", first_frame,
        "--output", cond_latent_path,
        "--width", str(target_w),
        "--height", str(target_h),
        "--device", device,
    ])
    cond_latent = torch.load(cond_latent_path, weights_only=True).to(torch_device)
    Path(cond_latent_path).unlink(missing_ok=True)
    print(f"  Condition latent: {cond_latent.shape}")
    _mem("after condition encode subprocess")

    # ── Phase 4: Denoise 3 distilled steps with condition image ─────────
    print(f"\n[4/5] Stage 2 denoising at {target_w}x{target_h} (3 steps)...")

    loras = []
    if lora_path:
        loras.append(LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP))
    loras.append(LoraPathStrengthAndSDOps(DISTILLED_LORA, 1.0, LTXV_LORA_COMFY_RENAMING_MAP))
    loras.append(LoraPathStrengthAndSDOps(DETAILER_LORA, 1.0, LTXV_LORA_COMFY_RENAMING_MAP))

    xf_ledger = ModelLedger(dtype=dtype, device=torch_device, checkpoint_path=MODEL_CHECKPOINT, loras=loras)
    transformer = xf_ledger.transformer()
    _mem("after transformer load")

    generator = torch.Generator(device=torch_device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    components = PipelineComponents(dtype=dtype, device=torch_device)
    distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(torch_device)

    # Condition frame 0 with the high-res first frame
    cond = VideoConditionByLatentIndex(latent=cond_latent, strength=1.0, latent_idx=0)

    def stage2_loop(sigmas, video_state, audio_state, stepper):
        """Stage 2 denoising: simple_denoising_func (no CFG), matching official pipeline."""
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=simple_denoising_func(
                video_context=v_ctx,
                audio_context=a_ctx,
                transformer=transformer,
            ),
        )

    stage_2_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=target_w, height=target_h, fps=frame_rate,
    )

    with torch.inference_mode():
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape,
            conditionings=[cond],
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=stage2_loop,
            components=components,
            dtype=dtype,
            device=torch_device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled,
        )

    print(f"  Denoising complete. Latent shape: {video_state.latent.shape}")
    del transformer, upscaled, xf_ledger, cond_latent
    _cleanup()
    _mem("after denoise")

    # Save denoised latent so decode can be retried without redoing diffusion
    output_dir = Path(output).parent
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    stage2_latent_path = str(latent_dir / Path(output).with_suffix(".pt").name)
    torch.save(video_state.latent.cpu(), stage2_latent_path)
    print(f"  Saved stage 2 latent: {stage2_latent_path}")
    del video_state, audio_state
    _cleanup()

    # ── Phase 5: Decode in subprocess ───────────────────────────────────
    print(f"\n[5/5] Decoding stage 2 video (subprocess)...")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    _run_subprocess([
        "decode_latent",
        "--latent-path", stage2_latent_path,
        "--output", output,
        "--device", device,
        "--seed", str(seed),
        "--frame-rate", str(frame_rate),
    ])

    print(f"\n  Stage 2 complete: {output} ({target_w}x{target_h})")


if __name__ == "__main__":
    fire.Fire({
        "upscale": upscale,
        "encode_text": encode_text,
        "encode_image": encode_image,
        "decode_latent": decode_latent,
    })

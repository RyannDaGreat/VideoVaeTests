#!/usr/bin/env python3
"""Stage 2 upscaling: takes a stage 1 video and produces a 2x resolution version.

Flow:
  1. Load stage 1 video → encode to latent with VAE encoder (tiled)
  2. Upsample latent 2x using spatial upsampler
  3. Add noise at distilled sigma level, denoise with distilled LoRA (3 steps)
  4. Decode with tiled VAE decoder at 2x resolution
  5. Save output video

Usage:
    python stage2_upscale.py \
        --input stage1_output.mp4 \
        --output stage2_output.mp4 \
        --device cuda:0 \
        --prompt "..." \
        --seed 42
"""

import sys
from pathlib import Path

import fire
import torch
from einops import rearrange

sys.path.insert(0, "/root/CleanCode")

# ── Model paths ─────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "/models/LTX2/ltx-2-19b-dev.safetensors"
TEXT_ENCODER = "/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized"
SPATIAL_UPSAMPLER = "/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA = "/models/LTX2/ltx-2-19b-distilled-lora-384.safetensors"

# Stage 2 uses only 4 sigma values (3 denoising steps)
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


def upscale(
    input: str,
    output: str,
    prompt: str,
    device: str = "cuda:0",
    seed: int = 42,
    lora_path: str = None,
    num_frames: int = 121,
    frame_rate: float = 25.0,
):
    """
    Run stage 2 upscaling on a stage 1 video.

    Args:
        input: Path to stage 1 output video
        output: Path to save stage 2 upscaled video
        prompt: Text prompt (same as stage 1)
        device: CUDA device
        seed: Random seed
        lora_path: Path to trained LoRA (applied on top of distilled LoRA)
        num_frames: Number of frames
        frame_rate: Frame rate
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.model.upsampler import upsample_video
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_pipelines.utils import ModelLedger
    from ltx_pipelines.utils.helpers import cleanup_memory, euler_denoising_loop, simple_denoising_func
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.types import PipelineComponents

    from ltx_trainer.video_utils import read_video

    torch_device = torch.device(device)
    dtype = torch.bfloat16

    print(f"Stage 2 upscaling: {input} -> {output}")
    print(f"  Device: {device}, Seed: {seed}")

    # Build LoRA list: distilled LoRA + optionally trained LoRA
    loras = []
    if lora_path:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps as LoraSpec
        loras.append((lora_path, 1.0, None))
    # Distilled LoRA for stage 2
    distilled_lora = [(DISTILLED_LORA, 1.0, None)]

    # Create model ledger for stage 2
    print("  Building model ledger...")
    stage_2_ledger = ModelLedger(
        dtype=dtype,
        device=torch_device,
        checkpoint_path=MODEL_CHECKPOINT,
        gemma_root_path=TEXT_ENCODER,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        loras=tuple(loras),
    )
    # Add distilled LoRA on top
    stage_2_ledger = stage_2_ledger.with_loras(loras=tuple(distilled_lora))

    pipeline_components = PipelineComponents(dtype=dtype, device=torch_device)

    generator = torch.Generator(device=torch_device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    # Step 1: Encode text
    print("  Encoding text...")
    text_encoder = stage_2_ledger.text_encoder()
    (v_context_p, a_context_p), _ = encode_text(text_encoder, prompts=[prompt, ""])
    del text_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    # Step 2: Load and encode stage 1 video to latent
    print(f"  Loading stage 1 video: {input}")
    video_frames, fps = read_video(input, max_frames=num_frames)
    print(f"    Loaded {video_frames.shape[0]} frames")

    video_encoder = stage_2_ledger.video_encoder()

    # Convert to [B, C, F, H, W] in [-1, 1]
    video_tensor = rearrange(video_frames, "T C H W -> 1 C T H W")
    # Trim to valid frame count (k*8 + 1)
    valid_frames = (video_tensor.shape[2] - 1) // 8 * 8 + 1
    video_tensor = video_tensor[:, :, :valid_frames]
    video_tensor = (video_tensor * 2.0 - 1.0).to(device=torch_device, dtype=torch.float32)

    print(f"  Encoding to latent space ({video_tensor.shape})...")
    video_encoder.to(torch_device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        stage1_latent = video_encoder(video_tensor)
    print(f"    Latent shape: {stage1_latent.shape}")

    # Step 3: Upsample latent 2x
    print("  Upsampling latent 2x...")
    upsampler = stage_2_ledger.spatial_upsampler()
    upscaled_latent = upsample_video(
        latent=stage1_latent,
        video_encoder=video_encoder,
        upsampler=upsampler,
    )
    print(f"    Upscaled latent shape: {upscaled_latent.shape}")
    del upsampler
    video_encoder.to("cpu")
    torch.cuda.synchronize()
    cleanup_memory()

    # Step 4: Denoise at 2x resolution with distilled sigmas
    print("  Loading transformer for stage 2 denoising...")
    transformer = stage_2_ledger.transformer()

    distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(torch_device)

    # Compute target resolution (2x stage 1)
    _, _, lt, lh, lw = upscaled_latent.shape
    # Latent to pixel: height = lh * 32, width = lw * 32, frames = (lt - 1) * 8 + 1
    target_h = lh * 32
    target_w = lw * 32
    target_frames = (lt - 1) * 8 + 1
    print(f"    Stage 2 resolution: {target_w}x{target_h}, {target_frames} frames")

    from ltx_core.types import LatentState, VideoPixelShape
    from ltx_pipelines.utils.helpers import denoise_audio_video

    stage_2_output_shape = VideoPixelShape(
        batch=1, frames=target_frames, width=target_w, height=target_h, fps=frame_rate,
    )

    def second_stage_denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=transformer,
            ),
        )

    print("  Running stage 2 denoising (3 steps)...")
    video_state, audio_state = denoise_audio_video(
        output_shape=stage_2_output_shape,
        conditionings=[],
        noiser=noiser,
        sigmas=distilled_sigmas,
        stepper=stepper,
        denoising_loop_fn=second_stage_denoising_loop,
        components=pipeline_components,
        dtype=dtype,
        device=torch_device,
        noise_scale=distilled_sigmas[0],
        initial_video_latent=upscaled_latent,
    )

    del transformer
    torch.cuda.synchronize()
    cleanup_memory()

    # Step 5: Decode at 2x resolution
    print("  Decoding stage 2 video...")
    tiling_config = TilingConfig.default()
    video_decoder = stage_2_ledger.video_decoder()
    decoded_video = vae_decode_video(
        video_state.latent, video_decoder, tiling_config, generator,
    )

    del video_decoder
    torch.cuda.synchronize()
    cleanup_memory()

    # Step 6: Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving stage 2 video to {output_path}...")
    encode_video(
        video=decoded_video,
        fps=frame_rate,
        audio=None,
        audio_sample_rate=None,
        output_path=str(output_path),
        video_chunks_number=1,
    )
    print(f"  Stage 2 complete: {output_path}")


if __name__ == "__main__":
    fire.Fire(upscale)

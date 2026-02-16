# Manual Tests - Jsonnet + Stage 2 Upscaling Manifest

## Goal
Convert manual test configuration from JSON to Jsonnet for compactness and flexibility, add support for "random N" keyframe shorthand, then add optional LTX stage 2 super-resolution upscaling to the test pipeline.

## Algorithm / Flow
1. tests.jsonnet → (jsonnet library) → Python dict list → generate references → run inference
2. Keyframes can be: explicit list `[0, 5, 10, ...]` OR string shorthand `"random 25"` (25 randomly distributed keyframes)
3. Stage 2 (optional per-test): after stage 1 inference, run spatial upsampler + distilled LoRA refinement at 2x resolution
4. Output JSON in test output folders remains vanilla JSON (archival)

## Key Paths
- **This directory**: `Training/nanobanna_flickertest_50k_121-361f_3seed_768x480/manual_tests/`
- **Tests config**: `tests.jsonnet` (source of truth)
- **Run script**: `run_tests.py` (reads jsonnet, generates references, launches inference)
- **Shell wrapper**: `../run_manual_tests.sh` → calls `run_tests.py`
- **LTX Trainer**: `../../LTX2/src/packages/ltx-trainer/`
- **Inference script**: `../../LTX2/src/packages/ltx-trainer/scripts/inference.py`
- **Two-stage pipeline**: `../../LTX2/src/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`
- **Model weights**:
  - Base: `/models/LTX2/ltx-2-19b-dev.safetensors`
  - Text encoder: `/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized`
  - Spatial upscaler: `/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors`
  - Distilled LoRA: `/models/LTX2/ltx-2-19b-distilled-lora-384.safetensors`
  - Trained LoRA: `../outputs/checkpoints/lora_weights_step_*.safetensors`
- **Hardware**: 8x A100-SXM4-80GB

## File Structure
- `tests.jsonnet` — Compact Jsonnet test definitions with template inheritance
- `run_tests.py` — Main pipeline: load jsonnet → resolve keyframes → generate refs → inference
- `test_outputs/` — Archival output folders with vanilla JSON + generated videos
- `raw_inputs/` — Source videos and first frames
- `generated_references/` — NN-filled reference videos + condition images
- `claude_instructions.md` — This manifest
- `concerns.md` — Progress log

## Stage 2 Architecture
- Stage 1: Generate at 1152x736 using trained LoRA (with pulse mask + padding)
- **Latent crop**: remove pulse mask (left 4 latent cols) + padding (top 5 latent rows) in latent space
- Spatial upsampler: 2x latent upsampling of cropped content → 2048x1152 clean output
- Condition image: original high-res first frame resized to 2048x1152, VAE-encoded, injected at frame 0
- Stage 2 denoising: distilled LoRA + detailer LoRA, 3 steps, simple_denoising_func (no CFG)
- Sigma schedule: [0.909375, 0.725, 0.421875, 0.0] (noise_scale = 0.909375)
- Subprocess isolation: text encode + image encode + decode each run in child processes (Fire subcommands)
  to avoid Gemma 63.9 GiB GPU memory leak and VAE decoder leak
- 2048x1152 = 36,864 tokens — fits on A100-80GB (notebooks proved 32,640 works)
- Also tested 2304x1472 (no crop) = 52,992 tokens — also fits but includes pulse mask in output

## Critical Constraints
- `num_frames % 8 == 1` (LTX requirement: 121, 241, 361, etc.)
- `height % 32 == 0` and `width % 32 == 0`
- Jsonnet outputs vanilla JSON; output test folders keep JSON archival format
- NO silent fallbacks (per CLAUDE.md)
- Use Fire for CLI (per CLAUDE.md)
- Use einops for tensor operations (per CLAUDE.md)

## Keyframe Shorthand Resolution
- List `[0, 5, 10, ...]` → use as-is
- String `"random N"` → generate N random keyframe indices, always including frame 0, seeded by test seed

## Test Matrix (8 tests, 1 per GPU)
User wants tests with keyframe counts: 8, 16, 32, and more up to current max (~80)
8 total tests, one per GPU.

## Verification Strategy
- VLM check first/middle/last frames of output videos
- Ensure reference videos have correct pulse mask pattern
- Verify output resolution matches expected dimensions
- Check that stage 2 outputs are 2x stage 1 resolution

## Success Criteria
- tests.jsonnet is readable and compact, supports "random N" shorthand
- run_tests.py reads jsonnet, resolves keyframes, runs inference
- 8 keyframe-sweep tests run successfully (stage 1)
- Stage 2 upscaling works and produces higher-resolution output
- Maximum achievable resolution determined via bisection
- All changes committed with proper [Claude] footer
- User notified via ntfy when complete

## Lessons Learned
(append-only — never delete)
- Re-encoding decoded video back to latent space is LOSSY → blue cast + grid artifacts
- Must use raw patchified latent from --save-latent, unpatchify, then upsample
- Saved latent is PATCHIFIED (B, N, C) — must unpatchify to (B, C, LT, LH, LW) before upsample_video
- Gemma text encoder leaks 63.9 GiB GPU even after del/gc → subprocess isolation required
- VAE decoder also leaks → subprocess isolation required
- Recursive Fire subcommands > nested Python strings (see ~/CleanCode/CLAUDE.md)
- Pulse mask (128px) and padding are divisible by 32 (VAE spatial factor) → clean latent-space crop
- DummyRegistry (default) loads only relevant weights per component — no cross-contamination
- 2304x1472 (52,992 tokens) fits on A100-80GB when GPU is clean (previous OOMs were from memory leaks)
- ffmpeg-python must be in ltx-trainer requirements for rp.save_video_mp4 to work in uv venv

# Concerns Log - Manual Tests Jsonnet + Stage 2

## 2026-02-16 08:45 - Project Start
- Explored entire codebase with 7+ parallel agents
- Identified all key files, model paths, and pipeline architecture
- LTX2 has TI2VidTwoStagesPipeline for 2-stage generation
- Stage 2 uses spatial upsampler (2x) + distilled LoRA refinement
- Models all present in /models/LTX2/
- Starting implementation: jsonnet dep → convert tests → update run script → run tests

## 2026-02-16 09:00 - Jsonnet + Stage 1 Implementation Complete
- Added `jsonnet` to ltx-trainer/pyproject.toml dependencies, installed system-wide and in uv venv
- Converted tests.jsonnet to use `"random N"` shorthand for keyframes
- Updated run_tests.py: reads .jsonnet, resolves keyframe shorthands, saves vanilla JSON in output folders
- `resolve_keyframes()` pure function: seeds RNG with test seed for reproducible random keyframes
- 8 tests configured: 8, 16, 32, 44, 56, 64, 72, 80 keyframes at 1152x736
- Stage 1 tests launched on 8 GPUs, running now (step_08450 checkpoint)
- Stage 2 infrastructure coded: stage2_upscale.py standalone script + integration in run_tests.py
- Stage 2 uses spatial upsampler (2x latent) + distilled LoRA (3 denoising steps)
- Distilled sigma values: [0.909375, 0.725, 0.421875, 0.0]
- CONCERN: Stage 2 re-encodes stage 1 video to latent (slightly lossy vs operating on raw latent)

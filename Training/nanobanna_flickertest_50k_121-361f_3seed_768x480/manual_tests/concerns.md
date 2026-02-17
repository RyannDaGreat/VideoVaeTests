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

## 2026-02-16 09:10 - Stage 1 Tests Complete, Stage 2 VRAM Issues
- All 8 stage 1 tests completed successfully on 8 GPUs
- Output dir: test_outputs/kf_sweep_1152x736_step_08450/
- VLM verified first/middle/last frames of 32kf test - looks good
- Committed Jsonnet changes as 678df51
- Stage 2 OOM issues at 2304x1472 (2x of 1152x736):
  - OOM #1: VAE encoder conv3d needed 41 GiB → Fixed with tiled_encode_video
  - OOM #2: Upsampler dtype mismatch → Fixed by casting latent to bf16
  - OOM #3: ModelLedger loading full checkpoint to GPU → Fixed with CPU-first loading
  - OOM #4: 19B transformer at 2304x1472 resolution → Too many tokens (52,992 vs 13,248 at 1152x736)
  - OOM #5: FP8 transformer still OOM at 2304x1472 even with 25 frames
  - OOM #6: Even with FP8, LoRA overhead fills GPU
- FINDING: 2304x1472 is NOT feasible on A100-80GB for 121 frames
- FINDING: Even 25 frames at 2304x1472 with FP8+LoRA doesn't fit
- Currently testing: 25 frames, FP8, no trained LoRA (just distilled)
- Need to determine max feasible resolution via bisection

## 2026-02-16 10:30 - Stage 2 Working at 1152x704
- BREAKTHROUGH: Stage 2 pipeline fully functional at 576x368 → 1152x704 (121 frames)
- Key fixes required to get it working:
  - Text encoder: must run in subprocess (Gemma leaks 63.9 GiB GPU even after del/gc)
  - VAE encoder: must use tiled_encode_video from process_videos.py
  - Models: must build on CPU first via ModelLedger(device=cpu), then .to(cuda)
  - Denoising: must use lean video-only loop (no audio Modality), torch.inference_mode()
  - VAE decoder: must also run in subprocess (same memory leak issue as text encoder)
  - Patchifier: VideoLatentPatchifier(patch_size=1), positions via VideoLatentTools
- QUALITY ISSUES in stage 2 output:
  - Blue color cast (VAE re-encoding round-trip issue, or normalization mismatch)
  - Tiling grid artifacts visible in middle/late frames
  - Resizing 1152x736→576x368 loses detail before re-encoding
- Max resolution findings for A100-80GB, 121 frames, 19B transformer:
  - 2304x1472: NOT FEASIBLE (even FP8, even 25 frames)
  - 1536x960: NOT FEASIBLE (OOM in transformer forward)
  - 1152x736: NOT FEASIBLE as direct 2x output (OOM by ~600MB)
  - 1152x704: WORKS (576x368 stage 1, 12,672 tokens, uses ~78/80 GiB peak)
- CONCLUSION: The 2x spatial upsampler is only practical when stage 1 is at half target resolution.
  Our trained LoRA works best at 1152x736 so going higher requires multi-GPU or different approach.

## 2026-02-16 14:00 - Final 100-Step Video Set Complete
- All 8 stage 1 tests at 100 diffusion steps completed on 8 GPUs (step_08850 checkpoint)
- All 8 stage 2 upscaling jobs completed (576x368 → 1152x704)
- Output dir: test_outputs/kf_sweep_1152x736_100step_step_08850/
- VLM verification of 32kf test (first/middle/last frames):
  - Stage 1 (100 steps): Excellent quality. Clear boat, dynamic choppy ocean, consistent across frames.
  - Stage 2: Still has blue color cast and grid tiling artifacts. Not production-ready.
- Stage 2 artifacts are consistent with initial 30-step test - the re-encoding approach introduces
  color shift regardless of step count. Root cause likely: VAE encode→decode round trip is lossy,
  AND the trained LoRA's style doesn't align well with the distilled LoRA's expectations.
- RECOMMENDATION: Stage 1 at 100 steps at 1152x736 is the best quality achievable on single GPU.
  Stage 2 needs either: (a) saving raw latent from stage 1 instead of re-encoding, or
  (b) training the LoRA at half resolution (576x368) so stage 2 produces 1152x736 natively.

## 2026-02-16 15:00 - Root Cause Analysis of Stage 2 Artifacts
- VLM comparison: stage 1 crop looks great, stage 2 crop shows blue cast + grid tiling + blur
- Read working implementations: LTX Stages Analysis notebook + t2v_2stage_canny_nodistill.py
- ROOT CAUSE IDENTIFIED: My approach re-encoded a DECODED video back to latent space.
  Working code NEVER re-encodes. It keeps the raw diffusion latent and passes it directly
  to the spatial upsampler. The VAE encode→decode round trip is lossy and introduces artifacts.
- CORRECT APPROACH: Save raw patchified latent from stage 1 inference BEFORE VAE decode.
  Use that raw latent → upsample → stage 2 denoise → decode.
- The working code uses denoise_audio_video() for stage 2 (not a custom lean loop).
  The OOMs I hit were from memory leaks, not from denoise_audio_video being too heavy.
- PLAN: Modify inference.py to save latent. Then stage 2 loads that latent directly.

## 2026-02-16 18:00 - Research Frenzy Complete, Full Diagnosis

### Research Findings (8 parallel agents):
1. **Saved latent is PATCHIFIED (1, 13248, 128)** for 1152x736, 121 frames
   - Must unpatchify to (1, 128, 16, 23, 36) before upsample_video
2. **upsample_video expects spatial (B, C, F, H, W)** — not tokens
3. **Notebooks proved 1088x1920 (32,640 tokens) works on A100-80GB** for stage 2
   - Our target 1152x736 = only 13,248 tokens — EASILY fits
   - Previous OOMs were from Gemma memory leak + VAE leak, NOT token count
4. **Correct two-stage pattern**: stage 1 at HALF resolution → upsample → stage 2 at FULL
5. **Stage 2 uses simple_denoising_func (NO CFG)** — single forward pass per step
6. **Stage 2 LoRAs**: distilled + detailer (additive on top of stage 1 LoRAs via with_loras)
7. **DummyRegistry (default)** loads only relevant weights per component — memory safe
8. **Gemma text encoder** MUST run in subprocess (leaks 63.9 GiB permanently)

### VLM Comparison (cropped boat area, frame 30):
- Stage 1: Beautiful, detailed boat on choppy ocean, natural colors
- Stage 2: Absolute garbage — blue cast, grid tiling, blur, oversaturated

### Root Cause (confirmed):
Previous stage 2 re-encoded decoded video back to latent → lossy round trip + wrong format
Plus: loaded wrong latent format (patchified) into spatial-expecting upsample_video

### Correct Fix Plan:
1. Run stage 1 at 576x368 (half of 1152x736) with --save-latent
2. Stage 2: load raw latent → unpatchify → upsample 2x → denoise 3 steps → decode to 1152x736
3. Text encoding in subprocess (Gemma memory leak)
4. Sequential model loading with cleanup between each phase
5. 1152x736 = 13,248 tokens — well within A100-80GB (proven 32,640 works in notebooks)

## 2026-02-17 - I2V CFG Guidance Exploration

### Approaches tried (all initially broken due to IC-LoRA bug):
1. **3-pass additive**: V_∅ + text_cfg*(V_T - V_∅) + i2v_cfg*(V_I - V_∅) — went white
2. **4-pass nested**: nested text CFG within image CFG — went white
3. **Token noising**: replace image tokens with flow-matching-noised versions — went white
4. **Token removal**: remove image tokens from negative pass sequence — went white

### ROOT CAUSE of all failures: IC-LoRA reference tokens were being stripped
- ALL approaches used `denoise_mask < 1.0` to identify "image conditioned" tokens
- IC-LoRA reference tokens ALSO have denoise_mask=0 (they're conditioning tokens too)
- Every "no image" pass was actually "no image AND no IC-LoRA reference"
- The transformer lost ALL context about what video to generate → white/washed output
- Fix: use ref_seq_len to distinguish reference tokens (always kept) from condition
  image tokens (target frame 0 only, optionally dropped)

### Research findings (10-agent hyper-frenzy + 4-agent mini-frenzy):
- STIV tested separate image/text CFG — concluded unified CFG works better
- InstructPix2Pix has battle-tested 3-pass dual CFG formula
- DynamiCrafter is the only I2V model with explicit dual image/text scales
- Most models don't do separate image CFG — the model must be trained for it
- Our IC-LoRA WAS trained with first_frame_conditioning_p=0.8 (20% dropout) — valid
- LTX-2 uses flow matching: noisy = (1-sigma)*clean + sigma*noise (NOT sqrt-based DDPM)
- Washed-out artifacts from CFG are well-known; fixes include APG, CFG rescaling

### Current implementation: cfg_drop_image (float)
- 0.0 = standard CFG (negative pass keeps image) — 2 passes
- 1.0 = full drop (condition image tokens physically removed from negative sequence) — 2 passes
- 0-1 = blend both negative results at non-conditioned target tokens — 3 passes
- IC-LoRA reference tokens are ALWAYS present in ALL passes
- CFG delta only applied to non-conditioned target tokens (frames 1-N)
- Condition image tokens (frame 0) forced clean by post-processing mask

### Lessons learned:
- denoise_mask=0 doesn't distinguish IC-LoRA reference from condition image
- Must use ref_seq_len to tell them apart
- Flow matching noising: (1-sigma)*clean + sigma*noise, NOT clean + sigma*noise
- Token removal is cleaner than token noising (model has seen shorter sequences in training)

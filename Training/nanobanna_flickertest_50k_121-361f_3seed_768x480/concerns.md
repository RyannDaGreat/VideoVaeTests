# Concerns Log - Nanobanna Flicker Test (50K, 121-361f, 3 Seeds, 768x480)

Generated: 2026-02-11

## Status: GPU PREPROCESSING IN PROGRESS

### Progress Log
- **CPU video generation**: COMPLETE - 29,540 train pairs generated
- **Caption embedding**: COMPLETE (re-ran 3 times due to poor planning - fix applied)
- **Video latent encoding**: IN PROGRESS - 8,297/29,540 files, GPUs at 53-55GB/80GB
- **VAE tiling**: ENABLED (required to prevent OOM on 361-frame videos)

### VRAM Findings
- Without VAE tiling: OOM at 768x480x361 (tried to allocate 53.54 GiB, only 52.73 GiB free)
- With VAE tiling + expandable_segments: STABLE at 53-55GB per GPU
- **Fix applied to make_dataset.py**: `--vae-tiling` flag added permanently

### TODO: Training Forward Pass Test (CRITICAL)
Before launching full training, test a single forward pass with each bucket size.
**Sequence lengths are MASSIVE:**
- 768x480x121 IC LoRA: 11,520 tokens (2.25x previous run)
- 768x480x241 IC LoRA: 22,320 tokens (4.4x previous run)
- 768x480x361 IC LoRA: 33,120 tokens (6.5x previous run!)
Previous run (512x320x121): 5,120 tokens - trained fine on 80GB A100s.

### CONTINGENCY: If Training OOMs on 361-Frame Sequences

**DO NOT just retry or guess. Launch TWO frenzies:**

1. **Research Frenzy** (Opus agent):
   - Search LTX trainer docs for FSDP support, memory optimization flags
   - Check if gradient_accumulation + gradient_checkpointing can help
   - Research DeepSpeed ZeRO stages for this model size
   - Check if the trainer supports sequence-length-aware batching (skip long sequences if OOM)

2. **Coding Frenzy** (Opus agent):
   - Test forward pass with each bucket size independently
   - Try: gradient checkpointing, FSDP, DeepSpeed ZeRO-2/3, bf16 + gradient scaling
   - Try: reducing LoRA rank for longer sequences
   - Try: falling back to 512x320 for 361-frame videos only (mixed resolution buckets)
   - Last resort: drop 361-frame bucket entirely, train only on 121+241

**ESCALATION PATH (in order):**
1. Try gradient checkpointing, bf16, expandable_segments
2. Try FSDP (Fully Sharded Data Parallel) instead of DDP
3. Try DeepSpeed ZeRO-2, then ZeRO-3
4. Try reducing LoRA rank for 361-frame batches
5. Up to **5 research frenzies** (Opus agents) exploring LTX trainer memory options
6. Up to **5 coding frenzies** (Opus agents) implementing fixes
7. **ONLY AFTER ALL ABOVE FAIL** → Notify user with option:
   - Downgrade 361-frame videos ONLY to 512x320
   - Keep 121/241 at 768x480
   - Mixed resolution buckets: 768x480x121; 768x480x241; 512x320x361
   - This requires reprocessing ~7,100 videos at lower resolution
   - Dataset.json tracks which bucket each video belongs to (can identify 361-frame samples)

**Resolution downgrade is ABSOLUTE LAST RESORT** - notify user if required.

### Sequence Length Comparison Table
| Config | Tokens | vs Previous |
|--------|--------|-------------|
| 512x320x121 (prev) | 5,120 | baseline |
| 768x480x121 | 11,520 | 2.25x |
| 768x480x241 | 22,320 | 4.4x |
| 768x480x361 | 33,120 | 6.5x |

## Known Risks

### VRAM at 768x480x361
768x480 at 361 frames is significantly larger than the previous 512x320x121.
Monitor for OOM during both latent precomputation and training.
See claude_instructions.md "VRAM Fallback Strategy" section.

### Dataset Size
50K samples x 3 seeds = up to ~150K video pairs. This will take significantly
longer to generate and precompute latents for than the previous 10K run.

### Variable Frame Lengths
First time using multiple frame buckets. Verify that the trainer handles
bucketed data correctly with resolution-buckets "768x480x121;768x480x241;768x480x361".

## Fixes Applied (Carried Forward From Previous Run)

### Fix 1: Cache Bug (use_cache=False)
All image loading uses use_cache=False. Prevents identical reference videos.

### Fix 2: None Caption Handling
Explicit None check on video_caption[0] with fallback to "A video".

### Fix 3: First Frame Conditioning
first_frame_conditioning_p set to 0.8 (was accidentally 0.2 initially in previous run).

## Progress Log

(Will be updated as pipeline runs)

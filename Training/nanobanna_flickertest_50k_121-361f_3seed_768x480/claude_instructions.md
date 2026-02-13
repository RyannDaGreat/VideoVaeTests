# Nanobanna Flicker Test - 50K, Variable Frames (121-361), 3 Seeds, 768x480

**LIVING DOCUMENT**: This manifest is dynamically updated as new requirements emerge.
Always check `concerns.md` for real-time issues, fixes, and progress updates.

## CRITICAL ALGORITHM DEFINITION

**This trains:** **Image + In-Context + Prompt -> Video**

Based on Nanobanna algorithm that transforms videos using keyframes.

**Key components:**
1. **First-frame conditioning** (80% of the time): Target video frame 0 used as image conditioning
2. **IC LoRA conditioning**: Flickery reference video with sparse keyframes + pulse mask
3. **Text prompt**: Real captions from Pexels metadata (`video_caption[0]`)

**Training strategy**: `first_frame_conditioning_p: 0.8` (CRITICAL - NON-NEGOTIABLE)

This enables the model to use both the first frame AND the flickery keyframes to generate smooth video.

---

## What Changed From nanobanna_flickertest (Previous Run)

| Parameter | Previous (nanobanna_flickertest) | This Run (50k_121-361f_3seed_768x480) |
|-----------|----------------------------------|---------------------------------------|
| NUM_SAMPLES | 10 (10K samples) | 50 (all 50K samples) |
| NUM_TEST_SAMPLES | 50 | 1000 |
| Frame count | Fixed 121 | Variable: 121, 241, or 361 (per video) |
| Seeds per sample | 1 | 3 (seeds 0, 1, 2) |
| Resolution | 512x320 | 768x480 |
| Resolution buckets | 512x320x121 | 768x480x121;768x480x241;768x480x361 |
| File naming | {sample_id}.mp4 | {sample_id}_s{seed}.mp4 |
| Validation dims | [512, 320, 121] | [768, 480, 121] |

### Variable Frame Length Logic
1. **First pass**: Scan all sample directories, check `rp.get_video_file_num_frames()` for each
2. **Bucket selection**: With 2x speedup, need 2*F source frames:
   - 361 frames -> need 722+ source frames (preferred, longest)
   - 241 frames -> need 482+ source frames
   - 121 frames -> need 242+ source frames
   - Too short -> skip sample
3. **Largest bucket wins**: Each video gets the longest frame count it can support
4. **3 seeds**: Each sample produces 3 different temporal crops (seeds 0, 1, 2)
5. **Naming**: `{sample_id}_s0.mp4`, `{sample_id}_s1.mp4`, `{sample_id}_s2.mp4`

### Training Data Volume
With 50K samples x 3 seeds = up to ~150K training video pairs (minus skipped samples).

---

## CRITICAL LESSONS LEARNED (From Previous Run)

### Lesson 1: THE CACHE BUG - use_cache=False ALWAYS

```python
# WRONG - ALL samples load the SAME cached image from first sample
keyframes = rp.load_image("after.png", use_cache=True)

# CORRECT - Each sample loads its own image
keyframes = rp.load_image("after.png", use_cache=False)
```

**What happened**: When processing samples in parallel, `use_cache=True` caused every single
sample to load the first sample's after.png from cache. Result: Wind turbine reference video
matched with office scene, identical reference videos across different samples.

**NEVER use `use_cache=True` in parallel processing. ALWAYS use `use_cache=False`.**

### Lesson 2: None Caption Handling

```python
# WRONG - Returns None if video_caption[0] is literally None
caption = meta.get('video_caption', ["A video"])[0]

# CORRECT - Explicit None check with fallback
video_captions = meta.get('video_caption', ["A video"])
caption = video_captions[0] if video_captions and video_captions[0] is not None else "A video"
```

**What happened**: 211 entries had None captions causing `TypeError: can only concatenate str (not "NoneType") to str` during latent precomputation.

### Lesson 3: First Frame Conditioning at 0.8

`first_frame_conditioning_p: 0.8` is CRITICAL for the nanobanna algorithm.
Was accidentally set to 0.2 in initial run. Must be 0.8.

---

## Key Paths

### Source Data
| Item | Path |
|------|------|
| Pexels paired data root | `/root/CleanCode/Sandbox/RP_Dumps/YashDump/jan10_last_50K_pexels_v2/paired_data/` |
| Total samples available | ~50,000 |
| Samples to process | **50,000 (ALL)** |
| Sample structure | `{sample_id}/` containing `before.png`, `after.png`, `raw_video.mp4`, `metadata.json` |

### Models
| Item | Path |
|------|------|
| Base model checkpoint | `/models/LTX2/ltx-2-19b-dev.safetensors` (41 GB) |
| Gemma text encoder | `/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized/` |

### LTX-2 Training Infrastructure
| Item | Path |
|------|------|
| LTX2 repo root | `/root/CleanCode/Github/VideoVaeTests/LTX2/` |
| Trainer package | `/root/CleanCode/Github/VideoVaeTests/LTX2/src/packages/ltx-trainer/` |
| Training script | `...ltx-trainer/scripts/train.py` |
| Dataset preprocessor | `...ltx-trainer/scripts/process_dataset.py` |
| Inference script | `...ltx-trainer/scripts/inference.py` |

### Hardware
- **8x NVIDIA A100-SXM4-80GB** (640 GB total VRAM)
- All GPUs used for training via DDP (Distributed Data Parallel)

---

## Output File Structure

Everything lives in this directory:
```
/root/CleanCode/Github/VideoVaeTests/Training/nanobanna_flickertest_50k_121-361f_3seed_768x480/
```

### Required Deliverables

```
nanobanna_flickertest_50k_121-361f_3seed_768x480/
|-- claude_instructions.md          # THIS FILE - manifest/instructions
|-- concerns.md                     # Real-time issues, fixes, progress
|-- datasets/
|   |-- transform.py                # Video pair generator (768x480, --seed parameter)
|   |-- make_dataset.py             # Dataset orchestration (50K, variable frames, 3 seeds)
|   |-- videos/                     # Target videos ({sample_id}_s{seed}.mp4)
|   |-- reference_videos/           # Flickery reference videos ({sample_id}_s{seed}.mp4)
|   |-- test_videos/                # 1000 test videos (x3 seeds)
|   |-- test_reference_videos/      # 1000 test references (x3 seeds)
|   |-- dataset.json                # Training set metadata for LTX trainer
|   |-- test_set.json               # Test set metadata
|   `-- .precomputed/               # LTX trainer preprocessed latents
|       |-- latents/
|       |-- conditions/
|       `-- reference_latents/
|-- configs/
|   |-- nanobanna_flicker_ic_lora.yaml  # Main training config
|   `-- accelerate_ddp.yaml             # Accelerate DDP config (8 GPUs)
|-- outputs/                        # Training outputs (checkpoints, logs)
|-- inference_outputs/              # Validation inference results
|-- debug_videos/                   # 20 debug videos for visual inspection
`-- .gitignore                      # Ignore generated artifacts
```

---

## Resolution & Frame Constraints (LTX-2 Requirements)

- **Spatial**: width and height must be divisible by 32
- **Temporal**: frame count must satisfy `F % 8 == 1` (valid: 121, 241, 361)
- **Training resolution**: 768x480 at variable frame counts (121/241/361)
- **Resolution buckets**: `768x480x121;768x480x241;768x480x361`
- **Sequence length formula**: `seq_len = (H/32) * (W/32) * ((F-1)/8 + 1)`

---

## User Notification Points (ONLY THESE 3 TIMES)

1. **When latent precomputation starts** (GPU processing: VAE encoding + text encoding)
2. **When training starts** (8 GPU DDP training launch)
3. **When first checkpoint saves** (proof training is working)

---

## Training Babysitting Protocol

**Phase 1: Until First Checkpoint (frequent checks)**
- Monitor every 60 seconds
- Watch for: process running, outputs directory, checkpoint creation
- When first checkpoint detected -> notify user, transition to Phase 2

**Phase 2: After First Checkpoint (hourly checks)**
- Monitor every 1 hour
- Watch for: process still running, no crashes in logs, checkpoints continuing to save
- Document any issues in concerns.md

---

## VRAM Contingency Plan

768x480 at 361 frames produces **33,120 IC LoRA tokens** per sample (6.5x the previous run).
This WILL be tight on 80GB A100s. Follow this escalation path IN ORDER.

### Sequence Length Reference
| Bucket | IC LoRA Tokens | vs Previous (5,120) |
|--------|---------------|---------------------|
| 768x480x121 | 11,520 | 2.25x |
| 768x480x241 | 22,320 | 4.4x |
| 768x480x361 | 33,120 | 6.5x |

### Escalation Path (follow in order, do NOT skip steps)

**Step 1: Basic optimizations** (already applied)
- Gradient checkpointing: enabled
- bf16 mixed precision: enabled
- batch_size: 1

**Step 2: Switch DDP → FSDP**
- Config ready: `configs/accelerate_fsdp.yaml` (8 GPUs, FULL_SHARD)
- FSDP shards model parameters across GPUs (~8x memory reduction for model weights)
- Launch: `uv run accelerate launch --config_file configs/accelerate_fsdp.yaml scripts/train.py ...`

**Step 3: FSDP + CPU parameter offload**
- Edit `accelerate_fsdp.yaml`: set `fsdp_offload_params: true`
- Offloads sharded parameters to CPU RAM (slower but frees significant GPU memory)

**Step 4: FSDP + INT8 quantization**
- Edit training config: `quantization: "int8-quanto"` (~50% model memory reduction)
- Compatible with LoRA training mode

**Step 5: FSDP + 8-bit AdamW optimizer**
- Edit training config: `optimizer_type: "adamw8bit"` (~75% optimizer memory reduction)

**Step 6: Reduce LoRA rank**
- Lower `rank: 128` to `rank: 64` or `rank: 32` for longer sequences

**Step 7: Research frenzies (up to 5)**
- Launch 10 Opus agents per frenzy exploring memory optimization approaches
- Agents search web, LTX docs, PyTorch FSDP docs, HuggingFace Accelerate docs
- Each agent gets diversified prompts (different angles/approaches)
- All outputs go in `.frenzy/` subfolder

**Step 8: Coding frenzies (up to 5)**
- Launch 10 Opus agents per frenzy, each in isolated `scratchpad_N/` folder
- Agents experiment with different configurations and test forward passes
- HIGH AUTONOMY: agents try things aggressively, fail fast, try alternatives

**Step 9: Determine maximum resolution per frame count**
- BEFORE downgrading resolution, determine what we CAN get away with
- For each frame count (121, 241, 361), find the max resolution that fits in VRAM
- Test with FSDP + all optimizations enabled
- Example results might be:
  - 121 frames: 768x480 fits ✅
  - 241 frames: 768x480 fits ✅
  - 361 frames: max is 640x384 (or whatever we discover)
- Use these findings to create OPTIMAL mixed-resolution buckets
- LTX constraint: width and height must be divisible by 32

**Step 10: ABSOLUTE LAST RESORT - Mixed resolution downgrade**
- Only 361-frame videos get downgraded (121 and 241 stay at 768x480)
- Use the max resolution found in Step 9 (not blindly 512x320)
- Requires reprocessing ~7,100 videos at lower resolution
- Dataset.json tracks which bucket each video belongs to
- Mixed buckets example: `768x480x121;768x480x241;640x384x361`
- **NOTIFY USER before executing this step**

### Key Research Findings (from Opus agent)
- **FSDP**: Fully supported, config exists at `ltx-trainer/configs/accelerate/fsdp.yaml`
- **DeepSpeed**: NOT supported (zero mentions in codebase)
- **INT8 quantization**: Supported via optimum-quanto
- **8-bit AdamW**: Supported via bitsandbytes
- **Low VRAM reference**: `configs/ltx2_av_lora_low_vram.yaml`
- **No per-sample OOM protection**: Training crashes if any sample too large

---

## Rules
- **DO NOT modify any files in the LTX2 repository** - use it via sys.path or CLI only
- **DO NOT modify files in the original nanobanna_flickertest folder**
- All scripts must be **runnable with no arguments**
- **ALWAYS use use_cache=False** (cache bug prevention)
- **Captions**: metadata.json['video_caption'][0] with None fallback to "A video"
- **first_frame_conditioning_p: 0.8** (non-negotiable)
- Keep everything self-contained in this folder

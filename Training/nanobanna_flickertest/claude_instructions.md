# Nanobanna Flicker Test - Frame Flicker IC LoRA with Pexels Data

**📋 LIVING DOCUMENT**: This manifest is dynamically updated as new requirements emerge.
Always check `concerns.md` for real-time issues, fixes, and progress updates.

## CRITICAL ALGORITHM DEFINITION ⚠️

**This trains:** **Image + In-Context + Prompt → Video**

Based on Nanobanna algorithm that transforms videos using keyframes.

**Key components:**
1. **First-frame conditioning** (80% of the time): Target video frame 0 used as image conditioning
2. **IC LoRA conditioning**: Flickery reference video with sparse keyframes + pulse mask
3. **Text prompt**: Real captions from Pexels metadata (`video_caption[0]`)

**Training strategy**: `first_frame_conditioning_p: 0.8` (CRITICAL - NON-NEGOTIABLE)

This enables the model to use both the first frame AND the flickery keyframes to generate smooth video.

---

## Goal

Train a **frame flicker IC LoRA from scratch** for LTX-2 19B base model using **Pexels paired data**.
The LoRA learns to generate smooth videos from sparse keyframe conditioning with pulse mask indicator
AND first-frame conditioning.

This is adapted from `FrameFlickerEnvatoTest` but uses the Pexels dataset processed by Yash instead
of Envato videos.

---

## What is This Training About?

### IC LoRA (In-Context LoRA)
IC LoRA is a specialized LoRA training mode for **video-to-video transformations**. During training:

- **Reference video** (conditioning input): Flickery video showing only keyframes with pulse mask indicator
- **Target video** (ground truth): Smooth original video with pulse mask indicator
- The reference and target latents are **concatenated in sequence** and fed to the transformer
- **Loss is computed only on the target portion** (not the reference)
- This doubles the sequence length, increasing training/inference time

### Frame Flicker Conditioning

For our flicker LoRA specifically:
- **Reference video** = Sparse keyframes (e.g. 1-16 frames) repeated to fill entire video using
  `rp.quantize_to_nearest_values()`, creating a "flickery" effect. Each frame shows the nearest
  keyframe in time. Plus a vertical pulse mask (white bar at keyframe indices, black otherwise).
- **Target video** = The smooth original video with the same pulse mask

At inference time, you give the model keyframes + pulse mask and it generates the smooth interpolated video.

**Higher-level goal**: Build an algorithm that can transform one video into another using just a few
keyframes. The keyframes are transformed versions (e.g. stylized), and the model learns to generate
smooth video hitting those keyframes.

---

## Why Pexels Data (Not Envato)?

The original `FrameFlickerEnvatoTest` used Envato stock videos with random keyframe selection.

This version uses **Yash's Pexels paired data** where:
- `before.png` = Original 16 keyframes in 4x4 grid
- `after.png` = Transformed/stylized 16 keyframes in 4x4 grid
- `raw_video.mp4` = Original smooth video (centralized storage with hardlinks)
- `metadata.json` = Contains `chosen_frame_indices` (which frames are keyframes) and `chosen_frame_timestamps`

**We train on**: `after.png` (transformed keyframes) → `raw_video.mp4` (original smooth video)

**Why**: We want to learn "given these transformed keyframes, generate smooth video that hits them"

---

## CRITICAL BUG LESSON (NEVER FORGET)

**THE CACHE BUG** - `use_cache=True` causes catastrophic failure in parallel processing:

```python
# ❌ WRONG - ALL samples load the SAME cached image from first sample
keyframes = rp.load_image("after.png", use_cache=True)

# ✅ CORRECT - Each sample loads its own image
keyframes = rp.load_image("after.png", use_cache=False)
```

**What happened**: When processing 10,000 samples in parallel, `use_cache=True` caused every single
sample to load the **first sample's after.png** from cache. Result: Wind turbine reference video
matched with office scene, identical reference videos across different samples.

**VLM caught it**: User inspected debug videos and noticed "wind turbine doesn't turn into a man that
transform makes no sense" and "several debug videos have the *same* reference video which *should*
be impossible".

**NEVER use `use_cache=True` in parallel processing. ALWAYS use `use_cache=False`.**

---

## Key Paths

### Source Data
| Item | Path |
|------|------|
| Pexels paired data root | `/root/CleanCode/Sandbox/RP_Dumps/YashDump/jan10_last_50K_pexels_v2/paired_data/` |
| Total samples available | ~50,000 |
| Samples to process | **10,000 only** (user requirement) |
| Sample structure | `{sample_id}/` containing `before.png`, `after.png`, `raw_video.mp4`, `metadata.json` |
| Centralized video storage | `/root/CleanCode/Datasets/Pexels/downloads/{2-char-subdir}/{video_id}.mp4` |

### Sample Structure (Per Directory)
```
8122962_20260111_075835/
├── before.png          # Original 16 keyframes (4x4 grid) - NOT USED
├── after.png           # Transformed 16 keyframes (4x4 grid) - WE USE THIS
├── raw_video.mp4       # Hardlink to centralized storage - WE USE THIS
└── metadata.json       # Contains chosen_frame_indices and chosen_frame_timestamps
```

### metadata.json Structure
```json
{
  "pexels_id": "8122962",
  "chosen_frame_indices": [5, 12, 18, 25, 31, 38, 45, 51, 58, 65, 71, 78, 85, 91, 98, 105],
  "chosen_frame_timestamps": [0.2, 0.48, 0.72, 1.0, 1.24, 1.52, 1.8, 2.04, ...],
  "fps": 25.0,
  ...
}
```

### Models
| Item | Path |
|------|------|
| Base model checkpoint | `/models/LTX2/ltx-2-19b-dev.safetensors` (41 GB) |
| Gemma text encoder | `/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized/` |
| Spatial upscaler (NOT used) | `/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors` |

**We are NOT starting from any existing IC LoRA.** We train from scratch on the base model.

### LTX-2 Training Infrastructure
| Item | Path |
|------|------|
| LTX2 repo root | `/root/CleanCode/Github/VideoVaeTests/LTX2/` |
| Trainer package | `/root/CleanCode/Github/VideoVaeTests/LTX2/src/packages/ltx-trainer/` |
| Training script | `...ltx-trainer/scripts/train.py` |
| Dataset preprocessor | `...ltx-trainer/scripts/process_dataset.py` |
| Inference script | `...ltx-trainer/scripts/inference.py` |
| IC LoRA config template | `...ltx-trainer/configs/ltx2_v2v_ic_lora.yaml` |

### Hardware
- **8x NVIDIA A100-SXM4-80GB** (640 GB total VRAM)
- All GPUs used for training via DDP (Distributed Data Parallel)
- CUDA 12.2, Driver 535.183.01

---

## Output File Structure

Everything lives in this directory:
```
/root/CleanCode/Github/VideoVaeTests/Training/nanobanna_flickertest/
```

### Required Deliverables

```
nanobanna_flickertest/
├── claude_instructions.md          # THIS FILE - manifest/instructions
├── .claude_todo.md                 # TODO list (if using task tracking)
├── datasets/
│   ├── transform.py                # Video pair generator (Pexels adaptation)
│   ├── make_dataset.py             # Dataset orchestration script (runnable with no args)
│   ├── videos/                     # Target videos (flat structure)
│   ├── reference_videos/           # Flickery reference videos (flat structure)
│   ├── test_videos/                # 50 test videos
│   ├── test_reference_videos/      # 50 test references
│   ├── dataset.json                # Training set metadata for LTX trainer
│   ├── test_set.json               # Test set metadata
│   └── .precomputed/               # LTX trainer preprocessed latents
│       ├── latents/
│       ├── conditions/
│       └── reference_latents/
├── configs/
│   ├── frame_flicker_ic_lora.yaml  # Main training config
│   └── accelerate_ddp.yaml         # Accelerate DDP config (8 GPUs)
├── outputs/                        # Training outputs (checkpoints, logs)
├── inference_outputs/              # Validation inference results
├── .scratchwork/                   # Hidden scratch folder (safe to delete)
└── .gitignore                      # Ignore generated artifacts
```

### Rules
- **DO NOT modify any files in the LTX2 repository** - use it via sys.path or CLI only
- All scripts must be **runnable with no arguments**
- Scratch files go in `.scratchwork/` only
- Keep file count minimal
- Everything self-contained for copy-paste reusability

---

## Detailed Task Breakdown

### Task 1: `datasets/transform.py` - Video Pair Generator

**Purpose**: Generate reference/target video pairs from a single Pexels sample directory.

**Algorithm**:
1. Load `metadata.json` to get `chosen_frame_indices` (e.g. [5, 12, 18, 25, ...])
2. Load `after.png` with **`use_cache=False`** (CRITICAL!)
3. Split into 4x4 grid (16 keyframes)
4. Load `raw_video.mp4`
5. Random temporal crop to 121 frames (with speedup adjustment based on fps)
6. Find which keyframes fall in temporal window
7. Shift keyframe indices to be relative to crop start
8. Create flickery reference using `rp.quantize_to_nearest_values(range(121), shifted_key_indices)`
9. Create pulse mask (white bar at keyframe indices, black elsewhere)
10. Resize both videos to target dimensions (preserving aspect ratio)
11. Concatenate pulse mask + video content horizontally
12. Save reference.mp4 and target.mp4

**Key differences from Envato version**:
- Input: Sample directory (not video file)
- Keyframes: Pre-determined from metadata (not random selection)
- Temporal window: Random crop within video (Envato didn't need this)
- Image loading: **MUST use `use_cache=False`**

### Task 2: `datasets/make_dataset.py` - Dataset Orchestration

**Purpose**: Orchestrate the full pipeline from Pexels samples to training-ready dataset.

**Algorithm**:
1. Read all sample directories from Pexels paired_data/
2. Random sample 10,000 directories (seed=42 for reproducibility)
3. Split: 9,950 train, 50 test
4. Create output directories
5. For train split:
   - Run transform.py on each sample in parallel (32 workers)
   - Skip already-processed samples (idempotent)
   - Show progress with tqdm
6. For test split: same process
7. Generate `dataset.json` with entries: `{"caption": "...", "media_path": "videos/{id}.mp4", "reference_path": "reference_videos/{id}.mp4"}`
8. Generate `test_set.json` with similar structure
9. Call LTX `process_dataset.py` on 8 GPUs in parallel to precompute latents

**Supports `--test` flag**: Limit to 24 samples for quick testing

**Key differences from Envato version**:
- Source: Pexels sample directories (not Envato CSV)
- Selection: Random 10K from 50K (not filtering existing videos)
- No caption CSV (Pexels samples don't have captions - we'll use placeholder or None)

### Task 3: `configs/frame_flicker_ic_lora.yaml` - Training Config

Copy from FrameFlickerEnvatoTest template, update paths:
- **Line 52**: `preprocessed_data_root: "/root/CleanCode/Github/VideoVaeTests/Training/nanobanna_flickertest/datasets/.precomputed"`
- **Line 94**: `output_dir: "/root/CleanCode/Github/VideoVaeTests/Training/nanobanna_flickertest/outputs"`

Keep all other settings:
- LoRA rank 128, learning rate 2e-5
- 100,000 training steps
- Resolution: 512x320x121 frames
- Checkpoint interval: 50 steps
- Keep all checkpoints (keep_last_n: -1)

### Task 4: `train.sh` - Training Launcher

Minimal shell script that launches training via accelerate:
- All 8 GPUs via DDP
- Uses `configs/accelerate_ddp.yaml`
- Calls `uv run accelerate launch scripts/train.py`
- Portable paths using `git rev-parse --show-toplevel`

### Task 5: `validate.sh` - Inference Script

Shell script that:
- Finds latest checkpoint
- Runs inference on 8 test videos (one per GPU in parallel)
- Uses LTX `scripts/inference.py` with `--lora-path` and `--reference-video`
- Saves outputs to `inference_outputs/{step_name}/`
- Includes reference in output for side-by-side comparison

---

## Resolution & Frame Constraints (LTX-2 Requirements)

- **Spatial**: width and height must be divisible by 32
- **Temporal**: frame count must satisfy `F % 8 == 1` (valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121)
- **Training resolution**: 512x320x121 frames at 25 fps
- **Sequence length formula**: `seq_len = (H/32) * (W/32) * ((F-1)/8 + 1)`

---

## Frame-Perfect Synchronization (CRITICAL)

**BOTH reference and target videos MUST have**:
- Identical resolution (width and height)
- Identical frame count (exactly 121 frames)
- Identical fps (25 fps)
- Frame-perfect temporal alignment (frame N in target = frame N in reference)

**Implementation**:
- Load raw_video.mp4 and randomly crop to 121 frames
- Apply same fps (25) to both reference and target
- Both videos created from same temporal window
- Only difference: reference uses quantized keyframes, target uses smooth video

---

## Bulldog Babysit Mode (CRITICAL)

User explicitly requested **bulldog mode** and **babysit mode**. These are NON-NEGOTIABLE.

### Bulldog Mode 🐕
- **Persistent and tenacious** - don't give up on errors
- **Keep trying until it works** - retry with different approaches
- **Don't stop halfway** - see tasks through to completion
- **Bite down and hold** - stay focused on the goal

### Babysit Mode 👁️
- **Monitor continuously** - watch for errors, crashes, anomalies
- **Methodical and slow** - verify each step before moving to next
- **Ultra careful** - double/triple check everything
- **VLM verification** - visually inspect outputs at multiple stages
- **Restart on crash** - never leave things broken
- **Verbose logging** - report what's happening at every step
- **Subject consistency checks** - verify data integrity
- **No silent failures** - raise all errors loudly

**State "BULLDOG BABYSIT MODE" at the top of every response.**

---

## VLM Verification Strategy (CRITICAL)

User explicitly requested **ultra careful VLM verification of 50+ samples**.

### Test Phase (24 samples)
After generating 24 test samples:
1. Select 10+ samples for inspection
2. Extract first, middle (frame 60), last (frame 120) frames from each
3. VLM check:
   - Subject consistency within video (same scene/subject)
   - No duplicate reference videos across samples
   - Flicker pattern visible in reference
   - Pulse mask correct (white at keyframes, black elsewhere)
   - Target is smooth (no flickering)
   - Reference/target show same content

### Full Phase (10,000 samples)
After generating full dataset:
1. Sample 55 videos **evenly distributed** (not just first 55)
2. Sampling strategy: `np.linspace(0, 9999, 55, dtype=int)`
3. Extract first/middle/last frames
4. VLM check ALL criteria from test phase
5. Document any anomalies
6. **CRITICAL**: Check for duplicate reference videos with different indices
7. **CRITICAL**: Ensure wind turbine doesn't transform into office scene (subject consistency)

### When to VLM Check
- After test run (10+ samples)
- After full dataset generation (55 samples)
- After first checkpoint inference (first/middle/last of generated videos)
- Any time something looks suspicious

---

## Verbose Progress Output (CRITICAL)

**All scripts must print verbose progress to stdout.** Never suppress output.

### `datasets/make_dataset.py` must print:
- How many total samples found
- How many selected for processing
- Train/test split counts
- For each video: progress bar with tqdm
- Summary: total processed, total skipped
- When calling LTX process_dataset.py, let its output flow through

### `train.sh` must:
- Print config paths
- Print GPU assignment
- Print command being launched
- Never pass `--disable-progress-bars`
- Let LTX trainer print step-by-step progress

### `validate.sh` must print:
- Which checkpoint is being used
- Which test videos are being processed
- Per-video progress
- Where outputs were saved

### General rules:
- **Never suppress stdout/stderr** from subprocesses
- **Never use `> /dev/null`** or `2>/dev/null`
- Use tqdm progress bars for all long operations
- Print clear banners at phase transitions

---

## User Requirements (Verbatim)

These are the explicit instructions from the user. ALL must be followed:

1. **Only 10,000 samples** from the 50K Pexels dataset - don't process all 50K
   - **CRITICAL**: NUM_SAMPLES must be a prominent variable at the TOP of make_dataset.py
   - **CRITICAL**: To change from 10K to 50K, user should only need to change ONE CHARACTER: "10" → "50"
   - Script should NOT require command-line arguments for sample count
   - Make it obvious and easy to find
2. **Use after.png** (transformed keyframes), not before.png (original keyframes)
3. **Ultra careful about parallelism bugs** - no cache bugs like before
4. **VLM verify 50+ samples** - check first, middle, last frames for subject consistency
5. **No duplicate videos with different indices** - each sample must be unique
6. **Subject consistency** - wind turbine shouldn't transform into office scene
7. **Babysitting mode** - methodical, slow, verify everything, many hours available
8. **All work isolated in training folder** - no touching other files
9. **Self-contained and copy-pasteable** - everything including debug videos in this folder
10. **Test with 24 samples first** (`--test` flag)
11. **Then run full 10K dataset** after test verification passes
12. **Train from scratch** - no existing LoRA weights
13. **All 8 GPUs for training** via DDP
14. **Built-in validation** generates videos every N steps via trainer
15. **50 videos for test split**, rest for training
16. **Base model only** (one-stage pipeline, ~480p) - NOT training the upscaler
17. **Inference uses latest checkpoint** as it's saved during training
18. **Monitor continuously** - restart on crash
19. **After first checkpoint**: run inference, VLM review, notify user
20. **Absolute minimum code** - call existing LTX scripts via shell wherever possible
21. **Paths at top** - all path definitions at very top of every file
22. **Portable paths** - use `git rev-parse --show-toplevel` so project is movable
23. **Flat dataset structure** - no deep nesting in datasets/
24. **Re-runnable scripts** - skip already-processed samples
25. **Checkpoint frequently** - every 50 steps, keep all checkpoints
26. **121 frames** at 25 fps (8*15+1 frames)
27. **Never use `use_cache=True`** in parallel processing (cache bug lesson)
28. **Document everything** in claude_instructions.md for future Claude sessions
29. **Create giant todo list** so I don't forget things mid-plan
30. **Higher-level goal**: Train algorithm that transforms videos using keyframes
31. **Periodically read claude_instructions.md** - Re-read this document periodically to avoid
    forgetting requirements when conversation compacts. This document is the source of truth.
32. **NUM_SAMPLES easily changeable** - Put NUM_SAMPLES = 10  # Change to 50 for full dataset
    at the TOP of make_dataset.py so user can change "10" to "50" with one character edit

---

## Important Notes

- **NO silent failures** - all errors must be raised loudly
- Use **progress bars** (tqdm) for all long operations
- Use **einops** for tensor reshaping
- Keep code **simple and minimal**
- **Do NOT modify the LTX2 repository** - use it via sys.path or CLI only
- **Do NOT start from existing LoRA weights** - train from scratch
- IC LoRA strength must be **1.0** at inference (required by architecture)
- Videos have **no audio** - set `generate_audio: false`
- **Pexels samples don't have captions** - we'll need to handle this (placeholder captions or None)
- **Resolution: 512x320x121** to match FrameFlickerEnvatoTest
- **Speedup factor**: Raw Pexels videos are ~50fps, we train at 25fps, so 2x speedup

---

## Captions Strategy - CRITICAL REQUIREMENT ⚠️

**CAPTIONS ARE NOT OPTIONAL** - They are in the metadata!

**Source**: `metadata.json['video_caption'][0]` contains detailed video descriptions

**Example caption**:
```
"A young Black woman, with dark skin and dark hair styled in an updo with braids,
is seen sitting on a light-colored rug on a shiny wooden floor, intensely focused
on a black laptop..."
```

**Implementation**:
- Load metadata.json from sample directory
- Extract `meta['video_caption'][0]` (first element of list)
- Index 1 is the caption source (e.g. "koichi"), we don't use that
- Fallback to "A video" only if metadata missing or corrupted

**This is NOT optional** - captions improve training quality significantly.

---

## If You Get Stuck

1. **Read the LTX trainer docs**: `...ltx-trainer/docs/` has comprehensive guides
2. **Read the AGENTS.md**: `...ltx-trainer/AGENTS.md` has AI-specific guidance
3. **Check the example configs**: `...ltx-trainer/configs/ltx2_v2v_ic_lora.yaml`
4. **Look at FrameFlickerEnvatoTest**: The template this is based on
5. **Look at ZoomOutEnvatoTest**: Has detailed requirements from previous Claude
6. **Check generate_training_samples.py**: In YashDump/preview_code/ - similar logic
7. **Run research frenzy** if truly stuck
8. **Remember the cache bug** - always use_cache=False

---

## Success Criteria

1. ✅ Directory structure created
2. ✅ All source files copied and adapted
3. ✅ Test run with 24 samples succeeds
4. ✅ VLM verification of test samples passes (no duplicates, subject consistency)
5. ✅ Full 10K dataset generated successfully
6. ✅ VLM verification of 5 full samples passes (document paths + assessments in concerns.md)
7. ✅ Training launches on 8 GPUs without OOM
8. ✅ First checkpoint saved successfully
9. ✅ Inference runs on first checkpoint
10. ✅ VLM review of generated videos shows progress
11. ✅ Documentation complete (README, .gitignore)
12. ✅ Everything self-contained and copy-pasteable

---

## VLM Verification & Launch Protocol

### Full Dataset Verification (5 Samples)

After full dataset generation completes:

1. **Randomly select 5 samples** from the generated dataset
2. **VLM inspect** both reference and target videos for each sample
3. **Document in concerns.md**:
   - Exact paths (reference_videos/X.mp4 and videos/X.mp4)
   - Assessment for each sample (pass/fail with reasoning)
   - Overall verdict
4. **Action**:
   - If obvious fixable bug found → fix it, regenerate if needed
   - Otherwise → **LAUNCH TRAINING** (don't let perfect be enemy of good)

**Philosophy**: Quick verification to catch obvious issues, but maintain forward momentum.
Document observations for user review, but don't block on minor concerns.

### User Notification Points (ONLY THESE 3 TIMES)

**User wants to be notified ONLY at these specific moments:**

1. **When latent precomputation starts** (GPU processing: VAE encoding + text encoding)
2. **When training starts** (8 GPU DDP training launch)
3. **When first checkpoint saves** (proof training is working)

**NO notifications for**: dataset generation progress, VLM verification, ongoing babysitting, etc.
Document everything in concerns.md, but only interrupt user at the 3 key moments above.

### Training Babysitting Protocol

**Phase 1: Until First Checkpoint (frequent checks)**
- Monitor every 60 seconds
- Watch for: process running, outputs directory, checkpoint creation
- When first checkpoint detected → **notify user** (one of the 3 notification points), transition to Phase 2

**Phase 2: After First Checkpoint (hourly checks)**
- Monitor every 1 hour
- Watch for: process still running, no crashes in logs, checkpoints continuing to save
- Document any issues in concerns.md
- **No user notification** unless critical issue found (crash, OOM, etc.)

**Purpose**: Catch crashes or training issues that might occur hours into training.
Bulldog mode = autonomous monitoring with smart backoff.

---

## Timeline Estimate

- Setup & file creation: 10-20 minutes
- Test dataset generation (24 samples): 5-10 minutes
- Test VLM verification: 10-15 minutes
- Full dataset generation (10K samples): 1-3 hours
- Full VLM verification (5 samples): 5-10 minutes
- Latent precomputation (8 GPUs): 2-4 hours
- First checkpoint (50 steps): 30-60 minutes
- Total to first checkpoint: ~6-10 hours

**User has given me "all night" and "many hours" - take my time and be methodical.**

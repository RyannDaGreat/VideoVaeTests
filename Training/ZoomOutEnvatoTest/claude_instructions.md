# Zoom-Out IC LoRA Training Manifest

## Goal

Train a **zoom-out IC LoRA from scratch** for LTX-2 19B base model. The LoRA learns to generate
videos that appear to zoom out from a cropped/zoomed-in input. This is a **test run** to verify
that the LTX-2 training pipeline works correctly with our Envato dataset.

---

## What is an IC LoRA?

IC LoRA (In-Context LoRA) is a specialized LoRA training mode for **video-to-video transformations**.
Unlike standard LoRAs that learn aesthetics/styles, IC LoRA learns to transform a **reference video**
into a **target video**. During training:

- **Reference video** (conditioning input): provided clean/unnoised, representing the "before" state
- **Target video** (ground truth): noised during training, representing the desired "after" state
- The reference and target latents are **concatenated in sequence** and fed to the transformer
- **Loss is computed only on the target portion** (not the reference)
- This doubles the sequence length, increasing training/inference time

For our zoom-out LoRA specifically:
- **Reference video** = original video zoomed in by 2x, then center-cropped to original resolution
- **Target video** = the original video as-is (the "zoomed out" result)

At inference time, you give the model a zoomed-in video and it generates the zoomed-out version.

---

## Why Base Model Only (Not Two-Stage Upscaler)

LTX-2 has a two-stage pipeline architecture:
- **Stage 1 (base model)**: Generates video at a target resolution (e.g. 512x768) using the full
  19B transformer with CFG guidance over ~40 denoising steps
- **Stage 2 (upscaler)**: Takes stage 1 output, upsamples 2x spatially using a spatial upscaler
  model + distilled LoRA for refinement at higher resolution

We are training **ONLY for Stage 1 (base model)** because:
1. The Envato dataset videos are ~480p (960x540 preview resolution), not 4K
2. Training the upscaler requires high-resolution paired data we don't have
3. This is a proof-of-concept test run
4. The IC LoRA conditioning happens at the base model level
5. Resolution constraints: one-stage requires divisible by 32; two-stage requires divisible by 64

Our inference script must use the **one-stage pipeline** (not two-stage) with our trained LoRA.

---

## Key Paths

### Source Data
| Item | Path |
|------|------|
| Envato dataset root | `/root/CleanCode/Datasets/Envato/` |
| CSV with 3.87M captions | `/root/CleanCode/Datasets/Envato/captioned_envato_3869336.csv` |
| Downloaded raw videos | `/root/CleanCode/Datasets/Envato/downloads/raw_videos/` |
| Video path format (relative) | `VP/2H/VP2HWMT.mp4` (2-letter hierarchical prefixes) |

### CSV Columns (17 total)
`id, date, framerate, duration, looped, has_alpha, height, width, resolution, video_url, author, category, subcategory, title, description, caption, videos`

- `caption` = detailed text description of video content
- `videos` = relative path to video file (e.g. `VP/2H/VP2HWMT.mp4`)
- Full video path = `/root/CleanCode/Datasets/Envato/downloads/raw_videos/{videos_column}`

### Dataset Status
- CSV has **3,869,336** rows total
- Only **~8,283** videos have been downloaded so far (~0.2%)
- Downloaded videos are **960x540** (preview resolution, not 4K)
- Many CSV rows will NOT have a corresponding video file - **the script must filter to only rows where the video file actually exists on disk**
- The dataset creation script must be **re-runnable** as more videos get downloaded later

### Models
| Item | Path |
|------|------|
| Base model checkpoint | `/models/LTX2/ltx-2-19b-dev.safetensors` (41 GB) |
| Gemma text encoder | `/models/LTX2/gemma-3-12b-it-qat-q4_0-unquantized/` |
| Spatial upscaler (NOT used for training) | `/models/LTX2/ltx-2-spatial-upscaler-x2-1.0.safetensors` |
| Distilled LoRA (NOT used) | `/models/LTX2/ltx-2-19b-distilled-lora-*.safetensors` |

**We are NOT starting from any existing IC LoRA.** We train from scratch on the base model.

### LTX-2 Training Infrastructure
| Item | Path |
|------|------|
| LTX2 repo root | `/root/CleanCode/Github/VideoVaeTests/LTX2/` |
| Trainer package | `/root/CleanCode/Github/VideoVaeTests/LTX2/src/packages/ltx-trainer/` |
| Training script | `...ltx-trainer/scripts/train.py` |
| Dataset preprocessor | `...ltx-trainer/scripts/process_dataset.py` |
| Reference video generator | `...ltx-trainer/scripts/compute_reference.py` |
| Inference script | `...ltx-trainer/scripts/inference.py` |
| IC LoRA config template | `...ltx-trainer/configs/ltx2_v2v_ic_lora.yaml` |
| DDP accelerate config | `...ltx-trainer/configs/accelerate/ddp.yaml` |
| FSDP accelerate config | `...ltx-trainer/configs/accelerate/fsdp.yaml` |
| Training docs | `...ltx-trainer/docs/` (quick-start, training-modes, dataset-preparation, etc.) |

### Existing Inference Reference Scripts
| Item | Path |
|------|------|
| One-stage T2V test | `/root/CleanCode/Github/VideoVaeTests/untracked/T2VTest/t2v.py` |
| Two-stage canny IC LoRA test | `/root/CleanCode/Github/VideoVaeTests/untracked/T2VTest/t2v_2stage_canny_nodistill.py` |
| IC LoRA pipeline module | `...ltx-pipelines/src/ltx_pipelines/ic_lora.py` |

### Hardware
- **8x NVIDIA A100-SXM4-80GB** (640 GB total VRAM)
- All GPUs currently idle
- CUDA 12.2, Driver 535.183.01
- Plan: **ALL 8 GPUs for training** (NON-NEGOTIABLE - do not waste resources)
- LTX trainer handles validation internally every N steps (no separate inference during training)
- Inference runs AFTER training or between runs, not in parallel

---

## Output File Structure

Everything lives in this directory:
```
/root/CleanCode/Github/VideoVaeTests/Training/ZoomOutEnvatoTest/
```

### Required Deliverables (ONLY these files at the top level)

```
ZoomOutEnvatoTest/
|-- claude_instructions.md          # THIS FILE - manifest/instructions
|-- .claude_todo.md                 # Synced TODO list
|-- prepare_dataset.py              # Dataset creation script (runnable with no args)
|-- train.sh                        # Training launch script (runnable with no args)
|-- inference.sh                    # Inference launch script (runnable with no args)
|-- datasets/                       # All dataset artifacts
|   |-- dataset.csv                 # CSV for LTX trainer (caption, media_path, reference_path)
|   |-- videos/                     # Symlinks or copies of target videos (flat structure)
|   |-- reference_videos/           # Generated zoom-in conditioning videos (flat structure)
|   |-- test_videos/                # 50 videos held out for testing
|   |-- test_reference_videos/      # Corresponding zoom-in versions for test set
|   |-- .precomputed/               # LTX trainer preprocessed latents (generated by prepare_dataset.py)
|   |   |-- latents/
|   |   |-- conditions/
|   |   |-- reference_latents/
|-- configs/                        # Training YAML configs
|   |-- zoom_out_ic_lora.yaml       # Main training config (static, not auto-generated)
|   |-- accelerate_ddp.yaml         # Accelerate DDP config (copy of LTX official, num_processes=8)
|-- outputs/                        # Training outputs (checkpoints, validation samples, logs)
|-- inference_outputs/              # Periodic inference results during training
|-- .scratchwork/                   # Hidden scratch folder (safe to delete, nothing critical)
```

### Rules
- **DO NOT modify any files in the LTX2 repository** - use it via sys.path or CLI only
- All scripts must be **runnable with no arguments** (`python prepare_dataset.py`, `bash train.sh`)
- Scratch files go in `.scratchwork/` only
- Keep file count minimal - no extra scripts lying around
- The `prepare_dataset.py` must be **idempotent/re-runnable** (skip existing, handle new videos)

### ABSOLUTE MINIMUM CODE (CRITICAL RULE)

**Write the absolute minimum amount of custom code.** If we can accomplish something by
calling an existing LTX script via shell, DO THAT instead of writing Python. The LTX trainer
already provides:
- `scripts/process_dataset.py` - preprocesses videos into latents
- `scripts/compute_reference.py` - generates reference videos (we adapt its approach)
- `scripts/train.py` - runs training
- `scripts/inference.py` - runs inference with LoRA

**Prefer shell scripts calling existing tools over custom Python code.** The only custom code
we should write is the bare minimum that doesn't already exist:
1. Filtering the Envato CSV to videos that exist on disk
2. Generating the zoom-in reference videos (since compute_reference.py does Canny, not zoom)
3. Producing the dataset CSV/JSON in the format process_dataset.py expects

Everything else (preprocessing, training, inference) should be done by calling the LTX
scripts directly. Less code = less to review = fewer bugs = faster iteration.

**If `train.sh` can just be a few lines calling `accelerate launch`, that's ideal.**
**If `inference.sh` can just call `scripts/inference.py` with the right flags, do that.**

---

## Detailed Task Breakdown

### Task 1: `prepare_dataset.py` - Dataset Creation Script

This single script does everything needed to go from raw Envato data to training-ready dataset:

1. **Read the Envato CSV** from `/root/CleanCode/Datasets/Envato/captioned_envato_3869336.csv`
2. **Filter to rows where video file actually exists** at
   `/root/CleanCode/Datasets/Envato/downloads/raw_videos/{videos_column}`
3. **Skip videos that are too short** (need at least enough frames for training, ~2 seconds minimum)
4. **Split into train/test**: randomly hold out **50 videos** for testing, rest for training
5. **For each video, create the zoom-in reference video**:
   - Load video at native resolution (likely 960x540)
   - Zoom in by 2x (scale up 2x) and center-crop back to original resolution
   - This simulates what a "zoomed in" version looks like - the training teaches the model to reverse this
   - Save as MP4 in `datasets/reference_videos/` (flat naming: `{video_id}.mp4`)
   - Similarly for test set into `datasets/test_reference_videos/`
6. **Create symlinks or copies** of the target videos into `datasets/videos/` (flat naming)
7. **Generate `datasets/dataset.csv`** for LTX trainer with columns:
   - `caption` - from the Envato CSV caption column
   - `media_path` - path to target video in `datasets/videos/`
   - `reference_path` - path to reference (zoom-in) video in `datasets/reference_videos/`
8. **Run LTX trainer preprocessing** (`process_dataset.py`) to compute:
   - Video latents (target)
   - Reference latents (conditioning)
   - Text embeddings (conditions)

Key details:
- Resolution bucket for preprocessing: choose based on Envato video resolution (~480p).
  Since videos are 960x540, a good bucket would be `544x320x49` or `512x320x49` (both dims
  divisible by 32, frames satisfying `F % 8 == 1`). Or `480x288x49`. Pick something that doesn't
  require massive upscaling of 960x540 content. The zoom-in reference will be the center crop
  after 2x zoom, so it will show the center 480x270 region of each frame.
- Frame count: use 49 frames (8*6+1) for a short clip, or 25 (8*3+1) for faster training
- Must show progress bars during all long operations
- Must be re-runnable: skip already-processed videos, only process new ones

### Task 2: `train.sh` - Training Script

Minimal shell script (~10 lines) that launches training via accelerate CLI flags (no config file):

```bash
cd $LTX_TRAINER
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run accelerate launch \
    --config_file $ACCEL_CFG \
    scripts/train.py $TRAIN_CFG
```

Uses `configs/accelerate_ddp.yaml` (copy of official LTX `ddp.yaml`, `num_processes: 8`)
passed via `--config_file` with an **absolute path** (since we `cd` into the trainer dir).
The default accelerate config at `~/.cache/huggingface/accelerate/` is deleted on each
launch to prevent interference.

All 8 GPUs are used for training. The LTX trainer handles validation internally
(generates validation videos every 500 steps using the reference videos and prompts
specified in the config). No separate inference script is needed during training.

### Task 3: `inference.sh` - Periodic Inference Script

This script runs in a **separate tmux pane** alongside training. It uses **10-20 test videos**
(from the 50-video test split) with their real Envato captions as prompts.

**GPU split**: GPU 6 handles half the test videos, GPU 7 handles the other half (in parallel).

**Behavior**:
1. Monitors the training output directory for new checkpoints
2. Every time a new checkpoint appears, runs inference on both GPUs:
   - GPU 6: test videos 1-10 (or 1-5 if doing 10 total)
   - GPU 7: test videos 11-20 (or 6-10 if doing 10 total)
3. Each GPU loads the base model + latest LoRA checkpoint
4. Feeds the zoom-in reference video as conditioning, with the real caption as prompt
5. Generates at the same resolution as training (one-stage pipeline only)

**Output naming** - files must be clearly named so you know exactly what they are:
```
inference_outputs/
  step_00500/
    gpu6_vid01_{video_id}.mp4
    gpu6_vid01_{video_id}_frames.png    # first/middle/last composite
    gpu6_vid02_{video_id}.mp4
    ...
    gpu7_vid11_{video_id}.mp4
    ...
```

**Implementation**: Call the LTX trainer's `scripts/inference.py` directly with `--lora-path`,
`--reference-video`, and appropriate flags. Do NOT rewrite inference logic in custom Python.
The key difference from stock inference is that we use one-stage only (no upscaler).

The script should loop forever, sleeping between checks, and pick up new checkpoints as they
appear. Use the existing `scripts/inference.py` CLI - it already supports `--lora-path` and
`--reference-video` flags.

### Task 4: Babysitting & Monitoring

After launching both training and inference:

1. **Babysit continuously** - monitor training progress, check for crashes
2. **After first checkpoint**, verify inference produces an MP4
3. **After first successful video generation**:
   - Use VLM to examine first, middle, and last frames
   - Generate a report on video quality
   - **Notify me**: `MSG='First zoom-out IC LoRA video generated!' ; say "$MSG" ; rp call ntfy_send --- "$MSG"`
4. Only notify AFTER manually confirming the MP4 exists and looks reasonable
5. Continue babysitting - do NOT stop even if first results look bad
6. If anything crashes, restart it immediately

---

## Resolution & Frame Constraints (LTX-2 Requirements)

- **Spatial**: width and height must be divisible by 32
- **Temporal**: frame count must satisfy `F % 8 == 1` (valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121)
- **Sequence length formula**: `seq_len = (H/32) * (W/32) * ((F-1)/8 + 1)`
- Keep sequence length reasonable (<4096) to fit in memory with IC LoRA (which doubles it)

## Zoom-In Reference Video Generation Logic

For each source video at resolution WxH:

**Simplest approach** (preferred):
1. For each frame, crop the center region of size (W/2 x H/2)
2. Resize that crop back up to WxH using bilinear/bicubic interpolation
3. This is a 2x zoom: each dimension is halved then scaled back up
4. By area it's 25% of the original frame, by width/height it's 50% each side

**Equivalent approach**:
1. Scale up each frame by 2x to (2W)x(2H)
2. Center-crop back to WxH

Both are mathematically identical. Use the simpler one (crop then resize).
The model learns: given this zoomed-in view, generate the full zoomed-out view.

### CRITICAL: Frame-Perfect Synchronization

**BOTH target and reference videos MUST be pre-processed by the SAME pipeline** to guarantee:
- Identical resolution (same width and height)
- Identical frame count (exactly 121 frames each)
- Identical fps (25 fps, LTX-2's default)
- Frame-perfect temporal alignment (frame N in target = frame N in reference)

**Do NOT symlink raw Envato videos as targets** - they may have different fps, extra frames,
or different resolution than the reference. Instead, process BOTH through ffmpeg with identical
`-frames:v 121 -r 25` settings. The only difference is the reference gets the crop+resize filter.

### Frame Rate

LTX-2 operates at **25 fps** by default (frame_rate: 25.0 in config).
121 frames at 25fps = **4.84 seconds** of video.
The minimum source video duration should be ~5 seconds to ensure we have enough frames.

---

## If You Get Stuck

1. **Read the LTX trainer docs**: `...ltx-trainer/docs/` has comprehensive guides
2. **Read the AGENTS.md**: `...ltx-trainer/AGENTS.md` (also symlinked as CLAUDE.md) has AI-specific guidance
3. **Check the example configs**: `...ltx-trainer/configs/ltx2_v2v_ic_lora.yaml` is the IC LoRA template
4. **Look at existing inference scripts**: `untracked/T2VTest/` has working examples
5. **Run a research frenzy** if truly stuck - search web and codebase in parallel
6. **Check the trainer source code**: `...ltx-trainer/src/ltx_trainer/` for implementation details
7. **The compute_reference.py script** shows how Lightricks generates reference videos (Canny edges by default) - adapt the concept for zoom-in

---

## Verbose Progress Output (CRITICAL)

**All scripts must print verbose progress to stdout.** The user must be able to see what's
happening at every step by watching the console. Specifically:

### `prepare_dataset.py` must print:
- How many CSV rows were loaded
- How many videos exist on disk (after filtering)
- How many were skipped (missing file / too short / already processed)
- For each video being processed: print the video ID and what step it's on
  (e.g. `[1234/8000] Processing VP2HWMT - creating zoom-in reference...`)
- tqdm progress bars for the overall loop AND for any sub-operations (ffmpeg, encoding)
- Summary at end: total processed, total skipped, train/test split counts
- When calling the LTX `process_dataset.py` preprocessor, do NOT suppress its output -
  let it print its own progress (it uses tqdm internally)

### `train.sh` must:
- Print the config paths being used
- Print the GPU assignment
- Print the command being launched
- **Do NOT pass `--disable-progress-bars`** to the trainer - we WANT progress bars
- The LTX trainer already prints step-by-step loss, validation, and checkpoint info via
  Rich and tqdm - let all of that flow through to stdout

### `inference.sh` must print:
- Which checkpoint it found and is loading
- Which GPU it's using
- Which test videos it's generating
- Per-video progress (tqdm or print statements)
- Where output files were saved

### General rules:
- **Never suppress stdout/stderr** from subprocesses
- **Never use `> /dev/null`** or `2>/dev/null` on any subprocess
- When calling LTX scripts via subprocess, use `subprocess.run()` without `capture_output=True`
  so output streams directly to console
- Print a clear banner/header at the start of each major phase
- Print timing info (how long each phase took)

---

## User Instructions (Verbatim Requirements)

These are the explicit instructions from the user. ALL must be followed:

1. **Train from scratch** - do NOT continue from any existing LoRA weights. Base model only.
2. **Envato dataset** has ~4M rows but only ~8K downloaded. Use only what exists on disk.
3. **Dataset script must be re-runnable** - when more videos are downloaded later, running
   `prepare_dataset.py` again should pick up new videos and process only those.
4. **All scripts runnable with no arguments** - `python prepare_dataset.py`, `bash train.sh`
5. **Periodically prune trash** - don't leave garbage files lying around. Clean up after yourself.
6. **All changes exclusively in this directory** - never modify the LTX2 repo.
7. **Minimal files** - use off-the-shelf components, don't reinvent the wheel.
8. **Flat dataset structure** inside `datasets/`
9. **One script to prepare datasets** (`prepare_dataset.py`) - the ONLY script for dataset creation
10. **One shell script to train** (`train.sh`)
11. **Scratch work goes in `.scratchwork/`** (hidden, safe to delete)
12. **Train with ALL 8 GPUs** via DDP (accelerate CLI flags, no config file)
13. **Built-in validation** generates videos every 500 steps via the trainer's validation config.
    No separate inference script needed during training.
14. **50 videos for test split**, all the rest for training
15. **Use captions from the CSV** as-is (the `caption` column)
16. **Zoom-out logic**: reference = original video zoomed in 2x and center-cropped to original
    resolution. Target = original video. The LoRA learns to "zoom out" from the reference.
17. **Base model only** (one-stage pipeline, ~480p) - we are NOT training the upscaler
18. **Inference script must work with our trained LoRA checkpoints** as they are saved during training
19. **Babysit continuously** - monitor training, restart on crash
20. **After first successful video generation**: VLM review of first/middle/last frames, then notify
21. **Notify ONLY after manual confirmation** that the MP4 exists and looks reasonable:
    `MSG='First zoom-out IC LoRA video generated!' ; say "$MSG" ; rp call ntfy_send --- "$MSG"`
22. **Videos have no audio** - that's fine, we don't care about audio
23. **If stuck, run a research frenzy** - but there should be plenty of documentation in the LTX repo
24. **Verbose progress output everywhere** - print what you're doing, show tqdm bars, never suppress output
25. **ABSOLUTE MINIMUM CODE** - call existing LTX scripts via shell wherever possible. Do NOT
    rewrite Python code that already exists. The only custom code should be: filtering the CSV
    to existing videos, generating zoom-in reference videos, and producing the dataset metadata.
    Everything else (preprocessing, training, inference) calls existing LTX tools directly.
    Less code = less to review = fewer bugs. This is a VERY important rule.
26. **100,000 training steps** - this is a long run. Non-negotiable.
27. **Save checkpoints frequently** - every ~10-30 minutes. We have unlimited disk space,
    so keep ALL checkpoints (`keep_last_n: -1`). Start with interval 500 and adjust based
    on how fast steps are.
28. **The task is non-negotiable**: zoom-out IC LoRA. Reference = zoomed-in video, target =
    original video. Checkpoints must be produced. Example videos must be generated during
    training (via the validation config). This is not optional.
29. **Validation prompts must be REAL captions** from the test split, not made-up prompts.
    Pick 4-6 from the 50-video test set for the trainer's built-in validation.
30. **Inference uses 10-20 test videos** split across GPU 6 and GPU 7 (half each, in parallel).
    Output filenames must clearly identify the checkpoint step and video ID.
31. **Train with 121 frames** (8*15+1). These are big A100 80GB GPUs, use them.
32. **Stay in the LTX2 conda environment** (`conda activate LTX2`). It has everything needed.
    If something is missing, add it to the requirements.txt at the top of VideoVaeTests/ and
    install via `uv pip install -r requirements.txt`.
33. **Existing inference scripts** throughout the codebase can be referenced if stuck (e.g.
    `untracked/T2VTest/` scripts). But the LTX trainer's `scripts/inference.py` should suffice.
34. **Notify after first significant trained MP4** - not just any test, but a video that actually
    came from a trained checkpoint.
35. **NO GLOBAL PATHS** (except for the Envato dataset which lives outside the repo). All paths
    must be derived from `git rev-parse --show-toplevel` so the project is portable. Models
    live at `{repo_root}/LTX2/models/`, trainer at `{repo_root}/LTX2/src/packages/ltx-trainer/`.
    This is **NON-NEGOTIABLE** - the folder must be movable without breaking things.
36. The entire task (zoom-out IC LoRA, 100K steps, checkpoints, example videos) is
    **NON-NEGOTIABLE**. Do not change the task, do not skip steps, do not reduce scope.
37. **Path convention**: ALL path definitions go at the VERY TOP of every file (Python and shell).
    In Python: import subprocess/sys first, get git toplevel, define all paths, THEN do remaining
    imports. In shell: define HERE and REPO_ROOT first, then all paths. This matches the convention
    used in the user's notebooks. Paths must be easy to refactor if the project moves.
38. **Babysit BOTH training AND inference** simultaneously. Both run in separate tmux panes and
    both must be monitored. If either crashes, restart it. This is not just about training -
    inference must also be kept alive and producing videos.
39. **Git ignore inference_outputs/** and other generated artifacts (outputs/, datasets/.precomputed/)
    so they don't bloat the repo.
40. **Inference output must include a horizontally concatenated comparison video** for each sample
    showing: [reference (zoomed-in input) | generated output | target (ground truth)]. This is
    the primary way we evaluate training progress. If training is working, the output should
    look increasingly like the target.
41. **Sanity check inference BEFORE launching training inference jobs**: test with an existing
    pretrained IC LoRA (e.g. canny control LoRA) on a few test examples in `.scratchwork/` to
    verify the inference pipeline works end-to-end. Use VLM to verify first/middle/last frames.
    Only then start the real inference monitoring loop.
42. **Test iteration 0**: Run inference with NO LoRA (or the base model) as a baseline before
    any training happens. This gives us a "before" to compare against.
43. **Distilled checkpoint can be used with IC LoRAs** - check the existing inference scripts
    (notebooks, T2VTest/) for examples of how this is done. For our inference, we can use
    either the dev or distilled checkpoint.

---

## Important Notes

- **NO silent failures** - all errors must be raised loudly
- Use **progress bars** (tqdm) for all long operations
- **Print status messages** when moving between phases (e.g. "Phase 1: Loading CSV...", "Phase 2: Filtering videos...")
- Use **einops** for tensor reshaping, capital letters for dimension names
- Keep code **simple and minimal** - this is a proof of concept
- **Do NOT modify the LTX2 repository** - use it via sys.path or CLI only
- **Do NOT start from existing LoRA weights** - train from scratch
- **Do NOT use the two-stage pipeline** for training or inference - base model only
- IC LoRA strength must be **1.0** at inference (required by architecture)
- Videos have **no audio** - we don't care about audio, set `generate_audio: false`
- The Envato captions describe the video content, not camera motion - use them as-is

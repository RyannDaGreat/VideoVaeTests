# Nanobanna Flicker Test - Frame Flicker IC LoRA with Pexels Data

Train a frame flicker IC LoRA for LTX-2 using Yash's Pexels paired data.

## What This Does

Trains an IC LoRA that learns to generate smooth videos from sparse keyframe conditioning. The model learns:
- **Input**: Flickery reference video (keyframes repeated with pulse mask indicator)
- **Output**: Smooth target video hitting those keyframes

**Higher-level goal**: Transform videos using just a few keyframes.

## Data Source

Uses Pexels paired data from `/root/CleanCode/Sandbox/RP_Dumps/YashDump/jan10_last_50K_pexels_v2/paired_data/`

Each sample contains:
- `after.png` - 16 transformed keyframes in 4x4 grid (WE USE THIS)
- `raw_video.mp4` - Original smooth video (WE USE THIS)
- `metadata.json` - Keyframe indices and timestamps
- `before.png` - Original keyframes (NOT USED)

**Training**: `after.png` keyframes → `raw_video.mp4` smooth video

## File Structure

```
nanobanna_flickertest/
├── claude_instructions.md      # Comprehensive documentation for future Claude sessions
├── datasets/
│   ├── transform.py            # Video pair generator (Pexels adaptation)
│   ├── make_dataset.py         # Dataset orchestration script
│   ├── videos/                 # Generated target videos
│   ├── reference_videos/       # Generated flickery references
│   ├── test_videos/            # 50 test samples
│   ├── test_reference_videos/  # 50 test references
│   ├── dataset.json            # Training metadata
│   ├── test_set.json           # Test metadata
│   └── .precomputed/           # Precomputed latents
├── configs/
│   ├── nanobanna_flicker_ic_lora.yaml  # Training config
│   └── accelerate_ddp.yaml             # 8-GPU DDP config
├── train.sh                    # Training launcher
├── validate.sh                 # Inference script
├── outputs/                    # Training checkpoints and logs
└── inference_outputs/          # Validation videos
```

## Usage

### 1. Generate Dataset (Test Mode - 24 samples)

```bash
cd /root/CleanCode/Github/VideoVaeTests/Training/nanobanna_flickertest
python datasets/make_dataset.py --test
```

This generates 24 samples for quick verification (~5-10 minutes).

### 2. Generate Full Dataset (10K samples)

```bash
python datasets/make_dataset.py
```

This processes 10,000 samples from the 50K Pexels dataset (~2-4 hours total):
- Video pair generation: 1-2 hours
- Latent precomputation on 8 GPUs: 1-2 hours

**To change sample count**: Edit `NUM_SAMPLES = 10` in `datasets/make_dataset.py` (change "10" to "50" for full dataset)

### 3. Launch Training

```bash
bash train.sh
```

Trains on 8 GPUs using DDP for 100,000 steps. Checkpoints saved every 50 steps.

### 4. Run Inference

```bash
bash validate.sh
```

Runs inference on 8 test samples using the latest checkpoint. Results in `inference_outputs/step_XXXXX/`

## Key Details

### Resolution & Frames
- **Training resolution**: 512x320
- **Frame count**: 121 frames (8*15+1)
- **Frame rate**: 25 fps
- **Video duration**: ~4.8 seconds

### Training Parameters
- **LoRA rank**: 128
- **Learning rate**: 2e-5
- **Steps**: 100,000
- **Batch size**: 1 per GPU (8 effective with DDP)
- **Checkpoint interval**: 50 steps
- **Keep all checkpoints**: Yes (unlimited disk space)

### CRITICAL: Cache Bug Prevention

**NEVER use `use_cache=True` in parallel processing!**

Previous bug: `rp.load_image(..., use_cache=True)` caused all samples to load the same cached image from the first sample, resulting in wind turbines matched with office scenes.

**All image loading uses `use_cache=False`** in this implementation.

### Captions

Pexels samples don't have captions. Using placeholder "A video" for all samples since IC LoRA conditioning is primarily via the reference video, not text.

## Dependencies

- LTX-2 trainer at `/root/CleanCode/Github/VideoVaeTests/LTX2/`
- Models at `/models/LTX2/`
- 8x NVIDIA A100-SXM4-80GB GPUs
- Python packages: rp, fire, tqdm, torch, einops, etc.

## Documentation

See `claude_instructions.md` for comprehensive documentation including:
- Detailed requirements
- Bulldog babysit mode protocols
- VLM verification strategies
- Frame-perfect synchronization requirements
- All user requirements
- Lessons learned from previous work

# Frame Flicker Envato Test - Gray Frame Ablation

## What is this?

This is an ablation test comparing different preprocessing strategies for IC LoRA training with LTX2.

## Setup Instructions (for future Claude instances)

**CRITICAL RULES:**
- Only modify files within `FrameFlickerEnvatoTestAblation/` directory
- Do NOT touch any files in `FrameFlickerEnvatoTest/` or other directories
- Can pull raw data from `/root/CleanCode/Datasets/Envato/downloads/raw_videos`
- Can reference ZoomOutEnvatoTest for train/test splits

**Steps to create this ablation:**
1. Copy only source code from FrameFlickerEnvatoTest (NOT the entire dataset):
   - Copy: configs/, datasets/*.py, train.sh, validate.sh
   - Skip: datasets/videos/, datasets/reference_videos/, datasets/.precomputed/ (will regenerate)
2. Modify `datasets/transform.py`:
   - Remove `rp.quantize_to_nearest_values()` call
   - Fill reference video frames with gray (128, 128, 128) by default
   - Only overlay actual video content at keyframe indices
   - Keep indicator bar logic unchanged (white=keyframe, black=non-keyframe)
3. Update paths in `configs/frame_flicker_ic_lora.yaml`:
   - Change `preprocessed_data_root` to point to this ablation directory
   - Change `output_dir` to point to this ablation directory
4. Test with single video first to verify gray frames work correctly
5. Run `python datasets/make_dataset.py --test` (24 videos, ~10-20 min)
6. Run `python datasets/make_dataset.py` (full dataset, 1+ hour - BABYSIT THIS)
7. Verify dataset generation completed successfully
8. Run `bash train.sh` to start training

## Difference from FrameFlickerEnvatoTest

Both experiments train on Envato videos with a black/white flicker indicator in the top-left corner. The key difference is in how **reference videos** are generated:

### Original FrameFlickerEnvatoTest
- **Reference video preprocessing**: Uses `quantize_to_nearest_values()` to repeat keyframes
- If keyframes are at indices [1, 3, 5]:
  - Frame 0: shows frame 1 content (repeated)
  - Frame 1: shows frame 1 content (keyframe)
  - Frame 2: shows frame 3 content (repeated)
  - Frame 3: shows frame 3 content (keyframe)
  - Frame 4: shows frame 5 content (repeated)
  - Frame 5: shows frame 5 content (keyframe)
- **Result**: Flickery video with repeated frames creating temporal artifacts

### This Ablation (Gray Frame)
- **Reference video preprocessing**: Only keyframes show content, all other frames are solid gray (128, 128, 128)
- If keyframes are at indices [1, 3, 5]:
  - Frame 0: **GRAY**
  - Frame 1: shows frame 1 content (keyframe)
  - Frame 2: **GRAY**
  - Frame 3: shows frame 3 content (keyframe)
  - Frame 4: **GRAY**
  - Frame 5: shows frame 5 content (keyframe)
- **Result**: Sparse video with only keyframes visible, rest is gray

## Hypothesis

Testing whether sparse keyframe conditioning (gray frames) is more effective than dense repeated-frame conditioning (quantized frames) for IC LoRA training.

## Technical Details

- Black/white indicator bar: unchanged in both experiments (white = keyframe, black = non-keyframe)
- Target videos: identical in both experiments (smooth original video + indicator)
- Only the reference video generation differs

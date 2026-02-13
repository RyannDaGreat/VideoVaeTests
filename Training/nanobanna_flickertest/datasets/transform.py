"""Frame flicker transform for Pexels paired data: generates ref/tgt video pairs for IC LoRA training.

Adapted from FrameFlickerEnvatoTest to work with Yash's Pexels paired data structure.

Reference = flickery video (nearest-keyframe replacement from after.png) with pulse mask indicator.
Target = smooth original video (raw_video.mp4) with pulse mask indicator.

Both videos have identical resolution, frame count, and fps (frame-perfect synchronization).

Usage:
    python transform.py /path/to/sample_dir --ref_out_path ref.mp4 --tgt_out_path tgt.mp4
"""

import rp
import fire
from pathlib import Path


def process_pexels_sample(
    sample_dir: str,
    ref_out_path: str = None,
    tgt_out_path: str = None,
    num_frames: int = 121,
    show_progress: bool = False,
    indicator_size: float = 0.2,
):
    """
    Generate reference/target video pair from a Pexels paired data sample.

    Args:
        sample_dir: Path to sample directory containing after.png, raw_video.mp4, metadata.json
        ref_out_path: Output path for reference (flickery) video
        tgt_out_path: Output path for target (smooth) video
        num_frames: Number of frames to generate (must satisfy F % 8 == 1)
        show_progress: Show progress bars
        indicator_size: Size of pulse mask indicator as fraction of video size
    """
    sample_dir = Path(sample_dir)

    # Load metadata to get keyframe indices
    meta = rp.load_json(sample_dir / "metadata.json")
    key_indices = meta['chosen_frame_indices']  # e.g. [5, 12, 18, 25, ...]
    raw_fps = meta.get('fps', 25.0)

    # LTX2 trains at 25fps, Pexels videos are typically 50fps
    ltx2_framerate = 25
    speedup_factor = round(raw_fps / ltx2_framerate)  # Typically 2x for Pexels

    # Load raw video
    raw_video_path = sample_dir / "raw_video.mp4"
    raw_num_frames = rp.get_video_file_num_frames(str(raw_video_path))

    # Calculate required span (accounting for speedup)
    raw_span = num_frames * speedup_factor

    if raw_num_frames < raw_span:
        raise ValueError(f"Video too short: {raw_num_frames} < {raw_span} (need {num_frames} frames at {speedup_factor}x speedup)")

    # Random temporal crop
    start_index = rp.random_int(0, raw_num_frames - raw_span)
    end_index = start_index + raw_span

    # Load video with temporal crop and speedup
    raw_video = rp.load_video_via_decord(
        str(raw_video_path),
        indices=slice(start_index, end_index, speedup_factor),
    )

    # Load keyframes from after.png (CRITICAL: use_cache=False to avoid cache bug)
    keyframes_path = sample_dir / "after.png"
    keyframes = rp.load_image(
        str(keyframes_path),
        use_cache=False,  # CRITICAL: Prevents loading same image for all samples in parallel
    )
    keyframes = rp.as_byte_image(rp.as_rgb_image(keyframes, copy=False), copy=False)
    keyframes = rp.split_tensor_into_regions(keyframes, 4, 4)  # 16 keyframes in 4x4 grid

    # Find which keyframes fall in our temporal window
    in_window = [start_index <= ki < end_index for ki in key_indices]
    shifted_key_indices = [
        (ki - start_index) // speedup_factor
        for ki, m in zip(key_indices, in_window) if m
    ]

    if len(shifted_key_indices) == 0:
        raise ValueError(f"No keyframes in temporal window [{start_index}, {end_index})")

    # Resize videos to target dimensions (512x320)
    keyframes, raw_video = rp.resize_videos_to_hold(keyframes, raw_video, width=512)
    height = rp.get_video_height(keyframes)
    raw_video = rp.crop_images(raw_video, height=height, origin="center")

    # Create flickery reference video using nearest-neighbor keyframe replacement
    windowed_keyframes = keyframes[in_window]
    nn_indices = rp.quantize_to_nearest_values(range(num_frames), shifted_key_indices)
    nn_frames = windowed_keyframes[[shifted_key_indices.index(x) for x in nn_indices]]

    # Create pulse mask indicator (white bar at keyframe indices, black elsewhere)
    mask_width = round(indicator_size * rp.get_video_width(keyframes))
    out_height = round((1 + indicator_size) * height)
    out_width = round((1 + indicator_size) * rp.get_video_width(keyframes))

    pulse_mask = rp.np.zeros((num_frames, out_height, mask_width, 3), dtype=rp.np.uint8)
    pulse_mask[shifted_key_indices] = 255  # White bar at keyframes

    # Build output frames with pulse mask + video content
    out_shape = (num_frames, out_height, out_width, 3)
    ref_out = rp.np.zeros(out_shape, dtype=rp.np.uint8)
    tgt_out = rp.np.zeros(out_shape, dtype=rp.np.uint8)

    # Keyframes at bottom-right, pulse mask at top-left
    ref_out[:, -height:, -rp.get_video_width(keyframes):] = nn_frames
    tgt_out[:, -height:, -rp.get_video_width(keyframes):] = raw_video
    ref_out[:, :out_height, :mask_width] = pulse_mask
    tgt_out[:, :out_height, :mask_width] = pulse_mask

    # Resize back to target dimensions (512x320)
    ref_out, tgt_out = rp.resize_videos(ref_out, tgt_out, size=(height, rp.get_video_width(keyframes)))

    # Save outputs
    if ref_out_path is not None:
        rp.save_video_mp4(ref_out, ref_out_path, framerate=ltx2_framerate, show_progress=show_progress)
    if tgt_out_path is not None:
        rp.save_video_mp4(tgt_out, tgt_out_path, framerate=ltx2_framerate, show_progress=show_progress)

    return {
        'sample_name': sample_dir.name,
        'num_keyframes': len(shifted_key_indices),
        'keyframe_indices': shifted_key_indices,
    }


if __name__ == "__main__":
    fire.Fire(process_pexels_sample)

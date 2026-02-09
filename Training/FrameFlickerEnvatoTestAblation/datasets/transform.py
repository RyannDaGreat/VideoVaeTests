"""Frame flicker transform (GRAY FRAME ABLATION): generates ref/tgt video pairs for IC LoRA training.

Reference = SPARSE video (only keyframes visible, rest is gray) with indicator bar.
Target = smooth original video with indicator bar.
Both resized back to original resolution after adding the indicator border.

This ablation tests sparse keyframe conditioning vs dense repeated-frame conditioning.

Usage:
    python transform.py input.mp4 --ref_out_path ref.mp4 --tgt_out_path tgt.mp4
"""

import rp
import fire


def process_video(
    input_video_path: str,
    max_num_keyframes: int = 16,
    indicator_size: float = 0.2,
    ref_out_path: str = None,
    tgt_out_path: str = None,
    num_frames: int = 121,
    show_progress: bool = False,
):
    input_video = rp.load_video(
        input_video_path,
        use_cache=False,
        show_progress=show_progress,
        length=num_frames,
    )

    height, width = rp.get_video_dimensions(input_video)

    num_keyframes = rp.random_int(1, max_num_keyframes)
    keyframe_indices = rp.random_batch(range(num_frames), num_keyframes)

    out_height = round((1 + indicator_size) * height)
    out_width = round((1 + indicator_size) * width)

    # Create indicator bar canvas: black for non-keyframes, white for keyframes
    out_shape = num_frames, out_height, out_width, 3
    out_frames = rp.np.zeros(out_shape, dtype=rp.np.uint8)
    out_frames[keyframe_indices] = 255

    # Reference video: START WITH GRAY FRAMES (128, 128, 128)
    ref_out = out_frames + 0
    ref_out[:, -height:, -width:] = 128  # Fill video area with gray
    # Then overlay ONLY keyframes with actual video content
    for kf_idx in keyframe_indices:
        ref_out[kf_idx, -height:, -width:] = input_video[kf_idx]

    # Target video: smooth original (unchanged from original logic)
    tgt_out = out_frames + 0
    tgt_out[:, -height:, -width:] = input_video

    ref_out, tgt_out = rp.resize_videos(ref_out, tgt_out, size=(height, width))

    if ref_out_path is not None:
        rp.save_video_mp4(ref_out, ref_out_path, show_progress=show_progress)
    if tgt_out_path is not None:
        rp.save_video_mp4(tgt_out, tgt_out_path, show_progress=show_progress)


if __name__ == "__main__":
    fire.Fire(process_video)

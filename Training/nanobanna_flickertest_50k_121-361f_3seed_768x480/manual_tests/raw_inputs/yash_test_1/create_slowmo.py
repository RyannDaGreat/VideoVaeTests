#!/usr/bin/env python3
"""Create 2x slowmo versions of videos using RIFE frame interpolation.

Usage:
    python create_slowmo.py *_before.mp4           # Glob pattern
    python create_slowmo.py video1.mp4 video2.mp4  # Explicit files
    python create_slowmo.py *.mp4                   # All mp4s
"""
import glob as globmod
import subprocess
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/CleanCode")
import rp


def main(*video_paths: str):
    """
    Create 2x slowmo versions of one or more videos using RIFE interpolation.

    Command. Reads each video, runs RIFE frame interpolation, saves as {stem}_slowmo2x.mp4.
    Skips files that already have a slowmo version. Accepts glob patterns.

    Args:
        *video_paths: One or more video file paths or glob patterns.
    """
    # Expand glob patterns
    expanded = []
    for p in video_paths:
        matches = sorted(globmod.glob(p))
        expanded.extend(matches if matches else [p])

    if not expanded:
        print("No video files specified.")
        return

    print(f"Processing {len(expanded)} videos...")
    for video_path in expanded:
        video_path = Path(video_path)
        output_path = video_path.with_name(f"{video_path.stem}_slowmo2x.mp4")

        if output_path.exists():
            print(f"SKIP {video_path.name}: {output_path.name} already exists")
            continue

        print(f"Loading {video_path}...")
        video = rp.load_video(str(video_path))
        print(f"  {len(video)} frames, running RIFE 2x interpolation...")
        slow = rp.slowmo_video_via_rife(video)
        print(f"  {len(slow)} frames after interpolation")

        # Get original fps (same fps on disk, doubled frames = half speed)
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
            capture_output=True, text=True,
        )
        info = json.loads(result.stdout)["streams"][0]
        fps_num, fps_den = map(int, info["r_frame_rate"].split("/"))
        fps = fps_num / fps_den

        print(f"  Saving to {output_path} at {fps}fps...")
        rp.save_video_mp4(slow, str(output_path), framerate=fps)
        print(f"  Done: {output_path}")

    print(f"\nAll done.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

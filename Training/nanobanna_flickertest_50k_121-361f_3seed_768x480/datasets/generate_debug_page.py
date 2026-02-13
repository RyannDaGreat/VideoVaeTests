#!/usr/bin/env python3
"""Generate an HTML debug video viewer page.

Creates a beautiful grid page showing all debug videos on loop with labels,
a slider to control grid columns, and a JSON mapping of labels to video paths for LLM review.

Labels: A, B, C, ..., Z, then A1, A2, ..., A9, B1, B2, ..., etc.

Run standalone:  python datasets/generate_debug_page.py
Also called automatically by make_dataset.py after debug video generation.
"""

import json
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEBUG_DIR = HERE / "debug_videos"


def label_for_index(i):
    """A, B, ..., Z, A1, A2, ..., A9, B1, B2, ..., B9, C1, ..."""
    if i < 26:
        return string.ascii_uppercase[i]
    i -= 26
    letter = string.ascii_uppercase[i // 9]
    digit = (i % 9) + 1
    return f"{letter}{digit}"


def generate_debug_page():
    videos = sorted(DEBUG_DIR.glob("*_debug.mp4"))
    if not videos:
        print("No debug videos found, skipping HTML generation")
        return

    # Build label mapping
    mapping = {}
    video_entries = []
    for i, vid in enumerate(videos):
        label = label_for_index(i)
        mapping[label] = {
            "debug_video": str(vid.name),
            "sample_id": vid.name.replace("_debug.mp4", ""),
        }
        video_entries.append({"label": label, "filename": vid.name})

    # Save mapping JSON
    mapping_path = DEBUG_DIR / "labels_to_videos.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved label mapping: {mapping_path} ({len(mapping)} entries)")

    # Generate HTML
    videos_json = json.dumps(video_entries)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debug Videos - Nanobanna Flickertest</title>
<style>
@import url('https://fonts.cdnfonts.com/css/futura-pt');
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: linear-gradient(135deg, #0a0e27 0%, #0d1a3a 40%, #0f1f4a 70%, #0a1230 100%); background-attachment: fixed; color: #e0e0e0; font-family: 'Futura PT', 'Futura', 'Century Gothic', sans-serif; min-height: 100vh; }}
.header {{ padding: 24px 32px; background: #111; border-bottom: 1px solid #222; position: sticky; top: 0; z-index: 10; }}
.header h1 {{ font-size: 20px; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; }}
.controls {{ display: flex; align-items: center; gap: 16px; }}
.controls label {{ font-size: 13px; letter-spacing: 1px; text-transform: uppercase; opacity: 0.6; }}
.controls input[type=range] {{ width: 200px; accent-color: #fff; }}
.controls span {{ font-size: 14px; font-weight: 600; min-width: 24px; }}
.grid {{ display: grid; gap: 12px; padding: 24px 32px; }}
.card {{ background: #161616; border-radius: 6px; overflow: hidden; border: 1px solid #222; position: relative; }}
.card:hover {{ border-color: #444; }}
.card video {{ width: 100%; display: block; }}
.card video::-webkit-media-controls {{ opacity: 0; transition: opacity 0.2s; }}
.card:hover video::-webkit-media-controls {{ opacity: 1; }}
.label {{ position: absolute; font-size: 48px; font-weight: 700; letter-spacing: 4px; color: rgba(255,255,255,0.85); text-shadow: 0 2px 12px rgba(0,0,0,0.9); pointer-events: none; z-index: 2; transition: all 0.2s; }}
.label.top-left {{ top: 8px; left: 12px; }}
.label.center {{ top: 50%; left: 50%; transform: translate(-50%, -50%); }}
.info {{ padding: 6px 12px; font-size: 11px; opacity: 0.4; letter-spacing: 0.5px; text-align: center; }}
.count {{ font-size: 13px; opacity: 0.4; }}
</style>
</head>
<body>
<div class="header">
  <h1>Debug Videos <span class="count" id="count"></span></h1>
  <div class="controls">
    <label>Columns</label>
    <input type="range" id="cols" min="1" max="6" value="3">
    <span id="colsVal">3</span>
    <span style="margin-left:24px"></span>
    <label><input type="checkbox" id="labelPos" checked> Label top-left</label>
  </div>
</div>
<div class="grid" id="grid"></div>
<script>
const videos = {videos_json};
const grid = document.getElementById('grid');
const slider = document.getElementById('cols');
const colsVal = document.getElementById('colsVal');
document.getElementById('count').textContent = '(' + videos.length + ' videos)';

videos.forEach(v => {{
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = '<div class="label top-left">' + v.label + '</div>'
    + '<video src="' + v.filename + '" autoplay loop muted playsinline controls></video>'
    + '<div class="info">' + v.filename + '</div>';
  grid.appendChild(card);
}});

function updateCols() {{
  grid.style.gridTemplateColumns = 'repeat(' + slider.value + ', 1fr)';
  colsVal.textContent = slider.value;
}}
slider.addEventListener('input', updateCols);
updateCols();

const labelCheck = document.getElementById('labelPos');
function updateLabels() {{
  document.querySelectorAll('.label').forEach(el => {{
    el.className = labelCheck.checked ? 'label top-left' : 'label center';
  }});
}}
labelCheck.addEventListener('change', updateLabels);
</script>
</body>
</html>"""

    html_path = DEBUG_DIR / "index.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Generated debug viewer: {html_path}")
    print(f"Serve with: python -m http.server 8080 -d {DEBUG_DIR}")


if __name__ == "__main__":
    generate_debug_page()

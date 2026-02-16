// ── Shared defaults ─────────────────────────────────────────────────────────
local base = {
  num_frames: 121,
  width: 1152,
  height: 736,
  num_diffusion_steps: 20,
  seed: 42,
  stage_2: { enabled: true },

  // Derived: batch_title = "kf_sweep_<batch_name>_<W>x<H>_<steps>step"
  batch_title: "kf_sweep_%s_%dx%d_%dstep" % [self.batch_name, self.width, self.height, self.num_diffusion_steps],
};

// ── Per-subject bases (only subject-specific fields) ────────────────────────
local boat = base {
  batch_name: "boat",
  input_video: "raw_inputs/cgi_boat_original.mp4",
  first_frame: "raw_inputs/cgi_boat_daytime_firstframe.png",
  caption: "A breathtaking daytime shot of a sleek white boat cutting through choppy ocean waves. The scene vividly captures the intricate details of the churning, voluminous sea foam and dynamic, misting water spray crashing against the hull, with brilliant sunlight reflecting off every crest, ripple, and droplet of the turbulent water.",
};

local fish = base {
  batch_name: "fish",
  input_video: "raw_inputs/cgi_fish_explode.mp4",
  first_frame: "raw_inputs/cgi_fish_explode_nanobanana.png",
  caption: "Cinematic slow-motion footage of a large fish shattering into distinct chunks of salmon-pink flesh, silver skin, and dark viscera. The explosion occurs just above the surface of a rough, choppy, slate-gray ocean. A massive crown of white foaming cavitation and mist erupts outward. Debris and droplets fly violently toward the camera lens with realistic motion blur. Overcast natural lighting, high contrast textures, 4k raw video, hyperrealistic, shot on high-speed camera.",
};

local snowdog = base {
  batch_name: "snowdog",
  input_video: "raw_inputs/dogrun.mp4",
  first_frame: "raw_inputs/dogrun_snow_firstframe.png",
  caption: "A Jack Russell bounds through deep powder along an icy shoreline, kicking up white snow with every stride. The freezing winter ocean churns in the background.",
};

// ── Keyframe sweep ──────────────────────────────────────────────────────────
local kf_counts = [8, 16, 32, 44, 56, 64, 72, 80];

local make_tests(subject) = [
  subject {
    name: "%df_%dkf_%dx%d_s%d_i%d" % [subject.num_frames, kf_counts[i], subject.width, subject.height, subject.seed, i],
    keyframes: "random %d" % kf_counts[i],
  }
  for i in std.range(0, std.length(kf_counts) - 1)
];

// ── Active tests (comment/uncomment to select) ─────────────────────────────
make_tests(fish)

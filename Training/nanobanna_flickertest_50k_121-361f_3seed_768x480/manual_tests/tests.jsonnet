local base = {
  batch_title: "kf_sweep_1152x736",
  input_video: "raw_inputs/cgi_boat_original.mp4",
  first_frame: "raw_inputs/cgi_boat_daytime_firstframe.png",
  caption: "A breathtaking daytime shot of a sleek white boat cutting through choppy ocean waves. The scene vividly captures the intricate details of the churning, voluminous sea foam and dynamic, misting water spray crashing against the hull, with brilliant sunlight reflecting off every crest, ripple, and droplet of the turbulent water.",
  num_frames: 121,
  width: 1152,
  height: 736,
  num_diffusion_steps: 30,
  seed: 42,
};

// Keyframe sweep: 8 tests from sparse (8kf) to dense (80kf)
// "random N" means N randomly distributed keyframes, resolved at runtime using the test seed
local kf_counts = [8, 16, 32, 44, 56, 64, 72, 80];

[
  base {
    name: "%df_%dkf_%dx%d_s%d_i%d" % [base.num_frames, kf_counts[i], base.width, base.height, base.seed, i],
    keyframes: "random %d" % kf_counts[i],
  }
  for i in std.range(0, std.length(kf_counts) - 1)
]

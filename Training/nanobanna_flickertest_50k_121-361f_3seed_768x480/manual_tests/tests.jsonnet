// ── Shared defaults ─────────────────────────────────────────────────────────
local defaults = {
  num_frames: 121,
  width: 1152,
  height: 736,
  num_diffusion_steps: 20,
  seed: 42,  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  keyframes: "random 8",  // list of ints, or "random N" string (randomness controlled by seed)
  ref_first_frame: false,  // true: ref video frame 0 = condition image; false: ref video is purely NN-filled keyframes
  stage_2: { enabled: true },
  batch_title: "kf_sweep_%s_%dx%d_%dstep" % [self.batch_name, self.width, self.height, self.num_diffusion_steps],
};

// ── Subject mixins (compose with defaults via +) ────────────────────────────
local subjects = {
  boat: {
    batch_name: "boat",
    input_video: "raw_inputs/cgi_boat_original.mp4",
    first_frame: "raw_inputs/cgi_boat_daytime_firstframe.png",
    caption: "A breathtaking daytime shot of a sleek white boat cutting through choppy ocean waves. The scene vividly captures the intricate details of the churning, voluminous sea foam and dynamic, misting water spray crashing against the hull, with brilliant sunlight reflecting off every crest, ripple, and droplet of the turbulent water.",
  },
  fish: {
    batch_name: "fish",
    input_video: "raw_inputs/cgi_fish_explode.mp4",
    first_frame: "raw_inputs/cgi_fish_explode_nanobanana.png",
    caption: "Cinematic slow-motion close-up of a massive fish carcass that has been blown completely open, forming a hollow, cavernous tunnel above the water. The exterior of the arch is a jagged, torn armor of dark, iridescent fish scales. The interior is a gruesome cavity of raw, fibrous salmon-pink muscle and tissue. Long, wet ribbons of shredded flesh and gore dangle from the roof of this tunnel like organic stalactites, swaying slightly and dripping into the sea. The camera looks directly through the gaping wound, framing the choppy, slate-gray ocean and the distant horizon clearly on the other side. The water below is rough and restless, with foam lapping against the torn edges of the fish. Soft, overcast natural lighting, hyperrealistic texture detail, 4k raw footage.",
  },
  snowdog: {
    batch_name: "snowdog",
    input_video: "raw_inputs/dogrun.mp4",
    first_frame: "raw_inputs/dogrun_snow_firstframe.png",
    caption: "A Jack Russell bounds through deep powder along an icy shoreline, kicking up white snow with every stride. The freezing winter ocean churns in the background.",
  },
};

// ── Keyframe sweep generator ────────────────────────────────────────────────
local kf_counts = [8, 16, 32, 44, 56, 64, 72, 80];

local make_tests(config) = [
  config {
    name: "%df_%dkf_%dx%d_s%d_i%d" % [config.num_frames, kf_counts[i], config.width, config.height, config.seed, i],
    keyframes: "random %d" % kf_counts[i],  // overrides default keyframes
  }
  for i in std.range(0, std.length(kf_counts) - 1)
];

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
make_tests(defaults + subjects.fish)

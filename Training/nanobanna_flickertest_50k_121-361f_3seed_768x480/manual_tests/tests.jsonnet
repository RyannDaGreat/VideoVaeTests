// ── Shared defaults ─────────────────────────────────────────────────────────
local defaults = {
  seed: 42,  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  num_frames: 121,
  width: 1152, // Must be divisible by 32
  height: 736, // Must be divisible by 32
  keyframes: "random 8",  // list of ints, "random N" (randomness controlled by seed), or "uniform N" (evenly spaced first-to-last)
  num_diffusion_steps: 20,
  guidance_scale: 4.0,  // text CFG (default 4.0); 1.0=disabled
  cfg_drop_image: 1,  // 0=standard CFG (neg pass keeps image), 1=neg pass drops image tokens entirely, 0-1 blends both (3 passes)
  ref_first_frame: false,  // true: ref video frame 0 = condition image; false: ref video is purely NN-filled keyframes. Empirically, not sure it matters...
  stage_2: { enabled: true },
  batch_title: "%s_%dx%d_%dstep" % [self.batch_name, self.width, self.height, self.num_diffusion_steps],
};

local res_480p = {
  width: 736,
  height: 480,
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
    input_video: "raw_inputs/dogrun_slomo4x.mp4",
    first_frame: "raw_inputs/dogrun_snow_firstframe.png",
    caption: "A Jack Russell bounds through deep powder along an icy shoreline, kicking up white snow with every stride. The freezing winter ocean churns in the background.",
  },
};

// 2x4 cross product: keyframes [20, 40] × cfg_drop_image [0, 0.5, 1, 2]
local sweep = [
  { keyframes: "random 20", cfg_drop_image: 0 },
  { keyframes: "random 20", cfg_drop_image: 0.5 },
  { keyframes: "random 20", cfg_drop_image: 1 },
  { keyframes: "random 20", cfg_drop_image: 2 },
  { keyframes: "random 40", cfg_drop_image: 0 },
  { keyframes: "random 40", cfg_drop_image: 0.5 },
  { keyframes: "random 40", cfg_drop_image: 1 },
  { keyframes: "random 40", cfg_drop_image: 2 },
];

// ── Test generator (applies a sweep to a base config) ───────────────────────
local make_tests(config, sweep) = [
  config + sweep[i] + {
    name: "%s_%d" % [config.batch_name, i],
  }
  for i in std.range(0, std.length(sweep) - 1)
];

local overrides = {
  keyframes: "random 40",
  // height: 320, // width: 512,
  
  num_frames: 121,
} + res_480p;

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
make_tests(defaults + subjects.fish + overrides, sweep)

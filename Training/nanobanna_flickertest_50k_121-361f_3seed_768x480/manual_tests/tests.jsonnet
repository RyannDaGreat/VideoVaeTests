// ── Shared defaults ─────────────────────────────────────────────────────────
local defaults = {
  seed: 42,  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  num_frames: 121,
  width: 1152, // Must be divisible by 32
  height: 736, // Must be divisible by 32
  keyframes: "random 8",  // list of ints, or "random N" string (randomness controlled by seed)
  num_diffusion_steps: 20,
  guidance_scale: 4.0,  // text CFG (default 4.0); 1.0=disabled
  i2v_guidance_scale: 1.0,  // image conditioning CFG (default 1.0=disabled, no extra passes); >1 amplifies image guidance (4 passes when both enabled)
  ref_first_frame: false,  // true: ref video frame 0 = condition image; false: ref video is purely NN-filled keyframes
  stage_2: { enabled: true },
  batch_title: "kf_sweep_%s_%dx%d_%dstep" % [self.batch_name, self.width, self.height, self.num_diffusion_steps],
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

// ── Sweep definitions (list of per-test override dicts) ─────────────────────
local kf_sweep = [
  { keyframes: "random 4" },
  { keyframes: "random 8" },
  { keyframes: "random 16" },
  { keyframes: "random 32" },
  { keyframes: "random 44" },
  { keyframes: "random 64" },
  { keyframes: "random 72" },
  { keyframes: "random 80" },
];

local i2v_sweep = [
  { i2v_guidance_scale: 1 },
  { i2v_guidance_scale: 1.00001 },
  { i2v_guidance_scale: 1.01 },
  { i2v_guidance_scale: 1.02 },
  { i2v_guidance_scale: 1.05 },
  { i2v_guidance_scale: 1.1 },
  { i2v_guidance_scale: 1.2 },
  { i2v_guidance_scale: 1.4 },
];

// ── Test generator (applies a sweep to a base config) ───────────────────────
local make_tests(config, sweep) = [
  config + sweep[i] + {
    name: "%s_%d" % [config.batch_name, i],
  }
  for i in std.range(0, std.length(sweep) - 1)
];

local overrides = {
  // i2v_guidance_scale: 4,
  keyframes: "random 32",  // list of ints, or "random N" string (randomness controlled by seed)
  height: 320,
  width:512,
  num_frames: 49,
} + res_480p;

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
make_tests(defaults + subjects.fish + overrides, i2v_sweep)

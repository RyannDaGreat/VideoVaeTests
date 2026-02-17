// ── Shared defaults ─────────────────────────────────────────────────────────
local defaults = {
  seed: 42,  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  num_frames: 121, // Number of frames in the video, must be 1 more than a number divisible by 8
  width: 1152, // Must be divisible by 32
  height: 736, // Must be divisible by 32
  keyframes: "random 8",  // list of ints, "random N" (randomness controlled by seed), or "uniform N" (evenly spaced first-to-last)
  num_diffusion_steps: 30,
  guidance_scale: 4.0,  // text CFG (default 4.0); 1.0=disabled
  cfg_drop_image: 1,  // 0=standard CFG (neg pass keeps image), 1=neg pass drops image tokens entirely, 0-1 blends both (3 passes)
  ref_first_frame: false,  // true: ref video frame 0 = condition image; false: ref video is purely NN-filled keyframes. Empirically, not sure it matters...
  stage_2: { enabled: true },
  save_stage2_comparison_video: true,  // save raw input vs stage 2 output side-by-side comparison
  batch_title: "%s_%dx%d_%dstep" % [self.batch_name, self.width, self.height, self.num_diffusion_steps],
};

local res_720p = { width: 1152, height: 736, num_frames: 121 };
local res_480p = { width: 736, height: 480, num_frames: 241 };
local res_cheap = {width: 512, height: 320, num_frames:121};

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
  horse_armor: {
    batch_name: "horse_armor",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      Cinematic slow-motion tracking shot, keeping the subject perfectly centered in a medium-wide frame under bright, natural sunlight coming from slightly above and in front of the lens. The camera glides smoothly alongside a magnificent, highly muscular solid black horse galloping powerfully from left to right across a soft, light-tan sandy riding arena.
      The horse is heavily outfitted in intricate, beautiful Mongolian-style armor, featuring a detailed leather lamellar neck guard and chest piece with gleaming metal accents, alongside an ornate traditional saddle and matching bridle. Where exposed, its sleek coat reflects the bright daylight, highlighting its defined musculature, while its long, thick black tail flows dramatically in the air. Small, distinct plumes of dust kick up from its rear hooves, which feature subtle white markings, as they strike the sand.
      The background features shallow depth of field, maintaining a warm, dusty, cinematic atmosphere. Behind a barrier of silver, galvanized steel horizontal pipe fencing, there is a long white painted cinderblock wall, a tan building with a brown pitched roof, and lush green foliage beneath a clear pale blue sky. As the camera tracks the horse, a small green plastic step stool briefly passes through the frame near the fence line, grounding the epic action in a realistic environment.
    |||,
  },
};


// ── Test generator (applies a sweep to a base config) ───────────────────────
local make_tests(config, sweep) = [
  config + sweep[i] + {
    name: "%s_%d" % [config.batch_name, i],
  }
  for i in std.range(0, std.length(sweep) - 1)
];

// 2x4 cross product: resolution [720p, 480p] × keyframes [8, 16, 32, 48]
local sweep = [
  res_720p + { keyframes: "random 8" },
  res_720p + { keyframes: "random 16" },
  res_720p + { keyframes: "random 32" },
  res_720p + { keyframes: "random 48" },
  res_480p + { keyframes: "random 8" },
  res_480p + { keyframes: "random 16" },
  res_480p + { keyframes: "random 32" },
  res_480p + { keyframes: "random 48" },
];

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
make_tests(defaults + subjects.horse_armor, sweep)

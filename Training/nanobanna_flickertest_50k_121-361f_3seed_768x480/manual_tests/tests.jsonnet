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
  horse_armor_with_mongol: {
    batch_name: "horse_armor_with_mongol",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      Cinematic slow-motion tracking shot, keeping the subjects perfectly centered in a medium-wide frame under bright, natural sunlight coming from slightly above and in front of the lens. The camera glides smoothly alongside a magnificent, highly muscular solid black horse galloping powerfully from left to right across a soft, light-tan sandy riding arena. A formidable Mongolian warrior sits confidently astride the horse, dressed in traditional leather lamellar armor, a fur-trimmed conical helmet, and intricately patterned silk garments, leaning slightly forward in a focused, commanding posture.
      The horse is heavily outfitted to match, wearing beautiful Mongolian-style armor that includes a detailed leather neck guard and chest piece with gleaming metal accents, resting under an ornate traditional saddle. Where exposed, the horse's sleek dark coat reflects the daylight, highlighting its defined musculature, while its long, thick black tail flows dramatically in the wind. As the horse's hooves strike the soft sand, small, distinct plumes of warm dust kick up into the air.
      The background features a shallow depth of field, maintaining a warm, dusty, cinematic atmosphere. Behind a barrier of silver, galvanized steel horizontal pipe fencing, there is a long white painted cinderblock wall, a tan building with a brown pitched roof, and lush green foliage beneath a clear pale blue sky. As the camera tracks the rider and horse, a small green plastic step stool briefly passes through the frame near the fence line, grounding the epic, historical subjects within a realistic, modern environment.
    |||,
  },
  horse_robot: {
    batch_name: "horse_robot",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      A wide-angle cinematic shot captures a lifelike robotic horse, constructed from high-polish brushed chrome and articulated steel plates, galloping through a dusty, sun-bleached outdoor corral. The lighting is harsh midday sun, creating sharp specular highlights on the horse's metallic flanks and deep, rhythmic shadows within the exposed gears of its joints. As the machine moves, a subtle heat haze shimmers from its internal cooling vents, and its optical sensors glow with a steady, piercing blue light.
      The camera performs a smooth lateral tracking shot, keeping pace with the robot's powerful stride at a low angle to emphasize its weight and mechanical force. Puffs of fine sand erupt from beneath its heavy steel hooves with every impact, lingering in the air as a golden mist. The background of weathered wooden fences and distant arid hills is rendered with a shallow depth of field, keeping the focus entirely on the fluid, hydraulic motion of the horse's piston-driven neck and wire-mesh mane.
      The scene is defined by high-contrast textures—the sleek, reflective surface of the "skin" against the gritty, matte reality of the desert sand. Every movement is captured with slight motion blur on the hooves to convey speed, while the torso remains stabilized and sharp. The atmosphere is grounded and industrial, stripping away sci-fi tropes in favor of a raw, documentary-style observation of advanced robotics in a rustic environment.
    |||,
  },
  horse_grass_field: {
    batch_name: "horse_grass_field",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      Cinematic tracking shot, wide angle, side profile. A magnificent, muscular solid black horse gallops with powerful strides from left to right across a vast, lush emerald-green grassy field. The horse's coat is exceptionally glossy, creating brilliant wet-look highlights that ripple over its shoulders and flanks under the bright, natural afternoon sun. Its thick black mane and tail billow and flow elegantly in slow motion, trailing behind its powerful frame.
      The camera tracks at a consistent speed alongside the horse, maintaining a steady, smooth lateral movement that keeps the animal centered against a soft-focus background. The ground is covered in dense, vibrant grass; as the horse's hooves strike the turf, small blades of grass and dew droplets are kicked into the air rather than dust. The environment is open and expansive, with distant rolling hills and a few scattered oak trees under a clear, pale blue sky.
      The lighting is high-contrast and warm, coming from a high side-angle to emphasize the horse's defined musculature and the texture of the grass. A slight motion blur on the background enhances the sense of speed, while a shallow depth of field keeps the focus sharply on the sleek, dark texture of the horse. The overall atmosphere is majestic and serene, capturing the raw power of the animal in a pristine natural setting.
    |||,
  },
  horse_nyancat: {
    batch_name: "horse_nyancat",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      Cinematic slow-motion tracking shot. A magnificent, highly muscular solid black unicorn gallops from left to right across a sandy outdoor riding arena, captured in a smooth handheld tracking shot that keeps the subject perfectly centered. A giant, spiraled horn protrudes prominently from its forehead, catching the sunlight. The unicorn's coat is exceptionally glossy and sleek, reflecting the bright daylight and highlighting its well-defined musculature, particularly along its shoulder, flanks, and ribcage. Its long, thick black mane flows dramatically in the air due to the speed and slow-motion effect. As the unicorn gallops, a vibrant, surreal stream of colorful rainbows is expelled from its rear, trailing behind it in the air, mixing with the small, distinct plumes of dust kicked up by its hooves hitting the light-tan sand.
      The unicorn is contained behind a barrier of silver, galvanized steel corral fencing made of horizontal metal pipes passing in a blur. The background features a bright, sunny daytime environment with a clear, pale blue sky. Behind the metal fencing, there is a mix of structures: a long, white painted cinderblock wall or low building, a tan building with a brown pitched roof, and various lush green trees and foliage. At one point in the tracking shot, a small, green plastic step stool or mounting block is visible inside the arena near the fence line.
      The lighting is bright natural sunlight coming from slightly above and in front of the camera, creating intense, wet-looking highlights on the unicorn's dark coat and casting sharp shadows on the sand, enhancing the surreal contrast between the realistic setting and the magical action. The atmosphere is warm and dusty but punctuated by the fantastical, vivid colors of the rainbow trail.
    |||,
  },
  horse_stadium: {
    batch_name: "horse_stadium",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_armor_firstframe.png",
    caption: |||
      Cinematic wide tracking shot in slow-motion, following a magnificent, muscular solid black horse galloping with power from left to right. The horse's coat is exceptionally glossy, reflecting the bright overhead stadium lights that highlight the ripple of its muscles along its shoulders and flanks. Its thick black mane and tail flow dramatically behind it. As its hooves strike the light-tan sand of the arena floor, small plumes of dust kick up into the air.
      The background is a packed, sun-drenched sports stadium. Rows of spectators fill the grandstands behind a professional white perimeter wall, their forms slightly soft-focused to create a shallow depth of field. The crowd is visible as a sea of movement, with people leaning forward and gesturing in excitement as the horse passes. The lighting is a mix of bright natural daylight and the high-intensity glow of stadium floodlights, creating a high-contrast, epic atmosphere with subtle lens flares.
      The camera moves with smooth, motorized precision, keeping pace with the horse's gallop to maintain a perfect profile view. The composition is grounded and professional, capturing the scale of the arena. There is a clear sense of speed and weight, emphasized by the motion blur of the stadium seating in the far distance and the sharp, crisp detail of the horse in the foreground.
    |||,
  },
  minecraft_horse_stable: {
    batch_name: "mc_horse_stable",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/minecraft_horse_stable.png",
    caption: |||
      A Minecraft-style blocky black horse with a blue and brown saddle trots from left to right inside a wooden-fenced stable area. The horse is rendered in the distinctive low-polygon, voxel aesthetic of Minecraft, with flat-shaded surfaces and sharp geometric edges. Its blocky legs move in a stiff, game-like animation cycle as it crosses the sandy ground.
      The environment is a classic Minecraft stable scene: oak wood plank fences form the corral, with a white quartz block wall and dark oak wood staircase structure in the background. The ground is smooth sandstone. The sky is a clear Minecraft blue with simple white rectangular clouds.
      The lighting is Minecraft's characteristic flat, ambient illumination with soft directional shadows. The entire scene maintains the game's iconic pixelated texture resolution and cubic geometry, creating a charming, toy-like atmosphere.
    |||,
  },
  minecraft_horse_saddle_stable: {
    batch_name: "mc_horse_saddle",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/minecraft_horse_saddle_stable.png",
    caption: |||
      A Minecraft-style blocky black horse without a saddle trots from left to right inside a wooden-fenced stable corral. The horse has the distinctive cubic, low-polygon Minecraft aesthetic with flat-shaded dark surfaces and angular geometric limbs. Its mane is a simple dark rectangular extrusion along its blocky neck.
      The stable environment features oak wood plank fences, a white quartz block wall, and dark oak wood staircase structures forming the roof and walls in the background. The sandy ground is flat sandstone blocks. A wooden gate is visible to the right side of the corral.
      The scene is lit with Minecraft's flat ambient lighting and gentle directional shadows. All textures maintain the game's signature pixelated, 16x16 resolution look. The sky shows classic Minecraft blue with blocky white clouds.
    |||,
  },
  minecraft_horse_steve_stable: {
    batch_name: "mc_steve_stable",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/minecraft_horse_steve_stable.png",
    caption: |||
      Minecraft's Steve character rides a blocky black horse from left to right inside a wooden-fenced stable corral. Steve sits upright on a brown saddle, wearing his iconic cyan t-shirt and dark pants, his blocky arms held forward. The horse moves with the stiff, game-like animation typical of Minecraft, its cubic legs cycling beneath it.
      The stable is built from oak wood plank fences and dark oak wood structures, with a white quartz block wall in the background. The ground is flat sandstone. The environment captures the cozy, enclosed feel of a player-built Minecraft horse stable.
      Minecraft's characteristic flat lighting illuminates the scene with soft shadows. All geometry is strictly cubic and voxel-based. Textures are the game's signature low-resolution pixel art. The atmosphere is warm and playful.
    |||,
  },
  minecraft_horse_steve_meadow: {
    batch_name: "mc_steve_meadow",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/minecraft_horse_steve_meadow.png",
    caption: |||
      Minecraft's Steve character rides a blocky black horse from left to right across a lush meadow biome. Steve sits atop a brown saddle in his iconic cyan shirt, riding confidently through the open landscape. The horse gallops with Minecraft's characteristic stiff-legged animation across green grass blocks dotted with small white and yellow flowers.
      The background is a panoramic Minecraft landscape: a winding blue river cuts through the grassy terrain, with a cluster of village buildings featuring tan walls and brown roofs nestled among dark oak and birch trees. Rolling green hills and blocky mountain peaks rise in the distance under a bright blue sky with flat white clouds.
      The scene is rendered in Minecraft's distinctive voxel style with pixelated textures and cubic geometry throughout. The lighting is bright and cheerful with the game's ambient illumination. The wide composition captures the sense of open-world adventure and exploration.
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

// One test per subject at 720p with 32 keyframes — all horses except horse_armor
local sweep = [{}];  // single entry, no overrides (defaults handle everything)

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
local base = defaults + res_720p + { keyframes: "random 32" };
make_tests(base + subjects.horse_armor_with_mongol, sweep)
+ make_tests(base + subjects.horse_robot, sweep)
+ make_tests(base + subjects.horse_grass_field, sweep)
+ make_tests(base + subjects.horse_nyancat, sweep)
+ make_tests(base + subjects.horse_stadium, sweep)
+ make_tests(base + subjects.minecraft_horse_stable, sweep)
// + make_tests(base + subjects.minecraft_horse_saddle_stable, sweep)
+ make_tests(base + subjects.minecraft_horse_steve_stable, sweep)
+ make_tests(base + subjects.minecraft_horse_steve_meadow, sweep)

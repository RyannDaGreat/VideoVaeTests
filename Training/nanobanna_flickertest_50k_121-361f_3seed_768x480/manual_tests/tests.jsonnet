// ── Shared defaults ─────────────────────────────────────────────────────────
local defaults = {
  // seed: 42,  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  seed: 'random',  // int or "random" (system-entropy); controls both diffusion noise and keyframe selection
  num_frames: 121, // Number of frames in the video, must be 1 more than a number divisible by 8
  width: 1152, // Must be divisible by 32
  height: 736, // Must be divisible by 32
  keyframes: "random 8",  // list of ints, "random N" (randomness controlled by seed), or "uniform N" (evenly spaced first-to-last)
  num_diffusion_steps: 30,
  guidance_scale: 4.0,  // text CFG (default 4.0); 1.0=disabled
  cfg_drop_image: 2,  // 0=standard CFG (neg pass keeps image), 1=neg pass drops image tokens entirely, 0-1 blends both (3 passes)
  ref_first_frame: false,  // true: ref video frame 0 = condition image; false: ref video is purely NN-filled keyframes. Empirically, not sure it matters...
  stage_2: { enabled: true },
  checkpoint: "latest",  // "latest" (highest step) or int step number (e.g. 50000)
  frame_rate: 25.0,  // FPS conditioning — normalizes temporal position embeddings. Match source video fps.
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
    first_frame: "raw_inputs/horse_armor_with_mongol_firstframe.png",
    caption: |||
      Cinematic slow-motion tracking shot, keeping the subjects perfectly centered in a medium-wide frame under bright, natural sunlight coming from slightly above and in front of the lens. The camera glides smoothly alongside a magnificent, highly muscular solid black horse galloping powerfully from left to right across a soft, light-tan sandy riding arena. A formidable Mongolian warrior sits confidently astride the horse, dressed in traditional leather lamellar armor, a fur-trimmed conical helmet, and intricately patterned silk garments, leaning slightly forward in a focused, commanding posture.
      The horse is heavily outfitted to match, wearing beautiful Mongolian-style armor that includes a detailed leather neck guard and chest piece with gleaming metal accents, resting under an ornate traditional saddle. Where exposed, the horse's sleek dark coat reflects the daylight, highlighting its defined musculature, while its long, thick black tail flows dramatically in the wind. As the horse's hooves strike the soft sand, small, distinct plumes of warm dust kick up into the air.
      The background features a shallow depth of field, maintaining a warm, dusty, cinematic atmosphere. Behind a barrier of silver, galvanized steel horizontal pipe fencing, there is a long white painted cinderblock wall, a tan building with a brown pitched roof, and lush green foliage beneath a clear pale blue sky. As the camera tracks the rider and horse, a small green plastic step stool briefly passes through the frame near the fence line, grounding the epic, historical subjects within a realistic, modern environment.
    |||,
  },
  horse_robot: {
    batch_name: "horse_robot",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_robot_firstframe.png",
    caption: |||
      A wide-angle cinematic shot captures a lifelike robotic horse, constructed from high-polish brushed chrome and articulated steel plates, galloping through a dusty, sun-bleached outdoor corral. The lighting is harsh midday sun, creating sharp specular highlights on the horse's metallic flanks and deep, rhythmic shadows within the exposed gears of its joints. As the machine moves, a subtle heat haze shimmers from its internal cooling vents, and its optical sensors glow with a steady, piercing blue light.
      The camera performs a smooth lateral tracking shot, keeping pace with the robot's powerful stride at a low angle to emphasize its weight and mechanical force. Puffs of fine sand erupt from beneath its heavy steel hooves with every impact, lingering in the air as a golden mist. The background of weathered wooden fences and distant arid hills is rendered with a shallow depth of field, keeping the focus entirely on the fluid, hydraulic motion of the horse's piston-driven neck and wire-mesh mane.
      The scene is defined by high-contrast textures—the sleek, reflective surface of the "skin" against the gritty, matte reality of the desert sand. Every movement is captured with slight motion blur on the hooves to convey speed, while the torso remains stabilized and sharp. The atmosphere is grounded and industrial, stripping away sci-fi tropes in favor of a raw, documentary-style observation of advanced robotics in a rustic environment.
    |||,
  },
  horse_grass_field: {
    batch_name: "horse_grass_field",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_grass_field_firstframe.png",
    caption: |||
      Cinematic tracking shot, wide angle, side profile. A magnificent, muscular solid black horse gallops with powerful strides from left to right across a vast, lush emerald-green grassy field. The horse's coat is exceptionally glossy, creating brilliant wet-look highlights that ripple over its shoulders and flanks under the bright, natural afternoon sun. Its thick black mane and tail billow and flow elegantly in slow motion, trailing behind its powerful frame.
      The camera tracks at a consistent speed alongside the horse, maintaining a steady, smooth lateral movement that keeps the animal centered against a soft-focus background. The ground is covered in dense, vibrant grass; as the horse's hooves strike the turf, small blades of grass and dew droplets are kicked into the air rather than dust. The environment is open and expansive, with distant rolling hills and a few scattered oak trees under a clear, pale blue sky.
      The lighting is high-contrast and warm, coming from a high side-angle to emphasize the horse's defined musculature and the texture of the grass. A slight motion blur on the background enhances the sense of speed, while a shallow depth of field keeps the focus sharply on the sleek, dark texture of the horse. The overall atmosphere is majestic and serene, capturing the raw power of the animal in a pristine natural setting.
    |||,
  },
  horse_nyancat: {
    batch_name: "horse_nyancat",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_nyancat_firstframe.png",
    caption: |||
      Cinematic slow-motion tracking shot. A magnificent, highly muscular solid black unicorn gallops from left to right across a sandy outdoor riding arena, captured in a smooth handheld tracking shot that keeps the subject perfectly centered. A giant, spiraled horn protrudes prominently from its forehead, catching the sunlight. The unicorn's coat is exceptionally glossy and sleek, reflecting the bright daylight and highlighting its well-defined musculature, particularly along its shoulder, flanks, and ribcage. Its long, thick black mane flows dramatically in the air due to the speed and slow-motion effect. As the unicorn gallops, a vibrant, surreal stream of colorful rainbows is expelled from its rear, trailing behind it in the air, mixing with the small, distinct plumes of dust kicked up by its hooves hitting the light-tan sand.
      The unicorn is contained behind a barrier of silver, galvanized steel corral fencing made of horizontal metal pipes passing in a blur. The background features a bright, sunny daytime environment with a clear, pale blue sky. Behind the metal fencing, there is a mix of structures: a long, white painted cinderblock wall or low building, a tan building with a brown pitched roof, and various lush green trees and foliage. At one point in the tracking shot, a small, green plastic step stool or mounting block is visible inside the arena near the fence line.
      The lighting is bright natural sunlight coming from slightly above and in front of the camera, creating intense, wet-looking highlights on the unicorn's dark coat and casting sharp shadows on the sand, enhancing the surreal contrast between the realistic setting and the magical action. The atmosphere is warm and dusty but punctuated by the fantastical, vivid colors of the rainbow trail.
    |||,
  },
  horse_stadium: {
    batch_name: "horse_stadium",
    input_video: "raw_inputs/horse_slowmo.mp4",
    first_frame: "raw_inputs/horse_stadium_firstframe.png",
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
  // asian_woman: {
  //   // REDUNDANT: Output = input here
  //   batch_name: "asian_woman",
  //   input_video: "raw_inputs/asian_woman_slowmo.mp4",
  //   first_frame: "raw_inputs/asian_woman_firstframe.png",
  //   caption: |||
  //     A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.
  //   |||,
  // },
  asian_man_male: {
    batch_name: "asian_man_male",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_man_male_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian man in the foreground wearing a pale green knit sweater, a thin gold chain necklace, and white wireless earbuds. Behind him, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the man begins with his head tilted back, his eyes squeezed shut, and his hands raised near his shoulders with fingers curled. Slowly, he brings both palms down to press firmly against his temples, his brow furrowing deeply while his eyes remain closed. Suddenly, he snaps his eyes open, dropping his hands and throwing them outward with splayed fingers. His jaw drops into an exaggerated, wide-eyed expression, his mouth open as he stares intently past the camera lens.",
  },
  asian_woman_1920s: {
    batch_name: "asian_woman_1920s",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_1920s_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with warm amber candlelight and dim overhead chandelier lighting, establishes a young Asian woman in the foreground wearing a sleeveless emerald green beaded flapper dress with fringe trim and intricate gold geometric detailing, a long multi-strand pearl necklace, and a tan cloche hat adorned with a green feather and jeweled pin. Behind her, slightly out of focus due to a shallow depth of field, a 1920s jazz speakeasy bar fills the background \u2014 a jazz band plays saxophone, double bass, and drums while elegantly dressed patrons dance and socialize beneath glowing amber table lamps and a warm chandelier. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_bold_lip: {
    batch_name: "asian_woman_bold_lip",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_bold_lip_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. She wears bold, dark red lipstick and polished makeup that gives her a striking, glamorous look, with straight dark hair. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_bold_makeup: {
    batch_name: "asian_woman_bold_makeup",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_bold_makeup_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. She wears bold dark burgundy-red lipstick, matching dark red nail polish on her fingers, and dramatic blush makeup that gives her a striking, polished look. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_clown: {
    batch_name: "asian_woman_clown",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_clown_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground dressed in a full clown costume \u2014 a bright yellow outfit covered in multicolored polka dots, a ruffled rainbow-striped collar, and a colorful bucket hat adorned with a large flower on top. Her face is painted in classic clown makeup: a white base, red nose, and blue triangular accents under the eyes. A thin gold chain necklace remains visible at her collar. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting with soft pink and beige furniture and green plants is faintly visible. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_elderly: {
    batch_name: "asian_woman_elderly",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_elderly_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes an elderly Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. She has long gray hair framing a warm, aged face with soft wrinkles. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_greenscreen: {
    batch_name: "asian_woman_greenscreen",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_greenscreen_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with bright even lighting, establishes a young Asian woman in the foreground wearing a cream-white knit sweater, a thin gold chain necklace, and white wireless earbuds. She has straight dark hair falling past her shoulders. Behind her, a solid bright green screen fills the entire background with no depth of field blur, the vivid green completely flat and evenly lit. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_male: {
    batch_name: "asian_woman_male",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_male_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young white man in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. He has short dark hair and light stubble. Behind him, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the man begins with his head tilted back, his eyes squeezed shut, and his hands raised near his shoulders with fingers curled. Slowly, he brings both palms down to press firmly against his temples, his brow furrowing deeply while his eyes remain closed. Suddenly, he snaps his eyes open, dropping his hands and throwing them outward with splayed fingers. His jaw drops into an exaggerated, wide-eyed expression, his mouth open as he stares intently past the camera lens.",
  },
  asian_woman_mime: {
    batch_name: "asian_woman_mime",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_mime_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground dressed as a mime: her face is painted completely white with a classic mime design featuring a teardrop under one eye and bold red lips, she wears a black beret, a black-and-white horizontal-striped long-sleeve shirt, and white gloves. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_moody: {
    batch_name: "asian_woman_moody",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_moody_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with dramatic directional lighting and deep shadows, establishes a young Asian woman in the foreground wearing a pale olive-green knit sweater, a thin gold chain, and white wireless earbuds. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. Hard shafts of warm sunlight cut across her face and body from one side, casting strong contrasting shadows and creating a moody, high-contrast atmosphere against the darker surroundings. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_pierced: {
    batch_name: "asian_woman_pierced",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_pierced_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain necklace, and dangling silver earrings. She has a blunt bob haircut with straight-across bangs, a small labret piercing below her lower lip, and stacked silver and gold bangles on both wrists. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_south_asian: {
    batch_name: "asian_woman_south_asian",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_south_asian_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young South Asian woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. Her long, wavy dark hair falls loosely around her shoulders. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_spaceship: {
    batch_name: "asian_woman_spaceship",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_spaceship_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with cool blue and purple sci-fi lighting, establishes a young Asian woman in the foreground wearing a pale green knit sweater, a thick gold chain necklace, and white wireless earbuds. Behind her, slightly out of focus due to a shallow depth of field, the interior of a futuristic spaceship stretches out \u2014 glowing blue holographic control panels, hexagonal wall panels, illuminated instrument screens, and a star-filled void visible through the windows. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_striped: {
    batch_name: "asian_woman_striped",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_striped_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Asian woman in the foreground wearing a navy blue and white horizontal striped t-shirt, a thin gold chain, silver bracelets on her wrist, and white wireless earbuds. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  asian_woman_white_woman: {
    batch_name: "asian_woman_white_woman",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/asian_woman_white_woman_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young white woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  black_woman: {
    batch_name: "black_woman",
    input_video: "raw_inputs/asian_woman_slowmo.mp4",
    first_frame: "raw_inputs/black_woman_firstframe.png",
    caption: "A cinematic medium close-up, shot in slow motion with soft studio lighting, establishes a young Black woman in the foreground wearing a pale green knit sweater, a thin gold chain, and white wireless earbuds. She has straight dark hair cut to shoulder length in a sleek bob. Behind her, slightly out of focus due to a shallow depth of field, a warm-toned indoor setting is faintly visible with soft furnishings and a green houseplant. The camera remains locked in a static position as the woman begins with her head tilted back, her eyes squeezed shut, and her hands raised near her shoulders with fingers curled. Slowly, she brings both palms down to press firmly against her temples, her brow furrowing deeply while her eyes remain closed. Suddenly, she snaps her eyes open, dropping her hands and throwing them outward with splayed fingers. Her jaw drops into an exaggerated, wide-eyed expression, her mouth open as she stares intently past the camera lens.",
  },
  // ── Previz subjects (24fps CG/VFX) ──────────────────────────────────────
  otis_previz: {
    batch_name: "otis_previz",
    input_video: "raw_inputs/OTIS_PREVIS_beautyJPG.mp4",
    first_frame: "raw_inputs/otis_beauty_firstframe.png",
    frame_rate: 24.0,  // native fps of source
    num_frames: 105,   // 105 % 8 == 1, matches source exactly
    caption: |||
      A dramatic elevated nighttime shot of a large white motor yacht struggling through a violent storm at sea. The camera holds at a slightly overhead angle, capturing the vessel's full profile as it pitches and rolls through enormous dark waves. A warm amber deck light near the bow cuts through the heavy spray, casting a sharp glow against the wet hull and churning foam while the rest of the scene is swallowed by pitch-black sky.
      Massive swells surge past the hull, sending thick curtains of white sea spray and mist streaming across the frame. The ocean surface is a chaotic landscape of deep troughs and cresting peaks, with turbulent foam wrapping around the waterline and trailing off behind the stern. The boat's radar mast, antennas, and cabin superstructure are silhouetted against the spray clouds, rocking with the vessel's heavy motion.
      The lighting is stark and cinematic — the single warm deck light creates a dramatic contrast against the cold, dark ocean and featureless black sky. Dense mist and airborne water droplets catch the light, producing a soft atmospheric haze around the vessel.
    |||,
  },
  tlst_previz: {
    batch_name: "tlst_previz",
    input_video: "raw_inputs/TLST_PREVIS_beautyJPG.mp4",
    first_frame: "raw_inputs/tlst_beauty_firstframe.png",
    frame_rate: 24.0,  // native fps of source
    num_frames: 121,   // 121 % 8 == 1; source has 244f but 1152x736 maxes at 121f (13,248 tokens)
    caption: |||
      A harrowing storm sequence follows a small wooden sailboat battling mountainous ocean swells. The scene opens from the deck in a low-angle close-up: a figure in a weathered rust-orange rain jacket grips the rigging with both hands, bracing against the violent rocking as sheets of rain and spray lash across the frame. Wet ropes, metal fittings, and a billowing white sail fill the foreground, with the chaotic gray sky barely visible through the deluge.
      The camera pulls back dramatically to a wide aerial perspective, revealing the full scale of the storm. The small sailboat, its single mast and pale sail now tiny against the scene, is nearly swallowed by towering blue-gray waves that dwarf the vessel from every direction. Massive swells crest and roll with heavy, churning white foam, and a bright orange life preserver ring is visible on the stern as the boat slides down into deep wave troughs.
      The color palette is cold and desaturated — steely blue-gray water, overcast sky blending into the ocean at the horizon, with the orange jacket and life ring providing the only warm accents. Heavy atmospheric haze from rain and spray reduces visibility, creating depth and scale.
    |||,
  },
  // ── Last-2/3 TLST (first third cut — bad footage) ──────────────────────
  // 163 normal frames, 325 slowmo frames. Different first frame and prompt.
  local _tlst_l23_caption = |||
    A wide aerial shot captures a small wooden sailboat engulfed by mountainous ocean swells during a violent storm. The vessel, its single mast and pale sail barely visible, is nearly swallowed by towering blue-gray waves that dwarf it from every direction. An orange life preserver ring on the stern provides a small point of warm color against the cold, churning sea. Massive swells crest and roll with heavy white foam as the boat pitches and slides through deep wave troughs.
    The camera slowly pulls back to reveal the full terrifying scale of the storm, the sailboat shrinking against the enormous wave faces. Thick curtains of rain and sea spray reduce visibility, creating layers of atmospheric haze that add depth to the scene. The waves move with slow, immense weight, their surfaces textured with wind-whipped spray and streaks of white foam.
    The color palette is cold and desaturated — steely blue-gray water blending seamlessly into an overcast sky at the horizon. The lighting is flat and diffused through thick storm clouds, with no direct sun.
  |||,
  tlst_previz_l23: {
    batch_name: "tlst_l23",
    input_video: "raw_inputs/TLST_PREVIS_beautyJPG_last2thirds.mp4",
    first_frame: "raw_inputs/tlst_beauty_last2thirds_firstframe.png",
    frame_rate: 24.0,
    num_frames: 121,  // max at 1152x736; source has 163f
    caption: _tlst_l23_caption,
  },
  // ── Slowmo previz subjects (2x slowdown, 48fps conditioning) ────────────
  // Same prompts and first frames as normal-speed, different driving videos.
  // frame_rate=48 because original 24fps content was interpolated to 2x frames.
  // Files are stored at 24fps on disk (compression), but model needs true temporal rate.
  local _otis_slowmo_base = {
    input_video: "raw_inputs/OTIS_PREVIS_beautyJPG.mp4_slowmo2x.mp4",
    first_frame: "raw_inputs/otis_beauty_firstframe.png",
    frame_rate: 48.0,
    caption: subjects.otis_previz.caption,
  },
  local _tlst_slowmo_base = {
    input_video: "raw_inputs/TLST_PREVIS_beautyJPG.mp4_slowmo2x.mp4",
    first_frame: "raw_inputs/tlst_beauty_firstframe.png",
    frame_rate: 48.0,
    caption: subjects.tlst_previz.caption,
  },
  // Short tier: 121f @ 1152x736 (13,248 tokens — proven safe)
  otis_slowmo_short: _otis_slowmo_base + {
    batch_name: "otis_slo_short",
    num_frames: 121,
    width: 1152, height: 736,
  },
  tlst_slowmo_short: _tlst_slowmo_base + {
    batch_name: "tlst_slo_short",
    num_frames: 121,
    width: 1152, height: 736,
  },
  // Medium tier: 201f @ 896x576 (13,104 tokens)
  otis_slowmo_medium: _otis_slowmo_base + {
    batch_name: "otis_slo_med",
    num_frames: 201,
    width: 896, height: 576,
  },
  tlst_slowmo_medium: _tlst_slowmo_base + {
    batch_name: "tlst_slo_med",
    num_frames: 201,
    width: 896, height: 576,
  },
  // Long tier: 297f @ 768x480 (13,110 tokens) — TLST only (OTIS slowmo is only 209f)
  tlst_slowmo_long: _tlst_slowmo_base + {
    batch_name: "tlst_slo_long",
    num_frames: 281,  // 281 % 8 == 1; max at 768x480 = 12,960 tokens
    width: 768, height: 480,
  },
  // ── Last-2/3 TLST slowmo subjects ──────────────────────────────────────
  local _tlst_l23_slowmo_base = {
    input_video: "raw_inputs/TLST_PREVIS_beautyJPG_slowmo2x_last2thirds.mp4",
    first_frame: "raw_inputs/tlst_beauty_last2thirds_firstframe.png",
    frame_rate: 48.0,
    caption: _tlst_l23_caption,
  },
  tlst_l23_slowmo_short: _tlst_l23_slowmo_base + {
    batch_name: "tlst_l23_slo_s",
    num_frames: 121,
    width: 1152, height: 736,
  },
  tlst_l23_slowmo_medium: _tlst_l23_slowmo_base + {
    batch_name: "tlst_l23_slo_m",
    num_frames: 201,
    width: 896, height: 576,
  },
  tlst_l23_slowmo_long: _tlst_l23_slowmo_base + {
    batch_name: "tlst_l23_slo_l",
    num_frames: 281,  // 281 % 8 == 1; max at 768x480 = 12,960 tokens
    width: 768, height: 480,
  },
  // ── William boat (CG previz → photoreal storm yacht) ───────────────────
  local _william_caption = |||
    A cinematic wide shot of a sturdy white motor yacht cutting through violent, dark ocean swells during a fierce storm. The vessel sits low in the frame, its two-deck cabin structure and radar mast silhouetted against a sky of dense, roiling charcoal storm clouds. Warm amber cabin lights glow through the windows and illuminate the deck railings, casting sharp reflections on the wet hull and churning water below.
    Massive dark waves roll past the hull, rocking the boat as white spray erupts along the waterline. The ocean surface is a turbulent landscape of deep troughs and wind-whipped crests, textured with streaks of foam. Rain streaks through the frame, caught in the warm deck lighting, while the background sky churns with dramatic cloud formations lit from within by distant lightning.
    The lighting is high-contrast and moody — warm amber from the vessel's lights against the cold, desaturated blue-gray of the storm. The camera holds steady at a low angle slightly off the bow, emphasizing the weight of the waves against the hull.
  |||,
  // Normal: 69f source → 65f valid, 24fps. Max res per tier computed from 13,248 token limit.
  william_boat_hi: {
    batch_name: "wboat_hi",
    input_video: "raw_inputs/william_boat_test_01.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 24.0,
    num_frames: 65,
    width: 1664, height: 864,  // 12,636 tokens (LT=9)
    caption: _william_caption,
  },
  william_boat_med: {
    batch_name: "wboat_med",
    input_video: "raw_inputs/william_boat_test_01.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 24.0,
    num_frames: 65,
    width: 1376, height: 736,  // 8,901 tokens (LT=9)
    caption: _william_caption,
  },
  william_boat_lo: {
    batch_name: "wboat_lo",
    input_video: "raw_inputs/william_boat_test_01.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 24.0,
    num_frames: 65,
    width: 1152, height: 608,  // 6,156 tokens (LT=9)
    caption: _william_caption,
  },
  // Slowmo: 137f valid as-is, 60fps. Max res per tier.
  william_boat_slo_hi: {
    batch_name: "wboat_slo_hi",
    input_video: "raw_inputs/william_boat_test_01_slowmo2x.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 60.0,
    num_frames: 137,
    width: 1184, height: 608,  // 12,654 tokens (LT=18)
    caption: _william_caption,
  },
  william_boat_slo_med: {
    batch_name: "wboat_slo_med",
    input_video: "raw_inputs/william_boat_test_01_slowmo2x.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 60.0,
    num_frames: 137,
    width: 1024, height: 544,  // 9,792 tokens (LT=18)
    caption: _william_caption,
  },
  william_boat_slo_lo: {
    batch_name: "wboat_slo_lo",
    input_video: "raw_inputs/william_boat_test_01_slowmo2x.mp4",
    first_frame: "raw_inputs/william_boat_gen_01.jpg",
    frame_rate: 60.0,
    num_frames: 137,
    width: 896, height: 480,  // 7,560 tokens (LT=18)
    caption: _william_caption,
  },
};


// ── Test generator (applies a sweep to a base config) ───────────────────────
local make_tests(config, sweep) = [
  config + sweep[i] + {
    name: "%s_%d" % [config.batch_name, i],
  }
  for i in std.range(0, std.length(sweep) - 1)
];

// ── Sweep: checkpoint [50k→1k] × cdi [0,1,2] × kf [16,32,64] ───────────
// Checkpoint is outermost: all tests for step 50000 run first, then 25000, etc.
local checkpoint_steps = [50000, 25000, 10000, 5000, 1000];  // descending
local cdi_values = [0, 1, 2];
local kf_values = [16, 32, 64];

local sweep = [
  { checkpoint: ckpt, cfg_drop_image: cdi, keyframes: "random %d" % kf }
  for ckpt in checkpoint_steps  // outermost loop
  for cdi in cdi_values
  for kf in kf_values
];

// Normal-speed previz (24fps)
local normal_subjects = [
  subjects.otis_previz,       // 105f @ 1152x736
  subjects.tlst_previz,       // 121f @ 1152x736 (truncated from 244f)
  subjects.tlst_previz_l23,   // 121f @ 1152x736 (last 2/3 of video)
];

// Slowmo previz (48fps, various resolution tiers)
local slowmo_subjects = [
  subjects.otis_slowmo_short,     // 121f @ 1152x736
  subjects.otis_slowmo_medium,    // 201f @ 896x576
  subjects.tlst_slowmo_short,     // 121f @ 1152x736
  subjects.tlst_slowmo_medium,    // 201f @ 896x576
  subjects.tlst_slowmo_long,      // 297f @ 768x480
  subjects.tlst_l23_slowmo_short,  // 121f @ 1152x736 (last 2/3)
  subjects.tlst_l23_slowmo_medium, // 201f @ 896x576  (last 2/3)
  subjects.tlst_l23_slowmo_long,   // 297f @ 768x480  (last 2/3)
];

// ── Active tests (compose: defaults + subject + optional overrides) ─────────
// 45 sweep points (5 ckpt × 3 cdi × 3 kf) × 11 subjects = 495 tests total
//
// Iteration order (outermost → innermost):
//   1. checkpoint  (50000 → 25000 → 10000 → 5000 → 1000)
//   2. cfg_drop_image  (0 → 1 → 2)
//   3. keyframes  (16 → 32 → 64)
//   4. subject  (all 11 subjects per sweep point)
//
// This means: all 99 tests at ckpt 50000 run before any at 25000, etc.
// Within a checkpoint, all subjects cycle through each cdi×kf combo.
local base = defaults + { num_diffusion_steps: 20 };
local all_subjects = normal_subjects + slowmo_subjects;
local previz_tests = [
  base + subj + sweep_point + {
    name: "%s_%d" % [subj.batch_name, i],
  }
  for i in std.range(0, std.length(sweep) - 1)
  for sweep_point in [sweep[i]]
  for subj in all_subjects
];

// ── William boat sweep: latest ckpt × cdi [0,1,2] × 6 resolution tiers ──
// 1 ckpt × 3 cdi × 6 subjects = 18 tests. Fixed 16 keyframes.
local william_cdi = [0, 1, 2];
local william_subjects = [
  subjects.william_boat_hi,       // 65f @ 1664x864
  subjects.william_boat_med,      // 65f @ 1376x736
  subjects.william_boat_lo,       // 65f @ 1152x608
  subjects.william_boat_slo_hi,   // 137f @ 1184x608
  subjects.william_boat_slo_med,  // 137f @ 1024x544
  subjects.william_boat_slo_lo,   // 137f @ 896x480
];
local william_sweep = [
  { checkpoint: "latest", cfg_drop_image: cdi, keyframes: "random 16" }
  for cdi in william_cdi
];
local william_tests = [
  base + subj + wp + {
    name: "%s_%d" % [subj.batch_name, i],
  }
  for i in std.range(0, std.length(william_sweep) - 1)
  for wp in [william_sweep[i]]
  for subj in william_subjects
];

// ── Combined output ──────────────────────────────────────────────────────
// previz_tests (495) + william_tests (18) = 513 total
// Uncomment to run all: previz_tests + william_tests
william_tests

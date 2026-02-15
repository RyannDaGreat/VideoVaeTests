"""Extra 20 batches - heavy on 361f, varied prompts. Run after scratchpad.py finishes."""
import random, json, subprocess, sys, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_JSON = HERE / "tests.json"

PROMPTS_EXTENDED = [
    # Original set
    "A Jack Russell bounds through deep powder along an icy shoreline, kicking up white snow with every stride. The freezing winter ocean churns in the background.",
    "A Jack Russell sprints through fresh powder along a frozen shoreline, sending up clouds of snow with each powerful stride.",
    "A small terrier dashes across a snowy beach, paws churning through powder. Waves crash on the frozen shore.",
    "A Jack Russell gallops through a winter wonderland along the coast, snow flying from his paws under a pale sky.",
    # Footprint-focused
    "A Jack Russell runs through pristine snow along the shoreline, leaving a trail of tiny pawprints behind. Each stride sends a puff of powder into the frigid ocean air.",
    "Deep pawprints trail behind a small dog bounding through coastal snow. The winter waves crash nearby as the terrier races ahead, never looking back.",
    "A terrier's legs blur as he sprints along the snowy beach, carving fresh tracks in the untouched powder. The cold sea mist hangs over the frozen sand.",
    # Motion-focused
    "In slow motion, a Jack Russell leaps through a drift of fresh snow, legs fully extended. Crystal snowflakes scatter in every direction against the steel-gray ocean.",
    "A blur of white and brown fur tears across the frozen beach, the little dog's ears streaming behind him like flags in the arctic wind.",
    "The Jack Russell's collar jingles as he bounds from one snow drift to the next along the frozen coastline, his shadow stretching long in the low winter sun.",
    # Atmosphere-focused
    "Under a heavy winter sky, a lone Jack Russell explores the snow-covered beach. The ocean is dark and churning, and snowflakes drift lazily onto his brown and white coat.",
    "Dawn light catches the snow as a small dog races along the water's edge. The beach is transformed into a frozen wonderland, every wave crest topped with ice.",
    "A Jack Russell pauses mid-stride on the snowy shore, breath visible in the cold air, before launching forward again through the deep white powder.",
    # Dramatic
    "Against the dramatic backdrop of crashing winter waves, a tiny Jack Russell charges fearlessly through knee-deep snow, ears pinned back by the bitter coastal wind.",
    "The frozen beach stretches endlessly as a small white dog sprints through fresh snowfall, his paws barely touching the ground between leaps.",
    "Snow explodes around a bounding terrier as he races along the winter shoreline, the setting sun painting the ice-covered waves in shades of gold and purple.",
    # Simple/clean
    "A dog runs through snow on a beach.",
    "Jack Russell terrier playing in snow by the ocean.",
    "Small dog running on a snowy beach at sunset.",
    "A happy terrier bounds through fresh powder beside frozen ocean waves.",
    # Detailed
    "A purebred Jack Russell terrier with a brown and white coat bounds energetically through six inches of fresh powder snow on a wide sandy beach. The North Atlantic ocean churns with winter waves in the background, and snowflakes continue to fall from the overcast sky.",
    "Close-up tracking shot of a Jack Russell terrier running at full speed through deep snow along a frozen beach. His muscles ripple under his short coat as each stride sends up a spray of white powder. The cold ocean is visible in the background.",
]
SEEDS = [42, 17, 99, 256, 7, 314, 1337, 2024, 555, 777, 888, 999, 1111, 2222, 3333, 4444]

def make_test(name, num_frames, nkf, seed_offset, prompt_idx, smooth=False):
    random.seed(seed_offset)
    max_kf = num_frames - 1
    nkf = min(nkf, max_kf)
    kf = sorted(random.sample(range(1, num_frames), nkf - 1) + [0])
    return {
        "name": name,
        "input_video": "raw_inputs/dogrun_slomo4x.mp4",
        "first_frame": "raw_inputs/dogrun_snow_firstframe.png",
        "output_video": f"test_outputs/{name}.mp4",
        "caption": PROMPTS_EXTENDED[prompt_idx % len(PROMPTS_EXTENDED)],
        "num_frames": num_frames,
        "keyframes": kf,
        "smooth": smooth,
        "num_diffusion_steps": 100,
        "seed": SEEDS[seed_offset % len(SEEDS)],
    }

def write_and_run(batch_name, tests):
    with open(TESTS_JSON, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"\n{'='*70}\n  BATCH: {batch_name} ({len(tests)} tests)\n{'='*70}", flush=True)
    for d in ["generated_references", "precomputed_latents"]:
        p = HERE / d
        if p.exists():
            shutil.rmtree(p)
    result = subprocess.run([sys.executable, str(HERE / "run_tests.py"), "run"], cwd=str(HERE.parent))
    print(f"  BATCH {'DONE' if result.returncode == 0 else 'FAILED'}: {batch_name}", flush=True)

batch_num = 11

# Batches 11-14: 361f NN, various kf densities, footprint/motion prompts
for kf_count in [32, 64, 128, 256]:
    tests = [make_test(f"361f_nn_{kf_count}kf_p{i}", 361, kf_count, batch_num*1000+i, 4+i)
             for i in range(8)]
    write_and_run(f"361f_nn_{kf_count}kf_footprint_motion", tests)
    batch_num += 1

# Batches 15-16: 361f smooth, dense kf, atmosphere prompts
for kf_count in [128, 256]:
    tests = [make_test(f"361f_sm_{kf_count}kf_p{i}", 361, kf_count, batch_num*1000+i, 10+i, smooth=True)
             for i in range(8)]
    write_and_run(f"361f_smooth_{kf_count}kf_atmosphere", tests)
    batch_num += 1

# Batches 17-18: 361f NN, dramatic prompts, varying seeds
for kf_count in [48, 96]:
    tests = [make_test(f"361f_nn_{kf_count}kf_dram{i}", 361, kf_count, batch_num*1000+i, 13+i)
             for i in range(8)]
    write_and_run(f"361f_nn_{kf_count}kf_dramatic", tests)
    batch_num += 1

# Batch 19: 361f NN, simple/clean prompts
tests = [make_test(f"361f_nn_64kf_simple{i}", 361, 64, batch_num*1000+i, 16+i)
         for i in range(8)]
write_and_run("361f_nn_64kf_simple_prompts", tests)
batch_num += 1

# Batch 20: 361f NN, detailed prompts
tests = [make_test(f"361f_nn_96kf_detail{i}", 361, 96, batch_num*1000+i, 20+i%2)
         for i in range(8)]
write_and_run("361f_nn_96kf_detailed_prompts", tests)
batch_num += 1

# Batches 21-22: 241f NN, footprint + atmosphere prompts
for kf_count in [48, 128]:
    tests = [make_test(f"241f_nn_{kf_count}kf_fp{i}", 241, kf_count, batch_num*1000+i, 4+i)
             for i in range(8)]
    write_and_run(f"241f_nn_{kf_count}kf_footprint", tests)
    batch_num += 1

# Batches 23-24: 121f NN, detailed + dramatic prompts
for kf_count in [24, 64]:
    tests = [make_test(f"121f_nn_{kf_count}kf_mix{i}", 121, kf_count, batch_num*1000+i, 7+i)
             for i in range(8)]
    write_and_run(f"121f_nn_{kf_count}kf_dramatic_detail", tests)
    batch_num += 1

# Batches 25-26: 361f NN extreme density (every 2-3 frames)
for kf_count in [180, 350]:
    tests = [make_test(f"361f_nn_{kf_count}kf_ext{i}", 361, kf_count, batch_num*1000+i, i)
             for i in range(8)]
    write_and_run(f"361f_nn_{kf_count}kf_extreme_density", tests)
    batch_num += 1

# Batches 27-28: 361f NN sparse (very few keyframes)
for kf_count in [4, 8]:
    tests = [make_test(f"361f_nn_{kf_count}kf_sparse{i}", 361, kf_count, batch_num*1000+i, 10+i)
             for i in range(8)]
    write_and_run(f"361f_nn_{kf_count}kf_sparse", tests)
    batch_num += 1

# Batches 29-30: Mixed frame counts, all prompts cycling
tests = [
    make_test("final_121f_nn_32kf", 121, 32, 290001, 0),
    # make_test("final_121f_sm_64kf", 121, 64, 290002, 5, smooth=True),
    make_test("final_241f_nn_48kf", 241, 48, 290003, 8),
    make_test("final_241f_nn_128kf", 241, 128, 290004, 12),
    make_test("final_361f_nn_64kf", 361, 64, 290005, 15),
    make_test("final_361f_nn_200kf", 361, 200, 290006, 18),
    # make_test("final_361f_sm_128kf", 361, 128, 290007, 20, smooth=True),
    make_test("final_361f_nn_300kf", 361, 300, 290008, 21),
]
write_and_run("final_mixed_all_prompts", tests)

tests = [
    make_test("grand_361f_nn_48kf", 361, 48, 300001, 4),
    make_test("grand_361f_nn_96kf", 361, 96, 300002, 7),
    make_test("grand_361f_nn_160kf", 361, 160, 300003, 11),
    # make_test("grand_361f_sm_96kf", 361, 96, 300004, 14, smooth=True),
    make_test("grand_361f_nn_256kf", 361, 256, 300005, 17),
    make_test("grand_241f_nn_96kf", 241, 96, 300006, 19),
    # make_test("grand_241f_sm_160kf", 241, 160, 300007, 21, smooth=True),
    make_test("grand_121f_nn_48kf", 121, 48, 300008, 3),
]
write_and_run("grand_finale_mixed", tests)

print("\n" + "="*70)
print("  ALL 20 EXTRA BATCHES COMPLETE")
print("="*70)

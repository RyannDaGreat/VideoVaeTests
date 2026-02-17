# Discoveries

**This document records novel findings and techniques discovered during research.
Written for someone with no prior context on this codebase.**

---

## Discovery 1: Image-Aware Classifier-Free Guidance via Token Removal

### Date: 2026-02-17

### The Problem

We wanted to control how strongly a first-frame image anchor influences video
generation in LTX-2, independently from the text prompt's influence. Standard
classifier-free guidance (CFG) only varies the text — the image anchor is always
present in both the positive and negative passes.

### Background: How LTX-2 Conditions on Images

LTX-2 uses **in-context conditioning** for first-frame image-to-video (I2V). This
is architecturally different from most other I2V models:

- The image is VAE-encoded into latent tokens (same format as video frame tokens)
- These tokens **replace** the video's frame 0 tokens in the sequence
- The `denoise_mask` at those positions is set to 0 (meaning "don't denoise, this is clean")
- The `timestep` at those positions is set to 0 (telling the transformer "trust this, it's known data")

The transformer sees a sequence like:

```
[frame0=clean_image, frame1=noisy, frame2=noisy, ..., frameN=noisy]
```

Frame 0 is clean (the anchor). All other frames are noisy and get denoised. The
transformer's attention mechanism propagates information from the clean anchor to
all noisy frames.

When IC-LoRA (Image-Conditioned LoRA) is also used, there's a second set of
conditioning tokens — the reference video — prepended to the sequence:

```
[ref_video_tokens (clean)] + [frame0=clean_image, frame1=noisy, ..., frameN=noisy]
```

Both the reference tokens and the condition image have `denoise_mask=0`, but they
serve completely different roles:
- **IC-LoRA reference**: provides style/content context for the entire video. MUST
  always be present or the model has no idea what to generate.
- **Condition image (frame 0)**: anchors the first frame to a specific image. This
  is what we want to vary for guidance.

### The Approach: Token Removal for the Negative Pass

Standard CFG runs two passes per denoising step:
1. **Positive**: text=prompt, image=anchor → prediction V_pos
2. **Negative**: text=negative_prompt, image=anchor → prediction V_neg
3. **Output**: V_pos + scale × (V_pos − V_neg)

The image anchor appears in BOTH passes, so the CFG delta only captures the text
prompt's effect. We wanted the delta to also capture the image's effect.

Our solution: for the negative pass, **physically remove the condition image tokens
from the sequence**. The transformer sees a shorter sequence:

```
[ref_video_tokens] + [frame1=noisy, frame2=noisy, ..., frameN=noisy]
```

No frame 0. No clean anchor. The IC-LoRA reference is still there (critical). The
positional embeddings for frames 1-N are kept as-is (they're absolute coordinates,
not relative). The transformer handles variable sequence lengths natively via attention.

This is valid because during training, the model saw videos without first-frame
conditioning 20% of the time (`first_frame_conditioning_p=0.8`). So the "no anchor"
state is a known input distribution.

### The Punnett Square

Four possible negative pass configurations form a 2×2 grid:

```
                    text = positive prompt     text = negative prompt
                    ─────────────────────     ──────────────────────
image = present     V_TI (positive pass)      V_I  (standard neg)
image = removed     V_T  (not used)           V_∅  (drop-image neg)
```

Standard CFG uses V_TI and V_I (top row only):
```
output = V_TI + (cfg_scale - 1) × (V_TI − V_I)
```

Drop-image CFG uses V_TI and V_∅ (diagonal):
```
output = V_TI + (cfg_scale - 1) × (V_TI − V_∅)
```

### The cfg_drop_image Parameter

We blend between the two negative passes using a single parameter α ∈ [0, ∞):

```
delta_standard = (cfg_scale - 1) × (V_TI − V_I)     [text delta only]
delta_dropped  = (cfg_scale - 1) × (V_TI − V_∅)     [text + image delta]

At non-conditioned tokens (frames 1-N):
  final_delta = (1 − α) × delta_standard + α × delta_dropped
```

| α value | Behavior | Forward passes |
|---------|----------|---------------|
| 0.0     | Standard CFG (image in both passes) | 2 (V_TI + V_I) |
| 1.0     | Full image drop (image only in positive) | 2 (V_TI + V_∅) |
| 0 < α < 1 | Blend: partial image influence in guidance | 3 (V_TI + V_I + V_∅) |
| α > 1   | Extrapolate: amplify image's guidance effect | 3 (V_TI + V_I + V_∅) |

At α=0 and α=1, only 2 forward passes are needed (the unused one is skipped).
At any other α, all 3 passes run and the deltas are blended.

The CFG delta is **only applied to non-conditioned target tokens** (frames 1-N).
The condition image tokens (frame 0) and the IC-LoRA reference tokens are always
forced clean by the post-processing `denoise_mask`.

### Critical Implementation Details

1. **IC-LoRA reference must ALWAYS be in the sequence.** An earlier version
   incorrectly stripped it (used `denoise_mask < 1` which matched both reference
   AND condition image tokens). This caused completely white/washed-out outputs
   because the model had no context for what to generate.

2. **Token removal, not token noising.** We tried replacing image tokens with
   flow-matching-noised versions `(1−σ)×clean + σ×noise` but this also produced
   artifacts. Physically removing the tokens is cleaner — the model has genuinely
   seen shorter sequences during training.

3. **Positional embeddings are absolute and independent.** Reference tokens and
   target tokens have separately computed positions (time, height, width). Removing
   target frame 0 doesn't require shifting anything — frames 1-N keep their
   original absolute coordinates.

4. **LTX-2 uses flow matching (rectified flow)**, not DDPM. The noising formula is
   `x_t = (1−σ)×x_0 + σ×noise` (linear interpolation, no square roots). This
   matters for any approach that constructs noised versions of clean tokens.

### Matrix Formulation

The four predictions form a 2×2 matrix. The final output at non-conditioned tokens
can be expressed as a weighted sum of all four:

```
output = w_TI × V_TI + w_I × V_I + w_T × V_T + w_∅ × V_∅
```

where the weights are determined by `cfg_scale` (s) and `cfg_drop_image` (α):

```
         ┌                              ┐
         │  text=pos        text=neg    │
    W =  │  ────────        ────────    │
  img=yes│  s               −(1−α)(s−1) │
  img=no │  0               −α(s−1)     │
         └                              ┘
```

Derivation: the blended delta at non-conditioned tokens is:
```
delta = (1−α) × (s−1) × (V_TI − V_I) + α × (s−1) × (V_TI − V_∅)
output = V_TI + delta
       = V_TI + (1−α)(s−1)(V_TI − V_I) + α(s−1)(V_TI − V_∅)
       = [1 + (s−1)] × V_TI − (1−α)(s−1) × V_I − α(s−1) × V_∅
       = s × V_TI − (1−α)(s−1) × V_I − α(s−1) × V_∅
```

So the coefficient matrix is:

```
         ┌                    ┐
    W =  │  s      −(1−α)(s−1)│   ← image present
         │  0      −α(s−1)    │   ← image removed
         └                    ┘
            ↑          ↑
         text=pos   text=neg
```

Verification:
- α=0 (standard): w_TI=s, w_I=−(s−1), w_T=0, w_∅=0 → output = s×V_TI − (s−1)×V_I ✓
- α=1 (full drop): w_TI=s, w_I=0, w_T=0, w_∅=−(s−1) → output = s×V_TI − (s−1)×V_∅ ✓
- α=0.5: both V_I and V_∅ contribute equally with half weight

In the future, the 2×2 weight matrix could be exposed directly as a configurable
parameter, allowing arbitrary combinations of the four predictions without being
constrained to the (s, α) parameterization.

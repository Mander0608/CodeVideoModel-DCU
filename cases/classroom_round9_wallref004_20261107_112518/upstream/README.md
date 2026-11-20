---
title: MiniMax-H3 Inpainting
emoji: 🖌️
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
short_description: Say what to change in a clip and MiniMax-H3 repaints it
suggested_hardware: zero-a10g
---

# MiniMax-H3 — masked video and audio inpainting

Name what should change — "the fox" — and MiniMax-H3 repaints it, keeping everything else in the frame and, by
default, the original soundtrack. SAM 3 segments the subject and tracks it through every frame, so nothing has to be
painted by hand.

This Space is the denoising half of the `ref2va` task with a mask: the `transformer_ref` partition and the two
autoencoders. The Qwen3-VL conditioner runs in
[`qwen3vl-conditioner`](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner), called over the gradio API,
so the 62 GiB encoder is never resident here.

## What the mask actually does

Not ordinary latent inpainting. A preserved row is handed to the model as **conditioning** — clean content pinned at
`t = 0.999`, the timestep MiniMax-H3's keyframe anchors ride at — rather than as the source re-noised to the step's
own sigma, which is a level the checkpoint has never seen a target row claim.

| mask | row timestep | content |
|---|---|---|
| repaint | the schedule's `t` | the model's |
| preserve | `max(t, 0.999)` video, `1.0` audio | the source, clean |
| feathered | `1 - m·σ` | blended to that level |

MiniMax-H3 already carries a per-row timestep vector for its anchors, so nothing about the transformer changes. The
mask is reduced onto three separate grids that a generic resize reproduces none of: the VAE's 16x spatial
compression, then the transformer's 2x2 patch; the VAE's `(1, 4, 4, 4, 4)` frame grouping; and, for audio, **40
latents per second rather than 24 frames per second**.

## Speed and quality

- **A few-step LoRA is on by default** — `lightx2v/Minimax-h3-Turbo`'s reference-partition 4-step adapter, run at six
  steps. Measured against a 28-step run on the same seed and canvas: 64 s against 221 s, with detail inside the mask
  within 1% (4.75 against 4.75 mean gradient). It carries attention and feed-forward modules only, no AdaLN, which is
  why it loads onto the pruned checkpoint at all.
- **Masks are hard by default.** A feathered mask leaves its edge rows at intermediate timesteps holding a *mixture*
  of source and repaint. Paste that through an upscale and crossfade it into a sharp plate and the result is a
  visible band along the mask — reproduced, isolated to that exact combination, and fixed by squaring the mask off
  and generating at the plate's own size.

## Notes from practice

- **Mask geometry decides what a prompt can do.** A box cut through the middle of a subject leaves that subject on all
  four sides, and the model will not contradict the border it was told to preserve.
- **Keep the soundtrack** and the model animates to the words already there. That is the lip-sync recipe.
- **Crop to the painted region.** Cost is set by the canvas, not by how much of the frame changes.
- **Grow the mask when you are replacing a subject, not editing one.** A segmentation hugs its outline, and the model
  will not contradict the border it was told to preserve: given a mask shaped like a walking animal it will put
  whatever you asked for into that shape. Room is what lets it choose its own.
- **Per-shot prompting is unavoidable.** Masking makes the prompt less strict, not optional.

## Credits

The workflow this follows was worked out by the ComfyUI community — Ablejones, Nekodificador and drozbay, whose
[MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments) node pack covers the same ground. No code is
carried over from it (it is GPLv3); the mask geometry here is re-derived from the checkpoint's own constants.

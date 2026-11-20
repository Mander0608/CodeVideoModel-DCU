"""Pixel-space masks onto MiniMax-H3's packed row grid.

MiniMax-H3 denoises one packed sequence whose video rows are frame-major then row-major over ``2 x 2`` latent
patches, and whose audio rows are channel-major over a 40 Hz latent clock. A mask that is going to select *rows*
therefore has to be reduced on three separate grids, none of which a generic resize reproduces:

* **spatially**, by the video VAE's 16x compression, and then again by the transformer's ``2 x 2`` patch — a row is
  one token, so it is masked as a whole and the reduction over its four latent pixels is a maximum;
* **temporally**, by the VAE's chunked causal grouping, which is *not* uniform: the first latent frame of every
  17-frame chunk covers a single pixel frame and the next four cover four each, i.e. the ``(1, 4, 4, 4, 4)`` cycle
  that also spaces MiniMax-H3's rotary time axis;
* **on the audio clock**, at 40 latents per second rather than 24 frames per second — the two streams share a rotary
  clock but not a rate, and aligning an audio mask to the video grid is the mistake that puts a masked soundtrack
  half a beat out.

Everything here is derived from the released checkpoint's own geometry (``clip_length = 17``, ``vae_ratio_t = 4``,
``token_drop = 3``, spatial compression 16, patch ``(1, 2, 2)``, 40 audio latents/s), read off the components rather
than hardcoded wherever a caller has them to hand.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# The VAE's chunked causal grouping, as pixel frames per latent frame within one 17-frame chunk. `clip_length = 17`
# pre-pads to 20 at `vae_ratio_t = 4` and drops the 3 leading tokens, so the chunk's first kept latent frame covers
# one real pixel frame and the remaining four cover four each.
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)


def latent_frame_groups(
    num_frames: int,
    num_latent_frames: int,
    frames_per_chunk: int = MINIMAX_H3_FRAMES_PER_CHUNK,
    frames_per_latent: tuple[int, ...] = MINIMAX_H3_FRAMES_PER_LATENT,
) -> list[tuple[int, int]]:
    r"""
    The pixel frames each latent frame is encoded from.

    Args:
        num_frames (`int`): Pixel frames of the clip, of the form `17 * n + 5`.
        num_latent_frames (`int`): Latent frames the VAE produces for them, `5 * n + 2`.
        frames_per_chunk (`int`, defaults to 17): Pixel frames per VAE chunk, its `clip_length`.
        frames_per_latent (`tuple[int, ...]`, defaults to `(1, 4, 4, 4, 4)`):
            Pixel frames each latent frame of a chunk covers, cycling from the first frame.

    Returns:
        `list[tuple[int, int]]`: one `(start, end)` half-open pixel-frame range per latent frame, covering the clip.
    """
    starts = [0]
    for span in frames_per_latent:
        starts.append(starts[-1] + span)
    cycle = len(frames_per_latent)

    groups = []
    for index in range(num_latent_frames):
        start = (index // cycle) * frames_per_chunk + starts[index % cycle]
        end = start + frames_per_latent[index % cycle]
        groups.append((min(start, num_frames - 1), min(end, num_frames)))
    # A frame count that ends mid-chunk leaves a tail no latent frame's own span reaches; it is encoded into the last
    # latent frame, so that is where its coverage belongs.
    groups[-1] = (groups[-1][0], num_frames)
    return groups


def pixel_mask_to_row_mask(
    mask: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    frames_per_chunk: int = MINIMAX_H3_FRAMES_PER_CHUNK,
    frames_per_latent: tuple[int, ...] = MINIMAX_H3_FRAMES_PER_LATENT,
) -> torch.Tensor:
    r"""
    Reduce a pixel-space video mask to one value per video row of the packed sequence.

    Every reduction is a maximum: a row regenerates as much as the most-masked pixel it covers asks it to. That is the
    safe direction — a token the model must repaint is never accidentally pinned to the source — and it is what makes
    a feathered mask behave, since the softest values survive into the row rather than being averaged away.

    Args:
        mask (`torch.Tensor` of shape `(num_frames, height, width)`):
            The mask over the source clip, `1` where the video regenerates and `0` where it is preserved. Values in
            between are honoured: they place the row part-way down its own schedule.
        num_latent_frames (`int`), latent_height (`int`), latent_width (`int`):
            The generated video's latent shape, i.e. what the layout step resolved.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`): The transformer's `(t, h, w)` patch.
        frames_per_chunk (`int`), frames_per_latent (`tuple[int, ...]`): See [`latent_frame_groups`].

    Returns:
        `torch.Tensor` of shape `(num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w),)`:
        one value per video row, in the frame-major then row-major order [`patchify_video_latents`] produces.
    """
    if mask.ndim != 3:
        raise ValueError(f"A video mask must be `(num_frames, height, width)`, got {tuple(mask.shape)}.")
    _, patch_h, patch_w = patch_size
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(
            f"A {latent_height}x{latent_width} latent canvas is not divisible by the patch {(patch_h, patch_w)}."
        )

    num_frames = mask.shape[0]
    mask = mask.to(torch.float32).clamp(0.0, 1.0)

    # 1. Spatially, onto the latent grid. `adaptive_max_pool2d` divides each axis into `latent_*` near-equal bands,
    # which is the VAE's own 16x split whenever the canvas is a multiple of 16 — and it stays sane when it is not.
    reduced = F.adaptive_max_pool2d(mask[:, None], (latent_height, latent_width))[:, 0]

    # 2. Temporally, onto the VAE's chunked grouping.
    groups = latent_frame_groups(num_frames, num_latent_frames, frames_per_chunk, frames_per_latent)
    reduced = torch.stack([reduced[start:end].amax(dim=0) for start, end in groups])

    # 3. Onto the transformer's patch. A row is one token: it carries one timestep and is written back as a whole, so
    # sub-patch detail cannot survive and the strongest value in the patch is what the row acts on.
    rows = reduced.reshape(
        num_latent_frames, latent_height // patch_h, patch_h, latent_width // patch_w, patch_w
    ).amax(dim=(2, 4))
    return rows.reshape(-1)


def audio_mask_to_row_mask(
    mask: torch.Tensor,
    num_audio_latents: int,
    audio_channels: int = 2,
) -> torch.Tensor:
    r"""
    Reduce a mask over the soundtrack's timeline to one value per audio row of the packed sequence.

    Args:
        mask (`torch.Tensor` of shape `(num_audio_latents,)` or `(n,)`):
            The mask over the generated soundtrack, `1` where the audio regenerates and `0` where it is preserved. A
            mask of another length is resampled onto the audio latent clock with a maximum, so a mask drawn at video
            frame rate — or as a timeline image — lands on the right latents rather than half a beat off.
        num_audio_latents (`int`): Audio latents per channel, i.e. what the layout step resolved.
        audio_channels (`int`, defaults to 2): Channels the soundtrack is packed channel-major over.

    Returns:
        `torch.Tensor` of shape `(num_audio_latents * audio_channels,)`: one value per audio row, channel-major.
    """
    mask = mask.reshape(-1).to(torch.float32).clamp(0.0, 1.0)
    if mask.shape[0] != num_audio_latents:
        mask = F.adaptive_max_pool1d(mask[None, None], num_audio_latents)[0, 0]
    # Channel-major: the layout lays both stereo channels out as two blocks of `num_audio_latents` rows, so one
    # timeline repeats rather than interleaves.
    return mask.repeat(audio_channels)


def quantize_mask(mask: torch.Tensor, levels: int = 256) -> torch.Tensor:
    r"""
    Snap a mask to a grid of `levels` steps, rounding *up* so a partly-masked row never rounds to fully preserved.

    Each distinct mask value becomes a distinct row timestep, and every distinct row timestep is an extra row in the
    transformer's modulation table. A feathered mask left at float32 can carry thousands of them; on this grid it
    carries at most `levels + 1`, and the difference is invisible at the noise levels involved.

    Args:
        mask (`torch.Tensor`): The mask to snap.
        levels (`int`, defaults to 256): Steps to snap to.

    Returns:
        `torch.Tensor`: The snapped mask.
    """
    return torch.ceil(mask * levels) / levels

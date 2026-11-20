"""Crop a masked subject out of a clip, inpaint it small, paste it back.

MiniMax-H3 generates on a 768-short-edge canvas and attends over the whole packed sequence at once, so the cost of a
request is set by its canvas rather than by how much of the frame actually changes. Repainting a face in a 4K plate at
full resolution is mostly spent on pixels the mask preserves — which is why every workflow in the wild crops to the
subject first, and why "reduce the resolution until it stops running out of memory" is the usual advice.

This is the static half of that: one box around everything the mask ever touches, held for the whole clip. A box that
never moves cannot be read as camera motion, which is the failure mode of cropping each frame to its own subject; the
cost is that a subject crossing the frame drags the box out to cover its whole travel. Tracking the subject with a box
that moves as little as possible does better on those clips and is a much larger piece of work.

Nothing here touches the model. Crop before the pipeline, paste after.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def mask_bounding_box(
    mask: np.ndarray,
    frame_height: int,
    frame_width: int,
    crop_scale: float = 0.5,
    multiple: int = 32,
    threshold: float = 0.02,
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.0,
) -> tuple[int, int, int, int]:
    r"""
    One box around everything the mask touches anywhere in the clip.

    Args:
        mask (`np.ndarray` of shape `(num_frames, height, width)`): The mask, over `[0, 1]`.
        frame_height (`int`), frame_width (`int`): The frame the box is cut from, which it is clamped to.
        crop_scale (`float`, defaults to 0.5):
            Padding around the subject, as a fraction of its size. The model needs context around what it repaints —
            a box cut tight to a head gives it nothing to match lighting or motion against.
        multiple (`int`, defaults to 32): What both axes are rounded up to, i.e. the pipeline's `canvas_multiple`.
        threshold (`float`, defaults to 0.02): Mask values at or below this are treated as unmasked.
        min_aspect_ratio (`float`, defaults to 0.25), max_aspect_ratio (`float`, defaults to 4.0):
            The ratios MiniMax-H3 was trained over. A box outside them is widened on its short axis.

    Returns:
        `tuple[int, int, int, int]`: the box as `(top, left, height, width)`, in source pixels.
    """
    if mask.ndim != 3:
        raise ValueError(f"A mask must be `(num_frames, height, width)`, got {tuple(mask.shape)}.")

    covered = mask.max(axis=0) > threshold
    if not covered.any():
        raise ValueError("The mask is empty: there is nothing to inpaint.")
    rows = np.flatnonzero(covered.any(axis=1))
    cols = np.flatnonzero(covered.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    # Padding, as a share of the subject rather than a fixed margin, so it scales with the shot.
    pad_y = (bottom - top) * crop_scale / 2.0
    pad_x = (right - left) * crop_scale / 2.0
    top, bottom = top - pad_y, bottom + pad_y
    left, right = left - pad_x, right + pad_x

    height, width = bottom - top, right - left
    # Back inside the ratios the checkpoint was trained over, by growing the short axis rather than cutting the long
    # one — the subject stays fully inside the box either way.
    ratio = width / height
    if ratio < min_aspect_ratio:
        width = height * min_aspect_ratio
    elif ratio > max_aspect_ratio:
        height = width / max_aspect_ratio

    return _snap_box(
        (top + bottom) / 2.0, (left + right) / 2.0, height, width, frame_height, frame_width, multiple
    )


def _snap_box(
    center_y: float,
    center_x: float,
    height: float,
    width: float,
    frame_height: int,
    frame_width: int,
    multiple: int,
) -> tuple[int, int, int, int]:
    """A centred box onto the `multiple` grid and inside the frame, keeping the subject centred where it can."""
    height = min(int(np.ceil(height / multiple)) * multiple, (frame_height // multiple) * multiple)
    width = min(int(np.ceil(width / multiple)) * multiple, (frame_width // multiple) * multiple)
    height, width = max(height, multiple), max(width, multiple)

    top = int(round(center_y - height / 2.0))
    left = int(round(center_x - width / 2.0))
    # A box pushed off the edge slides back in rather than being cut down: its size is already on the grid, and
    # shrinking it here would drop part of the subject.
    top = max(0, min(top, frame_height - height))
    left = max(0, min(left, frame_width - width))
    return top, left, height, width


def canvas_for_box(
    box_height: int,
    box_width: int,
    multiple: int = 32,
    short_edge: int = 768,
    max_pixels: int = 768 * 1344,
) -> tuple[int, int]:
    r"""
    The canvas to generate a box at.

    Same rule the pipeline applies to any request — short edge first, area capped, both axes rounded — with one
    addition: a box smaller than the canvas is generated at its own size rather than upscaled. Repainting a 320-pixel
    head at 768 and scaling it back down spends the difference on nothing.

    Lower `short_edge` and `max_pixels` together to trade quality for memory; the community workflows run at roughly
    0.4-0.5 MP on 24 GB cards.

    Args:
        box_height (`int`), box_width (`int`): The box, as [`mask_bounding_box`] returned it.
        multiple (`int`, defaults to 32): What both axes round to.
        short_edge (`int`, defaults to 768), max_pixels (`int`, defaults to `768 * 1344`): The canvas budget.

    Returns:
        `tuple[int, int]`: the `(height, width)` to generate at.
    """
    scale = short_edge / min(box_height, box_width)
    if box_height * box_width * scale * scale > max_pixels:
        scale = (max_pixels / (box_height * box_width)) ** 0.5
    scale = min(scale, 1.0)

    height = max(multiple, round(box_height * scale / multiple) * multiple)
    width = max(multiple, round(box_width * scale / multiple) * multiple)
    return height, width


def crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r"""Cut `box` out of a `(num_frames, height, width, ...)` clip or mask. Pixel-exact, no resampling."""
    top, left, height, width = box
    return array[:, top : top + height, left : left + width]


def paste_back(
    frames: np.ndarray,
    generated: np.ndarray,
    box: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    feather: int = 8,
) -> np.ndarray:
    r"""
    Paste an inpainted crop back into the frames it came from.

    Args:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The original `uint8` clip.
        generated (`np.ndarray` of shape `(num_frames, box_height, box_width, 3)` or the canvas size):
            The inpainted crop, rescaled to the box if it was generated at another size.
        box (`tuple[int, int, int, int]`): The box, as `(top, left, height, width)`.
        mask (`np.ndarray` of shape `(num_frames, box_height, box_width)`, *optional*):
            Confine the paste to the mask, blurred by `feather`. Without it the whole box is pasted, feathered only
            at its border. Confining is the safer default on a static plate: the video VAE's decode is not local, so
            a repainted region moves its surroundings slightly even where the mask preserved the latents exactly.
        feather (`int`, defaults to 8): Width of the blend ramp in box pixels.

    Returns:
        `np.ndarray` of shape `(num_frames, height, width, 3)`: the `uint8` clip with the crop pasted in.
    """
    top, left, box_height, box_width = box
    num_frames = frames.shape[0]
    if generated.shape[0] != num_frames:
        raise ValueError(f"The clip has {num_frames} frames and the inpainted crop {generated.shape[0]}.")

    if generated.shape[1:3] != (box_height, box_width):
        generated = np.stack(
            [
                np.asarray(Image.fromarray(frame).resize((box_width, box_height), Image.Resampling.LANCZOS))
                for frame in generated
            ]
        )

    weight = _border_ramp(box_height, box_width, feather)[None]
    if mask is not None:
        if mask.shape[1:3] != (box_height, box_width):
            raise ValueError(
                f"The mask is {mask.shape[1:3]} and the box {(box_height, box_width)}; crop the mask with the clip."
            )
        weight = weight * _feather_mask(mask, feather)

    out = frames.copy()
    region = out[:, top : top + box_height, left : left + box_width].astype(np.float32)
    blended = region + weight[..., None] * (generated.astype(np.float32) - region)
    out[:, top : top + box_height, left : left + box_width] = blended.round().clip(0, 255).astype(np.uint8)
    return out


def _border_ramp(height: int, width: int, feather: int) -> np.ndarray:
    """1 inside the box, ramping to 0 over `feather` pixels at its edges."""
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)

    def axis(length):
        edge = np.minimum(np.arange(length), np.arange(length)[::-1]).astype(np.float32)
        return np.clip((edge + 0.5) / feather, 0.0, 1.0)

    return axis(height)[:, None] * axis(width)[None, :]


def _feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    """The mask, blurred outwards so the paste fades into the plate rather than showing its own edge."""
    if feather <= 0:
        return mask.astype(np.float32)
    blurred = []
    for frame in mask:
        image = Image.fromarray((frame * 255.0).clip(0, 255).astype(np.uint8))
        image = image.filter(ImageFilter.MaxFilter(_odd(feather))).filter(ImageFilter.GaussianBlur(feather / 2.0))
        blurred.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(blurred)


def _odd(value: int) -> int:
    """`MaxFilter` takes an odd kernel size, and 1 is a no-op rather than an error."""
    return max(1, int(value) | 1)

"""Masked video+audio inpainting for MiniMax-H3, on the `ref2va` reference workflow.

`ref2va` conditions on reference images, clips and soundtracks and then generates a whole new video. This module keeps
that conditioning and makes the *target* partly given: a source clip is encoded, a mask says which of its rows the
model repaints, and everything the mask preserves is presented to the transformer the way a keyframe anchor is —
clean content pinned at the visual-conditioning timestep `0.999`, rather than the source noised down to the step's own
sigma.

That distinction is the whole trick, and it is what ComfyUI's MiniMax-H3 masking does too (`scale_latent_inpaint`
returns the anchor rather than the schedule's noise level, and the DiT relabels masked rows). MiniMax-H3 was trained
with clean visual conditioning at `0.999` and has never seen a partially denoised target row claiming to be something
else, so ordinary "re-noise the original to sigma_t" inpainting is off-distribution for it.

Three things follow, and they are the three blocks below:

* **the mask lands on rows, not pixels.** A row is one `2 x 2` latent patch of one latent frame, and the latent
  frames are grouped from pixel frames on the VAE's `(1, 4, 4, 4, 4)` cycle. `h3_inpaint_masks` does that reduction;
  a generic resize does not reproduce it, and on the audio side the clock is 40 latents/s rather than 24 fps.
* **preserved rows are relabelled.** MiniMax-H3 already carries a per-row timestep vector — that is how a keyframe
  anchor rides at `0.999` beside target rows stepping down the schedule — so a mask value `m` simply places its row at
  `1 - m * sigma`, clamped at the conditioning timestep. A fully preserved row lands exactly on `0.999`; a fully
  generated one on the schedule; a feathered one in between, honestly labelled.
* **preserved rows are written back every step.** The scheduler still steps them, and the write-back overrules it, the
  way `MiniMaxH3AudioConditionStep` imposes a given soundtrack in the audio-to-video Space.

The audio stream is masked on the same footing. Preserving the whole soundtrack — the default when a source
soundtrack is passed — is what makes the model animate a mouth *to the words that are already there* rather than
inventing new ones, which is the "keep the original spoken text" recipe from the community workflows.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import attention_backend
from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareConditionLatentsStep,
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3Ref2VAPrepareLatentsStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3DenoiseLoopWrapper,
    MiniMaxH3LoopSchedulerStep,
    MiniMaxH3Ref2VALoopDenoiser,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    encode_vae_condition,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3DecodeStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MiniMaxH3ModularPipeline,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from diffusers.modular_pipelines.modular_pipeline import (
    BlockState,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.utils import logging

from h3_inpaint_masks import audio_mask_to_row_mask, pixel_mask_to_row_mask, quantize_mask


logger = logging.get_logger(__name__)


# The timesteps a fully preserved row is pinned at. MiniMax-H3 holds visual conditioning just short of clean and
# audio conditioning exactly clean, and a preserved inpainting row is conditioning in every sense that matters, so it
# is presented at the levels the checkpoint was trained to read.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def resample_frame_indices(num_frames: int, fps: float, target_fps: float) -> np.ndarray:
    r"""
    The source frame every target-rate frame is taken from.

    Whole frames are held and dropped, never blended, reproducing `ffmpeg`'s `fps` filter the way the reference
    implementation resamples a reference clip. Returned as indices rather than frames so a clip and the mask drawn
    over it go through the same resample.

    Args:
        num_frames (`int`): Frames the source carries.
        fps (`float`): The rate it carries them at.
        target_fps (`float`): The rate to resample onto, i.e. `components.fps`.

    Returns:
        `np.ndarray`: One source index per resampled frame.
    """
    if fps <= 0:
        raise ValueError(f"A source clip must have a positive frame rate, got {fps}.")
    if fps == target_fps:
        return np.arange(num_frames)
    scale = target_fps / fps
    slots = np.floor(np.arange(num_frames) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(num_frames * scale + 0.5))
    return np.repeat(np.arange(num_frames), repeats)


def normalize_source_frames(frames) -> np.ndarray:
    r"""Any accepted clip layout as `uint8` `(num_frames, height, width, 3)`, matching the reference blocks."""
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"A source clip must be `(num_frames, height, width, 3)` RGB, got {tuple(frames.shape)}.")
    return frames


def normalize_source_mask(mask) -> np.ndarray:
    r"""Any accepted mask layout as float32 `(num_frames, height, width)` over `[0, 1]`."""
    if isinstance(mask, list):
        mask = np.stack([np.asarray(frame.convert("L")) for frame in mask])
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 4:
        # `(num_frames, height, width, 1)` or a 3-channel mask painted as an image.
        mask = mask.mean(axis=-1) if mask.shape[-1] in (3, 4) else mask[..., 0]
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise ValueError(f"A mask must be `(num_frames, height, width)`, got {tuple(mask.shape)}.")
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask.clip(0.0, 1.0)


def normalize_waveform(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    r"""A `(channels, num_samples)` waveform as MiniMax-H3's stereo, float32, `target_sample_rate` soundtrack."""
    import torchaudio

    if waveform.ndim == 1:
        waveform = waveform[None]
    waveform = waveform.to(torch.float32)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    waveform = waveform[:2]
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


class MiniMaxH3InpaintSourceGeometryStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Resolves the request's geometry from the source clip: the canvas its aspect ratio maps onto and the "
            "`17 * n + 5` frame count its length rounds to. `ref2va` otherwise generates on MiniMax-H3's own 16:9 "
            "canvas — references never bind the generated geometry — but an inpainting request has to land back on "
            "top of its own footage, so the source is what settles it."
        )

    @property
    def expected_configs(self):
        from diffusers.modular_pipelines.modular_pipeline_utils import ConfigSpec

        return [ConfigSpec("canvas_short_edge", 768), ConfigSpec("canvas_max_pixels", 768 * 1344)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="source_video",
                required=True,
                description=(
                    "The clip being inpainted: a list of images, a `(num_frames, height, width, 3)` array or a "
                    "`(num_frames, 3, height, width)` tensor, `uint8` or floating point over `[0, 1]`."
                ),
            ),
            InputParam(
                name="source_fps",
                type_hint=float,
                description="The rate `source_video` carries its frames at. Left out, MiniMax-H3's own 24 fps.",
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
            InputParam(
                name="num_frames",
                type_hint=int,
                description="Frames to generate. Left out, as many as the source clip has at 24 fps.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            OutputParam(
                "source_frame_indices",
                type_hint=np.ndarray,
                description=(
                    "The source frame every generated frame is taken from, after the 24 fps resample and the "
                    "alignment padding. The clip and the mask drawn over it are both read through it."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        frames = normalize_source_frames(block_state.source_video)
        fps = block_state.source_fps or components.fps

        indices = resample_frame_indices(frames.shape[0], fps, components.fps)
        num_frames = block_state.num_frames or int(indices.shape[0])

        # The ceiling is a model limit, not a preference: past it the rotary clock leaves the trained range.
        max_frames = int(components.max_duration * components.fps)
        if num_frames > max_frames:
            logger.warning(
                f"MiniMax-H3 generates at most {components.max_duration:g} seconds; the source clip is "
                f"{indices.shape[0] / components.fps:.2f}s at {components.fps} fps and is truncated to {max_frames} "
                "frames. Inpaint a longer clip in segments."
            )
            num_frames = max_frames
        num_frames = align_num_frames(num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk)
        if num_frames < int(components.min_duration * components.fps):
            raise ValueError(
                f"MiniMax-H3 generates at least {components.min_duration:g} seconds, i.e. "
                f"{int(components.min_duration * components.fps)} frames; the source clip resolves to {num_frames}."
            )

        # Alignment rounds *up*, so a clip that ends mid-chunk is held on its last frame rather than truncated down to
        # the chunk below — the tail is a repeated frame the caller can trim, not lost footage.
        if indices.shape[0] < num_frames:
            indices = np.concatenate([indices, np.full(num_frames - indices.shape[0], indices[-1])])
        block_state.source_frame_indices = indices[:num_frames]

        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                frames.shape[2],
                frames.shape[1],
                components.canvas_multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
            )
        block_state.num_frames = num_frames

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3InpaintEncodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes the source clip and its soundtrack, and reduces the masks onto the packed sequence's row grid. "
            "The clip goes through the same recipe a keyframe anchor does — ImageNet normalization, a posterior "
            "sampled under the fixed encode seed, float16 rounding, latent normalization — because that is what a "
            "preserved row is presented as. The masks are reduced with a maximum on all three grids: the VAE's 16x "
            "spatial compression, its `(1, 4, 4, 4, 4)` frame grouping, and the transformer's `2 x 2` patch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLMiniMaxH3),
            ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="source_video", required=True, description="The clip being inpainted. See above."),
            InputParam(
                name="source_frame_indices",
                type_hint=np.ndarray,
                required=True,
                description="The source frame every generated frame is taken from.",
            ),
            InputParam(
                name="mask",
                required=True,
                description=(
                    "Which of the source clip's pixels the model repaints: `(num_frames, height, width)` over "
                    "`[0, 1]`, `1` where the video regenerates and `0` where it is preserved, one frame per *source* "
                    "frame. Read at its own resolution — it does not have to match the canvas, only the framing."
                ),
            ),
            InputParam(
                name="source_audio",
                type_hint=torch.Tensor,
                description=(
                    "The source soundtrack, a `(channels, num_samples)` waveform. Left out, MiniMax-H3 writes the "
                    "whole soundtrack itself."
                ),
            ),
            InputParam(
                name="source_audio_sample_rate",
                type_hint=int,
                description="The rate `source_audio` carries its samples at. Left out, the audio VAE's own.",
            ),
            InputParam(
                name="audio_mask",
                type_hint=torch.Tensor,
                description=(
                    "Which of the soundtrack the model rewrites, over the generated duration: `1` regenerates, `0` "
                    "preserves. Any length, resampled onto the 40 Hz audio latent clock. Left out, a source "
                    "soundtrack is preserved whole — which is what makes the model animate to the words already "
                    "there rather than inventing new ones."
                ),
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(name="num_frames", type_hint=int, required=True, description="Resolved number of frames."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "source_latents",
                type_hint=torch.Tensor,
                description="The source clip encoded onto the target canvas, `(1, C, F, latent_h, latent_w)`.",
            ),
            OutputParam(
                "source_audio_rows",
                type_hint=torch.Tensor,
                description="The source soundtrack in channel-major row layout, or None.",
            ),
            OutputParam(
                "inpaint_row_mask",
                type_hint=torch.Tensor,
                description="One value per generated video row: `1` regenerates, `0` preserves.",
            ),
            OutputParam(
                "inpaint_audio_row_mask",
                type_hint=torch.Tensor,
                description="One value per generated audio row, channel-major.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        height, width, num_frames = block_state.height, block_state.width, block_state.num_frames
        ratio = components.vae_spatial_compression_ratio
        num_latent_frames = video_latent_num_frames(
            num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
        )
        num_audio_latents = audio_latent_num_frames(num_frames, components.fps)
        indices = block_state.source_frame_indices

        # 1. The clip, onto the 24 fps grid and the target canvas. LANCZOS is the reference implementation's rescale.
        frames = normalize_source_frames(block_state.source_video)[indices]
        if frames.shape[1:3] != (height, width):
            frames = np.stack(
                [
                    np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
                    for frame in frames
                ]
            )
        pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
        block_state.source_latents = encode_vae_condition(
            components.vae,
            pixels,
            components.pixel_mean,
            components.pixel_std,
            components.keyframe_encode_seed,
        )

        # 2. The mask, on the same 24 fps grid but at its own resolution — the reduction onto the latent grid is a
        # maximum, so resizing it to the canvas first would only cost detail.
        mask = torch.from_numpy(normalize_source_mask(block_state.mask)[indices].copy())
        block_state.inpaint_row_mask = quantize_mask(
            pixel_mask_to_row_mask(mask, num_latent_frames, height // ratio, width // ratio, components.patch_size)
        ).to(device)

        # 3. The soundtrack. A request without one generates its audio outright, so every audio row is masked in.
        num_audio_rows = num_audio_latents * components.audio_channels
        if block_state.source_audio is None:
            block_state.source_audio_rows = None
            block_state.inpaint_audio_row_mask = torch.ones(num_audio_rows, device=device)
        else:
            sample_rate = block_state.source_audio_sample_rate or components.audio_sampling_rate
            waveform = normalize_waveform(block_state.source_audio, sample_rate, components.audio_sampling_rate)
            # The audio VAE right-pads to a whole hop, so exactly `num_audio_latents` hops encode to exactly the rows
            # the layout reserves. Silence pads a soundtrack shorter than the video.
            num_samples = num_audio_latents * components.audio_vae.hop_length
            if waveform.shape[-1] < num_samples:
                waveform = F.pad(waveform, (0, num_samples - waveform.shape[-1]))
            waveform = waveform[:, :num_samples]

            # Under `native` attention explicitly: the audio VAE's encoder carries a causal-attention `pre_block` kept
            # in float32, and cuDNN has no float32 SDPA kernel, so the backend a pipeline selects for its transformer
            # aborts here. Only *encoding* audio reaches this path.
            with attention_backend("native"):
                posterior = components.audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
            latents = posterior.mode().float().transpose(1, 2)
            mean = torch.tensor(components.audio_vae.config.latents_mean, device=latents.device).view(1, 1, -1)
            std = torch.tensor(components.audio_vae.config.latents_std, device=latents.device).view(1, 1, -1)
            rows = ((latents - mean) / std).reshape(-1, components.audio_latent_channels)
            if rows.shape[0] != num_audio_rows:
                raise ValueError(
                    f"The source soundtrack encodes to {rows.shape[0]} audio rows but the layout holds "
                    f"{num_audio_rows}."
                )
            block_state.source_audio_rows = rows

            if block_state.audio_mask is None:
                block_state.inpaint_audio_row_mask = torch.zeros(num_audio_rows, device=device)
            else:
                block_state.inpaint_audio_row_mask = quantize_mask(
                    audio_mask_to_row_mask(
                        block_state.audio_mask, num_audio_latents, components.audio_channels
                    )
                ).to(device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3InpaintPrepareStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Packs the source latents into rows, checks them against the layout, keeps the noise the generated rows "
            "were drawn from, and imposes the source once before the first forward. From here the loop reads a "
            "sequence whose preserved rows already hold the source at the conditioning level."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="source_latents", type_hint=torch.Tensor, required=True, description="The encoded source clip."
            ),
            InputParam(
                name="source_audio_rows",
                type_hint=torch.Tensor,
                description="The encoded source soundtrack in row layout, or None.",
            ),
            InputParam(
                name="inpaint_row_mask", type_hint=torch.Tensor, required=True, description="Per video row."
            ),
            InputParam(
                name="inpaint_audio_row_mask", type_hint=torch.Tensor, required=True, description="Per audio row."
            ),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="The video rows, source imposed."),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="The audio rows, source imposed."),
            OutputParam("source_rows", type_hint=torch.Tensor, description="The source clip in row layout."),
            OutputParam(
                "inpaint_noise_rows",
                type_hint=torch.Tensor,
                description="The noise the generated video rows were drawn from.",
            ),
            OutputParam(
                "inpaint_audio_noise_rows",
                type_hint=torch.Tensor,
                description="The noise the generated audio rows were drawn from.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        num_condition_video_rows = block_state.num_condition_video_rows
        num_condition_audio_rows = block_state.num_condition_audio_rows

        source_rows = patchify_video_latents(
            block_state.source_latents.to(device, torch.float32), components.patch_size
        )
        generated_video_rows = block_state.latents.shape[0] - num_condition_video_rows
        if source_rows.shape[0] != generated_video_rows:
            raise ValueError(
                f"The source clip packs into {source_rows.shape[0]} video rows but the layout reserved "
                f"{generated_video_rows} generated ones. The canvas the layout was built from and the one the source "
                "was encoded at do not agree."
            )
        if block_state.inpaint_row_mask.shape[0] != generated_video_rows:
            raise ValueError(
                f"The mask reduces to {block_state.inpaint_row_mask.shape[0]} rows but the layout reserved "
                f"{generated_video_rows} generated video rows."
            )
        block_state.source_rows = source_rows

        block_state.inpaint_noise_rows = block_state.latents[num_condition_video_rows:].clone()
        block_state.inpaint_audio_noise_rows = block_state.audio_latents[num_condition_audio_rows:].clone()

        # The first forward reads the same presentation every later one does.
        block_state.latents[num_condition_video_rows:] = impose_source(
            block_state.latents[num_condition_video_rows:],
            source_rows,
            block_state.inpaint_noise_rows,
            block_state.inpaint_row_mask,
            VISUAL_COND_TIMESTEP,
            components.scheduler,
        )
        if block_state.source_audio_rows is not None:
            block_state.audio_latents[num_condition_audio_rows:] = impose_source(
                block_state.audio_latents[num_condition_audio_rows:],
                block_state.source_audio_rows.to(device, torch.float32),
                block_state.inpaint_audio_noise_rows,
                block_state.inpaint_audio_row_mask,
                AUDIO_COND_TIMESTEP,
                components.audio_scheduler,
            )

        self.set_block_state(state, block_state)
        return components, state


def impose_source(
    rows: torch.Tensor,
    source_rows: torch.Tensor,
    noise_rows: torch.Tensor,
    row_mask: torch.Tensor,
    condition_timestep: float,
    scheduler,
) -> torch.Tensor:
    r"""
    Blend the source into the rows the mask preserves.

    The preserved content enters at the *conditioning* timestep rather than the step's own — clean for audio, `0.999`
    for video — and the mask does the mixing, so a row with mask `m` ends up carrying content at roughly
    `1 - m * sigma`, which is exactly the timestep the layout labels it with.

    Args:
        rows (`torch.Tensor` of shape `(num_rows, dim)`): The generated rows as the loop last left them.
        source_rows (`torch.Tensor` of shape `(num_rows, dim)`): The source in the same layout.
        noise_rows (`torch.Tensor` of shape `(num_rows, dim)`): The noise those rows were drawn from.
        row_mask (`torch.Tensor` of shape `(num_rows,)`): `1` regenerates, `0` preserves.
        condition_timestep (`float`): The level the source is presented at.
        scheduler (`MiniMaxH3Scheduler`): The stream's scheduler, for its forward process.

    Returns:
        `torch.Tensor`: The rows with the source imposed.
    """
    mask = row_mask.to(rows.dtype).unsqueeze(-1)
    source = source_rows.to(rows)
    if condition_timestep < 1.0:
        source = scheduler.scale_noise(source, condition_timestep, noise_rows.to(rows))
    return mask * rows + (1.0 - mask) * source


def _place_rows(row_mask: torch.Tensor, timestep: float, pin: float) -> torch.Tensor:
    r"""
    Place every generated row of one stream on its own timestep: `1 - m * sigma`, pinned at `pin`.

    The two ends are *selected* rather than computed. Every distinct value here becomes a row of the transformer's
    modulation table, and float32 arithmetic lands `1 - 1 * (1 - t)` a ulp away from `t` — which would split each
    level in two and leave the fully generated rows claiming a timestep the scheduler is not actually stepping them
    on. Computing in float64 and selecting the endpoints keeps a hard mask exactly as cheap as no mask at all.

    Args:
        row_mask (`torch.Tensor` of shape `(num_rows,)`): `1` regenerates, `0` preserves.
        timestep (`float`): The stream's own timestep this step.
        pin (`float`): Where a fully preserved row sits, i.e. the conditioning timestep or the schedule if it is
            already cleaner.

    Returns:
        `torch.Tensor` of shape `(num_rows,)`: the float32 timestep of every row.
    """
    mask = row_mask.detach().cpu().to(torch.float64)
    rows = (1.0 - mask * (1.0 - timestep)).clamp(max=pin)
    rows = torch.where(mask >= 1.0, torch.tensor(timestep, dtype=torch.float64), rows)
    rows = torch.where(mask <= 0.0, torch.tensor(pin, dtype=torch.float64), rows)
    return rows.to(torch.float32)


class MiniMaxH3InpaintSetTimestepsStep(MiniMaxH3SetTimestepsStep):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "The `ref2va` timestep plan, with every generated row relabelled by the mask. A row masked `m` sits at "
            "`1 - m * sigma` of its own stream's schedule, clamped at the conditioning timestep: fully generated rows "
            "keep the schedule, fully preserved ones land on `0.999` (video) or `1.0` (audio) — where the checkpoint "
            "reads clean conditioning — and feathered ones fall honestly in between. MiniMax-H3 already carries a "
            "per-row timestep vector for its anchors, so nothing about the transformer changes."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(
                name="inpaint_row_mask", type_hint=torch.Tensor, required=True, description="Per video row."
            ),
            InputParam(
                name="inpaint_audio_row_mask", type_hint=torch.Tensor, required=True, description="Per audio row."
            ),
        ]

    @staticmethod
    def build_masked_row_timesteps(
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
        num_text_tokens: int,
        video_timestep: float,
        audio_timestep: float,
        condition_video_timestep: float,
        condition_audio_timestep: float,
        video_row_mask: torch.Tensor,
        audio_row_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        [`MiniMaxH3SetTimestepsStep.build_row_timesteps`] with the generated rows placed by the mask.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
        """
        sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
        row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)

        # `1 - t` is the stream's sigma; a row only carries `m` of it, and a fully preserved row is pinned wherever
        # the checkpoint reads conditioning — which the schedule itself overtakes on the last steps.
        video_rows = _place_rows(video_row_mask, video_timestep, max(video_timestep, condition_video_timestep))
        audio_rows = _place_rows(audio_row_mask, audio_timestep, max(audio_timestep, condition_audio_timestep))

        row_timesteps[video_indices[num_condition_video_rows:]] = video_rows
        row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
        row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_rows
        row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        components.audio_scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        block_state.row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in self.build_masked_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), components.keyframe_noise_aug),
                    AUDIO_COND_TIMESTEP,
                    block_state.inpaint_row_mask,
                    block_state.inpaint_audio_row_mask,
                )
            )
            for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
        ]

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3InpaintLoopSchedulerStep(MiniMaxH3LoopSchedulerStep):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Steps both streams, then re-imposes the source on every row the mask preserves. The scheduler is still "
            "stepped over those rows so its step index keeps up with the loop; the write-back is what the next "
            "forward reads. The last step imposes the source clean, so what the mask preserved decodes back to the "
            "footage it came from."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(name="source_rows", type_hint=torch.Tensor, required=True, description="The source clip."),
            InputParam(
                name="source_audio_rows", type_hint=torch.Tensor, description="The source soundtrack, or None."
            ),
            InputParam(
                name="inpaint_noise_rows", type_hint=torch.Tensor, required=True, description="Video row noise."
            ),
            InputParam(
                name="inpaint_audio_noise_rows", type_hint=torch.Tensor, required=True, description="Audio row noise."
            ),
            InputParam(
                name="inpaint_row_mask", type_hint=torch.Tensor, required=True, description="Per video row."
            ),
            InputParam(
                name="inpaint_audio_row_mask", type_hint=torch.Tensor, required=True, description="Per audio row."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        components, block_state = super().__call__(components, block_state, i, t)

        # On the last step the schedule has reached the clean end, and so does the source: the preserved rows are
        # handed to the decoder as the footage itself rather than as the anchor's 0.999.
        last = i + 1 >= block_state.timesteps.numel()
        video_level = 1.0 if last else VISUAL_COND_TIMESTEP

        num_condition_video_rows = block_state.num_condition_video_rows
        block_state.latents[num_condition_video_rows:] = impose_source(
            block_state.latents[num_condition_video_rows:],
            block_state.source_rows,
            block_state.inpaint_noise_rows,
            block_state.inpaint_row_mask,
            video_level,
            components.scheduler,
        )
        if block_state.source_audio_rows is not None:
            num_condition_audio_rows = block_state.num_condition_audio_rows
            block_state.audio_latents[num_condition_audio_rows:] = impose_source(
                block_state.audio_latents[num_condition_audio_rows:],
                block_state.source_audio_rows,
                block_state.inpaint_audio_noise_rows,
                block_state.inpaint_audio_row_mask,
                AUDIO_COND_TIMESTEP,
                components.audio_scheduler,
            )
        return components, block_state


class MiniMaxH3InpaintDenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VALoopDenoiser, MiniMaxH3InpaintLoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `ref2va` denoising loop with the source imposed on every preserved row, one forward per step."


class MiniMaxH3Ref2VAInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3PrepareConditionLatentsStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3Ref2VAPrepareLatentsStep,
        MiniMaxH3InpaintPrepareStep,
        MiniMaxH3InpaintSetTimestepsStep,
        MiniMaxH3InpaintDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        "prepare_layout",
        "prepare_condition_latents",
        "prepare_latents",
        "prepare_latents_ref2va",
        "prepare_inpaint",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self) -> str:
        return (
            "Core denoising workflow for masked `ref2va` inpainting: `MiniMaxH3Ref2VACoreDenoiseStep` with the source "
            "clip imposed on every row the mask preserves, and those rows relabelled onto the conditioning timestep."
        )


class MiniMaxH3Ref2VAInpaintBlocks(SequentialPipelineBlocks):
    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3InpaintSourceGeometryStep,
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VATextEncoderStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3InpaintEncodeStep,
        MiniMaxH3Ref2VAInpaintCoreDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = [
        "source_geometry",
        "setup",
        "text_encoder",
        "reference_encoder",
        "inpaint_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self) -> str:
        return (
            "Masked video and audio inpainting with MiniMax-H3, on the `ref2va` reference workflow: the source clip "
            "settles the canvas and the frame count, the references condition the repaint the way they condition any "
            "`ref2va` request, and the mask decides which rows the model writes. Everything the mask preserves rides "
            "through the loop as conditioning, at the levels the checkpoint reads it at."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos", description="The inpainted video."),
            OutputParam(
                "audio",
                type_hint=torch.Tensor,
                description="The soundtrack of the packed sequence, of shape `(1, 2, num_samples)`.",
            ),
            OutputParam("sampling_rate", type_hint=int, description="Sample rate of the soundtrack in Hz."),
        ]


class MiniMaxH3Ref2VAInpaintGeneratorBlocks(SequentialPipelineBlocks):
    model_name = "minimax-h3"
    # `MiniMaxH3Ref2VAInpaintBlocks` without its text-encoder step, so the 62 GiB Qwen3-VL can live somewhere else and
    # `prompt_embeds` / `text_token_tags` are the whole wire format — the split the reference and audio-to-video
    # Spaces already deploy behind. The geometry step stays: it is also what maps source frames onto the 24 fps grid,
    # and passing the conditioner's resolved `height` / `width` / `num_frames` through it leaves them untouched.
    block_classes = [
        MiniMaxH3InpaintSourceGeometryStep,
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3InpaintEncodeStep,
        MiniMaxH3Ref2VAInpaintCoreDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = ["source_geometry", "setup", "reference_encoder", "inpaint_encoder", "denoise", "decode"]

    @property
    def description(self) -> str:
        return (
            "The denoising half of a split MiniMax-H3 deployment, masked: `MiniMaxH3Ref2VAInpaintBlocks` without its "
            "text-encoder step. The caller has to pass the `height`, `width` and `num_frames` the conditioner "
            "resolved, because the presentation the embeddings were built from encodes that plan."
        )

    @property
    def outputs(self):
        return MiniMaxH3Ref2VAInpaintBlocks.outputs.fget(self)

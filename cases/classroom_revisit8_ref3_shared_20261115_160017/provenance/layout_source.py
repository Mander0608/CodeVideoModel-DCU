    @staticmethod
    def build_ref2va_packed_sequence(
        text_token_tags: torch.Tensor,
        references: list[MiniMaxH3Reference],
        condition_latents: list[torch.Tensor],
        audio_condition_latents: list[torch.Tensor],
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        audio_channels: int,
        audio_tag: int,
        video_tag: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        r"""
        Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision
                block, which MiniMax-H3 tags `0` (video).
            references (`list[MiniMaxH3Reference]`):
                The references, in packed order. Only their modality is read here; the geometry comes from the latents.
            condition_latents (`list[torch.Tensor]`):
                One `(1, channels, num_latent_frames, latent_height, latent_width)` tensor per image and video
                reference, in packed order, as [`MiniMaxH3Ref2VAReferenceEncoderStep`] produced them.
            audio_condition_latents (`list[torch.Tensor]`):
                One `(num_audio_latents * 2, audio_latent_channels)` tensor per audio-bearing reference, in packed
                order.
            num_latent_frames (`int`): Number of target latent frames.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
            audio_channels (`int`): Channels the soundtrack is packed channel-major over, i.e.
                `components.audio_channels`.
            audio_tag (`int`), video_tag (`int`):
                The modality tags an audio and a video row carry, i.e. `components.audio_tag` and
                `components.video_tag`.

        Returns:
            `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
            leading video and audio rows that are references rather than generated.
        """
        _, patch_h, patch_w = patch_size
        num_text_tokens = text_token_tags.shape[0]
        num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        num_target_audio_rows = num_audio_latents * audio_channels

        # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
        # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed as
        # iterators alongside the reference list rather than indexed by it.
        visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
        audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
        num_reference_video_rows = sum(
            frames * (height // patch_h) * (width // patch_w)
            for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
        )
        num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
        sequence_length = (
            num_text_tokens
            + num_reference_video_rows
            + num_reference_audio_rows
            + num_target_audio_rows
            + num_target_video_rows
        )

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
        target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
        # rows end and every block pushes it forward by the time that block occupies.
        video_indices, audio_indices = [], []
        cursor = num_text_tokens
        rotary_time = float(num_text_tokens)
        for reference in references:
            if reference.kind == "image":
                num_latent_frames_, reference_height, reference_width = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
                rows = slice(cursor, cursor + num_video_rows)
                cursor = rows.stop
                video_indices.append(torch.arange(rows.start, rows.stop))
                frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
                position_ids[rows, 0] = rotary_time
                position_ids[rows, 1:] = frame_grid
                # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
                rotary_time += 1.0
            elif reference.kind == "audio":
                num_audio_rows = next(audio_row_counts)
                reference_audio_latents = num_audio_rows // audio_channels
                rows = slice(cursor, cursor + num_audio_rows)
                cursor = rows.stop
                audio_indices.append(torch.arange(rows.start, rows.stop))
                _fill_audio_positions(
                    position_ids, rows, reference_audio_latents, rotary_time, target_width_grid, audio_channels
                )
                rotary_time += float(reference_audio_latents)
            elif reference.kind == "video":
                # A video reference's soundtrack rows are packed immediately before its video rows and share their
                # origin, so the two are rotary-aligned exactly as the generated audio and video are.
                num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
                reference_audio_latents = num_audio_rows // audio_channels
                num_latent_frames_, reference_height, reference_width = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
                audio_rows = slice(cursor, cursor + num_audio_rows)
                video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
                cursor = video_rows.stop
                audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
                video_indices.append(torch.arange(video_rows.start, video_rows.stop))

                frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
                _fill_audio_positions(
                    position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid, audio_channels
                )
                frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
                position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
                position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
                # The rotary time this reference advances the clock by, summed sequentially in float64. That is *not*
                # how `_temporal_position_span_pairwise` sums the same series — it reproduces a numpy pairwise sum,
                # and the two orders differ in the last ulp from 16 latent frames onwards. The reference
                # implementation keeps both, one per call site, so the port has to as well.
                video_span = sum(
                    _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                    for index in range(num_latent_frames_)
                )
                rotary_time += max(float(reference_audio_latents), video_span)
            else:
                raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

        # The generated rows. Target audio and target video share the origin the reference blocks left behind.
        audio_start = cursor
        video_start = audio_start + num_target_audio_rows
        _fill_audio_positions(
            position_ids,
            slice(audio_start, video_start),
            num_audio_latents,
            rotary_time,
            target_width_grid,
            audio_channels,
        )
        frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
        position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
        position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

        video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
        audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = audio_tag
        token_tags[video_indices] = video_tag

        return (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_reference_video_rows,
            num_reference_audio_rows,
        )

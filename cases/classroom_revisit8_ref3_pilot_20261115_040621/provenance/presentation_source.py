    @staticmethod
    def _build_presentation(
        tokenizer,
        prompt: str,
        references: list[MiniMaxH3Reference],
        image_token_counts: list[int],
        video_block_token_counts: list[int],
        video_block_timestamps: list[list[float]],
        text_tag: int = 1,
        video_tag: int = 0,
    ) -> tuple[list[int], list[int]]:
        r"""
        Tokenize MiniMax-H3's presentation of a `ref2va` request.

        Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision
        block for an image, `"<Audio j>: "` alone for audio — a waveform never reaches the conditioner — and `"<Video
        k>: "` plus one timestamped vision block per merged frame pair for a video. A video that carries sound is
        labelled `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The prompt
        follows verbatim, with no chat template and no special tokens.

        Args:
            tokenizer (`Qwen2TokenizerFast`): Tokenizer of the conditioner.
            prompt (`str`): The prompt, appended verbatim.
            references (`list[MiniMaxH3Reference]`): The normalized references, in packed order.
            image_token_counts (`list[int]`): Number of vision tokens of every image reference's block.
            video_block_token_counts (`list[int]`): Number of vision tokens per block of every video reference.
            video_block_timestamps (`list[list[float]]`): The timestamp of every vision block, per video reference.
            text_tag (`int`, *optional*, defaults to 1): MiniMax-H3's modality tag for a text row.
            video_tag (`int`, *optional*, defaults to 0): MiniMax-H3's modality tag for a vision block's rows.

        Returns:
            `tuple[list[int], list[int]]`: the token ids and their modality tags.
        """

        def text(value: str) -> tuple[list[int], list[int]]:
            token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
            return token_ids, [text_tag] * len(token_ids)

        def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
            token_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            return token_ids, [video_tag] * len(token_ids)

        token_ids, token_tags = [], []

        def emit(segment: tuple[list[int], list[int]]) -> None:
            token_ids.extend(segment[0])
            token_tags.extend(segment[1])

        counts = {"image": 0, "video": 0, "audio": 0}
        for reference in references:
            if reference.has_audio:
                counts["audio"] += 1
                emit(text(f"<Audio {counts['audio']}>: "))
            if reference.kind == "image":
                counts["image"] += 1
                emit(text(f"<Picture {counts['image']}>: "))
                emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
            elif reference.kind == "video":
                counts["video"] += 1
                emit(text(f"<Video {counts['video']}>: "))
                for timestamp in video_block_timestamps[counts["video"] - 1]:
                    # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                    emit(text(f"<{timestamp:.1f} seconds>"))
                    emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
        emit(text(prompt))
        return token_ids, token_tags

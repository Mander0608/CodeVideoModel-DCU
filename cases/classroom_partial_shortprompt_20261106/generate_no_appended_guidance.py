def generate(proxy, image, prompt, model_root, output, *, seed=SEED, keep_steps=KEEP_STEPS,
             text_device="cuda:1", generation_device="cuda:0"):
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= keep_steps <= STEPS:
        raise ValueError(f"keep_steps must lie in [0, {STEPS}].")
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    if not image.is_file():
        raise FileNotFoundError(image)
    check_video(proxy)
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames
    from diffusers.utils.export_utils import encode_video
    runner = H3(text_device, generation_device)
    temporary = output.with_suffix(".partial.mp4")
    try:
        with torch.inference_mode():
            runner.load(model_root)
            g = runner.generation
            ref_video, video_latent = runner.proxy(proxy)
            ref_image, image_latent = runner.image(image)
            shape = video_latent.shape
            audio_frames = audio_latent_num_frames(FRAMES)
            geometry = (*shape[2:], audio_frames)
            image_rows = runner.reference_rows(
                image_latent, stream_seed(seed, "target_first_frame_condition_noise", -20))
            video_rows = runner.reference_rows(video_latent, stream_seed(seed, "ref2va_condition_noise"))
            prompt = prompt.rstrip()
            base = runner.condition(prompt, [ref_image], [image_latent], image_rows, geometry)
            full = runner.condition(prompt, [ref_image, ref_video], [image_latent, video_latent],
                                    torch.cat([image_rows, video_rows]), geometry)
            for scheduler, shift in ((g.scheduler, 12), (g.audio_scheduler, 3)):
                scheduler.set_timesteps(STEPS + 1, device=runner.device)
                if len(scheduler.timesteps) != STEPS or scheduler.shift != shift:
                    raise ValueError("Expected the native H3 20-update schedule with shifts 12/3.")
            if runner.text_device == runner.device:
                runner.managers[0].disable_auto_cpu_offload()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rng = torch.Generator(device="cpu").manual_seed(stream_seed(seed, "sdedit_base_noise"))
            video = torch.randn(shape, generator=rng, dtype=torch.float32).to(runner.device)
            audio = torch.randn((audio_frames * g.audio_channels, g.audio_latent_channels),
                                generator=rng, dtype=torch.float32).to(runner.device)
            video = patchify_video_latents(video, g.patch_size)
            for i in range(STEPS):
                condition = full if i < keep_steps else base
                tv, ta = g.scheduler.timesteps[i], g.audio_scheduler.timesteps[i]
                vp, ap = runner.predict(condition, video, audio, float(tv), float(ta))
                video = advance(video, vp, tv, g.scheduler.sigmas[i], g.scheduler.sigmas[i + 1])
                audio = advance(audio, ap, ta, g.audio_scheduler.sigmas[i], g.audio_scheduler.sigmas[i + 1])
            frames, sound = runner.decode(unpatchify(video, shape, g.patch_size), audio, audio_frames)
            output.parent.mkdir(parents=True, exist_ok=True)
            encode_video(frames, fps=FPS, output_path=str(temporary), audio=sound,
                         audio_sample_rate=int(g.audio_sampling_rate))
            check_video(temporary, audio=True)
            temporary.replace(output)
    finally:
        runner.close()
        temporary.unlink(missing_ok=True)

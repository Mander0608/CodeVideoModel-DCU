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
            image_refs, image_latents, image_rows_list = [], [], []
            for spec in CASE_REFS:
                ref, latent = runner.image(EXPERIMENT_ROOT / spec['path'])
                rows = runner.reference_rows(latent, stream_seed(seed, spec['noise_stream'], spec['noise_step']))
                image_refs.append(ref); image_latents.append(latent); image_rows_list.append(rows)
                REFERENCE_AUDIT.append(dict(path=spec['path'], size=list(ref.image.size), latent_shape=list(latent.shape),
                    latent_sha256=TENSOR_SHA(latent), row_sha256=TENSOR_SHA(rows), rows=int(rows.shape[0])))
                SAVE_AUDIT()
                print('REFERENCE_ENCODED', spec['path'], REFERENCE_AUDIT[-1], flush=True)
            image_rows = torch.cat(image_rows_list)
            shape = video_latent.shape
            audio_frames = audio_latent_num_frames(FRAMES)
            geometry = (*shape[2:], audio_frames)
            # Independent, stable noise streams assigned above.
            video_rows = runner.reference_rows(video_latent, stream_seed(seed, "ref2va_condition_noise"))
            prompt = prompt.rstrip()
            base = runner.condition(prompt, image_refs, image_latents, image_rows, geometry)
            full = runner.condition(prompt, image_refs + [ref_video], image_latents + [video_latent],
                                    torch.cat([image_rows, video_rows]), geometry)
            CONDITION_AUDIT.update(image_count=len(image_refs), image_rows=[int(x.shape[0]) for x in image_rows_list],
                base_rows=int(base.rows.shape[0]), full_rows=int(full.rows.shape[0]), proxy_rows=int(video_rows.shape[0]),
                proxy_row_sha256=TENSOR_SHA(video_rows), base_embeds_shape=list(base.embeds.shape), full_embeds_shape=list(full.embeds.shape),
                reference_image_short_edge=int(g.config.get('reference_image_short_edge',2048)))
            assert CONDITION_AUDIT['base_rows'] == sum(CONDITION_AUDIT['image_rows'])
            assert CONDITION_AUDIT['full_rows'] == CONDITION_AUDIT['base_rows'] + CONDITION_AUDIT['proxy_rows']
            SAVE_AUDIT()
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
            CONDITION_AUDIT.update(target_initial_noise_sha256=TENSOR_SHA(video), audio_initial_noise_sha256=TENSOR_SHA(audio))
            SAVE_AUDIT()
            video = patchify_video_latents(video, g.patch_size)
            for i in range(STEPS):
                condition = full if i < keep_steps else base
                print('CONDITION_STEP',i+1,'images',len(image_refs),'proxy',i < keep_steps,flush=True)
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
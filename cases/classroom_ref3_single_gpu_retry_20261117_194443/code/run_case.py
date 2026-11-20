"""Independent 1/3-image adapter; never edits shared CVM code."""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, os, runpy, subprocess, sys, time

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def adapted_source(native):
    # Deliberately keep the original denoising/decode schedule byte-for-byte.
    tree = ast.parse(native)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
    src = ast.get_source_segment(native, fn)
    changes = [
        ('if not image.is_file():\n        raise FileNotFoundError(image)',
         "for item in REF_CONFIG:\n        if not (CASE_DIR / item['path']).is_file():\n            raise FileNotFoundError(item['path'])"),
        ('ref_image, image_latent = runner.image(image)',
         "ref_images, image_latents, all_image_rows = [], [], []\n"
         "            for index, item in enumerate(REF_CONFIG):\n"
         "                ref_image, image_latent = runner.image(CASE_DIR / item['path'])\n"
         "                ref_images.append(ref_image); image_latents.append(image_latent)\n"
         "                ref_seed = stream_seed(seed, item['noise_stream'], item['noise_step'])\n"
         "                all_image_rows.append(runner.reference_rows(image_latent, ref_seed))\n"
         "                REFERENCE_AUDIT.append(dict(picture=index+1, sha256=item['sha256'], role=item['role'],\n"
         "                    size=list(ref_image.image.size), latent_shape=list(image_latent.shape),\n"
         "                    rows=int(all_image_rows[-1].shape[0]), noise_seed=ref_seed,\n"
         "                    row_sha256=hashlib.sha256(all_image_rows[-1].detach().cpu().contiguous().numpy().tobytes()).hexdigest()))\n"
         "                print('REFERENCE_READY', index+1, REFERENCE_AUDIT[-1], flush=True)"),
        ('image_rows = runner.reference_rows(\n                image_latent, stream_seed(seed, "target_first_frame_condition_noise", -20))',
         'image_rows = torch.cat(all_image_rows)'),
        ('prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE', 'prompt = prompt.rstrip()'),
        ('base = runner.condition(prompt, [ref_image], [image_latent], image_rows, geometry)',
         'base = runner.condition(prompt, ref_images, image_latents, image_rows, geometry)'),
        ('full = runner.condition(prompt, [ref_image, ref_video], [image_latent, video_latent],',
         'full = runner.condition(prompt, ref_images + [ref_video], image_latents + [video_latent],'),
        ('for scheduler, shift in ((g.scheduler, 12), (g.audio_scheduler, 3)):',
         "CONDITION_AUDIT.update(base_reference_count=len(ref_images), full_reference_count=len(ref_images)+1,\n"
         "                base_rows=int(base.rows.shape[0]), full_rows=int(full.rows.shape[0]),\n"
         "                image_rows=int(image_rows.shape[0]), proxy_rows=int(video_rows.shape[0]),\n"
         "                base_text_tokens=int(base.embeds.shape[-2]), full_text_tokens=int(full.embeds.shape[-2]),\n"
         "                reference_short_edge=int(g.config.get('reference_image_short_edge',2048)))\n"
         "            assert CONDITION_AUDIT['base_rows'] == sum(x['rows'] for x in REFERENCE_AUDIT)\n"
         "            assert CONDITION_AUDIT['full_rows'] == CONDITION_AUDIT['base_rows'] + video_rows.shape[0]\n"
         "            save_audit()\n"
         "            for scheduler, shift in ((g.scheduler, 12), (g.audio_scheduler, 3)):"),
        ('video = patchify_video_latents(video, g.patch_size)',
         "CONDITION_AUDIT['target_initial_noise_sha256'] = hashlib.sha256(video.detach().cpu().contiguous().numpy().tobytes()).hexdigest()\n"
         "            CONDITION_AUDIT['audio_initial_noise_sha256'] = hashlib.sha256(audio.detach().cpu().contiguous().numpy().tobytes()).hexdigest()\n"
         "            save_audit()\n"
         "            video = patchify_video_latents(video, g.patch_size)"),
        ('condition = full if i < keep_steps else base',
         "condition = full if i < keep_steps else base\n"
         "                print('CONDITION_STEP', i+1, 'image_count', len(ref_images), 'proxy', i < keep_steps, flush=True)"),
    ]
    for old, new in changes:
        assert src.count(old) == 1, ('Native runner changed', old)
        src = src.replace(old, new)
    compile(src, '<adapted_generate>', 'exec')
    return src

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    a = args.case.resolve(); root = a.parent; cfg = read(a / 'config.json')
    project = Path('/public/home/lixin1/chenxy/CodeVideoModel')
    expected = read(root / 'shared_source_checksums.json')
    for rel, h in expected.items(): assert sha(project / rel) == h, rel
    for rel, h in read(root / 'frozen_manifest.json').items(): assert sha(root / rel) == h, rel
    for ref in cfg['references']: assert sha(a / ref['path']) == ref['sha256']
    assert sha(a / cfg['proxy']) == cfg['source_proxy_sha256']
    review = read(a / 'review/input_review.json')
    assert review['status'] == 'passed' and review['prompt_sha256'] == sha(a / cfg['prompt'])
    assert review['reference_sha256'] == [r['sha256'] for r in cfg['references']]
    assert review['proxy_sha256'] == sha(a / cfg['proxy'])
    assert cfg['seed'] == 42 and cfg['steps'] == 20 and cfg['keep_steps'] == 5
    assert cfg['reference_short_edge_override'] is None and not os.environ.get('CVM_SMOKE_TEST')
    source = adapted_source((project / 'cvm/video.py').read_text())
    if not args.execute and not args.preflight:
        print(json.dumps({'execute': False, 'case': str(a), 'image_count': len(cfg['references']),
                          'adapted_code_sha256': hashlib.sha256(source.encode()).hexdigest()})); return
    sys.path.insert(0, str(project))
    import torch
    from cvm import check_video, video
    check_video(a / cfg['proxy'])
    assert torch.cuda.device_count() == 1
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '1'
    free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(1)]
    policy = cfg['resource_policy']
    assert all(f >= m for f,m in zip(free,policy['minimum_free_GiB'])), free
    total = [torch.cuda.mem_get_info(i)[1] / 2**30 for i in range(1)]
    budgets = [min(free[i]-policy['leave_free_headroom_GiB'], policy['allocator_absolute_max_GiB'][i]) for i in range(1)]
    for i, budget in enumerate(budgets):
        torch.cuda.set_per_process_memory_fraction(budget / total[i], i)
    resource_report = {'free_GiB_at_start':free,'total_GiB':total,'allocator_cap_GiB':budgets,
                       'minimum_free_GiB':policy['minimum_free_GiB'],'headroom_GiB':policy['leave_free_headroom_GiB'],
                       'scope':'PyTorch allocator cap for this process; non-PyTorch allocations and other processes are not controlled.'}
    save(a / 'resource_budget.json', resource_report)
    print('RESOURCE_BUDGET', json.dumps(resource_report), flush=True)
    from PIL import Image
    for ref in cfg['references']:
        with Image.open(a / ref['path']) as im: assert im.size == (960, 544)
    if args.preflight:
        save(a / 'preflight.json', {'status': 'passed', 'free_GiB': free, 'image_count': len(cfg['references']),
                                   'input_checksums_match': True, 'shared_source_sha256': expected})
        print('PREFLIGHT_PASS', cfg['job_id'], flush=True); return
    output = a / cfg['output']; output.parent.mkdir(exist_ok=False)
    (output.parent / 'generate_effective.py').write_text(source, encoding='utf-8')
    prompt = (a / cfg['prompt']).read_text(encoding='utf-8').strip()
    (output.parent / 'effective_prompt.txt').write_text(prompt, encoding='utf-8')
    runpy.run_path(str(project / 'run_dcu.py'), run_name='dcu_adapter_import')
    save(output.parent / 'runtime_environment.json', {
        'visible_devices': os.environ.get('HIP_VISIBLE_DEVICES'),
        'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0),
        'text_device': 'cuda:0', 'generation_device': 'cuda:0',
        'python': sys.executable, 'torch_file': torch.__file__,
        'torch': torch.__version__, 'hip': torch.version.hip,
        'offload': 'Existing text leaf CPU offload and generation block CPU offload unchanged'})
    refaudit, condaudit = [], {}
    def audit(): save(output.parent / 'condition_audit.json', {'references': refaudit, 'conditions': condaudit})
    video.__dict__.update(REF_CONFIG=cfg['references'], CASE_DIR=a, REFERENCE_AUDIT=refaudit,
                          CONDITION_AUDIT=condaudit, save_audit=audit)
    exec(compile(source, str(output.parent / 'generate_effective.py'), 'exec'), video.__dict__)
    started = time.monotonic()
    save(output.parent / 'launch_record.json', {'config': cfg, 'source_sha256': expected,
         'runner_sha256': sha(Path(__file__)), 'started_at': datetime.datetime.now().astimezone().isoformat()})
    video.generate(a / cfg['proxy'], a / cfg['references'][0]['path'], prompt,
                   project / 'models/MiniMax-H3-ref2va', output, seed=42, keep_steps=5,
                   text_device='cuda:0', generation_device='cuda:0')
    media = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(output)]))
    stream = next(s for s in media['streams'] if s['codec_type'] == 'video')
    assert (stream['width'], stream['height'], stream['avg_frame_rate'], int(stream['nb_read_frames'])) == (960, 544, '24/1', 124)
    save(output.with_suffix('.run.json'), {'status': 'completed', 'elapsed_seconds': time.monotonic()-started,
        'media': media, 'sha256': sha(output), 'torch': torch.__version__, 'hip': torch.version.hip,
        'image_count': len(cfg['references']), 'actual_updates_expected': 20,
        'peak_allocated_GiB': [torch.cuda.max_memory_allocated(i)/2**30 for i in range(1)],
        'peak_reserved_GiB': [torch.cuda.max_memory_reserved(i)/2**30 for i in range(1)]})
    print('RUN_COMPLETE', cfg['job_id'], flush=True)

if __name__ == '__main__': main()

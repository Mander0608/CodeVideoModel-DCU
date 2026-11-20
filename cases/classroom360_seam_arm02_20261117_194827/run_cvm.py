"""Case-local one/three-image seam ablation adapter. Does not modify shared CVM source."""
from pathlib import Path
import argparse, hashlib, inspect, json, runpy, sys
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text()); project=a.project_root
    for rel,h in cfg['expected_source_sha256'].items():assert sha(project/rel)==h,rel
    refs=cfg['reference_images']; assert len(refs) == cfg['expected_image_count'] and len(refs) in (1,3)
    assert refs[0]['role']=='start' and refs[0]['noise_stream']=='additional_start_frame_condition_noise'
    assert len({r['noise_stream'] for r in refs})==len(refs)
    for ref in refs:assert (ROOT/ref['path']).is_file()
    argv=[str(project/'run_dcu.py'),'--code-video',str(ROOT/cfg['code_video']),'--reference-image',str(ROOT/refs[0]['path']),
          '--prompt-file',str(ROOT/cfg['prompt_file']),'--model-root',str(project/'models/MiniMax-H3-ref2va'),
          '--text-device',cfg['text_device'],'--generation-device',cfg['generation_device'],'--keep-steps',str(cfg['keep_steps']),
          '--seed',str(cfg['seed']),'--output',str(ROOT/cfg['output'])]
    sys.path.insert(0,str(project));from cvm import video
    assert (video.STEPS,video.FRAMES,video.WIDTH,video.HEIGHT,video.FPS)==(20,124,960,544,24)
    source=inspect.getsource(video.generate)
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,repr(old)
        source=source.replace(old,new)
    replace('            ref_image, image_latent = runner.image(image)',
'''            image_refs, image_latents, image_rows_list = [], [], []
            for spec in CASE_REFS:
                ref, latent = runner.image(CASE_ROOT / spec['path'])
                stream = spec['noise_stream']
                rows = runner.reference_rows(latent, stream_seed(seed, stream, -20))
                image_refs.append(ref); image_latents.append(latent); image_rows_list.append(rows)
                print('REFERENCE_ENCODED', spec['role'], spec['path'], list(latent.shape), flush=True)
            image_rows = torch.cat(image_rows_list)
'''.rstrip())
    replace('            image_rows = runner.reference_rows(\n                image_latent, stream_seed(seed, "target_first_frame_condition_noise", -20))','            # Destination noise unchanged; independent extra stream for starting reference.')
    replace('            prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE','            prompt = prompt.rstrip()')
    replace('            base = runner.condition(prompt, [ref_image], [image_latent], image_rows, geometry)',
            '            base = runner.condition(prompt, image_refs, image_latents, image_rows, geometry)')
    replace('            full = runner.condition(prompt, [ref_image, ref_video], [image_latent, video_latent],\n                                    torch.cat([image_rows, video_rows]), geometry)',
'''            full = runner.condition(prompt, image_refs + [ref_video], image_latents + [video_latent],
                                    torch.cat([image_rows, video_rows]), geometry)
            audit = {'image_order': CASE_REFS, 'image_count': len(image_refs),
                     'image_latent_shapes': [list(x.shape) for x in image_latents],
                     'image_rows': [int(x.shape[0]) for x in image_rows_list],
                     'base_reference_rows': int(base.rows.shape[0]), 'full_reference_rows': int(full.rows.shape[0]),
                     'base_embeds_shape': list(base.embeds.shape), 'full_embeds_shape': list(full.embeds.shape),
                     'reference_image_short_edge': g.config.get('reference_image_short_edge', 2048),
                     'schedule': {'updates_1_5': 'all images + proxy', 'updates_6_20': 'all images'}}
            (CASE_ROOT/'outputs/conditioning_audit.json').write_text(CASE_JSON.dumps(audit,indent=2))
            print('CONDITIONING_AUDIT', CASE_JSON.dumps(audit), flush=True)''')
    replace('            video = patchify_video_latents(video, g.patch_size)',
'''            noise_record = {'video': hashlib.sha256(video.cpu().numpy().tobytes()).hexdigest(),
                            'audio': hashlib.sha256(audio.cpu().numpy().tobytes()).hexdigest()}
            (CASE_ROOT/'outputs/initial_noise_sha256.json').write_text(CASE_JSON.dumps(noise_record,indent=2))
            video = patchify_video_latents(video, g.patch_size)''')
    compile(source,'generate_effective.py','exec')
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    (out/'generate_effective.py').write_text(source,encoding='utf-8')
    prompt=(ROOT/cfg['prompt_file']).read_text(encoding='utf-8').strip()
    (out/'effective_prompt.txt').write_text(prompt,encoding='utf-8')
    record={'argv':argv,'config':cfg,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'effective_generate_sha256':sha(out/'generate_effective.py'),'execute':a.execute}
    (out/'launch_record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
    if not a.execute:return
    video.__dict__.update(CASE_ROOT=ROOT,CASE_REFS=refs,CASE_JSON=json)
    exec(compile(source,str(out/'generate_effective.py'),'exec'),video.__dict__)
    sys.argv=argv;runpy.run_path(str(project/'run_dcu.py'),run_name='__main__')
if __name__=='__main__':main()

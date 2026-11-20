"""Case-local one/two-reference CVM adapter; shared sources stay unchanged."""
from pathlib import Path
import argparse, ast, datetime, hashlib, importlib.metadata, json, os, runpy, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2),encoding='utf-8');temp.replace(p)
def adapt(native):
    fn=next(x for x in ast.parse(native).body if isinstance(x,ast.FunctionDef) and x.name=='generate')
    src=ast.get_source_segment(native,fn)
    def replace(old,new):
        nonlocal src
        assert src.count(old)==1,('Changed native source',old)
        src=src.replace(old,new)
    replace('            ref_image, image_latent = runner.image(image)', '''            image_refs, image_latents, image_rows_list = [], [], []
            for spec in CASE_REFS:
                ref, latent = runner.image(EXPERIMENT_ROOT / spec['path'])
                rows = runner.reference_rows(latent, stream_seed(seed, spec['noise_stream'], spec['noise_step']))
                image_refs.append(ref); image_latents.append(latent); image_rows_list.append(rows)
                REFERENCE_AUDIT.append(dict(path=spec['path'], size=list(ref.image.size), latent_shape=list(latent.shape),
                    latent_sha256=TENSOR_SHA(latent), row_sha256=TENSOR_SHA(rows), rows=int(rows.shape[0])))
                SAVE_AUDIT()
                print('REFERENCE_ENCODED', spec['path'], REFERENCE_AUDIT[-1], flush=True)
            image_rows = torch.cat(image_rows_list)''')
    replace('            image_rows = runner.reference_rows(\n                image_latent, stream_seed(seed, "target_first_frame_condition_noise", -20))','            # Independent, stable noise streams assigned above.')
    replace('            prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE','            prompt = prompt.rstrip()')
    replace('            base = runner.condition(prompt, [ref_image], [image_latent], image_rows, geometry)','            base = runner.condition(prompt, image_refs, image_latents, image_rows, geometry)')
    replace('            full = runner.condition(prompt, [ref_image, ref_video], [image_latent, video_latent],\n                                    torch.cat([image_rows, video_rows]), geometry)', '''            full = runner.condition(prompt, image_refs + [ref_video], image_latents + [video_latent],
                                    torch.cat([image_rows, video_rows]), geometry)
            CONDITION_AUDIT.update(image_count=len(image_refs), image_rows=[int(x.shape[0]) for x in image_rows_list],
                base_rows=int(base.rows.shape[0]), full_rows=int(full.rows.shape[0]), proxy_rows=int(video_rows.shape[0]),
                proxy_row_sha256=TENSOR_SHA(video_rows), base_embeds_shape=list(base.embeds.shape), full_embeds_shape=list(full.embeds.shape),
                reference_image_short_edge=int(g.config.get('reference_image_short_edge',2048)))
            assert CONDITION_AUDIT['base_rows'] == sum(CONDITION_AUDIT['image_rows'])
            assert CONDITION_AUDIT['full_rows'] == CONDITION_AUDIT['base_rows'] + CONDITION_AUDIT['proxy_rows']
            SAVE_AUDIT()''')
    replace('            video = patchify_video_latents(video, g.patch_size)', '''            CONDITION_AUDIT.update(target_initial_noise_sha256=TENSOR_SHA(video), audio_initial_noise_sha256=TENSOR_SHA(audio))
            SAVE_AUDIT()
            video = patchify_video_latents(video, g.patch_size)''')
    replace('                condition = full if i < keep_steps else base', '''                condition = full if i < keep_steps else base
                print('CONDITION_STEP',i+1,'images',len(image_refs),'proxy',i < keep_steps,flush=True)''')
    compile(src,'<adapted_generate>','exec');return src

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    case=ROOT/'cases'/args.job;cfg=read(case/'config.json')
    for rel,h in read(ROOT/'frozen_manifest.json').items():assert sha(ROOT/rel)==h,rel
    for rel,h in read(ROOT/'source_checksums.json').items():assert sha(PROJECT/rel)==h,rel
    source=adapt((PROJECT/'cvm/video.py').read_text())
    assert len(cfg['references']) in (1,2) and (cfg['seed'],cfg['steps'],cfg['keep_steps'])==(42,20,5)
    assert not os.environ.get('CVM_SMOKE_TEST')
    if not args.execute and not args.preflight:
        print('DRY_RUN_OK',args.job,hashlib.sha256(source.encode()).hexdigest());return
    sys.path.insert(0,str(PROJECT))
    import torch
    from PIL import Image
    from cvm import video,check_video
    assert (video.STEPS,video.FRAMES,video.WIDTH,video.HEIGHT,video.FPS)==(20,124,960,544,24)
    assert torch.cuda.device_count()==2
    free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)]
    assert min(free)>60,free
    check_video(ROOT/cfg['proxy'])
    for r in cfg['references']:
        with Image.open(ROOT/r['path']) as im:assert im.size==(960,544)
    if args.preflight:
        save(case/'preflight.json',dict(status='passed',free_GiB=free,physical_devices=os.environ['HIP_VISIBLE_DEVICES']))
        print('PREFLIGHT_PASS',args.job,free,flush=True);return
    out=case/'outputs';out.mkdir(exist_ok=False)
    (out/'generate_effective.py').write_text(source)
    prompt=(ROOT/cfg['prompt']).read_text(encoding='utf-8').strip();(out/'effective_prompt.txt').write_text(prompt,encoding='utf-8')
    runpy.run_path(str(PROJECT/'run_dcu.py'),run_name='dcu_adapter_import')
    reference_audit=[];condition_audit={};cache_audit=[]
    def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def save_audit():save(out/'condition_audit.json',dict(references=reference_audit,conditions=condition_audit,cache=cache_audit))
    original_h3=video.H3
    cache=ROOT/'latent_cache'/cfg['scene'];cache.mkdir(parents=True,exist_ok=True)
    class CachedH3(original_h3):
        def encode_pixels(self,pixels):
            # A creates canonical VAE latents. B reuses identical pixels' latents,
            # avoiding inter-run VAE numerical differences in the paired comparison.
            key=tensor_sha(pixels);path=cache/(key+'.pt');meta_path=cache/(key+'.json')
            if path.exists():
                meta=read(meta_path);assert sha(path)==meta['file_sha256']
                latent=torch.load(path,map_location='cpu',weights_only=True)
                assert tensor_sha(latent)==meta['latent_sha256']
                latent=latent.to(self.device);mode='reused'
            else:
                latent=super().encode_pixels(pixels)
                assert torch.isfinite(latent).all()
                temporary=path.with_suffix('.tmp');torch.save(latent.detach().cpu(),temporary);temporary.replace(path)
                meta=dict(pixel_sha256=key,pixel_shape=list(pixels.shape),latent_sha256=tensor_sha(latent),latent_shape=list(latent.shape),file_sha256=sha(path),created_by=cfg['job_id'])
                save(meta_path,meta);mode='created'
            cache_audit.append(dict(mode=mode,path=str(path.relative_to(ROOT)),**meta));save_audit()
            print('CANONICAL_LATENT',mode,key,meta['pixel_shape'],flush=True)
            return latent
    video.H3=CachedH3
    video.__dict__.update(CASE_REFS=cfg['references'],EXPERIMENT_ROOT=ROOT,REFERENCE_AUDIT=reference_audit,CONDITION_AUDIT=condition_audit,TENSOR_SHA=tensor_sha,SAVE_AUDIT=save_audit)
    exec(compile(source,str(out/'generate_effective.py'),'exec'),video.__dict__)
    packages={}
    for name in ['torch','diffusers','transformers','accelerate','flash-attn']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='unknown'
    save(out/'launch_record.json',dict(config=cfg,physical_devices=os.environ['HIP_VISIBLE_DEVICES'],source_sha256=read(ROOT/'source_checksums.json'),runner_sha256=sha(Path(__file__)),packages=packages,torch_hip=torch.version.hip,started_at=datetime.datetime.now().astimezone().isoformat(),cache_policy='same-scene canonical per-pixel VAE latents shared by A and B; noise streams unchanged'))
    began=time.monotonic();output=out/'result.mp4'
    video.generate(ROOT/cfg['proxy'],ROOT/cfg['references'][0]['path'],prompt,PROJECT/'models/MiniMax-H3-ref2va',output,seed=42,keep_steps=5,text_device='cuda:1',generation_device='cuda:0')
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(output)]))
    v=next(x for x in media['streams'] if x['codec_type']=='video')
    assert (v['width'],v['height'],v['avg_frame_rate'],int(v['nb_read_frames']))==(960,544,'24/1',124)
    save(out/'result.run.json',dict(status='completed',elapsed_seconds=time.monotonic()-began,media=media,sha256=sha(output),peak_allocated_GiB=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],peak_reserved_GiB=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)]))
    if cfg['condition']=='B':
        a=read(ROOT/'cases'/(cfg['scene']+'_A')/'outputs/condition_audit.json')
        for k in ['target_initial_noise_sha256','audio_initial_noise_sha256','proxy_row_sha256']:assert a['conditions'][k]==condition_audit[k],k
        assert a['references'][0]['row_sha256']==reference_audit[0]['row_sha256']
        save(out/'paired_noise_check.json',dict(status='passed',target_noise_identical=True,audio_noise_identical=True,shared_reference_rows_identical=True,proxy_rows_identical=True))
    print('RUN_COMPLETE',args.job,flush=True)

if __name__=='__main__':main()

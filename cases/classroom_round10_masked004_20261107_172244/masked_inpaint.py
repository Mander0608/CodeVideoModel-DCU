"""DCU runner using the pinned linoyts inpainting blocks, with local conditioning."""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=Path('/public/home/lixin1/chenxy')
sys.path[:0]=[str(ROOT/'upstream'),str(BASE/'CodeVideoModel')]
import numpy as np
from PIL import Image
import torch
from cvm.dcu_runtime import configure
configure()
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep,get_qwen3vl_prompt_embeds
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
import h3_inpaint_blocks as blocks
from h3_inpaint_crop import paste_back


def normalized_audio(waveform,sample_rate,target_sample_rate):
    # Same upstream path when rates already match; resampling happens in FFmpeg on input.
    assert sample_rate==target_sample_rate,(sample_rate,target_sample_rate)
    if waveform.ndim==1:waveform=waveform[None]
    waveform=waveform.to(torch.float32)
    if waveform.shape[0]==1:waveform=waveform.repeat(2,1)
    return waveform[:2]

blocks.normalize_waveform=normalized_audio


def pin(module,name):
    copied=0;last=0
    for tensor in itertools.chain(module.parameters(),module.buffers()):
        if tensor.device.type=='cpu' and not tensor.is_pinned():
            tensor.data=tensor.detach().pin_memory()
            copied+=tensor.numel()*tensor.element_size()
            if copied-last>5e9:
                print('PIN',name,round(copied/1e9,2),'GB',flush=True);last=copied
    print('PIN_DONE',name,round(copied/1e9,2),'GB',flush=True)


def video(path):
    import av
    with av.open(str(path)) as c:
        fps=float(c.streams.video[0].average_rate)
        frames=np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
    return frames,fps


def save_video(frames,path,audio_source):
    h,w=frames.shape[1:3]
    cmd=['ffmpeg','-hide_banner','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
         '-r','24','-i','-','-i',str(audio_source),'-map','0:v:0','-map','1:a:0?',
         '-frames:v',str(len(frames)),'-t',str(len(frames)/24),'-c:v','libx264','-crf','16',
         '-preset','slow','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)]
    r=subprocess.run(cmd,input=frames.tobytes(),capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf-8','replace'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-turbo',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    source_path=args.case/'inputs/source_video.mp4'
    source,fps=video(source_path)
    mask_video,mask_fps=video(args.case/'inputs/repaint_mask.mp4')
    assert source.shape==mask_video.shape and fps==mask_fps==24
    mask=(mask_video.mean(-1)>127.5).astype(np.float32)
    prompt=(args.case/'prompt.txt').read_text(encoding='utf-8').strip()
    assert args.no_turbo and args.steps==20 and args.seed==42 and not args.smoke
    from diffusers import MiniMaxH3Scheduler
    for sub in ['scheduler','audio_scheduler']:
        sched=MiniMaxH3Scheduler.from_pretrained(str(BASE/'CodeVideoModel/models/MiniMax-H3-ref2va'),subfolder=sub)
        sched.set_timesteps(args.steps+1,device='cpu')
        assert len(sched.timesteps)==20,(sub,len(sched.timesteps))
    ref_paths=[args.case/'references/i9.png']
    reference_edge=768
    audio_offset=0
    if args.smoke:
        # Exercise actual holes, not the fully preserved first half of the source.
        audio_offset=93/24
        source=source[93:115]
        mask=mask[93:115]
        source=np.stack([np.asarray(Image.fromarray(f).resize((256,160),Image.Resampling.LANCZOS)) for f in source])
        mask=np.stack([np.asarray(Image.fromarray(f).resize((256,160),Image.Resampling.NEAREST)) for f in mask])
        reference_edge=256
        MiniMaxH3ModularPipeline.min_duration=property(lambda self:0.1)
    n,h,w=source.shape[:3]
    assert n%17==5 and h%32==w%32==0
    assert 0<mask.mean()<1
    refs=[]
    for path in ref_paths:
        im=Image.open(path).convert('RGB');iw,ih=im.size
        scale=reference_edge/min(iw,ih)
        im=im.resize((max(32,round(iw*scale/32)*32),max(32,round(ih*scale/32)*32)),Image.Resampling.LANCZOS)
        refs.append(MiniMaxH3ImageReference(image=im))
    a=subprocess.run(['ffmpeg','-v','error','-ss',str(audio_offset),'-i',str(source_path),
        '-t',str(n/24),'-map','0:a:0','-ac','2','-ar','32000','-f','f32le','-'],capture_output=True,check=True)
    audio=torch.from_numpy(np.frombuffer(a.stdout,np.float32).copy().reshape(-1,2).T.copy())
    report={'source_shape':list(source.shape),'mask_fraction':float(mask.mean()),'smoke':args.smoke,
        'steps':args.steps,'scheduler_requested_steps':args.steps+1,'seed':args.seed,'base_model':'existing full MiniMax-H3 Ref2VA (not Space pruned model)',
        'upstream_revision':json.loads((ROOT/'upstream/source_manifest.json').read_text())['revision'],
        'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'torch':torch.__version__,
        'hip':torch.version.hip,'turbo_requested':not args.no_turbo}
    if args.preflight_only:
        from h3_inpaint_masks import pixel_mask_to_row_mask
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
        latent_n=video_latent_num_frames(n,17,5)
        rows=pixel_mask_to_row_mask(torch.from_numpy(mask),latent_n,h//16,w//16,(1,2,2))
        assert rows.numel()==latent_n*(h//32)*(w//32)
        assert set(rows.unique().tolist())=={0.,1.}
        report['mask_rows']=rows.numel();report['status']='preflight_passed'
        print(json.dumps(report,indent=2),flush=True);return
    model=BASE/'CodeVideoModel/models/MiniMax-H3-ref2va'
    tm=ComponentsManager()
    text_workflow=ModularPipeline.from_pretrained(str(model)).blocks.get_workflow('t2va').sub_blocks['text_encoder']
    text_pipe=text_workflow.init_pipeline(str(model),components_manager=tm)
    print('LOAD_TEXT',flush=True)
    text_pipe.load_components(dtype=torch.bfloat16)
    pin(text_pipe.text_encoder,'text_encoder')
    from accelerate import cpu_offload
    cpu_offload(text_pipe.text_encoder,execution_device=torch.device('cuda:1'),offload_buffers=True)
    block=MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=2.0)
    vision,images,videos,timestamps=block._gather_vision_features(text_pipe.processor,refs,24)
    ids,tags=block._build_presentation(text_pipe.tokenizer,prompt,refs,images,videos,timestamps,
                                     text_tag=text_pipe.text_tag,video_tag=text_pipe.video_tag)
    print('ENCODE_LOCAL_TEXT',flush=True)
    with torch.inference_mode():
        embeds=get_qwen3vl_prompt_embeds(text_pipe.text_encoder,text_pipe.processor,ids,vision,
            text_encoder_layer=text_pipe.text_encoder_layer,device=torch.device('cuda:1'),dtype=text_pipe.text_encoder.dtype)
    assert torch.isfinite(embeds).all()
    embeds=embeds.cpu()
    print('TEXT_DONE',list(embeds.shape),flush=True)
    torch.save({'prompt_embeds':embeds,'text_token_tags':torch.tensor(tags,dtype=torch.long),
                'prompt':prompt},args.output.with_suffix('.conditioning.pt'))
    from accelerate.hooks import remove_hook_from_module
    with torch.inference_mode():remove_hook_from_module(text_pipe.text_encoder,recurse=True)
    for identifier in list(tm.get_ids()):tm.remove(identifier)
    del text_pipe,tm;gc.collect();torch.cuda.empty_cache()
    gm=ComponentsManager()
    pipe=blocks.MiniMaxH3Ref2VAInpaintGeneratorBlocks().init_pipeline(str(model),components_manager=gm)
    print('LOAD_GENERATOR',flush=True)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.register_to_config(reference_image_short_edge=reference_edge)
    if not args.no_turbo:
        lora=ROOT/'models/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors'
        assert lora.is_file(),lora
        pipe.load_lora_weights(str(lora),adapter_name='turbo')
        pipe.set_adapters(['turbo'],[1.0])
        report['turbo_loaded']=True
        print('TURBO_LOADED',flush=True)
    for name in ['vae','audio_vae','transformer_ref']:
        module=getattr(pipe,name)
        pin(module,name)
        module.set_attention_backend('native')
    gm.enable_auto_cpu_offload(device='cuda:0',memory_reserve_margin='16GB')
    pipe.transformer_ref.enable_group_offload(onload_device=torch.device('cuda:0'),
        offload_device=torch.device('cpu'),offload_type='block_level',num_blocks_per_group=2,use_stream=False)
    step=[0,time.monotonic()]
    def on_prediction(module,inputs,output):
        values=output if isinstance(output,tuple) else tuple(output.values())
        for value in values:
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise FloatingPointError('Non-finite H3 prediction')
        step[0]+=1
        print('DENOISE',step[0],'seconds',round(time.monotonic()-step[1],2),flush=True)
        step[1]=time.monotonic()
    pipe.transformer_ref.register_forward_hook(on_prediction)
    print('INPAINT_START',json.dumps(report),flush=True)
    with torch.inference_mode():
        state=pipe(prompt_embeds=embeds.to('cuda:0'),text_token_tags=torch.tensor(tags,dtype=torch.long),
            references=refs,source_video=source,source_fps=24,mask=mask,
            source_audio=audio,source_audio_sample_rate=32000,audio_mask=None,
            height=h,width=w,num_frames=n,num_inference_steps=args.steps+1,
            generator=torch.Generator('cpu').manual_seed(args.seed))
    generated=np.stack([np.asarray(frame) for frame in state.get('videos')[0]])
    assert generated.shape==source.shape,(generated.shape,source.shape)
    assert step[0]==20,step[0]
    save_video(generated,args.output.with_name('raw_generated_padded.mp4'),source_path)
    save_video(generated[:,32:-32,32:-32],args.output.with_name('raw_generated_960x544.mp4'),source_path)
    out=paste_back(source,generated,(0,0,h,w),mask=mask)
    print('SAVING_VIDEO',flush=True)
    save_video(out,args.output.with_name('composited_padded.mp4'),source_path)
    if not args.smoke:
        assert (w,h)==(1024,608)
        cropped=args.output
        save_video(out[:,32:-32,32:-32],cropped,source_path)
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(args.output)]))
    stream=next(s for s in media['streams'] if s['codec_type']=='video')
    assert int(stream['nb_read_frames'])==n
    report.update(status='completed',elapsed_seconds=time.monotonic()-started,media=media,
        denoiser_evaluations=step[0],predictions_finite_checked=True,
        inference_run=True,peak_allocated_GiB=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        protocol='explicit_masked_completion_not_original_cvm',turbo_loaded=False,
        source_references=['raw_original_i9_F123','original_i0_via_source_video','accepted_history_via_source_video'],known_region_motion_supplied=True,
        raw_generated_sha256=hashlib.sha256(args.output.with_name('raw_generated_960x544.mp4').read_bytes()).hexdigest())
    args.output.with_suffix('.run.json').write_text(json.dumps(report,indent=2))
    print('INPAINT_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()

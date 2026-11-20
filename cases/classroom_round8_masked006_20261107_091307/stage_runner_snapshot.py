"""Freeze an isolated same-weight H3 masked runner; does not execute inference."""
from pathlib import Path
import json,shutil,hashlib,datetime
B=Path(__file__).resolve().parent;R=B/'round8';RUN=R/'05_cvm/attempt_006_masked';OLD=R/'05_cvm/attempt_005'
assert not (RUN/'deployment.json').exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(RUN/'preparation_validation.json')['status']=='passed_offline'
source=(RUN/'original_inpaint_runner.py').read_text(encoding='utf-8')
def replace(a,b):
 global source
 assert source.count(a)==1,a
 source=source.replace(a,b)
replace('BASE=ROOT.parent',"BASE=Path('/public/home/lixin1/chenxy')")
replace("ref_paths=[args.case/'references/01_front.png',args.case/'references/02_left30.png']","ref_paths=[args.case/'references/i7.png']")
replace("p.add_argument('--steps',type=int,default=8)","p.add_argument('--steps',type=int,default=20)")
replace("prompt=(args.case/'prompt.txt').read_text().strip()","prompt=(args.case/'prompt.txt').read_text(encoding='utf-8').strip()\n    assert args.no_turbo and args.steps==20 and args.seed==42 and not args.smoke\n    from diffusers import MiniMaxH3Scheduler\n    for sub in ['scheduler','audio_scheduler']:\n        sched=MiniMaxH3Scheduler.from_pretrained(str(BASE/'CodeVideoModel/models/MiniMax-H3-ref2va'),subfolder=sub)\n        sched.set_timesteps(args.steps+1,device='cpu')\n        assert len(sched.timesteps)==20,(sub,len(sched.timesteps))")
replace("num_frames=n,num_inference_steps=args.steps,","num_frames=n,num_inference_steps=args.steps+1,")
replace("'steps':args.steps,'seed':args.seed,","'steps':args.steps,'scheduler_requested_steps':args.steps+1,'seed':args.seed,")
replace("out=paste_back(source,generated,(0,0,h,w),mask=mask)","assert step[0]==20,step[0]\n    save_video(generated,args.output.with_name('raw_generated_padded.mp4'),source_path)\n    save_video(generated[:,32:-32,32:-32],args.output.with_name('raw_generated_960x544.mp4'),source_path)\n    out=paste_back(source,generated,(0,0,h,w),mask=mask)")
replace("save_video(out,args.output,source_path)","save_video(out,args.output.with_name('composited_padded.mp4'),source_path)")
replace("cropped=args.output.with_name(args.output.stem+'_960x544.mp4')","cropped=args.output")
replace("output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())","output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),\n        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),\n        protocol='explicit_masked_completion_not_original_cvm',turbo_loaded=False,\n        source_references=['accepted_i7'],known_region_motion_supplied=True,\n        raw_generated_sha256=hashlib.sha256(args.output.with_name('raw_generated_960x544.mp4').read_bytes()).hexdigest())")
compile(source,'masked_inpaint.py','exec');(RUN/'masked_inpaint.py').write_text(source,encoding='utf-8',newline='\n')
wrapper='''from pathlib import Path
import argparse,sys,runpy,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--project-root');p.add_argument('--execute',action='store_true');a=p.parse_args()
assert a.execute
cfg=json.loads((ROOT/'cvm_config.json').read_text(encoding='utf-8'));out=ROOT/'outputs';out.mkdir(exist_ok=True)
prompt=(ROOT/'inputs/prompt.txt').read_text(encoding='utf-8').strip()
assert prompt==(ROOT/'prompt.txt').read_text(encoding='utf-8').strip()
(out/'effective_prompt.txt').write_text(prompt,encoding='utf-8')
shutil.copy2(ROOT/'masked_inpaint.py',out/'generate_effective.py')
argv=[str(ROOT/'masked_inpaint.py'),'--case',str(ROOT),'--output',str(ROOT/cfg['output']),'--steps','20','--seed','42','--no-turbo']
(out/'launch_record.json').write_text(json.dumps({'argv':argv,'config':cfg,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'protocol':'explicit_masked_completion_not_original_cvm'},indent=2),encoding='utf-8')
sys.argv=argv;runpy.run_path(str(ROOT/'masked_inpaint.py'),run_name='__main__')
'''
(RUN/'run_cvm.py').write_text(wrapper,encoding='utf-8',newline='\n')
preflight='''from pathlib import Path
import sys,json,hashlib,subprocess
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
expected=json.loads((ROOT/'input_checksums.json').read_text())
for rel,digest in expected.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
assert torch.cuda.device_count()==2
free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)];assert min(free)>60,free
source={r:hashlib.sha256((PROJECT/r).read_bytes()).hexdigest() for r in ['cvm/video.py','run_dcu.py']}
assert source=={'cvm/video.py':'98b451d987812d6f1caec3c36208fa280866bf40be803cd4474813ad0db40e4c','run_dcu.py':'a46644c0e095eff2ee9046b08c1cb1d3096b3e61d0232dd5783fbea42e6d671f'}
p=subprocess.run([sys.executable,str(ROOT/'masked_inpaint.py'),'--case',str(ROOT),'--output',str(ROOT/'outputs/preflight_not_generated.mp4'),'--steps','20','--seed','42','--no-turbo','--preflight-only'],capture_output=True,text=True)
(ROOT/'mask_preflight.log').write_text(p.stdout+'\\n'+p.stderr,encoding='utf-8');p.check_returncode()
assert 'preflight_passed' in p.stdout
report={'status':'passed','input_checksums_match':True,'free_GiB':free,'torch':torch.__version__,'hip':torch.version.hip,'shared_source_sha256':source,'protocol':'explicit_masked_completion_not_original_cvm','mask_preflight':'mask_preflight.log','actual_updates_verified_from_scheduler':20,'turbo':False}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2));print('CASE_PREFLIGHT',json.dumps(report),flush=True)
'''
(RUN/'preflight.py').write_text(preflight,encoding='utf-8',newline='\n')
cfg={'status':'prepared_to_launch','backend':'existing full MiniMax-H3 Ref2VA weights with pinned explicit masked-completion blocks','protocol':'explicit_masked_completion_branch_not_original_CVM_baseline','inputs':{'code_video':'inputs/partial_proxy.mp4','source_video_padded':'inputs/source_video.mp4','repaint_mask':'inputs/repaint_mask.mp4','reference_image':'references/i7.png','prompt_file':'inputs/prompt.txt'},'seed':42,'steps':20,'scheduler_requested_steps':21,'keep_steps':None,'turbo':False,'frames':124,'fps':24,'width':960,'height':544,'processing_size':[1024,608],'output':'outputs/round8_seed42.mp4','mask_input':True,'camera_parameter_input':False,'style_reference_guidance_appended':False,'known_region_motion_supplied':True,'notes':'Changes target conditioning algorithm and prompt, not a paired one-variable comparison; only accepted i7 supplies known appearance. Preserve raw generated video before paste-back. Source trajectory still -210 to-240. Do not call it raw Blender RGB CVM success.'}
save(RUN/'cvm_config.json',cfg)
shutil.copy2(OLD/'scene_snapshot_reference.json',RUN/'scene_snapshot_reference.json')
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round8_masked006_{stamp}';od=read(OLD/'deployment.json')
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(RUN/name).write_text((OLD/name).read_text(encoding='utf-8').replace(od['remote_case'],remote),encoding='utf-8',newline='\n')
# Collect the additional spatial-mask preflight log as well as all outputs.
p=RUN/'watch_and_collect.py';txt=p.read_text(encoding='utf-8');txt=txt.replace("'finished_at.txt','preflight.json'","'finished_at.txt','preflight.json','mask_preflight.log'");p.write_text(txt,encoding='utf-8',newline='\n')
shutil.copy2(Path(__file__),RUN/'stage_runner_snapshot.py')
for p in RUN.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(RUN/'input_checksums.json',{p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file()})
nd={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r8masked_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(RUN/'deployment.json',nd)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round8_masked_branch_prepared',current_deployment=nd,updated_at=nd['prepared_at'],last_verified_phase='Masked branch runner frozen, local syntax and124frame mask checks pass; not submitted. Remote preflight must validate20actual updates, masks, source hashes and resource locks.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(nd,indent=2))

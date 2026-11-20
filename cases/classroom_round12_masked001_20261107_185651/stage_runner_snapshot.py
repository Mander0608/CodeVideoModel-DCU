"""Freeze one reviewed masked continuation, preserving accepted template configuration."""
from pathlib import Path
import argparse,json,hashlib,shutil,datetime
from prompt_render_review import require_prompt_render_review
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);p.add_argument('--attempt',required=True);args=p.parse_args();n=args.round
assert n in [11,12]
R=B/f'round{n}';A=R/'05_cvm'/args.attempt;T=B/'round10/05_cvm/attempt_004_masked_original_i9'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (A/'deployment.json').exists()
assert read(A/'offline_visual_review.json')['accepted_for_inference']
review=require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
for rel,h in review['additional_condition_sha256'].items():assert sha(A/rel)==h
assert sha(A/'inputs/reference.png')==sha(A/f'references/i{n-1}.png')
assert (A/'prompt.txt').read_bytes()==(A/'inputs/prompt.txt').read_bytes()
frozen=read(R/'CHECKSUMS_INPUTS.json')
for rel,h in frozen.items():assert sha(R/rel)==h,rel
assert read(R/'blend_validation.json')['status']=='passed'
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');now=datetime.datetime.now().astimezone().isoformat()
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round{n}_masked001_{stamp}';oldremote=read(T/'deployment.json')['remote_case']
for name in ['masked_inpaint.py','run_cvm.py','preflight.py','supervise.py','watch_and_collect.py','launch.sh']:
 code=(T/name).read_text(encoding='utf-8').replace(oldremote,remote).replace('round10_seed42',f'round{n}_seed42')
 if name=='masked_inpaint.py':
  code=code.replace('references/i9.png',f'references/i{n-1}.png').replace("source_references=['raw_original_i9_F123','original_i0_via_source_video','accepted_history_via_source_video']",f"source_references=['accepted_current_i{n-1}_masked_composite','original_i0_via_source_video','accepted_history_with_pixel_lineage']")
 if name=='watch_and_collect.py':
  code=code.replace("phases[-1] if phases else 'loading'", "('DENOISE '+steps[-1][0]+' seconds '+steps[-1][1]) if steps else (phases[-1] if phases else 'loading')")
 (A/name).write_text(code,encoding='utf-8',newline='\n')
cfg=read(T/'cvm_config.json');cfg.update(status='prepared_to_launch',output=f'outputs/round{n}_seed42.mp4')
cfg['inputs']['reference_image']=f'references/i{n-1}.png';cfg['source_image_reference']=f'accepted current-branch i{n-1}'
cfg['source_video_history']=['originali0 priority','latestaccepted history with per-pixel inherited original/generated labels']
cfg['notes']=f'Round{n} continuation of accepted masked004 algorithm. NoTurbo,seed42,20actualupdates,scheduler21,keep_stepsnull; known regions supplied/pasted back. Unknown8px+source seams24px. Nominal{-30*(n-1)}to{-30*n}; not independent model memory/pose/depth.'
intended=read(R/'provenance/intended_protocol.json')
assert cfg['seed']==intended['seed']==42 and cfg['steps']==intended['actual_updates']==20 and cfg['keep_steps'] is None and cfg['turbo'] is False
save(A/'cvm_config.json',cfg)
save(A/'scene_snapshot_reference.json',{'scene':f'../../02_blender/scene/classroom_v{n-1}.blend','scene_sha256':sha(R/f'02_blender/scene/classroom_v{n-1}.blend'),'code_sha256':{p.relative_to(R).as_posix():sha(p) for p in (R/'02_blender').glob('*.py')},'all_render_input_manifest':'../../CHECKSUMS_INPUTS.json','all_render_input_manifest_sha256':sha(R/'CHECKSUMS_INPUTS.json'),'accepted_references':read(R/'00_input/accepted_references.json')})
save(A/'protocol_manifest.json',{'parent_method':'round10/05_cvm/attempt_004_masked_original_i9','same_masked_core_algorithm':True,'same_sampling':True,'trajectory_changed':True,'prompt_adapted_to_actual_view':True,'source_reference_changed':True,'not_paired_single_variable_control':True,'known_pixels_supplied':True})
save(A/'implementation_source.json',{'template':T.relative_to(B).as_posix(),'template_runner_sha256':sha(T/'masked_inpaint.py'),'changes':['case and output paths','reference and source-lineage metadata','collector phase shows denoise'], 'masked_core_algorithm_changed':False})
shutil.copy2(B/'round10/provenance/check_prelaunch_resources.py',A/'check_prelaunch_resources.py');shutil.copy2(Path(__file__),A/'stage_runner_snapshot.py')
for q in A.glob('*.py'):compile(q.read_text(encoding='utf-8'),str(q),'exec')
save(A/'input_checksums.json',{q.relative_to(A).as_posix():sha(q) for q in A.rglob('*') if q.is_file() and '__pycache__' not in q.parts})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r{n}_mask001_{stamp}','status':'prepared_to_launch','prepared_at':now};save(A/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status=f'round{n}_masked_prepared',current_round=n,current_run=str(A.resolve()),current_run_relative=A.relative_to(B).as_posix(),current_deployment=dep,updated_at=now,last_verified_phase=f'Round{n} source/mask/prompt reviewed and frozen; not submitted.');s[f'round{n}']='masked_prepared';save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(dep,indent=2))

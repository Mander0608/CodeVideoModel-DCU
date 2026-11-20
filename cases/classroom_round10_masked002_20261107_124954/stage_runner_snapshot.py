"""Freeze round10 masked branch using previously audited round8 implementation."""
from pathlib import Path
import json,hashlib,shutil,datetime
B=Path(__file__).resolve().parent;R=B/'round10';RUN=R/'05_cvm/attempt_002_masked';T=B/'round8/05_cvm/attempt_006_masked';OLD=R/'05_cvm/attempt_001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (RUN/'deployment.json').exists()
assert read(RUN/'preparation_validation.json')['status']=='passed_offline'
assert read(RUN/'offline_visual_review.json')['accepted_for_inference']
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round10_masked002_{stamp}'
oldremote=read(T/'deployment.json')['remote_case']
for name in ['masked_inpaint.py','run_cvm.py','preflight.py','supervise.py','watch_and_collect.py','launch.sh']:
 code=(T/name).read_text(encoding='utf-8').replace(oldremote,remote).replace('round8_seed42','round10_seed42')
 if name=='masked_inpaint.py':
  code=code.replace("references/i7.png","references/i9.png").replace("source_references=['accepted_i7']","source_references=['accepted_i9_manual_composite','original_i0_via_source_video']")
 if name=='watch_and_collect.py':
  code=code.replace("r'^DENOISE (\\d+) seconds ([0-9.]+)'","r'\\bDENOISE (\\d+) seconds ([0-9.]+)'")
 (RUN/name).write_text(code,encoding='utf-8',newline='\n')
cfg=read(T/'cvm_config.json');cfg.update(status='prepared_to_launch',output='outputs/round10_seed42.mp4')
cfg['inputs']['reference_image']='references/i9.png'
cfg['notes']='Round10 explicit masked fallback after original RGB CVM lost the returning front board and retained gray wedges. Original i0 overrides generated overlap in source video; v001 latest-priority and v002 corrected source both preserved. One corrected i9 image. Full H3, seed42,20actualupdates,noTurbo; keep_steps is not applicable. Known pixels pasted back; not original CVM success or independent recall. Nominal -270 to -300.'
cfg['source_image_reference']='one corrected i9 F123 image'
cfg['source_video_history']=['corrected i9 composite plus accepted history','original i0 authoritative returning rays']
save(RUN/'cvm_config.json',cfg)
shutil.copy2(OLD/'scene_snapshot_reference.json',RUN/'scene_snapshot_reference.json')
save(RUN/'paired_manifest.json',{'comparison_scope':'Sequential masked fallback, not a single-variable ablation','original_CVM_baseline_passed':False,'same_base_weights_as_round1':True,'same_seed_and_actual_updates_as_round1':True,'same_conditioning_algorithm_as_round1':False,'same_proxy_representation_as_round1':False,'explicit_original_i0_pixels_return':True,'raw_Blender_scene_unchanged':True,'nominal_trajectory':[-270,-300]})
save(RUN/'implementation_source.json',{'baseline':'round8/05_cvm/attempt_006_masked','baseline_masked_runner_sha256':sha(T/'masked_inpaint.py'),'changes':['Image ref i7->corrected i9','Source metadata records original i0 supplied via video','Round10 output names/case paths; original-i0-first source priority; inherited manual provenance','Collector recognizes tqdm-prefixed DENOISE lines'],'masked_core_algorithm_changed':False})
shutil.copy2(Path(__file__),RUN/'stage_runner_snapshot.py')
shutil.copy2(OLD/'check_prelaunch_resources.py',RUN/'check_prelaunch_resources.py')
for p in RUN.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(RUN/'input_checksums.json',{p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r10masked_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(RUN/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round10_masked_branch_prepared',current_run=str(RUN.resolve()),current_run_relative=RUN.relative_to(B).as_posix(),current_deployment=dep,updated_at=dep['prepared_at'],last_verified_phase='Masked002 offline visual review and exact124frame mask checks passed; runner frozen but not submitted. Remote preflight must check source hashes/devices/20actual updates. No round11.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(dep,indent=2))

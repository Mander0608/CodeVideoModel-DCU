"""Freeze round9 masked branch using previously audited round8 implementation."""
from pathlib import Path
import json,hashlib,shutil,datetime
B=Path(__file__).resolve().parent;R=B/'round9';RUN=R/'05_cvm/attempt_002_masked';T=B/'round8/05_cvm/attempt_006_masked';OLD=R/'05_cvm/attempt_001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (RUN/'deployment.json').exists()
assert read(RUN/'preparation_validation.json')['status']=='passed_offline'
assert read(RUN/'offline_visual_review.json')['accepted_for_inference']
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round9_masked002_{stamp}'
oldremote=read(T/'deployment.json')['remote_case']
for name in ['masked_inpaint.py','run_cvm.py','preflight.py','supervise.py','watch_and_collect.py','launch.sh']:
 code=(T/name).read_text(encoding='utf-8').replace(oldremote,remote).replace('round8_seed42','round9_seed42')
 if name=='masked_inpaint.py':
  code=code.replace("references/i7.png","references/i8.png").replace("source_references=['accepted_i7']","source_references=['accepted_i8_composite','original_i0_via_source_video']")
 if name=='watch_and_collect.py':
  code=code.replace("r'^DENOISE (\\d+) seconds ([0-9.]+)'","r'\\bDENOISE (\\d+) seconds ([0-9.]+)'")
 (RUN/name).write_text(code,encoding='utf-8',newline='\n')
cfg=read(T/'cvm_config.json');cfg.update(status='prepared_to_launch',output='outputs/round9_seed42.mp4')
cfg['inputs']['reference_image']='references/i8.png'
cfg['notes']='Round9 explicit masked fallback after original raw RGB CVM motion failure. Same full H3 weights/seed42/20actual updates/noTurbo, but changed algorithm/representation/prompt; not paired ablation or original CVM success. i8 composite supplies latest appearance. Original i0 pixels explicitly return in source video; this is supplied history, not autonomous memory. Raw generated beforepasteback retained. Nominal -240 to -270.'
cfg['source_image_reference']='i8 composite only'
cfg['source_video_history']=['i8 composite','original i0 returning rays']
save(RUN/'cvm_config.json',cfg)
shutil.copy2(OLD/'scene_snapshot_reference.json',RUN/'scene_snapshot_reference.json')
save(RUN/'paired_manifest.json',{'comparison_scope':'Sequential masked fallback, not a single-variable ablation','original_CVM_baseline_passed':False,'same_base_weights_as_round1':True,'same_seed_and_actual_updates_as_round1':True,'same_conditioning_algorithm_as_round1':False,'same_proxy_representation_as_round1':False,'explicit_original_i0_pixels_return':True,'raw_Blender_scene_unchanged':True,'nominal_trajectory':[-240,-270]})
save(RUN/'implementation_source.json',{'baseline':'round8/05_cvm/attempt_006_masked','baseline_masked_runner_sha256':sha(T/'masked_inpaint.py'),'changes':['Image ref i7->i8','Source metadata records original i0 supplied via video','Round9 output names/case paths','Collector recognizes tqdm-prefixed DENOISE lines'],'masked_core_algorithm_changed':False})
shutil.copy2(Path(__file__),RUN/'stage_runner_snapshot.py')
for p in RUN.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(RUN/'input_checksums.json',{p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r9masked_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(RUN/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round9_masked_branch_prepared',current_run=str(RUN.resolve()),current_run_relative=RUN.relative_to(B).as_posix(),current_deployment=dep,updated_at=dep['prepared_at'],last_verified_phase='Masked002 offline visual review and exact124frame mask checks passed; runner frozen but not submitted. Remote preflight must check source hashes/devices/20actual updates. No round10.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(dep,indent=2))

"""Stage the visually inspected image-condition control once; no inference here."""
from pathlib import Path
import datetime, hashlib, json, shutil
B=Path(__file__).resolve().parent
OLD=B/'round9/05_cvm/attempt_003_masked'
NEW=B/'round9/05_cvm/attempt_004_wallref'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert NEW.is_dir() and not (NEW/'deployment.json').exists()
s=read(B/'CONTINUOUS_LEFT_STATE.json')
assert s['current_run_relative']==NEW.relative_to(B).as_posix() and s['current_deployment'] is None
pair=read(NEW/'paired_manifest.json')
assert all(sha(NEW/p)==h==sha(OLD/p) for p,h in pair['fixed_file_sha256'].items())
assert sha(NEW/'references/wall_crop_i0.png')==pair['new_image_sha256']
assert read(OLD/'timing.json')['status']=='completed'
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round9_wallref004_{stamp}'
olddep=read(OLD/'deployment.json')
for name in ['supervise.py','watch_and_collect.py','launch.sh']:
    (NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(olddep['remote_case'],remote),encoding='utf-8',newline='\n')
shutil.copy2(OLD/'check_prelaunch_resources.py',NEW/'check_prelaunch_resources.py')
save(NEW/'offline_visual_review.json',{
    'reviewed_at':datetime.datetime.now().astimezone().isoformat(),
    'inspected':['reference_crop_context.png','references/wall_crop_i0.png'],
    'passed':True,
    'finding':'Crop is cream wall to the right of original green board, excludes board/window/clock/furniture. Small wall marks and illumination gradient remain original image content.',
    'actual_model_image':'references/wall_crop_i0.png',
    'motion_audit_image':'inputs/reference.png (full accepted i8, not model Picture1)',
    'scope':'Image-content/aspect control only; supplied original appearance, no autonomous memory or depth claim.',
    'process_check':'Remote 2026-11-07T11:24:49+08:00: no classroom inference/supervisor; only unrelated h3_inpaint_web screen and unrelated cosmos supervisor. Local no Python/Blender collector.'})
cfg=read(NEW/'cvm_config.json');cfg['status']='prepared_to_launch';save(NEW/'cvm_config.json',cfg)
shutil.copy2(Path(__file__),NEW/'stage_snapshot.py')
for p in NEW.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='input_checksums.json'})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r9wallref4_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()}
save(NEW/'deployment.json',dep)
s.update(status='round9_wallref004_prepared',round9='three_rejected_image_control_prepared',current_deployment=dep,updated_at=dep['prepared_at'],last_verified_phase='Bare-wall crop visually inspected; fixed files and model image hashes passed. Staged locally, not submitted; check remote resources then deploy once. No i9/round10.')
save(B/'CONTINUOUS_LEFT_STATE.json',s)
plan=B/'CONTINUOUS_LEFT_PLAN.md'
plan.write_text(f'LATEST {s["updated_at"]}: Round9 wallref004 prepared after crop visual inspection. Actual model Picture1=original i0 bare wall crop; full i8 remains source/motion audit only. Static input hashes frozen. No i9/round10. Resource check and single deployment pending. On completion run all three audit scripts and inspect novel gap; if unsupported panel persists, diagnose source/latent masks, no reference/prompt sweep.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dep,indent=2))

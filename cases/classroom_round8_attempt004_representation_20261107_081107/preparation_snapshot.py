"""Stage one video-representation-only control after three archived failures."""
from pathlib import Path
import json,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'round8';D=R/'06_control_diagnosis'
OLD=R/'05_cvm/attempt_003';NEW=R/'05_cvm/attempt_004'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(D/'offline_validation.json')['status']=='passed'
diag=read(D/'representation_diagnosis.json');assert sha(D/'appearance_proxy.mp4')==diag['video_sha256']
dep=read(OLD/'deployment.json');assert dep['status']=='completed'
assert read(OLD/'analysis/visual_review.json')['accepted_for_modeling'] is False
report='''# Round8: three rejected results; controlled representation diagnostic

No round8 result has been accepted and no round9 scene has been created. Attempts 001 and 002 filled missing areas but drifted in motion/layout; increasing video conditioning from 5 to 10 denoising updates did not resolve this pair. Attempt003 retained 99.8966% near-gray pixels inside endpoint missing support and transitioned toward the Blender proxy appearance. All three endpoint direct matches fail; no total yaw is asserted.

Raw outputs, configurations, effective prompts, environments, timing, logs and checksums are retained in the three attempt folders. Archive verification passed for 59, 59 and 60 files respectively. Three-way output comparison: `06_control_diagnosis/three_attempt_comparison.jpg`.

Offline diagnosis found raw Blender first-frame MAE 23.713/255 relative to i7. A separate appearance-reprojected proxy uses the same Blender camera matrices and only accepted source images. Its unencoded first frame equals i7, all 124 unknown masks equal the raw proxy masks, and endpoint unknown fraction remains 38.8898%. Only i7 contributes pixels in this segment. Seven decoded-video motion checks yield 122–370 inliers, median residual below 0.352px, maximum sampled yaw difference 0.032 degrees. These confirm the constructed control, not any CVM output or accurate scene geometry.

Attempt004 changes only the model's video input relative to attempt003. Reference image, exact prompt, seed42, steps20 and keep_steps10 stay fixed. This explicitly changes control representation from raw Blender material RGB to accepted-appearance fixed-center reprojection; it is a diagnostic branch, not an unchanged Blender RGB baseline. It does not synthesize missing content offline, use a full-frame scene billboard, add depth evidence, or establish performance under translation. Raw Blender scene/renders remain preserved. Video conditioning10 remains an explicit departure from round1 baseline5.

Hypothesis: reducing appearance conflict may avoid the observed transition and improve limited-turn adherence. Cause is unproven. Acceptance requires coherent completion, plausible limited motion, valid correspondence and visual inspection; gray removal alone is insufficient. If this controlled representation test also fails, preserve its result and reassess mechanism before any additional submission. Never propagate a rejected frame. If it passes, explicitly decide and record how the diagnostic control relates to subsequent Blender RGB rounds; do not silently change the experiment definition.
'''
(R/'RESULT.md').write_text(report,encoding='utf-8');(D/'DIAGNOSIS.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'validate_round8_representation.py',D/'validate_representation_snapshot.py')
save(D/'CHECKSUMS_POST_REVIEW.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file() and p.name!='CHECKSUMS_POST_REVIEW.json'})
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir():shutil.copy2(p,NEW/'inputs'/p.name)
shutil.copy2(D/'appearance_proxy.mp4',NEW/'inputs/partial_proxy.mp4')
shutil.copy2(D/'representation_diagnosis.json',NEW/'inputs/representation_provenance.json')
shutil.copy2(D/'offline_validation.json',NEW/'inputs/representation_validation.json')
manifest=read(NEW/'inputs/input_manifest.json');manifest['status']='representation_control_prepared_not_run'
manifest['video_representation']='Fixed-center accepted-appearance reprojection using unchanged Blender camera matrices; NOT raw Blender material RGB. Only i7 contributes in this segment.'
for name,rec in manifest['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',manifest)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round8_attempt004_representation_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='Round8 attempt004 representation-only control vs003. Appearance reprojection replaces raw Blender RGB proxy; raw scene preserved. Same exact prompt/image/seed42/20steps/keep_steps10. keep_steps10 differs from round1 baseline5. See inputs/representation_provenance.json.');save(NEW/'cvm_config.json',cfg)
same=['reference.png','camera_trajectory.json','prompt.txt']
for name in same:assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_003','changed_model_conditions':['partial_proxy.mp4'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'old_video_sha256':sha(OLD/'inputs/partial_proxy.mp4'),'new_video_sha256':sha(NEW/'inputs/partial_proxy.mp4'),'same_runner':sha(OLD/'run_cvm.py')==sha(NEW/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'deviation_from_round1':['keep_steps10 instead of5','accepted-appearance reprojection instead of raw Blender RGB control'],'diagnosis':'../../06_control_diagnosis/DIAGNOSIS.md','retry_limit':'One representation control after offline validation. No blind repeats; failures must be diagnosed and never propagated.'})
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
nd={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r8a4_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',nd)
save(R/'EXPERIMENT_STATUS.json',{'status':'three_rejected_representation_control_prepared','accepted_attempt':None,'rejected_attempts':['05_cvm/attempt_001','05_cvm/attempt_002','05_cvm/attempt_003'],'active_attempt':'05_cvm/attempt_004','diagnosis':'06_control_diagnosis/DIAGNOSIS.md'})
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round8_representation_control_prepared',round8='three_rejected_attempt004_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=nd,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'gray missing regions retained and appearance transition','output_sha256':read(OLD/'analysis/validation.json')['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat(),last_verified_phase='Three attempts rejected and archives verified. Offline representation diagnosis complete. One video-only control prepared; no inference active, no round9.');save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
plan=BASE/'CONTINUOUS_LEFT_PLAN.md';text=plan.read_text(encoding='utf-8');marker='## Current state and scope\n'
entry=f'\nLATEST {s["updated_at"]}: round8 attempt003 REJECTED (99.8966% gray within endpoint missing area, invalid endpoint8inliers, no netyaw). All three failures archived/verified. Offline diagnosis complete at round8/06_control_diagnosis/DIAGNOSIS.md. Appearance-reprojected candidate retains all124 masks and camera matrices; firstframe MAE0 vs raw23.713, seven decoded motion checks max error0.032deg. ONE attempt004 video-only representation control PREPARED, same prompt/image/seed42/20steps/keep10 as003; NOT raw Blender RGB, explicit diagnostic branch. Read current state for launch status. If it fails diagnose before further tests; do not propagate or create round9. If it passes explicitly record control-policy decision before continuation.\n'
assert marker in text;plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8')
print(json.dumps(nd,indent=2))

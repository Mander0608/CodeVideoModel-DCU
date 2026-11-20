"""Archive rejected round8 motion drift and stage one keep_steps-only control."""
from pathlib import Path
import json, hashlib, shutil, datetime

BASE = Path(__file__).resolve().parent
R = BASE/'round8'
OLD = R/'05_cvm/attempt_001'
NEW = R/'05_cvm/attempt_002'
assert not NEW.exists(), 'Never duplicate a prepared or running attempt'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=True), encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

v = read(OLD/'analysis/validation.json')
t = read(OLD/'timing.json')
assert v['technical_status']=='passed' and t['exit_code']==0
assert sha(OLD/'outputs/round8_seed42.mp4')==v['output_sha256']
review = {
    'accepted_for_modeling': False,
    'inspected_frames': [0,31,62,92,116,123],
    'appearance': 'Gray holes are filled, but the dark wall panel sweeps across the frame far beyond the modest input rotation. Rear wall fixtures disappear/reappear on a substantially different layout; endpoint has a new left window and reordered desks.',
    'motion': 'At F31 direct reference matching estimates -11.20deg versus -3.72deg commanded; later direct matching fails. Endpoint has only6 inliers. No valid end-to-end net yaw estimate exists.',
    'completion': 'Endpoint near-gray fraction inside input-unknown is0.0330; visual full-frame completion does not compensate for motion/layout mismatch.',
    'decision': 'Reject for next-round reconstruction. Do not label the endpoint as a valid -240deg reference or create round9 from it.',
    'local_diagnostic': 'local_motion_diagnostic.json contains output-to-output pair estimates. Some middle pairs fail; cumulative diagnostic is deliberately null.',
    'causality': 'Source inspection confirms video conditioning only for first5 of20 denoising updates, with image/text for the rest. Insufficient motion control is a hypothesis, not established causation.'
}
save(OLD/'analysis/visual_review.json', review)
v.update(quality_status='rejected_for_next_round', review=review)
save(OLD/'analysis/validation.json', v)
save(OLD/'result_status.json', {'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'validation':'analysis/validation.json','timing':t})
dep = read(OLD/'deployment.json')
dep.update(status='completed',finished_at=t['finished_at'],exit_code=0)
save(OLD/'deployment.json',dep)
diagnosis = {
    'failure': 'Motion magnitude/timing and room layout drift; image completion succeeds.',
    'verified_source': 'outputs/generate_effective.py: condition = full if i < keep_steps else base',
    'baseline': {'keep_steps':5,'steps':20,'video_condition_updates':[0,1,2,3,4]},
    'controlled_retry': {'keep_steps':10,'steps':20,'video_condition_updates':list(range(10))},
    'unchanged': ['prompt bytes','reference image','partial video','trajectory','Blender scene','runner','seed42','20 denoising updates','no appended style guidance'],
    'tradeoff': 'Longer video conditioning may improve motion adherence but may also preserve gray holes or proxy appearance. Both motion and completion must pass visual review.',
    'interpretation': 'This attempt deviates explicitly from round1 keep_steps5. It is a one-parameter diagnostic retry, not evidence that the baseline pipeline succeeded.',
    'retry_limit': 'One keep_steps10 control. If motion drift or gray-copy persists, preserve and diagnose before any further change; do not propagate bad frames.'
}
save(OLD/'analysis/conditioning_diagnosis.json', diagnosis)
report = f'''# Round8 attempt001 rejected: motion/layout drift

Inference {t['inference_started_at']} to {t['finished_at']}, {t['inference_process_elapsed_seconds']:.2f}s. Exit0;124 frames decode; effective prompt matches; output SHA256 {v['output_sha256']}.

Gray holes are filled but output motion substantially exceeds the commanded30-degree left turn. F31 direct estimate is -11.20deg versus -3.72deg nominal. Endpoint direct match has only6 inliers, so end-to-end yaw remains null. Sampled views show changed layout and a newly invented endpoint window. No failed frame is accepted as -240deg or propagated to Blender.

One controlled retry changes only keep_steps5 to10. This is the number of denoising updates with video conditioning, not video frames. Prompt, image/video, scene, seed42 and20 total steps remain unchanged. The change may improve motion but also risks gray-copy; both must be checked. Source inspection supports this test, not a definitive cause. All original records are retained.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8')
(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
save(R/'EXPERIMENT_STATUS.json',{'status':'attempt001_rejected_conditioning_retry_preparing','accepted_attempt':None,'rejected_attempt':'05_cvm/attempt_001','reason':'motion magnitude and layout drift','timing':t})
save(R/'CHECKSUMS_ATTEMPT001_REJECTED.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='CHECKSUMS_ATTEMPT001_REJECTED.json' and '__pycache__' not in str(p)})
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir(): shutil.copy2(p,NEW/'inputs'/p.name)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round8_attempt002_keep10_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']: shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:
    (NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json')
cfg.update(status='prepared_to_launch',keep_steps=10,
    keep_steps_semantics='Video-reference conditioning for first10 of20 denoising updates; not a pixel preservation constraint or video-frame count.',
    notes='Round8 controlled retry: keep_steps5->10 only. Explicit deviation from round1 baseline. Same prompt/image/video/scene/seed42/20steps; inspect motion AND completion before modeling.')
save(NEW/'cvm_config.json',cfg)
same=['partial_proxy.mp4','reference.png','prompt.txt','camera_trajectory.json']
for name in same: assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
assert sha(OLD/'run_cvm.py')==sha(NEW/'run_cvm.py')
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_001','changed_model_conditions':['keep_steps:5->10'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'same_runner':True,'same_model_and_sampling_as_round1':False,'deviation_from_round1':'keep_steps10 instead of5; totalsteps20/seed42 unchanged','parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'retry_limit':diagnosis['retry_limit']})
save(NEW/'conditioning_diagnosis.json',diagnosis)
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
newdep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r8a2_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()}
save(NEW/'deployment.json',newdep)
s=read(BASE/'CONTINUOUS_LEFT_STATE.json')
s.update(status='round8_retry_prepared',round8='attempt001_rejected_attempt002_keep10_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=newdep,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'motion/layout drift','output_sha256':v['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat())
save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(newdep,indent=2))

"""Archive the rejected completion and prepare one prompt-only retry; no launch."""
from pathlib import Path
import json,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'round4';OLD=R/'05_cvm/attempt_001';NEW=R/'05_cvm/attempt_002'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');assert v['technical_status']=='passed'
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],
 'appearance':'CVM changes visible classroom appearance but retains the expanding gray cutout and hard boundary.',
 'motion':'Left turn occurs; approximate net yaw -30.52 degrees, not calibrated pose recovery.',
 'completion':'Endpoint near-gray fraction inside input-unknown mask is 0.9973855; visual inspection confirms missing left/top/bottom regions remain gray.',
 'decision':'Reject all frames as new scene evidence. Preserve failure and perform one prompt-only retry before any round5 geometry update.',
 'causality':'Cause is not established. Copying the partial video boundary is consistent with the output; a text-only change is a controlled probe, not a guaranteed remedy.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
timing=read(OLD/'timing.json');assert timing['exit_code']==0
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'validation':'analysis/validation.json','timing':timing})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=timing['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round4 attempt001: completion rejected

Inference {timing['inference_started_at']} to {timing['finished_at']}; {timing['inference_process_elapsed_seconds']:.2f} seconds. Exit 0, all 124 frames decode, effective prompt matches, remote/local output SHA256 matches: {v['output_sha256']}.

Sampled visual review confirms the expanding gray region was copied into the output. At F123, 99.74% of pixels inside the input unknown mask are near gray. The approximate turn is -30.52 degrees, so motion alone is insufficient for acceptance. No generated frame from this attempt will update Blender.

One prompt-only retry explicitly requests a fully filled rear classroom wall, connected floor and ceiling in the left, top and bottom missing regions. Image/video, seed42, steps20 and keep_steps5 remain unchanged. The cause of failure is not established. All original inputs and outputs remain preserved.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
save(R/'EXPERIMENT_STATUS.json',{'status':'attempt001_rejected_retry_preparing','accepted_attempt':None,'rejected_attempt':'05_cvm/attempt_001','reason':'Gray missing regions retained','timing':timing})
save(R/'CHECKSUMS_ATTEMPT001_REJECTED.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='CHECKSUMS_ATTEMPT001_REJECTED.json' and '__pycache__' not in str(p)})
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir():shutil.copy2(p,NEW/'inputs'/p.name)
prompt='''subject_definitions:
<Subject 1> is the classroom in <Picture 1>.
<Video 1> is an incomplete camera preview. Its flat gray areas are holes requiring generated classroom content.

summary:
[video editing + reference generation] Fill every missing region of <Video 1> to produce a complete, photorealistic classroom video.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - classroom identity, wooden desks, windows and daylight.
<Video 1>: partially_preserved - retain chronological camera motion and visible layout; completely replace gray areas and their hard boundaries.

detailed_description:
[Shot 1] Begin at <Picture 1>, then turn LEFT another 30 degrees in place along <Video 1>. Existing windows move RIGHT. As the left side is revealed, extend the classroom into that area with a continuous cream rear wall and plausible stationary desks. Extend the ceiling across the upper missing area and the floor across the lower missing area. Connect these surfaces to the visible room with matching perspective, daylight and occlusion. Every frame fills the entire image with realistic room content, including all left, upper and lower gray holes. No gray background, cutout border or empty region remains. One continuous shot without returning to the starting view.

overall_soundscape:
Quiet classroom ambience.

non_diegetic_music:
N/A
'''
(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
save(NEW/'inputs/prompt_adaptation.json',{'baseline':'../attempt_001/inputs/prompt.txt','changed_condition':'prompt only','reason':'Explicitly replace persistent gray cutout with rear wall, floor and ceiling','guaranteed_success':False})
manifest=read(NEW/'inputs/input_manifest.json');manifest['status']='retry_prepared_not_run'
for name,rec in manifest['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',manifest)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round4_attempt002_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='Round4 prompt-only retry: explicit rear-wall/floor/ceiling completion of left/top/bottom gray holes. Same image/video, scene, seed and sampling as rejected attempt001.');save(NEW/'cvm_config.json',cfg)
same=['partial_proxy.mp4','reference.png','camera_trajectory.json']
for name in same:assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_001','changed_model_conditions':['prompt.txt'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'same_runner':sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'retry_limit':'One prompt-only retry; inspect result before any further adaptation.'})
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
newdep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r4a2_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',newdep)
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round4_retry_prepared',round4='attempt001_rejected_attempt002_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=newdep,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'persistent gray unknown areas','output_sha256':v['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat());save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(newdep,indent=2))

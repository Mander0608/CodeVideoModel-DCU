"""Archive the rejected completion and prepare one prompt-only retry; no launch."""
from pathlib import Path
import json,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'round7';OLD=R/'05_cvm/attempt_001';NEW=R/'05_cvm/attempt_002'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');assert v['technical_status']=='passed'
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],
 'appearance':'CVM changes visible classroom appearance but retains the expanding gray cutout and hard boundary.',
 'motion':'Left turn occurs; approximate net yaw -31.84 degrees, not calibrated pose recovery.',
 'completion':'Endpoint near-gray fraction inside input-unknown mask is 0.9927030300643039; visual inspection confirms missing left/top/bottom regions remain gray.',
 'decision':'Reject all frames as new scene evidence. Preserve failure and perform one prompt-only retry before any round8 geometry update.',
 'causality':'Cause is not established. Copying the partial video boundary is consistent with the output; a text-only change is a controlled probe, not a guaranteed remedy.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
timing=read(OLD/'timing.json');assert timing['exit_code']==0
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'validation':'analysis/validation.json','timing':timing})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=timing['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round7 attempt001: completion rejected

Inference {timing['inference_started_at']} to {timing['finished_at']}; {timing['inference_process_elapsed_seconds']:.2f} seconds. Exit 0, all 124 frames decode, effective prompt matches, remote/local output SHA256 matches: {v['output_sha256']}.

Sampled visual review confirms the expanding gray region was copied into the output. At F123, 99.27% of pixels inside the input unknown mask are near gray. The approximate turn is -31.84 degrees, so motion alone is insufficient for acceptance. No generated frame from this attempt will update Blender.

One prompt-only retry explicitly requests outpainting the adjoining side wall, connected floor and ceiling in the left, top and bottom missing regions. The original prompt already requested completion, so this tests wording specificity, not the presence versus absence of a completion instruction. Image/video, seed42, steps20 and keep_steps5 remain unchanged. The cause of failure is not established. All original inputs and outputs remain preserved.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
save(R/'EXPERIMENT_STATUS.json',{'status':'attempt001_rejected_retry_preparing','accepted_attempt':None,'rejected_attempt':'05_cvm/attempt_001','reason':'Gray missing regions retained','timing':timing})
save(R/'CHECKSUMS_ATTEMPT001_REJECTED.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='CHECKSUMS_ATTEMPT001_REJECTED.json' and '__pycache__' not in str(p)})
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir():shutil.copy2(p,NEW/'inputs'/p.name)
prompt='subject_definitions:\n<Subject 1> is the classroom in <Picture 1>, with cream walls, wooden desks, a wall clock and soft daylight.\n<Video 1> provides the left-turn camera motion and the already visible classroom layout. Flat gray pixels and the sharp cutout outline are missing image content to be painted over.\n\nsummary:\n[video editing + reference generation] Outpaint <Video 1> into a full-frame classroom video: synthesize the missing room on the left, above and below the visible preview.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - classroom appearance, furnishings and lighting.\n<Video 1>: partially_preserved - preserve its continuous left-turn motion and the spatial arrangement inside the visible classroom. Replace the entire gray background and cutout outline with generated room surfaces.\n\ndetailed_description:\n[Shot 1] Start with the view in <Picture 1>. The camera stays at the same position and turns LEFT by 30 degrees, following the timing in <Video 1>. The clock, wall plaque and desks travel RIGHT across the image. The newly revealed left portion shows the adjoining cream classroom wall continuing around the rear corner, with a baseboard and plausible stationary wooden desks. Continue the wall around the existing dark side-wall feature without introducing a new doorway. Paint the upper gray wedge as the connected ceiling and the lower gray wedge as the continuous classroom floor. The gray area is replaced by these textured, naturally lit surfaces from the first moment it appears. The final frame is an edge-to-edge interior view of the same classroom, with wall, ceiling and floor reaching the image borders. Keep perspective, occlusion and lighting coherent throughout this single shot. No gray field or inset rectangular classroom image remains. No cut, return turn or camera translation.\n\noverall_soundscape:\nQuiet classroom ambience.\n\nnon_diegetic_music:\nN/A\n'
(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
save(NEW/'inputs/prompt_adaptation.json',{'baseline':'../attempt_001/inputs/prompt.txt','changed_condition':'prompt only','reason':'Explicit outpainting with concrete adjoining wall, ceiling and floor; preserve current clock/plaque motion; no new doorway requested','guaranteed_success':False})
manifest=read(NEW/'inputs/input_manifest.json');manifest['status']='retry_prepared_not_run'
for name,rec in manifest['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',manifest)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round7_attempt002_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='Round7 prompt-only retry: concrete adjoining wall/floor/ceiling outpainting and removal of inset-cutout appearance. Same image/video, scene, seed and sampling as rejected attempt001.');save(NEW/'cvm_config.json',cfg)
same=['partial_proxy.mp4','reference.png','camera_trajectory.json']
for name in same:assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_001','changed_model_conditions':['prompt.txt'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'same_runner':sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'retry_limit':'One prompt-only retry; inspect result before any further adaptation.'})
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
newdep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r7a2_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',newdep)
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round7_retry_prepared',round7='attempt001_rejected_attempt002_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=newdep,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'persistent gray unknown areas','output_sha256':v['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat());save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(newdep,indent=2))

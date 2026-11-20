"""Preserve keep_steps control failure; stage a bounded prompt-only motion probe."""
from pathlib import Path
import json,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'round8';OLD=R/'05_cvm/attempt_002';NEW=R/'05_cvm/attempt_003'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');t=read(OLD/'timing.json');assert v['technical_status']=='passed' and t['exit_code']==0
assert sha(OLD/'outputs/round8_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],
 'appearance':'Broad gray holes filled, but panel sweeps across the frame and scene rearranges into a different endpoint layout, similar to attempt001.',
 'motion':'F31 valid direct estimate -10.726deg versus -3.723deg nominal. Endpoint only7 inliers, invalid; net yaw remains null.',
 'decision':'Reject; no frame assigned to -240 for modeling and no round9 created.',
 'controlled_result':'Increasing keep_steps5 to10 did not resolve motion/layout drift in this fixed input/seed. Both attempts filled gray regions. This does not prove all conditioning strengths fail.',
 'limits':'Approximate homography diagnostics, no calibrated pose/depth. Failed endpoint angle is not used as a measurement.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'timing':t,'validation':'analysis/validation.json'})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=t['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round8 keep_steps10 control rejected

Inference {t['inference_started_at']} to {t['finished_at']}, {t['inference_process_elapsed_seconds']:.2f}s. Exit0,124 frames decode, prompt/hash verified. SHA256 {v['output_sha256']}.

Motion/layout drift persists despite increasing video-conditioned denoising updates from5 to10. At F31, direct relative yaw is -10.726deg versus -3.723 nominal (attempt001 -11.197deg). Endpoint7inliers is invalid; end-to-end yaw is null. Full-frame completion alone is insufficient. No new geometry propagated.

One further controlled prompt-only probe retains keep_steps10, image/video, scene, seed42 and20steps. It explicitly anchors the existing board's limited screen displacement and final oblique view, and excludes a tour, extra clock/window or new corner. This tests descriptive motion constraints, not a guaranteed correction. If it also fails, do not keep varying prompts/seeds blindly: preserve a consolidated diagnostic and reassess the control representation before another generation.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
save(R/'EXPERIMENT_STATUS.json',{'status':'attempt001_and002_rejected_motion_prompt_probe_preparing','accepted_attempt':None,'rejected_attempts':['05_cvm/attempt_001','05_cvm/attempt_002'],'reason':'motion/layout drift','timing':t})
save(R/'CHECKSUMS_ATTEMPT002_REJECTED.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name!='CHECKSUMS_ATTEMPT002_REJECTED.json' and '__pycache__' not in str(p)})
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir():shutil.copy2(p,NEW/'inputs'/p.name)
prompt='''subject_definitions:
<Subject 1> is the classroom in <Picture 1>, including the dark navy board on the left side wall and the desks in front of it.
<Video 1> is the exact camera-motion guide. Its gray pixels are missing room content.

summary:
[video editing + reference generation] Complete the gray areas of <Video 1> while keeping its slow, small camera movement and existing classroom layout.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - same stationary board, desks, walls and daylight.
<Video 1>: partially_preserved - follow the position and movement of each visible surface throughout the video; replace gray pixels with connected room content.

detailed_description:
[Shot 1] Begin at <Picture 1>. From a fixed position, the camera makes only the small, slow LEFT pan shown in <Video 1>, totaling 30 degrees. Keep the same oblique view of the dark board and the same rear corner throughout. The board slides gradually to the RIGHT: its right edge starts about one-third across the image and ends only slightly right of the image center, exactly as in <Video 1>. The board remains on the same side wall; the camera never turns to face it squarely and never passes it. The clock exits through the right border and does not reappear elsewhere. Existing desks stay fixed and follow the preview positions. Only a narrow additional section of the same cream side wall and continuous floor and ceiling becomes visible on the left. Fill all gray areas, including the upper and lower wedges, with these connected, naturally lit surfaces. No gray background or cutout border remains. One continuous small pan with the same initial and final holds as the guide. No room tour, large sweep, new corner, extra clock, new window, camera translation, cut or return turn.

overall_soundscape:
Quiet classroom ambience.

non_diegetic_music:
N/A
'''
(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
save(NEW/'inputs/prompt_adaptation.json',{'baseline':'../attempt_002/inputs/prompt.txt','changed_condition':'prompt only','reason':'Both previous runs overrotate. Anchor board right-edge screen position from input preview and retain oblique view/rear corner; no new window/clock.','keep_steps':10,'guaranteed_success':False})
manifest=read(NEW/'inputs/input_manifest.json');manifest['status']='retry_prepared_not_run'
for name,rec in manifest['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',manifest)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round8_attempt003_motion_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='Round8 attempt003 prompt-only probe relative to attempt002: explicit small-pan screen anchors. keep_steps10 retained; deviation from round1 keep_steps5 remains disclosed. Same image/video/scene/seed42/20steps.');save(NEW/'cvm_config.json',cfg)
same=['partial_proxy.mp4','reference.png','camera_trajectory.json']
for name in same:assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_002','changed_model_conditions':['prompt.txt'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'same_runner':sha(OLD/'run_cvm.py')==sha(NEW/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'deviation_from_round1':'keep_steps10 retained; no longer baseline5','retry_limit':'One screen-anchored prompt control. On failure preserve diagnostics and reassess representation; no blind additional prompt/seed retries.'})
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
nd={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r8a3_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',nd)
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round8_retry_prepared',round8='attempt001_002_rejected_attempt003_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=nd,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'motion/layout drift persists at keep10','output_sha256':v['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat());save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(nd,indent=2))

"""Archive appearance-control failure and stage one keep10->5 paired test."""
from pathlib import Path
import json,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'round8';OLD=R/'05_cvm/attempt_004';NEW=R/'05_cvm/attempt_005'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');t=read(OLD/'timing.json');assert v['technical_status']=='passed' and t['exit_code']==0
assert sha(OLD/'outputs/round8_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],
 'motion':'Approximate net yaw -29.97084 degrees; endpoint101 inliers,0.7275px. Motion now follows the limited turn.',
 'appearance':'Accepted i7 appearance remains recognizable; previous late transition to raw Blender material appearance is absent in sampled frames.',
 'completion':'Broad gray persists across all later sampled frames; endpoint99.64155% near-gray inside missing support. No novel valid modeling content.',
 'decision':'Reject; no selected i8 and no round9. Good rotation diagnostics alone do not satisfy completion.',
 'limits':'Single paired observation. Reprojection fidelity and output rotation are not metric geometry or translation evidence.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'validation':'analysis/validation.json','timing':t})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=t['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
diagnosis={'observation':'Appearance-only video replacement vs003 restores valid reference matching and approximately correct motion, but gray completion still fails.',
 'source_evidence':{'path':'outputs/generate_effective.py','sha256':sha(OLD/'outputs/generate_effective.py'),'video_encoding_line':21,'full_condition_lines':[31,32],'condition_switch_line':48,'expression':'condition = full if i < keep_steps else base','spatial_mask_input':False},
 'mechanism_limit':'All gray pixels are part of ordinary RGB video conditioning; there is no spatial inpainting mask or hard geometry constraint. Text requests completion but cannot guarantee it. This is a code observation, not proof of why gray is retained.',
 'hypothesis':'With appearance conflict reduced, lowering whole-video conditioning from10 to baseline5 updates may allow missing-region completion while retaining limited motion. Motion may deteriorate; unproven.',
 'controlled_retry':'Only keep_steps10->5 vs004. Same appearance video, prompt bytes, reference, camera matrices, scene provenance, seed42,20 total steps, model and runner.',
 'interpretation':'Restores round1 sampling parameters but remains an appearance-reprojection diagnostic branch, NOT raw Blender RGB baseline. Not a full factorial comparison because earlier prompts differ.',
 'acceptance':'Require full coherent completion and stable layout, plausible motion/direct or explicitly validated chained correspondence, visually inspect before accepting any frame.',
 'retry_limit':'One baseline-conditioning test on this representation. If it fails, consolidate all five results and inspect alternatives; do not make further blind prompt/seed/keep sweeps.'}
save(OLD/'analysis/conditioning_followup.json',diagnosis)
report=f'''# Round8 appearance-reprojection test rejected: motion passes, completion fails

Inference {t['inference_started_at']} to {t['finished_at']}, {t['inference_process_elapsed_seconds']:.2f}s. Exit0;124 frames decode; prompt and remote/local output hashes match. SHA256 {v['output_sha256']}.

Sampled output keeps i7 appearance and follows the small turn. Approximate net yaw is -29.97084 degrees; endpoint101 inliers,0.7275px. However,99.64155% of endpoint missing support remains near-gray, visually confirmed. No frame is accepted for reconstruction and no round9 exists.

Source inspection confirms the complete RGB video (gray included) conditions the first10 updates; image/text condition the rest. There is no spatial inpainting mask. The prior raw Blender appearance conflict appears reduced in this paired test, but completion remains unresolved. Cause is not established.

One paired attempt005 restores keep_steps5, retaining the exact prompt/image/reprojected video/seed42/20steps. This tests the completion versus motion tradeoff with reduced appearance conflict. It restores round1 sampling settings, not the raw Blender RGB control representation. Raw scenes and failed results remain unchanged. If this test also fails, consolidate all five results and reassess alternatives before further generation; do not blindly sweep prompts/seeds/keep settings or propagate failures.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(BASE/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py')
shutil.copy2(Path(__file__),OLD/'followup_preparation_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
for rel,digest in read(OLD/'ARCHIVE_CHECKSUMS.json').items():assert sha(OLD/rel)==digest,rel
D=R/'06_control_diagnosis';(D/'APPEARANCE_TEST_RESULT.md').write_text(report,encoding='utf-8');save(D/'appearance_conditioning_followup.json',diagnosis)
(NEW/'inputs').mkdir(parents=True)
for p in (OLD/'inputs').iterdir():shutil.copy2(p,NEW/'inputs'/p.name)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round8_attempt005_appearance_keep5_{stamp}'
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',keep_steps=5,keep_steps_semantics='Full RGB video-reference conditioning for first5 of20 updates; not a spatial mask or frame count.',notes='Attempt005 changes only keep_steps10->5 relative to004. Same appearance-reprojected video/prompt/image/seed42/20steps. Sampling restored to round1; video representation remains an explicit diagnostic deviation from raw Blender RGB.');save(NEW/'cvm_config.json',cfg)
same=['partial_proxy.mp4','reference.png','prompt.txt','camera_trajectory.json']
for name in same:assert sha(OLD/'inputs'/name)==sha(NEW/'inputs'/name)
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_004','changed_model_conditions':['keep_steps:10->5'],'same_inputs':{name:sha(NEW/'inputs'/name) for name in same},'same_runner':sha(OLD/'run_cvm.py')==sha(NEW/'run_cvm.py'),'same_sampling_as_round1':True,'same_control_representation_as_round1':False,'deviation_from_round1':'Accepted-appearance fixed-center reprojection instead of raw Blender material RGB.','parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'retry_limit':diagnosis['retry_limit']})
save(NEW/'conditioning_diagnosis.json',diagnosis);shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
nd={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r8a5_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',nd)
save(R/'EXPERIMENT_STATUS.json',{'status':'four_rejected_appearance_keep5_control_prepared','accepted_attempt':None,'rejected_attempts':[f'05_cvm/attempt_{i:03d}' for i in range(1,5)],'active_attempt':'05_cvm/attempt_005','diagnosis':'06_control_diagnosis/APPEARANCE_TEST_RESULT.md'})
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round8_appearance_keep5_prepared',round8='four_rejected_attempt005_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=nd,last_completed_attempt={'path':OLD.relative_to(BASE).as_posix(),'quality':'rejected','reason':'motion follows -30 but missing support remains gray','output_sha256':v['output_sha256']},updated_at=datetime.datetime.now().astimezone().isoformat(),last_verified_phase='Attempt004 completed/rejected/archived. One keep_steps10->5 paired test prepared; no inference active, no round9.');save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
plan=BASE/'CONTINUOUS_LEFT_PLAN.md';text=plan.read_text(encoding='utf-8');marker='## Current state and scope\n'
entry=f'\nLATEST {s["updated_at"]}: round8 attempt004 REJECTED, despite approximate netyaw-29.97084, endpoint101inliers/0.7275px:99.64155% gray inside missing support. Inference08:11:45–08:37:38,1552.86s; output SHA{v["output_sha256"]}. Archived and verified; no frame propagated. Code confirms whole-video conditioning, no spatial inpaint mask. ONE attempt005 keep_steps10->5 paired control PREPARED, same appearance video/prompt/image/seed42/20steps. Restores round1 sampling but remains a separate appearance-reprojection branch, NOT raw Blender RGB. See round8/06_control_diagnosis/APPEARANCE_TEST_RESULT.md. Read state for launch. If it fails consolidate all five and reassess alternatives; no blind sweep. No round9.\n'
plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8');print(json.dumps(nd,indent=2))

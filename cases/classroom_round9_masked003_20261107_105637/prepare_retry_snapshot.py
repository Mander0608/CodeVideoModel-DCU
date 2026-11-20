"""Archive masked002 structural failure; stage one prompt-only masked003 control."""
from pathlib import Path
import json,hashlib,shutil,datetime
B=Path(__file__).resolve().parent;R=B/'round9';OLD=R/'05_cvm/attempt_002_masked';NEW=R/'05_cvm/attempt_003_masked'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');m=read(OLD/'analysis/masked_validation.json');t=read(OLD/'timing.json')
assert v['technical_status']=='passed' and m['technical_status']=='passed_extra_mask_audit' and t['exit_code']==0
assert sha(OLD/'outputs/round9_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],'raw_and_composite_review':'Inspected source/raw-beforepasteback/composite. New blue board and round clock appear in masked left-side gap, despite prompt forbidding new clock/duplicate board. They are visible in both raw and composited output; pasteback is not their source.','completion':'Missing gray is replaced, but novel objects and return-wall junction are not supported. Do not infer valid structure from completion or temporal stability.','motion':'Full-image net -30.05341 is primarily supplied by preserved source pixels. Novel-region matched points show good late stability but do not establish semantic correctness.','decision':'Reject; no i9, no round10. Preserve full raw/composite/masks/source/provenance. One prompt-only paired retry of masked002: same input videos/masks/reference/runner/seed42/20actualupdates/noTurbo, concrete bare-wall-only gap content. If duplication persists, diagnose before another attempt.','limits':'Nominal fixed-center rotation, no depth/translation claim; original i0 return is explicitly supplied memory.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'timing':t,'reason':'New clock and blue board in unknown gap','validation':'analysis/validation.json'})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=t['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round9 masked002 rejected: extra objects in generated gap

Inference {t['inference_started_at']} to {t['finished_at']}, {t['inference_process_elapsed_seconds']:.3f}s. Exit0;124 frames;20actualupdates/noTurbo; remote/local composite hash verified: {v['output_sha256']}. Raw-beforepasteback SHA256: {m['raw_model_video_sha256']}.

Source, raw and composite comparisons reveal an added clock and second blue board in the previously unknown left-side gap, visible from early/middle frames to endpoint. These objects are generated, not supplied by original i0 or i8. The prompt already explicitly prohibited a new clock and duplicate board. Gray removal and temporal stability cannot validate them. The invented junction also needs scrutiny against future return views. No frame propagated.

Full-image rotation diagnostic -30.05341 degrees is dominated by forced source preservation. Novel-region diagnostics: F62->92 37matches,median2.250px; F92->11672matches,median0.658px; finalhold93matches,median0.256px,90below3px; finalholdMAE1.100/255. These indicate local image continuity, not valid new structure. Endpoint raw/composite neargray unknown0.001667/0.001793; raw/composite preserved-sourceMAE2.343/1.134.

One prompt-only control is prepared. All pixels/masks/reference, full weights, runner, seed42 and20actual updates remain identical. The revised prompt positively specifies bare cream plaster, ceiling and floor, with only completion of already clipped furniture. It clarifies that the only navy board is already present in the preserved area. This is still an explicit masked method-change branch, not original CVM success. If unsupported object duplication recurs, do not continue prompt sweeping or create round10; consolidate and diagnose.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8');(R/'06_control_diagnosis/MASKED002_FAILURE.md').write_text(report,encoding='utf-8')
for name in ['audit_cvm_attempt.py','audit_masked_completion.py','audit_novel_temporal.py']:shutil.copy2(B/name,OLD/(Path(name).stem+'_snapshot.py'))
shutil.copy2(Path(__file__),OLD/'failure_archive_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json' and '__pycache__' not in str(p)})
assert all(sha(OLD/p)==h for p,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items())
for name in ['inputs','references','upstream']:shutil.copytree(OLD/name,NEW/name)
prompt='''subject_definitions:
<Subject 1> is the same quiet classroom already visible in the source clip and <Picture 1>.

summary:
Fill only the masked gaps with plain cream plaster wall, continuous ceiling, and continuous floor.

retention_analysis:
Preserve every supplied region exactly. The existing navy board and furnishings stay where they already appear in the source clip. The brighter strip returning on the left is an existing view of this same room.

detailed_description:
One stationary classroom and one continuous camera rotation following the supplied source clip. The unknown gap between the left returning strip and the main preserved image is a bare cream plaster wall surface. Its upper area continues the plain ceiling; its lower area continues the classroom floor and simple dark baseboard. This newly exposed wall is completely empty, undecorated and uninterrupted, with natural lighting varying smoothly between the supplied regions. There are no additional mounted objects, displays, signs, clocks, panels, openings, pillars or projecting wall structures. The only navy board remains the one already present in the preserved image; do not reproduce it in the empty gap. Complete only portions of desks or chairs that visibly cross the mask boundary, without adding separate furniture. Maintain stable perspective and surfaces in every frame. Replace all gray pixels with these plain adjoining surfaces. No camera translation, cuts, or change in rotation timing.

overall_soundscape:
Keep the supplied silent soundtrack.

non_diegetic_music:
N/A
'''
for p in [NEW/'prompt.txt',NEW/'inputs/prompt.txt']:p.write_text(prompt,encoding='utf-8',newline='\n')
for name in ['masked_inpaint.py','run_cvm.py','preflight.py','scene_snapshot_reference.json','preparation_validation.json','method_provenance.json','offline_validation.json','implementation_source.json']:shutil.copy2(OLD/name,NEW/name)
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch');cfg['notes']+=' Prompt-only masked002->masked003 control targeting unsupported clock/blue-board duplication.';save(NEW/'cvm_config.json',cfg)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round9_masked003_{stamp}'
for name in ['supervise.py','watch_and_collect.py','launch.sh']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(dep['remote_case'],remote),encoding='utf-8',newline='\n')
fixed=['inputs/partial_proxy.mp4','inputs/source_video.mp4','inputs/repaint_mask.mp4','inputs/masks.npz','inputs/reference.png','references/i8.png','masked_inpaint.py','run_cvm.py','preflight.py']
assert all(sha(OLD/p)==sha(NEW/p) for p in fixed)
save(NEW/'paired_manifest.json',{'parent':'attempt_002_masked','changed_condition':'prompt_only','fixed_file_sha256':{p:sha(NEW/p) for p in fixed},'old_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'new_prompt_sha256':sha(NEW/'inputs/prompt.txt'),'same_seed':42,'same_actual_updates':20,'turbo':False,'original_CVM_baseline_passed':False,'source_preparation_frames':'../attempt_002_masked/preparation_frames (unchanged; canonical source/mask videos copied here)','post_retry_policy':'If structural duplication persists, consolidate/diagnose; no blind prompt sweep or propagation.'})
shutil.copy2(Path(__file__),NEW/'prepare_retry_snapshot.py')
for p in NEW.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
nd={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r9masked3_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat()};save(NEW/'deployment.json',nd)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round9_masked003_prepared',round9='masked002_rejected_prompt_control_prepared',current_run=str(NEW.resolve()),current_run_relative=NEW.relative_to(B).as_posix(),current_deployment=nd,last_completed_attempt={'path':OLD.relative_to(B).as_posix(),'quality':'rejected','output_sha256':v['output_sha256']},updated_at=nd['prepared_at'],last_verified_phase='Masked002 technically passed but rejected for extra clock/blue board. Full archive verified. One prompt-only masked003 prepared; not launched. No round10.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
save(R/'EXPERIMENT_STATUS.json',{'status':s['status'],'rejected_attempts':['05_cvm/attempt_001','05_cvm/attempt_002_masked'],'accepted_attempt':None,'original_cvm_baseline_passed':False})
plan=B/'CONTINUOUS_LEFT_PLAN.md';plan.write_text(f'LATEST {s["updated_at"]}: Round9 masked002 REJECTED after full audit/visualreview: new clock+blueboard in unknown gap.1203.889s,compositeSHA{v["output_sha256"]},rawSHA{m["raw_model_video_sha256"]}. Temporal stability does not establish semantics. No i9/round10. One prompt-only masked003 PREPARED, exact same source/mask/reference/runner/seed42/20actualupdates/noTurbo. New prompt specifies bare cream wall/ceiling/floor, no additional mounted objects. Inspect paired_manifest. Check resources then deploy once. If thirdattempt fails, consolidate/diagnose, no blind sweep. Read actual state for launch status.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(nd,indent=2))

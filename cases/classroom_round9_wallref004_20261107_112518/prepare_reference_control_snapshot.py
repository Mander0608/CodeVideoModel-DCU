"""Consolidate failures and prepare a targeted reference-content control, offline only."""
from pathlib import Path
import json,hashlib,datetime,shutil
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;R=B/'round9';OLD=R/'05_cvm/attempt_003_masked';NEW=R/'05_cvm/attempt_004_wallref';D=R/'06_control_diagnosis'
assert not NEW.exists()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');t=read(OLD/'timing.json');m=read(OLD/'analysis/masked_validation.json')
assert v['technical_status']=='passed' and t['exit_code']==0 and m['technical_status']=='passed_extra_mask_audit'
assert sha(OLD/'outputs/round9_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],'raw_and_composite_review':'Extra clock disappears compared with masked002, but a second blue panel remains in the unknown left gap in both raw and composite. Bare-wall instruction was not satisfied.','motion_and_temporal':'Full-image -30.07814deg largely supplied. New-region finalhold91matches/89below3px/median0.272px andMAE1.037 indicate local stability, not supported structure.','decision':'Reject. Preserve all results, no i9/round10. Consolidate three attempts and examine conditioning before any further trial.','limits':'Rotation-only, supplied original i0 return, composite i8 history; no depth,translation or unaided-memory evidence.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'timing':t,'validation':'analysis/validation.json'})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=t['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round9: three rejected attempts; reference-content diagnosis

Masked003 completed {t['inference_started_at']} to {t['finished_at']}, {t['inference_process_elapsed_seconds']:.3f}s; exit0,124frames,20actualupdates,noTurbo. Composite SHA256 {v['output_sha256']}; raw-beforepasteback {m['raw_model_video_sha256']}.

Original raw RGB attempt001 overrotates (~-41.32deg local chain). Masked002 restores supplied motion but adds a clock and a blue panel. Prompt-only masked003 removes the clock while retaining the extra panel despite explicit bare-wall wording. All three are rejected for continuation. A stable hallucination is not validated geometry. No i9/round10 exists.

Code inspection: masked_inpaint.py passes the same full i8 image through _gather_vision_features/_build_presentation into Qwen text conditioning, then passes references=refs into the inpaint generator. The input source clip independently supplies appearance through source_video and its preserved rows. The full i8 image contains a salient navy panel. Therefore reference-driven panel replication is a specific hypothesis, not an established cause; fixed noise, classroom priors and source-gap geometry may also contribute. No fourth prompt/seed/keep sweep is justified.

A targeted image-content control is prepared: keep masked003 prompt, source video, masks, nominal camera, full weights, seed42,20actualupdates and noTurbo identical; replace only the model Picture1 image with a bare-wall crop from original i0, coordinates [544,24,632,168) in the original640x384 image. Crop remains88x144; normal existing reference resizing applies. It contains wall material and no board, clock or furniture. It supplies original image appearance explicitly and must never be called independent memory. The full accepted i8 remains the source-video appearance and the audit motion reference, but is no longer the separate model Picture1. Those two roles are recorded distinctly.

This changes the image condition, including its content/field of view/shape, not the inpaint algorithm. A single result can test whether this reference choice avoids duplication in this case; it cannot isolate all semantic versus aspect-ratio effects or establish general causality. If it fails, do not continue interchangeable reference/prompt sweeps: analyze the source/latent-mask behavior and consider a clearly labeled manual structural correction with pre/post versions or report that this branch cannot provide an accepted new view. Do not propagate unsupported panels.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8');(D/'THREE_ATTEMPT_DIAGNOSIS.md').write_text(report,encoding='utf-8')
for name in ['audit_cvm_attempt.py','audit_masked_completion.py','audit_novel_temporal.py']:shutil.copy2(B/name,OLD/(Path(name).stem+'_snapshot.py'))
shutil.copy2(Path(__file__),OLD/'failure_archive_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json' and '__pycache__' not in str(p)})
summary=[]
for name in ['attempt_001','attempt_002_masked','attempt_003_masked']:
 run=R/'05_cvm'/name;checks=read(run/'ARCHIVE_CHECKSUMS.json');assert all(sha(run/k)==h for k,h in checks.items())
 val=read(run/'analysis/validation.json');summary.append({'attempt':name,'accepted':False,'output_sha256':val['output_sha256'],'seconds':read(run/'timing.json')['inference_process_elapsed_seconds'],'prompt_sha256':sha(run/'inputs/prompt.txt'),'reference_sha256':sha(run/'references/i8.png') if (run/'references/i8.png').exists() else sha(run/'inputs/reference.png'),'files_verified':len(checks)})
save(D/'three_attempt_comparison.json',summary)
sheet=Image.new('RGB',(1440,608),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,entry in enumerate(summary):
 for k,f in enumerate([62,123]):sheet.paste(Image.open(R/f'05_cvm/{entry["attempt"]}/analysis/output_{f:04d}.png').resize((480,272)),(j*480,k*304+32));draw.text((j*480+5,k*304+8),f'{entry["attempt"]} F{f}',fill='black')
sheet.save(D/'three_attempt_endpoints.jpg')
for folder in ['inputs','references','upstream']:shutil.copytree(OLD/folder,NEW/folder)
# Keep the former image under an explicit audit-only name; only wall_crop_i0 is model input.
(NEW/'references/i8.png').rename(NEW/'references/i8_previous_audit_only.png')
crop=Image.open(R/'00_input/i0.png').convert('RGB').crop((544,24,632,168));crop.save(NEW/'references/wall_crop_i0.png')
context=Image.open(R/'00_input/i0.png').convert('RGB');ImageDraw.Draw(context).rectangle((544,24,631,167),outline='red',width=3);context.save(NEW/'reference_crop_context.png')
save(NEW/'reference_crop_provenance.json',{'source':'../../00_input/i0.png','source_sha256':sha(R/'00_input/i0.png'),'source_size':[640,384],'crop_xyxy_exclusive':[544,24,632,168],'crop_size':[88,144],'crop_sha256':sha(NEW/'references/wall_crop_i0.png'),'model_reference':'references/wall_crop_i0.png','diagnostic_full_view_reference':'inputs/reference.png (accepted i8, NOT model Picture1)','supplied_original_memory':True,'source_video_unchanged':True,'interpretation':'Material-only crop from originally observed bare wall; no unseen texture generated offline.'})
for name in ['prompt.txt','run_cvm.py','preflight.py','scene_snapshot_reference.json','preparation_validation.json','method_provenance.json','offline_validation.json']:shutil.copy2(OLD/name,NEW/name)
code=(OLD/'masked_inpaint.py').read_text(encoding='utf-8');assert code.count("references/i8.png")==1
code=code.replace('references/i8.png','references/wall_crop_i0.png').replace("known_region_motion_supplied=True,","known_region_motion_supplied=True,model_image_reference='original_i0_bare_wall_crop',")
(NEW/'masked_inpaint.py').write_text(code,encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='offline_candidate_not_staged',source_image_reference='original i0 bare-wall crop [544,24,632,168)');cfg['inputs']['reference_image']='references/wall_crop_i0.png';cfg['notes']='Image-content control relative to masked003: only model Picture1 replaced by original-i0 bare-wall material crop; full i8 remains diagnostic reference/source appearance. Prompt/source/mask/seed/20actualupdates/fullweights/noTurbo unchanged. Explicit original appearance supply, not autonomous recall.';save(NEW/'cvm_config.json',cfg)
fixed=['inputs/prompt.txt','inputs/partial_proxy.mp4','inputs/source_video.mp4','inputs/repaint_mask.mp4','inputs/masks.npz','inputs/reference.png','run_cvm.py','preflight.py']
assert all(sha(OLD/k)==sha(NEW/k) for k in fixed)
save(NEW/'paired_manifest.json',{'parent':'attempt_003_masked','changed_condition':'model_image_reference_only','fixed_file_sha256':{k:sha(NEW/k) for k in fixed},'old_image_sha256':sha(OLD/'references/i8.png'),'new_image_sha256':sha(NEW/'references/wall_crop_i0.png'),'runner_change':'Only reference file path and report metadata; algorithm unchanged.','seed':42,'actual_updates':20,'turbo':False,'original_CVM_baseline_passed':False,'audit_reference_is_not_model_image':True})
shutil.copy2(Path(__file__),NEW/'prepare_reference_control_snapshot.py')
s=read(B/'CONTINUOUS_LEFT_STATE.json');now=datetime.datetime.now().astimezone().isoformat();s.update(status='round9_reference_control_offline',round9='three_rejected_reference_control_prepared',current_run=str(NEW.resolve()),current_run_relative=NEW.relative_to(B).as_posix(),current_deployment=None,last_completed_attempt={'path':OLD.relative_to(B).as_posix(),'quality':'rejected','output_sha256':v['output_sha256']},updated_at=now,last_verified_phase='Three failed archives verified. Image-only wall-crop control prepared OFFLINE; inspect crop before staging. No inference running; no i9/round10.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
save(R/'EXPERIMENT_STATUS.json',{'status':s['status'],'accepted_attempt':None,'rejected_attempts':[x['attempt'] for x in summary],'original_cvm_baseline_passed':False})
p=B/'CONTINUOUS_LEFT_PLAN.md';p.write_text(f'LATEST {now}: masked003 REJECTED: clock removed but unsupported bluepanel persists.1222.841s,compositeSHA{v["output_sha256"]},rawSHA{m["raw_model_video_sha256"]}. Three failed archives verified; no i9/round10. Read round9/06_control_diagnosis/THREE_ATTEMPT_DIAGNOSIS.md. One targeted MODEL IMAGE control attempt004_wallref prepared OFFLINE, not staged: originali0 bare-wall crop[544,24,632,168), image88x144 replaces fulli8 Picture1; prompt/source/masks/seed42/20actualupdates/noTurbo unchanged. Fulli8 still sourcevideo and audit reference, distinct from actual model image. Inspect crop then stage/checkresources/deploy once. If this fails, diagnose source/latent-mask behavior or explicitly labeled structural correction; no blind reference/prompt sweeps.\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'status':'offline_prepared','archives':summary,'crop':str(NEW/'references/wall_crop_i0.png')},indent=2))

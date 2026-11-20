"""Archive all five CVM failures; prepare a separately identified masked branch."""
from pathlib import Path
import sys,json,hashlib,shutil,datetime,subprocess,math
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;W=B.parent;R=B/'round8';OLD=R/'05_cvm/attempt_005';NEW=R/'05_cvm/attempt_006_masked';D=R/'06_control_diagnosis'
assert not NEW.exists()
sys.path.insert(0,str(W/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');t=read(OLD/'timing.json');assert t['exit_code']==0 and v['technical_status']=='passed';assert sha(OLD/'outputs/round8_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],'motion':'Approximate net yaw -30.79218deg, endpoint99inliers/0.78055px. Limited turn remains recognizable.','completion':'Broad gray retained:99.63170% near-gray within endpoint missing support.','decision':'Reject; no i8 or round9. Restoring keep_steps5 does not solve completion in this paired setting.','limitations':'One seed/setting; no universal conclusion or depth/translation claim.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'timing':t,'validation':'analysis/validation.json'})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=t['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
(OLD/'RESULT.md').write_text(f'# Round8 attempt005 rejected\n\nInference {t["inference_started_at"]} to {t["finished_at"]}, {t["inference_process_elapsed_seconds"]:.2f}s;124 frames decode, prompt and remote/local hashes match. SHA256 {v["output_sha256"]}.\n\nApproximate netyaw -30.79218 degrees, endpoint99inliers/0.78055px;99.63170% of missing support remains near-gray. No frame propagated. keep10->5 did not resolve missing-area completion with this appearance proxy and prompt. No further blind parameter sweep.\n',encoding='utf-8')
shutil.copy2(B/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py');shutil.copy2(Path(__file__),OLD/'failure_archive_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
summary=[]
for i in range(1,6):
 p=R/f'05_cvm/attempt_{i:03d}';val=read(p/'analysis/validation.json');cfg=read(p/'cvm_config.json')
 for rel,h in read(p/'ARCHIVE_CHECKSUMS.json').items():assert sha(p/rel)==h,(i,rel)
 summary.append({'attempt':p.name,'representation':'raw Blender RGB' if i<4 else 'accepted-i7 appearance reprojection','keep_steps':cfg['keep_steps'],'prompt_sha256':sha(p/'inputs/prompt.txt'),'video_sha256':sha(p/'inputs/partial_proxy.mp4'),'output_sha256':val['output_sha256'],'elapsed_seconds':read(p/'timing.json')['inference_process_elapsed_seconds'],'net_yaw':val['estimated_yaw_change'],'endpoint_inliers':val['last_frame'].get('inliers'),'endpoint_gray_fraction_in_missing':val['last_frame']['gray_fraction_in_input_unknown'],'accepted':False})
save(D/'five_attempt_comparison.json',summary)
sheet=Image.new('RGB',(1440,608),'#edf0f3');draw=ImageDraw.Draw(sheet)
for j,x in enumerate(summary):
 y=j//3*304;xpos=j%3*480;sheet.paste(Image.open(R/f'05_cvm/{x["attempt"]}/analysis/output_0123.png').resize((480,272)),(xpos,y+32));draw.text((xpos+5,y+8),f'{x["attempt"]} keep{x["keep_steps"]} F123',fill='black')
draw.text((965,365),'All five rejected. No accepted i8.',fill='black');sheet.save(D/'five_attempt_endpoints.jpg')
report='''# Round8: five rejected CVM attempts and a separate masked-completion branch

All five attempts are technically completed and fully archived; none supplies an accepted i8. Attempts001/002 fill gray but drift in motion/layout. Attempt003 preserves gray and switches toward proxy appearance. Appearance reprojection in004/005 restores recognizable limited motion (~-29.97/-30.79 degrees), but ~99.64/99.63% of missing support remains gray. Lowering keep10 to5 does not repair completion in this paired test. This is not a full factorial comparison: prompts differ among earlier runs.

The CVM runner encodes gray as normal RGB video; it has no spatial mask. The existing local pinned H3 inpainting implementation instead imposes preserved target rows as clean visual anchors, relabels their timesteps, and rewrites them after each scheduler update. Its separate mask determines where content may be regenerated. This provides a concrete alternative to further blind prompt/seed/keep sweeps. It does not prove a single cause of the previous failures.

An independent attempt006_masked branch is prepared with the SAME existing full MiniMax-H3 Ref2VA weights, seed42, no Turbo/LoRA, intended20 actual denoising updates, unchanged124-frame24fps -210 to-240 camera trajectory and output960x544. It supplies only accepted i7 as an image reference, accepted-appearance ray-reprojected source video, and a new explicit repaint mask. No rejected output is a source. Unknown angular support is dilated8px into the known area to allow seams. A32px replicated guard border gives a1024x608 processing canvas; it is cropped after processing. Original undilated support and per-pixel source labels are retained. No earlier images contribute pixels in this segment.

This is a METHOD CHANGE, not a successful run of the original unmasked Blender RGB->CVM baseline, not a one-variable paired ablation, and not model-only camera/memory performance. Known-region appearance/motion is supplied and then pasted back; matching there is a construction diagnostic. Only content inside the repaint region is new model output. Preserve raw generated frames/video BEFORE paste-back, composited output and cropped output. Inspect novel-region structure, seam consistency and temporal stability before accepting; never count gray removal alone as success. Generated surfaces remain priors, not measured depth, and translated-view accuracy is untested.

Alternatives considered: more blind prompt/keep sweeps have weak evidence; filling unseen pixels with the unmasked Blender shell would impose unobserved geometry/appearance and confound completion with hand-built assumptions. Explicit masked completion keeps the unknown support auditable while separating newly generated content from forced preservation. Raw Blender scene, raw RGB trajectory, all five failures and the unmasked pipeline remain unchanged.
'''
(D/'FIVE_ATTEMPT_DIAGNOSIS.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
for folder in ['inputs','references','upstream','preparation_frames/source','preparation_frames/mask']:(NEW/folder).mkdir(parents=True)
for name in ['reference.png','camera_trajectory.json','partial_proxy.mp4']:shutil.copy2(OLD/'inputs'/name,NEW/'inputs'/name)
shutil.copy2(OLD/'inputs/reference.png',NEW/'references/i7.png');shutil.copy2(R/'00_input/accepted_references.json',NEW/'inputs/accepted_references.json')
prompt='''subject_definitions:
<Subject 1> is the classroom in <Picture 1>, with a navy board on the cream side wall, wooden desks, a continuous floor and ceiling.

summary:
Complete the missing classroom surroundings of the supplied source clip within the repaint mask.

retention_analysis:
<Subject 1>: fully_preserved - keep the existing fixed furniture, board, wall edges, lighting and classroom identity.

detailed_description:
One continuous fixed-position left turn, following the supplied clip. Generate natural adjoining cream wall, continuous floor and ceiling in the masked gaps. Continue any clipped furniture or board edge only where its visible structure supports it. Keep the scene stationary and perspective aligned with the existing surfaces across every frame. No gray cutout remains. Do not add a new clock, window, room corner or duplicate board. Preserve the small oblique view and the clip's timing. No camera translation or cuts.

overall_soundscape:
Keep the supplied silent soundtrack.

non_diegetic_music:
N/A
'''
(NEW/'prompt.txt').write_text(prompt,encoding='utf-8',newline='\n');(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
fractions=[];masks=[]
for f in range(124):
 source=np.array(Image.open(D/f'appearance_proxy_frames/{f:04d}.png').convert('RGB'))
 labels=np.array(Image.open(D/f'appearance_source_labels/{f:04d}.png'));unknown=labels==0
 mask=cv2.dilate(unknown.astype(np.uint8),np.ones((17,17),np.uint8))>0
 assert not np.any(unknown & ~mask)
 padded=np.pad(source,((32,32),(32,32),(0,0)),mode='edge');mp=np.pad(mask,((32,32),(32,32)),mode='edge')
 Image.fromarray(padded).save(NEW/f'preparation_frames/source/{f:04d}.png');Image.fromarray(mp.astype(np.uint8)*255).save(NEW/f'preparation_frames/mask/{f:04d}.png');masks.append(mp)
 fractions.append({'frame':f,'unknown_fraction':float(unknown.mean()),'repaint_fraction_core':float(mask.mean()),'repaint_fraction_padded':float(mp.mean())})
np.savez_compressed(NEW/'inputs/masks.npz',repaint=np.stack(masks))
ff=W/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
for name in ['source','mask']:
 out=NEW/'inputs'/('source_video.mp4' if name=='source' else 'repaint_mask.mp4')
 cmd=[str(ff),'-hide_banner','-n','-framerate','24','-i',str(NEW/f'preparation_frames/{name}/%04d.png')]
 if name=='source':cmd+=['-f','lavfi','-i','anullsrc=r=32000:cl=stereo']
 cmd+=['-frames:v','124','-t',str(124/24),'-c:v','libx264','-crf','17' if name=='source' else '0','-preset','slow','-pix_fmt','yuv420p']
 cmd+=['-c:a','aac'] if name=='source' else ['-an'];cmd+=['-movflags','+faststart',str(out)]
 p=subprocess.run(cmd,capture_output=True);(NEW/f'encode_{name}.log').write_bytes(p.stderr);p.check_returncode()
 cap=cv2.VideoCapture(str(out));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(608,1024,3)
  if name=='mask':assert np.array_equal(im.mean(-1)>127.5,masks[n]),n
  n+=1
 cap.release();assert n==124
for name in ['h3_inpaint_blocks.py','h3_inpaint_masks.py','h3_inpaint_crop.py','source_manifest.json','README.md']:shutil.copy2(W/'H3-Inpainting-deploy/upstream'/name,NEW/'upstream'/name)
manifest=read(NEW/'upstream/source_manifest.json')
for name,info in manifest['files'].items():
 if (NEW/'upstream'/name).exists():assert sha(NEW/'upstream'/name)==info['sha256'],name
save(NEW/'preparation_validation.json',{'status':'passed_offline','source_frames':124,'mask_frames':124,'fps':24,'processing_size':[1024,608],'final_crop_size':[960,544],'mask_decode_exact':True,'dilation_pixels':8,'guard_pixels':32,'all_unknown_pixels_covered':True,'coverage':fractions,'source_reference':'accepted round7 attempt002 F123 i7 only; no rejected result used','method_change':True,'actual_inference_started':False})
shutil.copy2(Path(__file__),NEW/'prepare_inputs_snapshot.py');shutil.copy2(W/'H3-Inpainting-deploy/run_inpaint.py',NEW/'original_inpaint_runner.py')
save(NEW/'method_provenance.json',{'base_model':'existing full MiniMax-H3 Ref2VA','protocol':'explicit_masked_completion_branch','original_cvm_baseline_passed':False,'source_i7_sha256':sha(NEW/'references/i7.png'),'source_appearance_video_sha256':sha(NEW/'inputs/partial_proxy.mp4'),'pinned_inpaint_revision':manifest['revision'],'turbo':False,'intended_actual_updates':20,'seed':42,'geometry_and_nominal_trajectory_unchanged':True,'known_pixel_motion_is_supplied':True,'raw_generated_before_paste_back_required':True,'new_content_is_generated_prior_not_observation':True})
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round8_masked_branch_offline_preparation',round8='five_rejected_masked_branch_preparing',current_run=str(NEW),current_run_relative=NEW.relative_to(B).as_posix(),current_deployment=None,last_completed_attempt={'path':OLD.relative_to(B).as_posix(),'quality':'rejected','output_sha256':v['output_sha256'],'reason':'gray persists at keep5'},updated_at=datetime.datetime.now().astimezone().isoformat(),last_verified_phase='No inference running. All five rejected archives verified. Masked-completion branch input prepared; runner adaptation/preflight needed before any launch. No round9.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
save(R/'EXPERIMENT_STATUS.json',{'status':s['status'],'accepted_attempt':None,'rejected_attempts':[f'05_cvm/attempt_{i:03d}' for i in range(1,6)],'prepared_branch':'05_cvm/attempt_006_masked','original_cvm_baseline_passed':False})
plan=B/'CONTINUOUS_LEFT_PLAN.md';txt=plan.read_text(encoding='utf-8');marker='## Current state and scope\n';entry=f'\nLATEST {s["updated_at"]}: attempt005 REJECTED:netyaw-30.79218, endpoint99inliers/0.78055px,99.63170% missing-region gray;1457.56s,SHA{v["output_sha256"]}. All five attempts archived/verified. No inference running. Offline explicit masked-completion branch attempt_006_masked prepared; runner adaptation/preflight still required. Read round8/06_control_diagnosis/FIVE_ATTEMPT_DIAGNOSIS.md. SAME existing full H3 weights, no Turbo, intended20actual updates; NEW explicit mask and source anchors/paste-back, accepted-i7 appearance only. This is METHOD CHANGE, NOT original CVM baseline success; known motion is supplied, no camera-memory/depth/translation evidence. Preserve raw generated beforepasteback. No round9.\n';plan.write_text(txt.replace(marker,marker+entry,1),encoding='utf-8')
print(json.dumps({'prepared':str(NEW),'mask_last':fractions[-1],'inference_started':False},indent=2))

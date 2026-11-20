"""Archive rejected round9 original CVM; prepare a distinct masked candidate offline."""
from pathlib import Path
import json,hashlib,shutil,datetime,sys,math,subprocess
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;W=B.parent;R=B/'round9';OLD=R/'05_cvm/attempt_001';NEW=R/'05_cvm/attempt_002_masked';D=R/'06_control_diagnosis'
assert not NEW.exists()
sys.path.insert(0,str(W/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'analysis/validation.json');timing=read(OLD/'timing.json');motion=read(OLD/'analysis/local_motion_diagnostic.json')
assert v['technical_status']=='passed' and timing['exit_code']==0 and motion['all_pairs_valid']
assert sha(OLD/'outputs/round9_seed42.mp4')==v['output_sha256']
review={'accepted_for_modeling':False,'inspected_frames':[0,31,62,92,116,123],
 'completion':'Gray is filled; endpoint unknown near-gray fraction 0.0000550583. Color alone does not establish structural accuracy.',
 'motion':'Endpoint direct18inliers is invalid. Separate output-to-output chain has every pair valid, accumulated yaw diagnostic -41.320249deg vs -30 nominal. F62 direct86inliers gives -20.9935 vs nominal -15.2064; F92 direct30inliers -36.4023 vs -26.2772.',
 'structure':'New broad vertical return/recess left of board and furniture reconfiguration are not established by accepted geometry; combined with overrotation they cannot be propagated.',
 'decision':'Reject for fixed30-degree continuation. Preserve all outputs; no selected i9 and no round10.',
 'limits':'All image estimates assume intrinsics and rigidity. Invalid direct endpoint angle is retained only as raw diagnostic; no calibrated depth or translation evidence.'}
save(OLD/'analysis/visual_review.json',review);v.update(quality_status='rejected_for_next_round',review=review);save(OLD/'analysis/validation.json',v)
save(OLD/'result_status.json',{'status':'inference_completed_quality_rejected','accepted_for_next_round':False,'sha256':v['output_sha256'],'timing':timing,'validation':'analysis/validation.json'})
dep=read(OLD/'deployment.json');dep.update(status='completed',finished_at=timing['finished_at'],exit_code=0);save(OLD/'deployment.json',dep)
report=f'''# Round9 original CVM rejected; offline masked alternative

Original raw Blender RGB attempt001 completed {timing['inference_started_at']} to {timing['finished_at']}, {timing['inference_process_elapsed_seconds']:.2f}s. Exit0,124 frames decoded, effective prompt and remote/local SHA256 match: {v['output_sha256']}.

Completion succeeds visually but motion is inconsistent with fixed -240 to -270 boundary. Direct endpoint18inliers is invalid; separate local output-to-output chain passes all pairs and accumulates approximately -41.32025deg. Intermediate reliable direct matches also overrotate. The new vertical wall return/recess remains unverified. No failed frame is propagated, and no round10 exists.

Round8 already tested raw RGB keep5/10 and prompt adaptations, then appearance-only controls, before a separately identified mask branch completed missing areas. This does not prove the cause of round9 failure or that a keep10 control here would fail, but provides a concrete reason to prefer the established explicit-mask fallback over repeating that entire sweep. The next candidate changes representation, conditioning algorithm and prompt; it is NOT a paired ablation or original CVM success.

The new masked candidate uses unchanged camera matrices, accepted i8 appearance reprojected at the fixed center, and original i0 only where its frustum returns. Original i0 is resized to960x576 (including its16px vertical excess on each side), preserving the original support convention. Latest accepted sources take priority. Per-pixel source labels and i8's inherited composite provenance are saved. No rejected attempt is used. Known pixels and motion are supplied and pasted back; returning i0 is explicitly supplied memory, not model recall. Only repaint-region content is new output. Full weights, seed42,20actual updates,noTurbo; preserve raw generated beforepasteback and crop960x544.

No inference is launched by this script. Inspect appearance proxy, masks, seams and source support before staging. If the fallback also creates severe structural contradictions, diagnose instead of accepting. Metric depth and translated-view correctness remain untested.
'''
(OLD/'RESULT.md').write_text(report,encoding='utf-8');(R/'RESULT.md').write_text(report,encoding='utf-8')
shutil.copy2(B/'audit_cvm_attempt.py',OLD/'audit_cvm_attempt_snapshot.py');shutil.copy2(Path(__file__),OLD/'failure_archive_snapshot.py')
save(OLD/'ARCHIVE_CHECKSUMS.json',{p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
assert all(sha(OLD/p)==h for p,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items())
D.mkdir(exist_ok=True);(D/'DIAGNOSIS.md').write_text(report,encoding='utf-8')
for folder in ['appearance_proxy_frames','appearance_source_labels','inherited_pixel_origin']:(D/folder).mkdir()
for folder in ['inputs','references','upstream','preparation_frames/source','preparation_frames/mask']:(NEW/folder).mkdir(parents=True)
refs=read(R/'00_input/accepted_references.json');rows=read(R/'02_blender/inputs/camera_trajectory.json')['frames'];fx=rows[0]['K'][0][0]
yy,xx=np.mgrid[:544,:960];rays=np.stack([(xx+.5-480)/fx,-(yy+.5-272)/fx,-np.ones_like(xx)],-1)
assets=[]
for index,ref in enumerate(refs,1):
 im=Image.open(R/ref['path']).convert('RGB');extra=ref['vertical_extra_pixels_at_render_scale']
 if im.size==(640,384):im=im.resize((960,576))
 assert im.size==(960,544+2*extra)
 a=-math.radians(ref['yaw']);c,s=math.cos(a),math.sin(a);rot=np.array([[c,0,s],[s,0,-c],[0,1,0]])
 assets.append((index,ref,rot,np.array(im),extra))
inherited=np.array(Image.open(R/'00_input/i8_repaint_mask.png'))>127
coverage=[];masks=[]
for row in rows:
 f=row['frame'];world=rays@np.array(row['camera_to_world_blender'])[:3,:3].T
 result=np.full((544,960,3),128,np.uint8);labels=np.zeros((544,960),np.uint8);origins=np.zeros_like(labels)
 for index,ref,rot,im,extra in reversed(assets):
  local=world@rot;forward=-local[...,2];den=np.maximum(forward,1e-9);px=fx*local[...,0]/den+480;py=-fx*local[...,1]/den+272
  hit=(forward>0)&(px>=0)&(px<960)&(py>=-extra)&(py<544+extra)&(labels==0)
  mapped=cv2.remap(im,(px-.5).astype(np.float32),(py+extra-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  result[hit]=mapped[hit];labels[hit]=index
  if ref['id']=='i0':origins[hit]=1
  elif ref['id']=='i8':
   prior=cv2.remap(inherited.astype(np.uint8),(px-.5).astype(np.float32),(py-.5).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)>0
   origins[hit&~prior]=2;origins[hit&prior]=3
  else:origins[hit]=4
 original=np.array(Image.open(R/f'02_blender/renders/coverage_labels/{f:04d}.png'))
 assert np.array_equal(labels==0,original==0),f
 unknown=labels==0;mask=cv2.dilate(unknown.astype(np.uint8),np.ones((17,17),np.uint8))>0
 Image.fromarray(result).save(D/f'appearance_proxy_frames/{f:04d}.png');Image.fromarray(labels).save(D/f'appearance_source_labels/{f:04d}.png');Image.fromarray(origins).save(D/f'inherited_pixel_origin/{f:04d}.png')
 padded=np.pad(result,((32,32),(32,32),(0,0)),mode='edge');mp=np.pad(mask,((32,32),(32,32)),mode='edge')
 Image.fromarray(padded).save(NEW/f'preparation_frames/source/{f:04d}.png');Image.fromarray(mp.astype(np.uint8)*255).save(NEW/f'preparation_frames/mask/{f:04d}.png');masks.append(mp)
 coverage.append({'frame':f,'yaw':row['yaw_degrees'],'unknown_fraction':float(unknown.mean()),'repaint_fraction_core':float(mask.mean()),'source_pixels':{ref['id']:int((labels==i).sum()) for i,ref,_,_,_ in assets},'inherited_origin_fraction':{str(i):float((origins==i).mean()) for i in range(5)}})
assert np.array_equal(np.array(Image.open(D/'appearance_proxy_frames/0000.png')),np.array(Image.open(R/'00_input/i8.png')))
np.savez_compressed(NEW/'inputs/masks.npz',repaint=np.stack(masks))
ff=W/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
for name,folder,out in [('appearance',D/'appearance_proxy_frames',D/'appearance_proxy.mp4'),('source',NEW/'preparation_frames/source',NEW/'inputs/source_video.mp4'),('mask',NEW/'preparation_frames/mask',NEW/'inputs/repaint_mask.mp4')]:
 cmd=[str(ff),'-hide_banner','-n','-framerate','24','-i',str(folder/'%04d.png')]
 if name=='source':cmd+=['-f','lavfi','-i','anullsrc=r=32000:cl=stereo']
 cmd+=['-frames:v','124','-t',str(124/24),'-c:v','libx264','-crf','0' if name=='mask' else '17','-preset','slow','-pix_fmt','yuv420p']
 cmd+=['-c:a','aac'] if name=='source' else ['-an'];cmd+=['-movflags','+faststart',str(out)]
 p=subprocess.run(cmd,capture_output=True);(NEW/f'encode_{name}.log').write_bytes(p.stderr);p.check_returncode()
 cap=cv2.VideoCapture(str(out));count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==((544,960,3) if name=='appearance' else (608,1024,3))
  if name=='mask':assert np.array_equal(im.mean(-1)>127.5,masks[count])
  count+=1
 assert count==124;cap.release()
shutil.copy2(D/'appearance_proxy.mp4',NEW/'inputs/partial_proxy.mp4')
for name in ['reference.png','camera_trajectory.json']:shutil.copy2(OLD/'inputs'/name,NEW/'inputs'/name)
shutil.copy2(R/'00_input/i8.png',NEW/'references/i8.png')
for name in ['accepted_references.json','i8_repaint_mask.png','i8_original_unknown_mask.png']:shutil.copy2(R/'00_input'/name,NEW/'inputs'/name)
shutil.copy2(R/'provenance/i8_composite_sources.json',NEW/'inputs/i8_composite_sources.json')
template=B/'round8/05_cvm/attempt_006_masked'
for p in (template/'upstream').iterdir():
 if p.is_file():shutil.copy2(p,NEW/'upstream'/p.name)
prompt=(template/'prompt.txt').read_text(encoding='utf-8').replace('Do not add a new clock, window, room corner or duplicate board.','Do not add a new clock, window, pillar, wall recess, doorway, room corner or duplicate board. Join the known side-wall surface to the returning front-wall content with stationary room surfaces.')
(NEW/'prompt.txt').write_text(prompt,encoding='utf-8');(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8')
save(D/'representation_diagnosis.json',{'status':'offline_candidate_not_submitted','coverage':coverage,'reference_provenance':refs,'mask_disagreement_vs_raw_proxy':0,'first_frame_MAE_to_i8':0,'original_i0_return_is_explicitly_supplied_memory':True,'inherited_origin_labels':{'0':'unknown','1':'original_i0','2':'i8_preserved_region_from_i7','3':'i8_masked_generated_region','4':'other_accepted_prior'},'video_sha256':sha(D/'appearance_proxy.mp4'),'limits':'Fixed-center ray projection with nominal poses. Composite source inheritance retained. Not depth recovery or independent camera memory.'})
save(NEW/'preparation_validation.json',{'status':'passed_offline','source_frames':124,'mask_frames':124,'fps':24,'processing_size':[1024,608],'final_crop_size':[960,544],'mask_decode_exact':True,'dilation_pixels':8,'guard_pixels':32,'all_unknown_pixels_covered':True,'coverage':coverage,'method_change':True,'actual_inference_started':False})
save(NEW/'method_provenance.json',{'protocol':'explicit_masked_completion_branch','original_cvm_baseline_passed':False,'source_reference_sha256':sha(NEW/'references/i8.png'),'source_video_sha256':sha(D/'appearance_proxy.mp4'),'source_policy':'Latest accepted i8 then older fallback; original i0 returns explicitly. No rejected result used.','original_i0_supplied_memory':True,'known_pixel_motion_is_supplied':True,'raw_generated_before_paste_back_required':True,'turbo':False,'intended_actual_updates':20,'seed':42})
sheet=Image.new('RGB',(1440,912),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,f in enumerate([0,62,123]):
 for k,p in enumerate([R/f'02_blender/renders/partial/{f:04d}.png',D/f'appearance_proxy_frames/{f:04d}.png',NEW/f'preparation_frames/mask/{f:04d}.png']):
  im=Image.open(p).convert('RGB');im=im.crop((32,32,992,576)) if k==2 else im;sheet.paste(im.resize((480,272)),(480*j,304*k+32));draw.text((480*j+5,304*k+7),f'{["Raw RGB","Accepted appearance","Explicit repaint mask"][k]} F{f}',fill='black')
sheet.save(D/'representation_comparison.jpg');shutil.copy2(Path(__file__),NEW/'prepare_inputs_snapshot.py')
s=read(B/'CONTINUOUS_LEFT_STATE.json');now=datetime.datetime.now().astimezone().isoformat();s.update(status='round9_masked_branch_offline_preparation',round9='original_cvm_rejected_masked_candidate',current_run=str(NEW.resolve()),current_run_relative=NEW.relative_to(B).as_posix(),current_deployment=None,last_completed_attempt={'path':OLD.relative_to(B).as_posix(),'quality':'rejected','output_sha256':v['output_sha256']},updated_at=now,last_verified_phase='Round9 original CVM rejected for overrotation. Offline masked candidate prepared; visual review and runner staging required. No inference running. No round10.');save(B/'CONTINUOUS_LEFT_STATE.json',s)
save(R/'EXPERIMENT_STATUS.json',{'status':s['status'],'rejected_attempts':['05_cvm/attempt_001'],'accepted_attempt':None,'original_cvm_baseline_passed':False})
plan=B/'CONTINUOUS_LEFT_PLAN.md';plan.write_text(f'LATEST {now}: Round9 original CVM attempt001 REJECTED: chain approx-41.32025deg(allpairsvalid), directendpoint18invalid, no frame propagated.1454s approx; exact1444.242s in timing. Offline attempt002_masked prepared; not staged/launched. Need inspect round9/06_control_diagnosis/representation_comparison.jpg, then freeze adapted round8 masked runner and preflight. Source i8 composite + explicit returning i0,124frusta exactmatch; original i0 supplied memory, not recall. No round10.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'status':'offline_prepared','endpoint':coverage[-1],'source_video_sha256':sha(D/'appearance_proxy.mp4')},indent=2))

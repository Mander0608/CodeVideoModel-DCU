"""One-shot offline round10 explicit-mask candidate; original attempt already frozen."""
from pathlib import Path
import json,hashlib,shutil,datetime,sys,math,subprocess
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;W=B.parent;R=B/'round10';OLD=R/'05_cvm/attempt_001';NEW=R/'05_cvm/attempt_002_masked';D=R/'06_control_diagnosis'
assert not NEW.exists()
sys.path.insert(0,str(W/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OLD/'result_status.json')['accepted_for_next_round'] is False
assert all(sha(OLD/p)==h for p,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items())
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
prior_maps={r['id']:np.array(Image.open(R/f"00_input/{r['id']}_pixel_provenance_labels.png")) for r in refs if r['id'] in ['i9','i9_coverage']}
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
  elif ref['id'] in prior_maps:
   pm=cv2.remap(prior_maps[ref['id']],(px-.5).astype(np.float32),(py-.5).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
   origins[hit]=pm[hit]
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
 coverage.append({'frame':f,'yaw':row['yaw_degrees'],'unknown_fraction':float(unknown.mean()),'repaint_fraction_core':float(mask.mean()),'source_pixels':{ref['id']:int((labels==i).sum()) for i,ref,_,_,_ in assets},'inherited_origin_fraction':{str(i):float((origins==i).mean()) for i in range(7)}})
assert np.array_equal(np.array(Image.open(D/'appearance_proxy_frames/0000.png')),np.array(Image.open(R/'00_input/i9.png')))
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
shutil.copy2(R/'00_input/i9.png',NEW/'references/i9.png')
# Preserve full accepted reference assets and every per-pixel/manual provenance file.
shutil.copy2(R/'00_input/accepted_references.json',NEW/'inputs/accepted_references.json')
for ref in refs:
 paths=[ref['path']]
 if ref.get('pixel_provenance'):
  paths.append(ref['pixel_provenance']);meta=read(R/ref['pixel_provenance'])
  paths += [meta[k] for k in ['repaint_mask','original_unknown_mask'] if meta.get(k)]
  paths += meta.get('extra_provenance_files',[])
 for rel in set(paths):
  dst=NEW/'source_history'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/rel,dst)
template=B/'round8/05_cvm/attempt_006_masked'
for p in (template/'upstream').iterdir():
 if p.is_file():shutil.copy2(p,NEW/'upstream'/p.name)
prompt=(B/'round9/05_cvm/attempt_002_masked/prompt.txt').read_text(encoding='utf-8')
prompt=prompt.replace('Join the known side-wall surface to the returning front-wall content with stationary room surfaces.', 'Preserve the returning original green front chalkboard and its tree illustration supplied in the source video. The navy side-wall board is a separate existing object. Fill only the unknown ceiling and floor wedges with continuous surfaces. Do not replace the returning green board with blank wall or a clock.')
(NEW/'prompt.txt').write_text(prompt,encoding='utf-8');(NEW/'inputs/prompt.txt').write_text(prompt,encoding='utf-8')
save(D/'representation_diagnosis.json',{'status':'offline_candidate_not_submitted','coverage':coverage,'reference_provenance':refs,'mask_disagreement_vs_raw_proxy':0,'first_frame_MAE_to_i9':0,'original_i0_return_is_explicitly_supplied_memory':True,'inherited_origin_labels':{'0':'unknown','1':'original_i0','2':'i8_preserved_region_from_i7','3':'i8_masked_generated_region','4':'other_accepted_prior','5':'round9_masked_generator','6':'round9_manual_correction_support'},'video_sha256':sha(D/'appearance_proxy.mp4'),'limits':'Fixed-center ray projection with nominal poses. Composite source inheritance retained. Not depth recovery or independent camera memory.'})
save(NEW/'preparation_validation.json',{'status':'passed_offline','source_frames':124,'mask_frames':124,'fps':24,'processing_size':[1024,608],'final_crop_size':[960,544],'mask_decode_exact':True,'dilation_pixels':8,'guard_pixels':32,'all_unknown_pixels_covered':True,'coverage':coverage,'method_change':True,'actual_inference_started':False})
save(NEW/'method_provenance.json',{'protocol':'explicit_masked_completion_branch','original_cvm_baseline_passed':False,'source_reference_sha256':sha(NEW/'references/i9.png'),'source_video_sha256':sha(D/'appearance_proxy.mp4'),'source_policy':'Latest accepted corrected i9 then i9_coverage and older fallback; original i0 returns explicitly. No rejected result used.','original_i0_supplied_memory':True,'known_pixel_motion_is_supplied':True,'raw_generated_before_paste_back_required':True,'turbo':False,'intended_actual_updates':20,'seed':42})
sheet=Image.new('RGB',(1440,912),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,f in enumerate([0,62,123]):
 for k,p in enumerate([R/f'02_blender/renders/partial/{f:04d}.png',D/f'appearance_proxy_frames/{f:04d}.png',NEW/f'preparation_frames/mask/{f:04d}.png']):
  im=Image.open(p).convert('RGB');im=im.crop((32,32,992,576)) if k==2 else im;sheet.paste(im.resize((480,272)),(480*j,304*k+32));draw.text((480*j+5,304*k+7),f'{["Raw RGB","Accepted appearance","Explicit repaint mask"][k]} F{f}',fill='black')
sheet.save(D/'representation_comparison.jpg');shutil.copy2(Path(__file__),NEW/'prepare_inputs_snapshot.py')

# Verify that the returning board ROI is known and is protected from repaint.
labels=np.array(Image.open(D/'appearance_source_labels/0123.png'))
mask=masks[-1][32:576,32:992]
roi=np.zeros((544,960),bool);roi[140:350,10:280]=True
assert np.all(labels[roi]>0) and not mask[roi].any()
save(D/'return_board_protection.json',{'endpoint_board_interior_roi':[10,140,280,350],'known_fraction':float((labels[roi]>0).mean()),'repaint_fraction':float(mask[roi].mean()),'note':'Interior ROI only; inspect entire board visually. Protection is supplied evidence, not model recall.'})
s=read(B/'CONTINUOUS_LEFT_STATE.json');now=datetime.datetime.now().astimezone().isoformat();s.update(status='round10_masked_offline_prepared',round10='original_rejected_masked_candidate',current_run=str(NEW.resolve()),current_run_relative=NEW.relative_to(B).as_posix(),current_deployment=None,updated_at=now,last_verified_phase='Explicit-mask candidate offline only;124source/maskframes validated. Single model image corrected i9; returning original i0 supplied via source. Full inherited/manual provenance copied. Must visually inspect return-board and seams before staging. No round11.')
save(B/'CONTINUOUS_LEFT_STATE.json',s)
p=B/'CONTINUOUS_LEFT_PLAN.md';p.write_text(f'LATEST {now}: Round10 masked002 OFFLINE PREPARED, not submitted. Original001 remains rejected/frozen. Inspect representation_comparison.jpg and return_board_protection.json; then stage/check resources/deploy once. New source-history/mask protocol, one corrected i9 image; raw-before-pasteback required. No depth/unaidedmemory claim; no round11.\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'status':s['status'],'endpoint':coverage[-1]},indent=2))

"""Preserve offline v001; replace overlapping generated history with authoritative i0."""
from pathlib import Path
import json,hashlib,shutil,sys,subprocess,datetime
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;W=B.parent;R=B/'round10';A=R/'05_cvm/attempt_002_masked';D=R/'06_control_diagnosis';V=A/'offline_v001_latest_priority'
assert not V.exists() and not (A/'deployment.json').exists()
sys.path.insert(0,str(W/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
V.mkdir(); paths=[D/'appearance_proxy_frames',D/'appearance_source_labels',D/'inherited_pixel_origin',D/'appearance_proxy.mp4',D/'representation_diagnosis.json',D/'representation_comparison.jpg',A/'preparation_frames/source',A/'inputs/source_video.mp4',A/'inputs/partial_proxy.mp4',A/'preparation_validation.json',A/'method_provenance.json']
for src in paths:
 dst=V/src.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True)
 # Each explicitly enumerated path is inside round10; no computed recursive deletion.
 assert src.resolve().is_relative_to(R.resolve()) and dst.resolve().is_relative_to(R.resolve())
 shutil.move(str(src),str(dst))
for folder in ['appearance_proxy_frames','appearance_source_labels','inherited_pixel_origin']:(D/folder).mkdir()
(A/'preparation_frames/source').mkdir()
refs=read(R/'00_input/accepted_references.json');ref=next(r for r in refs if r['id']=='i0');idx=refs.index(ref)+1
rows=read(A/'inputs/camera_trajectory.json')['frames'];fx=rows[0]['K'][0][0]
im=Image.open(R/ref['path']).convert('RGB').resize((960,576));im=np.array(im);extra=ref['vertical_extra_pixels_at_render_scale'];assert extra==16
yy,xx=np.mgrid[:544,:960];rays=np.stack([(xx+.5-480)/fx,-(yy+.5-272)/fx,-np.ones_like(xx)],-1);rot=np.array([[1,0,0],[0,0,-1],[0,1,0]])
oldD=V/'06_control_diagnosis';diagnosis=read(oldD/'representation_diagnosis.json');coverage=[];change=[]
for row in rows:
 f=row['frame'];old=np.array(Image.open(oldD/f'appearance_proxy_frames/{f:04d}.png'));out=old.copy();labels=np.array(Image.open(oldD/f'appearance_source_labels/{f:04d}.png'));origins=np.array(Image.open(oldD/f'inherited_pixel_origin/{f:04d}.png'))
 local=(rays@np.array(row['camera_to_world_blender'])[:3,:3].T)@rot;forward=-local[...,2];den=np.maximum(forward,1e-9);px=fx*local[...,0]/den+480;py=-fx*local[...,1]/den+272
 hit=(forward>0)&(px>=0)&(px<960)&(py>=-extra)&(py<544+extra)
 assert not np.any(hit&(labels==0))
 mapped=cv2.remap(im,(px-.5).astype(np.float32),(py+extra-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
 out[hit]=mapped[hit];labels[hit]=idx;origins[hit]=1
 for folder,img in [('appearance_proxy_frames',out),('appearance_source_labels',labels),('inherited_pixel_origin',origins)]:Image.fromarray(img).save(D/f'{folder}/{f:04d}.png')
 Image.fromarray(np.pad(out,((32,32),(32,32),(0,0)),mode='edge')).save(A/f'preparation_frames/source/{f:04d}.png')
 c=diagnosis['coverage'][f].copy();c['source_pixels']={r['id']:int((labels==i).sum()) for i,r in enumerate(refs,1)};c['inherited_origin_fraction']={str(i):float((origins==i).mean()) for i in range(7)};coverage.append(c)
 change.append({'frame':f,'original_i0_fraction':float(hit.mean()),'changed_pixel_fraction':float(np.any(out!=old,axis=-1).mean()),'MAE_vs_v001':float(np.abs(out.astype(float)-old).mean())})
ff=W/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
for name,folder,out in [('appearance',D/'appearance_proxy_frames',D/'appearance_proxy.mp4'),('source',A/'preparation_frames/source',A/'inputs/source_video.mp4')]:
 cmd=[str(ff),'-hide_banner','-n','-framerate','24','-i',str(folder/'%04d.png')]
 if name=='source':cmd+=['-f','lavfi','-i','anullsrc=r=32000:cl=stereo']
 cmd+=['-frames:v','124','-t',str(124/24),'-c:v','libx264','-crf','17','-preset','slow','-pix_fmt','yuv420p'];cmd+=['-c:a','aac'] if name=='source' else ['-an'];cmd+=['-movflags','+faststart',str(out)]
 p=subprocess.run(cmd,capture_output=True);(A/f'encode_v002_{name}.log').write_bytes(p.stderr);p.check_returncode()
 cap=cv2.VideoCapture(str(out));n=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==((544,960,3) if name=='appearance' else (608,1024,3));n+=1
 cap.release();assert n==124
shutil.copy2(D/'appearance_proxy.mp4',A/'inputs/partial_proxy.mp4')
diagnosis.update(coverage=coverage,source_priority='Original i0 first within its valid frustum, otherwise latest accepted prior; original support/unknown/repaint masks unchanged.',first_frame_MAE_to_i9=change[0]['MAE_vs_v001'],video_sha256=sha(D/'appearance_proxy.mp4'),offline_revision='v002_original_i0_priority');save(D/'representation_diagnosis.json',diagnosis)
prep=read(V/'05_cvm/attempt_002_masked/preparation_validation.json');prep.update(coverage=coverage,source_priority_revision='v002',first_source_equals_reference=False,reason='Explicit original i0 override corrects returning historical seam; previous v001 F0 exactly matched i9.');save(A/'preparation_validation.json',prep)
meta=read(V/'05_cvm/attempt_002_masked/method_provenance.json');meta.update(source_video_sha256=sha(D/'appearance_proxy.mp4'),source_policy=diagnosis['source_priority'],source_revision='v002_original_i0_priority',before_correction_archive='offline_v001_latest_priority');save(A/'method_provenance.json',meta)
sheet=Image.new('RGB',(1440,912),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,f in enumerate([0,62,123]):
 for k,p in enumerate([oldD/f'appearance_proxy_frames/{f:04d}.png',D/f'appearance_proxy_frames/{f:04d}.png',A/f'preparation_frames/mask/{f:04d}.png']):
  im=Image.open(p).convert('RGB');im=im.crop((32,32,992,576)) if k==2 else im;sheet.paste(im.resize((480,272)),(j*480,k*304+32));draw.text((j*480+5,k*304+7),f'{["v001 latest priority","v002 original i0 priority","Unchanged repaint mask"][k]} F{f}',fill='black')
sheet.save(D/'source_priority_before_after.jpg')
save(D/'source_priority_correction.json',{'change':change,'original_i0_pixels_explicitly_supplied':True,'unknown_and_repaint_masks_changed':False,'reference_image_count':1,'independent_model_recall_claim':False,'reason':'v001 includes conflicting early-generated i1 at returning board/desk; original image overrides generated overlapping appearance. Manual correction to conditioning, not a generated result.'})
shutil.copy2(Path(__file__),A/'source_priority_repair_snapshot.py');save(V/'ARCHIVE_CHECKSUMS.json',{p.relative_to(V).as_posix():sha(p) for p in V.rglob('*') if p.is_file() and p.name!='ARCHIVE_CHECKSUMS.json'})
assert all(sha(V/p)==h for p,h in read(V/'ARCHIVE_CHECKSUMS.json').items())
print(json.dumps({'status':'v002_offline_ready_for_visual_review','first':change[0],'last':change[-1],'endpoint_sources':coverage[-1]['source_pixels']},indent=2))

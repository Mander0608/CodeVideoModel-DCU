"""Offline round12 originali0-first/accepted i11 source with inherited pixel-lineage; no inference."""
from pathlib import Path
import sys,json,hashlib,shutil,math,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parent;R=B/'round12';A=R/'05_cvm/attempt_001_masked';assert not A.exists()
sys.path.insert(0,str(B.parent/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f in ['inputs','references','upstream','preparation_frames/source','preparation_frames/mask','source_audit/frames','source_audit/source_labels','source_audit/inherited_origin']:(A/f).mkdir(parents=True,exist_ok=True)
refs=read(R/'00_input/accepted_references.json');rows=read(R/'02_blender/inputs/camera_trajectory.json')['frames'];fx=rows[0]['K'][0][0]
yy,xx=np.mgrid[:544,:960];rays=np.stack([(xx+.5-480)/fx,-(yy+.5-272)/fx,-np.ones_like(xx)],-1)
assets=[]
for index,ref in enumerate(refs,1):
 im=Image.open(R/ref['path']).convert('RGB');extra=ref['vertical_extra_pixels_at_render_scale']
 if im.size==(640,384):im=im.resize((960,576))
 assert im.size==(960,544+2*extra)
 a=-math.radians(ref['yaw']);c,s=math.cos(a),math.sin(a);rot=np.array([[c,0,s],[s,0,-c],[0,1,0]])
 assets.append((index,ref,rot,np.array(im),extra))
ordered=[x for x in assets if x[1]['id']=='i0']+[x for x in reversed(assets) if x[1]['id']!='i0']
inherited=np.array(Image.open(R/'00_input/i8_repaint_mask.png'))>127
origin_maps={}
for ref in refs:
 if ref.get('pixel_provenance'):
  meta=read(R/ref['pixel_provenance'])
  if meta.get('final_pixel_labels'):
   origin_maps[ref['id']]=np.array(Image.open(R/meta['final_pixel_labels']));assert set(np.unique(origin_maps[ref['id']])).issubset(set(range(7)))
coverage=[];masks=[]
for row in rows:
 f=row['frame'];world=rays@np.array(row['camera_to_world_blender'])[:3,:3].T
 result=np.full((544,960,3),128,np.uint8);labels=np.zeros((544,960),np.uint8);origin=np.zeros_like(labels)
 for index,ref,rot,im,extra in ordered:
  local=world@rot;forward=-local[...,2];den=np.maximum(forward,1e-9);px=fx*local[...,0]/den+480;py=-fx*local[...,1]/den+272
  hit=(forward>0)&(px>=0)&(px<960)&(py>=-extra)&(py<544+extra)&(labels==0)
  mapped=cv2.remap(im,(px-.5).astype(np.float32),(py+extra-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  result[hit]=mapped[hit];labels[hit]=index
  if ref['id']=='i0':origin[hit]=1
  elif ref['id'] in origin_maps:
   prior=cv2.remap(origin_maps[ref['id']],(px-.5).astype(np.float32),(py-.5).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
   origin[hit]=prior[hit]
   assert not np.any(origin[hit]==0)
  elif ref['id']=='i8':
   prior=cv2.remap(inherited.astype(np.uint8),(px-.5).astype(np.float32),(py-.5).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)>0
   origin[hit&~prior]=2;origin[hit&prior]=3
  else:origin[hit]=4
 unknown=labels==0;original=np.array(Image.open(R/f'02_blender/renders/coverage_labels/{f:04d}.png'));assert np.array_equal(unknown,original==0)
 mask=cv2.dilate(unknown.astype(np.uint8),np.ones((17,17),np.uint8))>0
 for folder,data in [('frames',result),('source_labels',labels),('inherited_origin',origin)]:Image.fromarray(data).save(A/f'source_audit/{folder}/{f:04d}.png')
 padded=np.pad(result,((32,32),(32,32),(0,0)),mode='edge');mp=np.pad(mask,((32,32),(32,32)),mode='edge');masks.append(mp)
 Image.fromarray(padded).save(A/f'preparation_frames/source/{f:04d}.png');Image.fromarray(mp.astype(np.uint8)*255).save(A/f'preparation_frames/mask/{f:04d}.png')
 coverage.append({'frame':f,'nominal_yaw':row['yaw_degrees'],'unknown_fraction':float(unknown.mean()),'repaint_fraction_core':float(mask.mean()),'source_pixels':{ref['id']:int((labels==i).sum()) for i,ref,_,_,_ in assets}})
np.savez_compressed(A/'inputs/masks.npz',repaint=np.stack(masks))
ff=B.parent/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
for name,folder,target in [('appearance','source_audit/frames','inputs/partial_proxy.mp4'),('source','preparation_frames/source','inputs/source_video.mp4'),('mask','preparation_frames/mask','inputs/repaint_mask.mp4')]:
 cmd=[str(ff),'-v','error','-n','-framerate','24','-i',str(A/folder/'%04d.png')]
 if name=='source':cmd+=['-f','lavfi','-i','anullsrc=r=32000:cl=stereo']
 cmd+=['-frames:v','124','-t',str(124/24),'-c:v','libx264','-crf','0' if name=='mask' else '17','-preset','slow','-pix_fmt','yuv420p']
 cmd+=['-c:a','aac'] if name=='source' else ['-an'];cmd+=[str(A/target)]
 p=subprocess.run(cmd,capture_output=True);(A/f'encode_{name}.log').write_bytes(p.stderr);p.check_returncode()
 cap=cv2.VideoCapture(str(A/target));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==((544,960,3) if name=='appearance' else (608,1024,3))
  if name=='mask':assert np.array_equal(im.mean(-1)>127.5,masks[n])
  n+=1
 cap.release();assert n==124
for name in ['reference.png','camera_trajectory.json','camera_trajectory.csv']:shutil.copy2(R/'02_blender/inputs'/name,A/'inputs'/name)
shutil.copy2(R/'00_input/i11.png',A/'references/i11.png')
assert sha(A/'references/i11.png')==next(ref['sha256'] for ref in refs if ref['id']=='i11')
shutil.copy2(R/'00_input/accepted_references.json',A/'inputs/accepted_references.json')
for ref in refs:
 paths=[ref['path']]
 if ref.get('pixel_provenance'):
  paths.append(ref['pixel_provenance']);meta=read(R/ref['pixel_provenance']);paths += [meta[k] for k in ['repaint_mask','original_unknown_mask'] if meta.get(k)];paths+=meta.get('extra_provenance_files',[])
 for rel in set(paths):
  dst=A/'source_history'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/rel,dst)
template=B/'round8/05_cvm/attempt_006_masked'
for p in (template/'upstream').iterdir():
 if p.is_file():shutil.copy2(p,A/'upstream'/p.name)
prompt=(R/'provenance/cvm_prompt.txt').read_text(encoding='utf-8')
(A/'prompt.txt').write_text(prompt,encoding='utf-8');(A/'inputs/prompt.txt').write_text(prompt,encoding='utf-8')
save(A/'preparation_validation.json',{'status':'passed_offline','source_frames':124,'mask_frames':124,'fps':24,'processing_size':[1024,608],'final_crop_size':[960,544],'mask_decode_exact':True,'dilation_pixels':8,'guard_pixels':32,'all_unknown_pixels_covered':True,'coverage':coverage,'method_change':True,'actual_inference_started':False})
save(A/'method_provenance.json',{'protocol':'explicit_masked_completion_branch','original_cvm_baseline_passed':False,'source_reference_sha256':sha(A/'references/i11.png'),'source_video_sha256':sha(A/'inputs/partial_proxy.mp4'),'source_policy':'Original i0 first in valid angular support, otherwise latest accepted composite i11 then raw i9 and older accepted history; i11 per-pixel original/generated lineage is reprojected from its saved final origin map. No historical manually corrected i9 and no rejected output used.','source_labels':{str(i):ref['id'] for i,ref,_,_,_ in assets},'inherited_origin_labels':{'0':'unknown','1':'original_i0','2':'i8_preserved_earlier_generated','3':'i8_masked_generated','4':'other_accepted_generated_prior','5':'round10_masked_generated_gaps_and_seams','6':'round11_masked_generated_gaps_and_seams'},'original_i0_supplied_memory':True,'known_pixel_motion_is_supplied':True,'raw_generated_before_paste_back_required':True,'turbo':False,'intended_actual_updates':20,'seed':42,'limits':'Nominal rotation-only projection, approximate source poses and scene layout; no depth or translation evidence. Source priority is an explicit conditioning choice.'})
sheet=Image.new('RGB',(1600,3*205),'#eeeeee');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for j,f in enumerate([0,31,62,92,123]):
 for k,path in enumerate([R/f'02_blender/renders/partial/{f:04d}.png',A/f'source_audit/frames/{f:04d}.png',A/f'preparation_frames/mask/{f:04d}.png']):
  im=Image.open(path).convert('RGB');im=im.crop((32,32,992,576)) if k==2 else im;sheet.paste(im.resize((320,181)),(320*j,205*k+24));draw.text((320*j+4,205*k+3),f'{["Blender input","i0-first source","Repaint mask"][k]} F{f}',font=font,fill='black')
sheet.save(A/'source_audit/representation_comparison.jpg');shutil.copy2(Path(__file__),A/'prepare_inputs_snapshot.py')
print(json.dumps({'path':str(A),'endpoint':coverage[-1],'status':'offline_pending_visual_review'},ensure_ascii=False))

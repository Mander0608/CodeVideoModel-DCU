"""Preserve initial unknown-only mask; add explicit source-transition repaint bands."""
from pathlib import Path
import sys,json,shutil,hashlib,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parent;A=B/'round12/05_cvm/attempt_001_masked';sys.path.insert(0,str(B.parent/'Blender-Classroom-Reconstruction/runtime'));import cv2
V=A/'offline_v001_unknown_only_mask';V.mkdir(exist_ok=False)
for rel in ['inputs/masks.npz','inputs/repaint_mask.mp4','preparation_validation.json','method_provenance.json','source_audit/representation_comparison.jpg']:
 target=V/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/rel,target)
shutil.copytree(A/'preparation_frames/mask',V/'preparation_frames/mask')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
old=np.load(A/'inputs/masks.npz')['repaint'];new=[];records=[]
for f in range(124):
 labels=np.array(Image.open(A/f'source_audit/source_labels/{f:04d}.png'));edge=np.zeros_like(labels,np.uint8)
 dx=(labels[:,:-1]!=labels[:,1:])&(labels[:,:-1]>0)&(labels[:,1:]>0);edge[:,:-1]|=dx;edge[:,1:]|=dx
 dy=(labels[:-1]!=labels[1:])&(labels[:-1]>0)&(labels[1:]>0);edge[:-1]|=dy;edge[1:]|=dy
 band=cv2.dilate(edge,np.ones((49,49),np.uint8))>0
 mask=old[f,32:-32,32:-32]|band;pad=np.pad(mask,((32,32),(32,32)),mode='edge');new.append(pad)
 Image.fromarray(pad.astype(np.uint8)*255).save(A/f'preparation_frames/mask/{f:04d}.png')
 Image.fromarray(band.astype(np.uint8)*255).save(A/f'source_audit/seam_band_{f:04d}.png')
 records.append({'frame':f,'unknown_border_mask_fraction':float(old[f,32:-32,32:-32].mean()),'source_transition_band_fraction':float(band.mean()),'combined_repaint_fraction_core':float(mask.mean())})
np.savez_compressed(A/'inputs/masks.npz',repaint=np.stack(new))
ff=B.parent/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
q=subprocess.run([str(ff),'-v','error','-y','-framerate','24','-i',str(A/'preparation_frames/mask/%04d.png'),'-frames:v','124','-an','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(A/'inputs/repaint_mask.mp4')],capture_output=True);(A/'encode_seam_mask.log').write_bytes(q.stderr);q.check_returncode()
c=cv2.VideoCapture(str(A/'inputs/repaint_mask.mp4'));n=0
while True:
 ok,im=c.read()
 if not ok:break
 assert np.array_equal(im.mean(-1)>127.5,new[n]);n+=1
c.release();assert n==124
prep=json.loads((A/'preparation_validation.json').read_text());prep['source_transition_repaint_dilation_pixels']=24;prep['source_transition_repaint']=records
for row,m in zip(prep['coverage'],new):row['repaint_fraction_core']=float(m[32:-32,32:-32].mean())
save(A/'preparation_validation.json',prep)
meta=json.loads((A/'method_provenance.json').read_text());meta['source_seam_policy']='24px each side of known-source identity transitions added to repaint mask; unknown dilated8px. Both mask revisions retained. Known interiors preserved, transition pixels may be generated.';meta['prior_mask_revision']='offline_v001_unknown_only_mask';save(A/'method_provenance.json',meta)
prompt=(A/'prompt.txt').read_text().replace('Generate only the masked ceiling and floor gaps and their narrow transition borders.','Generate only the masked ceiling and floor gaps and the narrow source-transition seam bands. Blend the cream wall, baseboard, floor and any desk edges crossing these bands naturally, without changing the unmasked greenboard or tree interior.')
(A/'prompt.txt').write_text(prompt,encoding='utf-8');(A/'inputs/prompt.txt').write_text(prompt,encoding='utf-8')
sheet=Image.new('RGB',(1600,410),'#eeeeee');d=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for j,f in enumerate([0,31,62,92,123]):
 im=np.array(Image.open(A/f'source_audit/frames/{f:04d}.png'));mask=new[f][32:-32,32:-32];overlay=im.copy();overlay[mask]=(im[mask].astype(float)*.45+np.array([255,60,60])*.55).astype(np.uint8)
 for k,data in enumerate([im,overlay]):sheet.paste(Image.fromarray(data).resize((320,181)),(j*320,k*205+24));d.text((j*320+3,k*205+3),f'{["Supplied source","Red=repaint + seams"][k]} F{f}',font=font,fill='black')
sheet.save(A/'source_audit/seam_mask_review.jpg')
roi=np.zeros((544,960),bool);roi[140:300,300:700]=True
assert not new[123][32:-32,32:-32][roi].any()
save(A/'source_audit/return_board_protection.json',{'endpoint_interior_roi_xyxy':[300,140,700,300],'repaint_fraction':float(new[123][32:-32,32:-32][roi].mean()),'scope':'Interior ROI only; visual inspection also needed. Board is supplied i0, not model memory.'})
shutil.copy2(Path(__file__),A/'source_seam_mask_snapshot.py');save(V/'ARCHIVE_CHECKSUMS.json',{p.relative_to(V).as_posix():sha(p) for p in V.rglob('*') if p.is_file()})
print(json.dumps({'first':records[0],'last':records[-1],'mask_decode_exact':True}))

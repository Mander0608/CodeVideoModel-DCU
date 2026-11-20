"""Offline deterministic removal of unsupported panel; never alter original output."""
from pathlib import Path
import json,hashlib,datetime,sys,subprocess,shutil
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;R=B/'round9';A=R/'05_cvm/attempt_004_wallref';O=R/'07_structural_correction/v002'
assert not O.exists()
sys.path.insert(0,str(B.parent/'Blender-Classroom-Reconstruction/runtime'));import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(A/'analysis/visual_review.json')['accepted_for_modeling'] is False
for d in ['frames','correction_masks','diagnostics']:(O/d).mkdir(parents=True)
rows=read(A/'inputs/camera_trajectory.json')['frames'];K=np.array(rows[0]['K']);Ki=np.linalg.inv(K);D=np.diag([1,-1,-1])
cap=cv2.VideoCapture(str(A/'outputs/round9_seed42.mp4'));frames=[]
while True:
 ok,f=cap.read()
 if not ok:break
 frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
cap.release();assert len(frames)==124
terminal=frames[-1];yy,xx=np.mgrid[:544,:960]
# Patch coordinates were chosen after viewing full resolution F123 and source labels.
# Preserve the existing corner at x164; continuation uses only bare-wall boundaries.
alpha=np.minimum.reduce([np.clip((yy-35)/10,0,1),np.clip((380-yy)/10,0,1),np.clip((225-xx)/10,0,1),(xx>=164).astype(float)])
# Harmonic continuation from bare-wall top/bottom/right boundary; left Neumann
# preserves corner discontinuity instead of bleeding brighter front-wall pixels.
p=terminal[35:381,164:226].astype(np.float32)
u=np.repeat(p[:,-1:,:],p.shape[1],axis=1);u[0]=p[0];u[-1]=p[-1]
for iteration in range(5000):
 u[1:-1,1:-1]=(u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:])*.25
 u[1:-1,0]=u[1:-1,1]
donor=terminal.copy();donor[35:381,164:226]=np.rint(u).clip(0,255).astype(np.uint8)
corrected_terminal=np.rint(terminal*(1-alpha[...,None])+donor*alpha[...,None]).astype(np.uint8)
Image.fromarray(corrected_terminal).save(O/'diagnostics/terminal_patch_preview.png')
rt=np.array(rows[-1]['camera_to_world_blender'])[:3,:3]@D
stats=[];picked={}
for i,frame in enumerate(frames):
 rf=np.array(rows[i]['camera_to_world_blender'])[:3,:3]@D;h=K@rf.T@rt@Ki
 warped_alpha=cv2.warpPerspective(alpha.astype(np.float32),h,(960,544),flags=cv2.INTER_LINEAR)
 unknown=np.array(Image.open(R/f'02_blender/renders/coverage_labels/{i:04d}.png'))==0
 warped_alpha*=unknown
 patch=cv2.warpPerspective(donor,h,(960,544),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
 corrected=np.rint(frame*(1-warped_alpha[...,None])+patch*warped_alpha[...,None]).astype(np.uint8)
 modified=warped_alpha>0
 assert not np.any(modified&~unknown)
 assert np.array_equal(corrected[~modified],frame[~modified])
 Image.fromarray(corrected).save(O/f'frames/{i:04d}.png')
 Image.fromarray(np.rint(warped_alpha*255).astype(np.uint8)).save(O/f'correction_masks/{i:04d}.png')
 stats.append({'frame':i,'nominal_yaw':rows[i]['yaw_degrees'],'correction_fraction':float(modified.mean()),'known_pixels_changed_lossless':0})
 if i in [0,31,62,92,116,123]:picked[i]=corrected
assert stats[0]['correction_fraction']==0
ff=B.parent/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
cmd=[str(ff),'-hide_banner','-n','-framerate','24','-i',str(O/'frames/%04d.png'),'-i',str(A/'outputs/round9_seed42.mp4'),'-map','0:v:0','-map','1:a:0','-frames:v','124','-c:v','libx264','-crf','17','-preset','slow','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(O/'corrected_derivative.mp4')]
p=subprocess.run(cmd,capture_output=True);(O/'encode.log').write_bytes(p.stderr);p.check_returncode()
sheet=Image.new('RGB',(1440,1216),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,i in enumerate([0,31,62,92,116,123]):
 for k,(label,im) in enumerate([('Original rejected',frames[i]),('Corrected synthetic derivative',picked[i])]):
  x=(j%3)*480;y=(j//3)*608+k*304;sheet.paste(Image.fromarray(im).resize((480,272)),(x,y+32));draw.text((x+5,y+8),f'{label} F{i}',fill='black')
sheet.save(O/'diagnostics/before_after_contact.jpg')
shutil.copy2(Path(__file__),O/'correction_source_snapshot.py')
manifest={'status':'offline_candidate_requires_visual_review','created_at':datetime.datetime.now().astimezone().isoformat(),
 'parent_attempt':'05_cvm/attempt_004_wallref','parent_status':'rejected_original_remains_rejected','parent_composite_sha256':sha(A/'outputs/round9_seed42.mp4'),
 'parent_raw_sha256':sha(A/'outputs/raw_generated_960x544.mp4'),'derivative_sha256':sha(O/'corrected_derivative.mp4'),
 'correction':'Manually bounded extra-panel removal: harmonic bare-wall continuation from original top/bottom/right patch boundaries, zero-normal-gradient left corner boundary,5000 Jacobi iterations; blend top/bottom/right10px, preserve corner x164. Transport patch and alpha by supplied nominal pure-rotation homographies; intersect with original unknown support every frame.',
 'terminal_region_xyxy':[164,35,225,380],'fill_algorithm':'harmonic_continuation_5000_iterations',
 'lossless_frames':'frames','alpha_masks':'correction_masks','all_frame_stats':stats,
 'known_pixels_unchanged_in_lossless_frames':True,'encoded_video_warning':'H264 re-encoding can change decoded pixels outside alpha; lossless frame sequence is the exact correction record.',
 'interpretation':'Deterministic structural correction using boundary colors of adjacent generated wall, not a successful CVM output or newly observed geometry. Main board/corner/furniture retained; no new panel modeled. No depth/translation or independent camera-memory claim.',
 'acceptance':False,'source_geometry_limit':'Corner location remains the existing assumed rectangular-room prior.'}
save(O/'manifest.json',manifest)
save(O/'CHECKSUMS_PRE_REVIEW.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='CHECKSUMS_PRE_REVIEW.json'})
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round9_structural_correction_offline_review',updated_at=manifest['created_at'],last_verified_phase='Offline v002 harmonic correction prepared, not accepted. Inspect before_after_contact and terminal patch, check panel removal and seams. Parent attempt004 rejected and frozen; no i9/round10/inference.');s['correction_candidate']='round9/07_structural_correction/v002';save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps({'path':str(O),'terminal_correction_fraction':stats[-1]['correction_fraction'],'derivative_sha256':manifest['derivative_sha256']},indent=2))

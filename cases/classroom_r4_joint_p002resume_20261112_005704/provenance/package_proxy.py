"""Package Blender RGB + angular-coverage partial proxy. Never writes original rounds."""
from pathlib import Path
import numpy as np,json,math,hashlib,shutil,sys,subprocess
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;P=O.parent;B=P.parent.parent;W=B.parent
sys.path.insert(0,str(W/'Blender-Classroom-Reconstruction/runtime'))
import cv2
T=json.loads((O/'trajectory.json').read_text(encoding='utf-8'));D=json.loads((P/'fit.json').read_text(encoding='utf-8'))
preview='--preview' in sys.argv
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for sub in ['partial','coverage','references','review']:(O/sub).mkdir(exist_ok=True)
def nominal_rotation(yaw):
 a=-math.radians(yaw);c,s=math.cos(a),math.sin(a)
 return np.array([[c,0,s],[s,0,-c],[0,1,0]])
refs=[];old=json.loads((B/'round4/00_input/accepted_references.json').read_text(encoding='utf-8'))
shutil.copy2(B/'round4/00_input/accepted_references.json',O/'references/original_reference_metadata.json')
f100=480/math.tan(math.radians(50))
for r in old[:3]:
 src=B/'round4'/r['path'];dest=O/'references'/src.name;shutil.copy2(src,dest)
 assert sha(dest)==r['sha256']
 refs.append({'id':r['id'],'file':str(dest.relative_to(O)),'sha256':sha(dest),'calibration':'historical nominal camera; not refitted','R_blender_to_world':nominal_rotation(r['yaw']).tolist(),'K':[[f100,0,480],[0,f100,272],[0,0,1]],'vertical_extra':r['vertical_extra_pixels_at_render_scale'],'source_metadata':r})
for pose in D['poses']:
 f=pose['frame'];src=P/'observations'/f'{f:04d}.png';dest=O/'references'/f'round3_F{f:03d}.png';shutil.copy2(src,dest)
 refs.append({'id':f'round3_F{f:03d}','file':str(dest.relative_to(O)),'sha256':sha(dest),'calibration':'Round3 fitted shared intrinsics and per-observation rotation','R_blender_to_world':(np.array(pose['camera_cv_to_world'])@np.diag([1,-1,-1])).tolist(),'K':D['K'],'vertical_extra':0,'source_frame':f,'source_video_sha256':sha(P/'observations/round3_generated.mp4')})
(O/'references.json').write_text(json.dumps({'policy':'Union of i0/i1/i2 nominal frusta and five fitted Round3 observed frusta; original i3 F121 replaced by actual fitted views including F123. Coverage is angular image support, not geometry validation or pixel provenance. No reference pixels pasted into Blender RGB.','references':refs},indent=2),encoding='utf-8')
yy,xx=np.mgrid[:544,:960];K=np.array(T['K']);rays=np.stack([(xx+.5-K[0,2])/K[0,0],-(yy+.5-K[1,2])/K[1,1],-np.ones_like(xx)],-1)
labels_all=[];stats=[]
if not preview:assert len(list((O/'rgb').glob('*.png')))==124
for pose in T['trajectory']:
 if preview and pose['frame'] not in [0,62,123]:continue
 f=pose['frame'];rgb=np.array(Image.open(O/'rgb'/f'{f:04d}.png').convert('RGB'));assert rgb.shape==(544,960,3)
 world=rays@np.array(pose['camera_blender_to_world'])[:3,:3].T;labels=np.zeros((544,960),np.uint8)
 for idx,r in enumerate(refs,1):
  local=world@np.array(r['R_blender_to_world']);z=-local[...,2];k=np.array(r['K']);px=k[0,0]*local[...,0]/np.maximum(z,1e-9)+k[0,2];py=-k[1,1]*local[...,1]/np.maximum(z,1e-9)+k[1,2];ex=r['vertical_extra']
  hit=(z>0)&(px>=0)&(px<960)&(py>=-ex)&(py<544+ex)&(labels==0);labels[hit]=idx
 unknown=labels==0;partial=rgb.copy();partial[unknown]=128
 assert np.array_equal(partial[~unknown],rgb[~unknown]);assert np.all(partial[unknown]==128)
 Image.fromarray(partial).save(O/'partial'/f'{f:04d}.png');Image.fromarray(labels).save(O/'coverage'/f'{f:04d}.png')
 labels_all.append(labels);stats.append({'frame':f,'unknown_pixels':int(unknown.sum()),'unknown_fraction':float(unknown.mean())})
assert stats[0]['unknown_pixels']==0
if preview:
 (O/'review/preview_coverage.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
 print(json.dumps(stats));sys.exit(0)
np.savez_compressed(O/'coverage_labels.npz',labels=np.stack(labels_all))
ff=W/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
def encode(folder,out):
 cmd=[str(ff),'-y','-framerate','24','-start_number','0','-i',str(O/folder/'%04d.png'),'-frames:v','124','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(O/out)]
 p=subprocess.run(cmd,capture_output=True,text=True);(O/'logs'/f'encode_{folder}.log').write_text(p.stderr,encoding='utf-8');assert p.returncode==0
encode('rgb','full_rgb.mp4');encode('partial','partial_proxy.mp4')
checks={}
for name in ['full_rgb.mp4','partial_proxy.mp4']:
 cap=cv2.VideoCapture(str(O/name));fps=cap.get(cv2.CAP_PROP_FPS);count=0;shape=None;errors=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  shape=list(bgr.shape);assert shape==[544,960,3]
  expected=np.array(Image.open(O/('rgb' if name=='full_rgb.mp4' else 'partial')/f'{count:04d}.png').convert('RGB'))
  errors.append(float(np.abs(bgr[:,:,::-1].astype(float)-expected).mean()));count+=1
 cap.release();assert count==124 and abs(fps-24)<1e-8
 checks[name]={'decoded_frames':count,'fps':fps,'shape':shape,'lossy_decode_mae_mean':float(np.mean(errors)),'lossy_decode_mae_max':float(np.max(errors)),'sha256':sha(O/name)}
font=None
sheet=Image.new('RGB',(960,5*296),'white');draw=ImageDraw.Draw(sheet)
for row,f in enumerate([0,31,62,92,123]):
 for col,folder in enumerate(['rgb','partial']):sheet.paste(Image.open(O/folder/f'{f:04d}.png').resize((480,272)),(col*480,row*296+24))
 draw.text((8,row*296+5),f'F{f:03d}  full Blender RGB | partial proxy   unknown {stats[f]["unknown_fraction"]:.2%}',fill='black')
sheet.save(O/'review/contact_sheet.jpg',quality=94)
sheet=Image.new('RGB',(960,3*296),'white');draw=ImageDraw.Draw(sheet)
for row,(label,path) in enumerate([('Round3 generated F123',P/'observations/0123.png'),('Revised scene at Round3 F123',P/'renders/revised_0123.png'),('Round4 new proxy F0 (same camera and scene)',O/'rgb/0000.png')]):
 sheet.paste(Image.open(path).resize((480,272)),(0,row*296+24));draw.text((8,row*296+5),label,fill='black')
sheet=sheet.crop((0,0,480,888));sheet.save(O/'review/start_comparison.jpg',quality=94)
prior=np.array(Image.open(P/'renders/revised_0123.png').convert('RGB')).astype(float);start=np.array(Image.open(O/'rgb/0000.png').convert('RGB')).astype(float);source=np.array(Image.open(P/'observations/0123.png').convert('RGB')).astype(float)
source_manifest=json.loads((P/'CHECKSUMS.json').read_text(encoding='utf-8'));source_check={name:sha(P/name)==h for name,h in source_manifest.items()};assert all(source_check.values())
verification={'videos':checks,'coverage':stats,'start_vs_previous_revised_render_mae':float(np.abs(start-prior).mean()),'start_vs_generated_round3_endpoint_mae':float(np.abs(start-source).mean()),'source_revision_manifest_files_verified':len(source_check),'source_revision_unchanged':True,'known_region_png_exactly_blender_rgb':True,'unknown_region_png_rgb':[128,128,128],'cvm_run':False,'note':'H264 compression changes exact pixel values; PNGs and coverage NPZ are authoritative. Angular support does not prove scene geometry; no photographic pasteback.'}
(O/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in verification.items() if k!='coverage'},indent=2));print('UNKNOWN_START_END',stats[0],stats[-1])

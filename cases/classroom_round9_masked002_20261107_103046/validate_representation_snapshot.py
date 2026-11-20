"""Validate the separate appearance-control candidate before any model use."""
from pathlib import Path
import sys,json,hashlib,math,datetime
import numpy as np
BASE=Path(__file__).resolve().parent;R=BASE/'round9';D=R/'06_control_diagnosis'
sys.path.insert(0,str(BASE.parent/'Blender-Classroom-Reconstruction/runtime'))
import cv2
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8')
archives={}
for name in ['attempt_001']:
    run=R/'05_cvm'/name;checks=read(run/'ARCHIVE_CHECKSUMS.json')
    for rel,digest in checks.items():assert sha(run/rel)==digest,(name,rel)
    archives[name]=len(checks)
diag=read(D/'representation_diagnosis.json');assert diag['mask_disagreement_vs_raw_proxy']==0
assert all(x['source_pixels']['i8']+x['source_pixels']['i0']+round(x['unknown_fraction']*960*544)==960*544 for x in diag['coverage'])
traj=read(R/'02_blender/inputs/camera_trajectory.json')['frames'];K=np.array(traj[0]['K'])
ref=cv2.imdecode(np.fromfile(R/'02_blender/inputs/reference.png',np.uint8),0)
sift=cv2.SIFT_create(nfeatures=3000);kp0,de0=sift.detectAndCompute(ref,None)
cap=cv2.VideoCapture(str(D/'appearance_proxy.mp4'));rows=[];count=0
while True:
    ok,im=cap.read()
    if not ok:break
    f=count;count+=1
    if f not in [0,7,31,62,92,116,123]:continue
    kp,de=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
    matches=[pair[0] for pair in cv2.BFMatcher().knnMatch(de0,de,k=2) if len(pair)==2 and pair[0].distance<.72*pair[1].distance]
    a=np.float32([kp0[m.queryIdx].pt for m in matches]);b=np.float32([kp[m.trainIdx].pt for m in matches])
    hom,mask=cv2.findHomography(a,b,cv2.RANSAC,3);assert hom is not None
    keep=mask.ravel().astype(bool);err=np.linalg.norm(cv2.perspectiveTransform(a[:,None,:],hom)[:,0,:]-b,axis=1)
    u,sv,vt=np.linalg.svd(np.linalg.inv(K)@hom@K);rot=u@vt
    yaw=-math.degrees(math.atan2(rot[0,2],rot[2,2]));nom=traj[f]['yaw_degrees']+240
    rows.append(dict(frame=f,inliers=int(keep.sum()),median_error_px=float(np.median(err[keep])),estimated_yaw=yaw,nominal_yaw=nom,absolute_yaw_error=abs(yaw-nom)))
    assert keep.sum()>=30 and np.median(err[keep])<2 and abs(yaw-nom)<1,rows[-1]
cap.release();assert count==124
report={'status':'passed','checked_at':datetime.datetime.now().astimezone().isoformat(),'verified_archived_files':archives,'video_decoded_frames':count,'motion_samples':rows,'max_sample_yaw_error':max(x['absolute_yaw_error'] for x in rows),'mask_disagreement_all_frames':0,'actual_appearance_source':'i8 composite supplies newest accepted appearance; original i0 returns with14.545% endpoint support. Other sources supply zero. This is supplied history.','visual_review':'F0/F62/F123 inspected: expected small rightward feature displacement; reference appearance retained; no offline completion of gray. Raw proxy and rejected attempt001 inspected separately. Source brightness/desk differences across the unknown gap remain explicit; completion must be reviewed.','limitations':'Known exact fixed-center reprojection only; this validates control construction, not CVM adherence or geometry/depth accuracy.'}
save(D/'offline_validation.json',report)
print(json.dumps(report,indent=2))

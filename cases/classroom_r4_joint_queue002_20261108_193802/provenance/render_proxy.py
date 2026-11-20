"""Independent Round4 rendering from the v2-based joint Round3 revision."""
from pathlib import Path
import bpy, numpy as np, json, math, sys, argparse, hashlib, time, shutil
from mathutils import Matrix
O=Path(__file__).resolve().parent
P=O.parent
D=json.loads((P/'fit.json').read_text(encoding='utf-8'))
ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
for sub in ['rgb','logs']: (O/sub).mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=P/'classroom_v3_joint.blend';source_sha=sha(source)
bpy.ops.wm.open_mainfile(filepath=str(source))
s=bpy.context.scene;c=s.camera;c.animation_data_clear();c.data.animation_data_clear()
s.timeline_markers.clear();s.frame_start=0;s.frame_end=123;s.render.fps=24;s.render.fps_base=1
s.render.resolution_x=960;s.render.resolution_y=544;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB'
s.render.use_persistent_data=True;s.render.threads_mode='FIXED';s.render.threads=8
s.cycles.samples=24;s.cycles.use_denoising=True;s.cycles.use_animated_seed=False
c.data.sensor_fit='HORIZONTAL';c.data.sensor_width=36;c.data.lens=36*D['K'][0][0]/960
c.data.shift_x=0;c.data.shift_y=0;s.render.pixel_aspect_x=1;s.render.pixel_aspect_y=1
R0=np.array(next(p for p in D['poses'] if p['frame']==123)['camera_cv_to_world'])@np.diag([1,-1,-1])
trajectory=[]
for f in range(124):
 u=float(np.clip((f-7)/109,0,1));theta=30*(3*u*u-2*u*u*u);t=math.radians(theta)
 rz=np.array([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1]])
 m=np.eye(4);m[:3,:3]=rz@R0;m[:3,3]=D['camera_position']
 s.frame_set(f);c.matrix_world=Matrix(m.tolist());c.keyframe_insert(data_path='rotation_euler',frame=f);c.keyframe_insert(data_path='location',frame=f)
 trajectory.append({'frame':f,'nominal_yaw_degrees':-90-theta,'left_increment_degrees':theta,'camera_blender_to_world':m.tolist()})
for name,f in [('Round3 endpoint / -90 nominal',0),('Start left turn',7),('End left turn / -120 nominal',116)]:s.timeline_markers.new(name,frame=f)
s['camera_review']='Round4: exact fitted Round3 F123 pose, then world-Z +30 degree turn; fitted intrinsics retained'
s['proxy_scope']='Mesh render only; angular support mask packaged separately; no generated-photo pasteback'
s.frame_set(0);bpy.context.view_layer.update()
assert np.max(np.abs(np.array(c.matrix_world)[:3,:3]-R0))<1e-6
s.render.filepath=str(O/'rgb/0000.png')
meta={'source_scene':str(source),'source_scene_sha256':source_sha,'fit_sha256':sha(P/'fit.json'),'K':D['K'],'horizontal_fov':D['horizontal_fov_fitted_degrees'],'resolution':[960,544],'fps':24,'frames':124,'camera_position':D['camera_position'],'trajectory':trajectory,'rotation_scope':'nominal -90 to -120; start is fitted Round3 F123, not original nominal camera','samples':24}
(O/'trajectory.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if not a.preview:
 bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'round4_proxy.blend'))
frames=[0,62,123] if a.preview else list(range(124));start=time.time();cache={};rendered=0
for f in frames:
 s.frame_set(f);bpy.context.view_layer.update();m=np.array(c.matrix_world)
 assert np.max(np.abs(m-np.array(trajectory[f]['camera_blender_to_world'])))<1e-6
 dest=O/'rgb'/f'{f:04d}.png';key=tuple(np.round(m.ravel(),7))
 if dest.exists():cache[key]=dest;print('EXISTING_VERIFIED_RUN_FRAME',f,flush=True);continue
 if key in cache:shutil.copy2(cache[key],dest);print('IDENTICAL_POSE_COPY',f,flush=True);continue
 s.render.filepath=str(dest);bpy.ops.render.render(write_still=True);cache[key]=dest;rendered+=1
 print('FRAME_DONE',f,flush=True)
assert sha(source)==source_sha
(O/'logs'/('preview_timing.json' if a.preview else 'render_timing.json')).write_text(json.dumps({'seconds':time.time()-start,'rendered_this_run':rendered,'frames_present':len(list((O/'rgb').glob('*.png'))),'source_unchanged':True},indent=2),encoding='utf-8')
print('RENDER_COMPLETE',flush=True)

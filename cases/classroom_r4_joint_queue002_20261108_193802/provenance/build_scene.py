"""Editable v2-based revision fitted to Round3. Run in Blender, no CVM inference.
The generated video supplies image evidence, not ground-truth depth. No full-frame billboard.
"""
from pathlib import Path
import bpy,json,math,sys,argparse
import numpy as np
from mathutils import Vector,Matrix
P=Path(__file__).resolve().parent
D=json.loads((P/'fit.json').read_text(encoding='utf-8'))
parser=argparse.ArgumentParser();parser.add_argument('--frames',default='0,31,62,92,123');parser.add_argument('--baseline',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
bpy.ops.wm.open_mainfile(filepath=str(P/'base/classroom_v2.blend'))
scene=bpy.context.scene;cam=scene.camera;cam.animation_data_clear();C=np.array(D['camera_position'])
removed=[];created=[]
def tag(o):
 o['evidence']='Round3 generated output, frames0/31/62/92/123';o['depth']='assumed scale; fixed-center rotation cannot recover metric depth';created.append(o.name);return o
def mat(name,color,rough=.6,metal=0):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
 p=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal;return m
def cube(name,loc,dim,m,bevel=0):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.name=name;o.dimensions=dim;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(m)
 if bevel:mod=o.modifiers.new('Soft edges','BEVEL');mod.width=bevel;mod.segments=2;o.modifiers.new('Normals','WEIGHTED_NORMAL')
 return tag(o)
def quad(name,points,m,thickness=0,texture=False):
 mesh=bpy.data.meshes.new(name);mesh.from_pydata([tuple(v) for v in points],[],[tuple(range(len(points)))]);mesh.update()
 o=bpy.data.objects.new(name,mesh);scene.collection.objects.link(o);o.data.materials.append(m)
 if thickness:mod=o.modifiers.new('Real slab thickness','SOLIDIFY');mod.thickness=thickness
 if texture:
  mesh.uv_layers.new(name='Local surface UV')
  for loop,uv in zip(mesh.uv_layers.active.data,[(0,1),(1,1),(1,0),(0,0)]):loop.uv=uv
 return tag(o)
def bar(name,a,b,width,m):
 a=Vector(a);b=Vector(b);o=cube(name,(a+b)/2,(width,width,(b-a).length),m,.006);o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler();return o
def patchmat(name,file):
 m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
 t=n.new('ShaderNodeTexImage');t.image=bpy.data.images.load(str(P/'assets'/file));e=n.new('ShaderNodeEmission');e.inputs['Strength'].default_value=1.;out=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(t.outputs['Color'],e.inputs[0]);m.node_tree.links.new(e.outputs[0],out.inputs[0]);m['scope']='Local window patch only, baked lighting from generated evidence';return m
def delete(o):removed.append(o.name);bpy.data.objects.remove(o,do_unlink=True)
def setpose(cv):
 r=np.array(cv)@np.diag([1,-1,-1]);m=np.eye(4);m[:3,:3]=r;m[:3,3]=C;cam.matrix_world=Matrix(m.tolist());bpy.context.view_layer.update()
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True
scene.render.resolution_x=960;scene.render.resolution_y=544;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.use_persistent_data=True
scene.render.threads_mode='FIXED';scene.render.threads=8
cam.data.lens=36*D['focal_pixels_fitted']/960;cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=36
cam['calibration_scope']='Fitted to generated Round3, not a calibrated real camera';cam['v2_original_horizontal_fov']=100
if not args.baseline:
 # Remove inherited hypotheses in the sector actually observed in Round3.
 furniture_roots=[o for o in scene.objects if o.type=='EMPTY' and (o.name.startswith('desk_') or o.name.startswith('chair_')) and o.location.x<0]
 delete_set=set(furniture_roots)
 for root in furniture_roots:delete_set.update(root.children_recursive)
 for o in list(scene.objects):
  if o in delete_set or o.name.startswith(('left_window','left_radiator','room.left_','room.left_window','window.','radiator.')):delete(o)
 wall=mat('R3 fitted cream plaster',(.60,.56,.45));wood=mat('R3 reddish desk laminate',(.18,.084,.045),.18)
 wood_bsdf=next(n for n in wood.node_tree.nodes if n.type=='BSDF_PRINCIPLED');wood_bsdf.inputs['Coat Weight'].default_value=.3
 steel=mat('R3 dark tubular steel',(.018,.021,.023),.38,.45);white=mat('R3 window white',(.55,.57,.54));floor=mat('R3 grey floor',(.25,.235,.215));ceil=mat('R3 acoustic ceiling',(.55,.55,.57));grid=mat('R3 ceiling joints',(.27,.28,.29))
 board_mat=bpy.data.materials.get('front_board_patch')
 if board_mat:
  for node in board_mat.node_tree.nodes:
   if node.type=='EMISSION':node.inputs['Strength'].default_value=.35
  board_mat['revision_note']='Photometric appearance adjustment to Round3 visible board; original artwork retained, no geometric evidence of replacement board.'
 # Fit the observed left edge and top/bottom extent of the front board in F0.
 # Its unseen right extent and original artwork remain inherited v2 priors.
 K=np.array(D['K']);R0=np.array(next(x for x in D['poses'] if x['frame']==0)['camera_cv_to_world'])
 rays=np.array([[677.,144.,1.],[674.,319.,1.]])@np.linalg.inv(K).T@R0.T
 edge=C+rays*((-.091-C[1])/rays[:,1])[:,None]
 new_left=float(edge[:,0].mean());new_bot=float(edge[1,2]);new_top=float(edge[0,2])
 old_left=-2.66;old_right=3.12;old_bot=1.255;old_top=3.215
 for ob in list(scene.objects):
  if ob.type=='MESH' and ob.name.startswith('blackboard.'):
   inv=ob.matrix_world.inverted()
   for vertex in ob.data.vertices:
    pos=ob.matrix_world@vertex.co;pos.x=new_left+(pos.x-old_left)*(old_right-new_left)/(old_right-old_left);pos.z=new_bot+(pos.z-old_bot)*(new_top-new_bot)/(old_top-old_bot);vertex.co=inv@pos
   ob['revision']='F0 visible left/top/bottom fit; unseen right boundary inherited'
 back=-10.8;top=D['ceiling_height'];left=D['left_wall_x'];right=5.15
 for name,loc,dim in [('room.floor',((left+right)/2,back/2,-.055),(right-left,-back,.10)),('room.ceiling',((left+right)/2,back/2,top+.055),(right-left,-back,.10)),('room.back_wall',((left+right)/2,back-.075,top/2),(right-left,.15,top)),('room.front_wall',((left+right)/2,.075,top/2),(right-left,.15,top)),('room.right_wall',(right+.075,back/2,top/2),(.15,-back,top))]:
  o=bpy.data.objects.get(name)
  if o:delete(o)
  cube(name,loc,dim,floor if name.endswith('floor') else ceil if name.endswith('ceiling') else wall)
 # Actual left-wall openings, frame/sill/radiator meshes; local window surfaces.
 wins=D['windows'];cursor=back;previous_edge=[back,back]
 def edge_extent(a,b):
  return [float(a[1]+(z-a[2])*(b[1]-a[1])/(b[2]-a[2])) for z in [0,top]]
 for w in wins:
  pts=np.array(w['vertices']);name='R3.window.'+w['id'];lo=min(pts[:,1]);hi=max(pts[:,1]);bot=float(pts[2:,2].mean());upper=float(pts[:2,2].mean())
  current_edge=edge_extent(pts[3],pts[0])
  quad(name+'.pier',[[left,previous_edge[0],0],[left,current_edge[0],0],[left,current_edge[1],top],[left,previous_edge[1],top]],wall,.12)
  right_edge=edge_extent(pts[2],pts[1])
  quad(name+'.below',[pts[3],pts[2],[left,right_edge[0],0],[left,current_edge[0],0]],wall,.1)
  quad(name+'.above',[pts[0],pts[1],[left,right_edge[1],top],[left,current_edge[1],top]],wall,.1)
  quad(name+'.glass',pts+[-.025,0,0],patchmat(name+'.local observed appearance',f'window_{w["id"]}.png'),texture=True)
  for i in range(4):bar(name+f'.frame{i}',pts[i]+[.04,0,0],pts[(i+1)%4]+[.04,0,0],.075,white)
  crossleft=pts[0]*.48+pts[3]*.52;crossright=pts[1]*.48+pts[2]*.52
  bar(name+'.crossbar',crossleft+[.055,0,0],crossright+[.055,0,0],.072,white)
  t=w['mullion_fraction'];bar(name+'.mullion',pts[0]*(1-t)+pts[1]*t+[.06,0,0],pts[3]*(1-t)+pts[2]*t+[.06,0,0],.055,white)
  cube(name+'.sill',(left+.13,(lo+hi)/2,bot-.075),(.38,hi-lo+.20,.09),white,.01)
  cube(name+'.radiator',(left+.18,(lo+hi)/2,.48),(.20,hi-lo-.10,.64),white,.015)
  for i in range(max(5,int((hi-lo)/.10))):cube(name+f'.radiator_rib{i}',(left+.29,lo+.08+i*.10,.48),(.06,.025,.60),white,.005)
  cursor=hi;previous_edge=edge_extent(pts[2],pts[1])
 quad('R3.window.front_corner_pier',[[left,previous_edge[0],0],[left,0,0],[left,0,top],[left,previous_edge[1],top]],wall,.12)
 cube('R3.left_baseboard',(left+.025,back/2,.075),(.05,-back,.15),steel)
 # Front aperture also updated from F0, while the original blackboard remains v2.
 fp=np.array(D['front_window']['vertices']);fp[:,1]=-.04
 quad('R3.front_window.glass',fp,patchmat('R3 front observed window','front_window_round3.png'),texture=True)
 for i in range(4):bar(f'R3.front_window.frame{i}',fp[i]+[0,-.02,0],fp[(i+1)%4]+[0,-.02,0],.075,white)
 bar('R3.front_window.crossbar',fp[0]*.5+fp[3]*.5+[0,-.03,0],fp[1]*.5+fp[2]*.5+[0,-.03,0],.07,white)
 lo,hi=float(fp[:,0].min()),float(fp[:,0].max());bot=float(fp[2:,2].mean())
 cube('R3.front_window.sill',((lo+hi)/2,-.16,bot-.08),(hi-lo+.18,.35,.10),white)
 cube('R3.front_window.radiator',((lo+hi)/2,-.17,.40),(hi-lo,.20,.58),white)
 # Each furniture set has a shared frame. Rectangular desk and chair components
 # are built in local coordinates, with measured relative offsets and rotations.
 def root(name,location,parent=None,yaw=0):
  ob=bpy.data.objects.new(name,None);scene.collection.objects.link(ob);ob.parent=parent;ob.location=location;ob.rotation_euler.z=yaw;return tag(ob)
 def localcube(name,loc,dim,material,parent,bevel=0):
  ob=cube(name,loc,dim,material,bevel);ob.parent=parent;return ob
 def localbar(name,a,b,width,material,parent):
  ob=bar(name,a,b,width,material);ob.parent=parent;return ob
 for row in D['desks']:
  name='R3.set.'+row['id'];x,y,w,depth,yaw=row['parameters']
  group=root(name,(x,y,0),yaw=yaw);group['observations']=json.dumps(row['reprojection']);group['dimensions_are_fitted']=True
  desk=root(name+'.desk',(0,0,0),group)
  o=localcube(name+'.desk.top',(0,0,.8125),(w,depth,.055),wood,desk,.018);o['rectangular']=True
  for sx in [-1,1]:
   for sy in [-1,1]:localbar(name+f'.desk.leg{sx}_{sy}',[sx*(w/2-.09),sy*(depth/2-.07),0],[sx*(w/2-.09),sy*(depth/2-.07),.79],.042,steel,desk)
  for sy in [-1,1]:localcube(name+f'.desk.apron{sy}',(0,sy*(depth/2-.035),.73),(w-.08,.035,.10),steel,desk,.005)
  for sx in [-1,1]:localcube(name+f'.desk.rail{sx}',(sx*(w/2-.075),0,.71),(.035,depth-.07,.035),steel,desk,.005)
  spec=row['chair'];dx,dy=spec['local_offset'];chair=root(name+'.chair',(dx,dy,0),group,spec['yaw_relative']);chair['evidence']=spec['evidence'];chair['table_pair']=desk.name
  localcube(name+'.chair.seat',(0,0,.45),(.48,.43,.045),wood,chair,.03)
  ht=spec['back_top_z'];localcube(name+'.chair.back',(0,-.27,ht-.12),(.50,.05,.24),wood,chair,.035)
  for sx in [-1,1]:
   for sy in [-1,1]:localbar(name+f'.chair.leg{sx}_{sy}',[sx*.205,sy*.19,0],[sx*.205,sy*.19,.43],.033,steel,chair)
   localbar(name+f'.chair.back_support{sx}',[sx*.205,-.245,.26],[sx*.205,-.245,ht-.03],.030,steel,chair)
 # Visible ceiling panels: geometry, no ceiling screenshot.
 for x in np.arange(left,right+.1,1.2):cube(f'R3.ceiling_joint_x{x:.2f}',(float(x),back/2,top-.008),(.018,-back,.012),grid)
 for y in np.arange(back,.1,1.2):cube(f'R3.ceiling_joint_y{y:.2f}',((left+right)/2,float(y),top-.008),(right-left,.018,.012),grid)
 for o in scene.objects:
  if o.type=='LIGHT':
   if o.name=='ceiling_bounce':o.location=(-3,-4,top-.35);o.data.energy=230;o.data.size=7
   elif o.name=='window_soft_light':o.location=(-5.8,-4,3.6);o.data.energy=180;o.data.size=5
 scene.world.node_tree.nodes['Background'].inputs[1].default_value=.24
 scene['revision']='v3_from_v2_round3_joint';scene['scope']='Multi-frame rectangular desk/chair set fit; not calibrated reconstruction'
 scene['base']='Archived classroom_v2.blend; no old v3 loaded';scene['remaining_limits']='Nonrigid source video; assumed depth, clipped tabletop extents, chair locations and rear shell.'
 (P/'changes.json').write_text(json.dumps({'removed_from_working_copy':removed,'created':created,'base':'base/classroom_v2.blend'},indent=2),encoding='utf-8')

for f in [int(x) for x in args.frames.split(',')]:
 pose=next(x for x in D['poses'] if x['frame']==f);setpose(pose['camera_cv_to_world'])
 scene.render.filepath=str(P/'renders'/f'{"v2" if args.baseline else "revised"}_{f:04d}.png');bpy.ops.render.render(write_still=True)
 print('RENDER_DONE',f,flush=True)
if not args.baseline:
 # Review animation is Round3 fitted observation poses, NOT an unvalidated Round4 proxy.
 for pose in D['poses']:
  setpose(pose['camera_cv_to_world']);cam.keyframe_insert(data_path='rotation_euler',frame=pose['frame'])
 scene.frame_start=0;scene.frame_end=123;scene.frame_set(123);scene.render.fps=24
 scene['camera_review']='Five fitted Round3 observations; interpolation is for review, not a validated full camera trajectory'
 for pose in D['poses']:scene.timeline_markers.new('Round3 observation '+str(pose['frame']),frame=pose['frame'])
 for screen in bpy.data.screens:
  for area in screen.areas:
   if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.shading.type='MATERIAL';area.spaces.active.overlay.show_overlays=False
 bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(P/'classroom_v3_joint.blend'))
 print('SAVED_REVISED',flush=True)

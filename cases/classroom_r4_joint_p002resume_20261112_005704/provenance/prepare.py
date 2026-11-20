"""Stage an independent RGB CVM attempt for the new Round4 joint-scene proxy."""
from pathlib import Path
import json,hashlib,shutil,datetime,sys
C=Path(__file__).resolve().parent;P=C.parent;B=P.parent.parent;V=P/'round4_proxy_v1';A=C/'attempt_001'
sys.path.insert(0,str(B));from prompt_render_review import require_prompt_render_review
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not A.exists(),'Do not overwrite or restage an attempt'
for rel,h in read(V/'CHECKSUMS.json').items():assert sha(V/rel)==h,rel
for rel,h in read(P/'CHECKSUMS.json').items():assert sha(P/rel)==h,rel
for sub in ['inputs','provenance']:(A/sub).mkdir(parents=True,exist_ok=True)
(C/'logs').mkdir(exist_ok=True)
shutil.copy2(V/'partial_proxy.mp4',A/'inputs/partial_proxy.mp4')
shutil.copy2(P/'observations/0123.png',A/'inputs/reference.png')
prompt='''subject_definitions:
<Subject 1> is the classroom in <Picture 1>, with reddish brown wooden desks and chairs on dark metal frames, white-framed windows, cream walls and daylight.
<Video 1> is an incomplete Blender camera preview. Its flat gray areas mark missing image regions, not physical surfaces.

summary:
[video editing + reference generation] Complete <Video 1> as one photorealistic classroom shot, following its camera motion and visible object layout.

retention_analysis:
<Subject 1>: retain classroom identity, materials, window scenery and natural daylight from <Picture 1>.
<Video 1>: retain the chronological camera path, perspective and positions of visible desks, chairs, windows and radiators; replace gray gaps and their artificial boundaries with coherent room content.

detailed_description:
[Shot 1] Start at the window-facing orientation shown in the first frame of <Video 1>, corresponding to the ending view in <Picture 1>. Hold briefly, turn LEFT another 30 degrees in place following <Video 1>, and hold at the final orientation. There is no camera translation, zoom, cut, reframing or return to the starting view. Existing windows and furniture move RIGHT in the image as the camera turns; all furniture remains stationary in the room. The large window on the right exits to the right, while the other windows remain toward the right side of the final composition. Use <Picture 1> for photographic appearance and classroom identity. Where the reference and the approximate preview differ in object shape or projection, follow <Video 1> for camera motion and visible layout; do not snap between two layouts. Preserve the visible desks and their paired chairs with consistent occlusion through the turn. Fill the newly exposed left region with a continuous cream rear wall and room details consistent with the adjoining visible scene. Extend the ceiling into the upper gaps and the floor into the lower gaps with matching perspective and daylight. Fill all gray gaps throughout the shot, smoothly connecting them to existing content; leave no gray background, cutout boundary or empty strip. Keep the room and its lighting coherent throughout the video.

overall_soundscape:
Quiet classroom ambience.

non_diegetic_music:
N/A
'''
(A/'inputs/prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
for name in ['trajectory.json','references.json','verification.json','saved_scene_verification.json','round4_proxy.blend','render_proxy.py','package_proxy.py']:
 shutil.copy2(V/name,A/'provenance'/name)
for name in ['contact_sheet.jpg','start_comparison.jpg','VISUAL_REVIEW.md']:shutil.copy2(V/'review'/name,A/'provenance'/name)
for name in ['fit.json','build_scene.py','scene_verification.json']:shutil.copy2(P/name,A/'provenance'/name)
shutil.copy2(B/'prompt_render_review.py',A/'prompt_render_review.py')
now=datetime.datetime.now().astimezone();stamp=now.strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_r4_joint_rgb001_{stamp}'
T=B/'round4/05_cvm/attempt_002';old=read(T/'deployment.json')['remote_case']
for name in ['run_cvm.py','preflight.py','supervise.py','launch.sh']:
 text=(T/name).read_text(encoding='utf-8').replace(old,remote).replace('round4_seed42','round4_joint_seed42')
 if name=='preflight.py':
  text += '\nfrom prompt_render_review import require_prompt_render_review\nrequire_prompt_render_review(ROOT/"inputs/prompt_render_review.json",prompt=ROOT/"inputs/prompt.txt",video=ROOT/"inputs/partial_proxy.mp4",reference=ROOT/"inputs/reference.png")\n'
 (A/name).write_text(text,encoding='utf-8',newline='\n')
cfg=read(T/'cvm_config.json');cfg.update(output='outputs/round4_joint_seed42.mp4',scheduler_steps_requested=21,actual_denoising_updates=20,turbo=False,notes='New joint-scene Round4 RGB proxy and Round3 F123 reference. Full H3; existing RGB keep_steps=5, no hard mask, no anchoring, no output pasteback. New scene, calibration, reference frame and prompt differ from historical Round4; not a single-variable ablation.')
save(A/'cvm_config.json',cfg)
review={'status':'passed','reviewed_by':'Codex visual inspection','reviewed_at':now.isoformat(),'inspected_frames':[0,31,62,92,123],'sha256':{'prompt':sha(A/'inputs/prompt.txt'),'video':sha(A/'inputs/partial_proxy.mp4'),'reference':sha(A/'inputs/reference.png')},'checks':{k:True for k in ['visible_objects_and_returning_landmarks','entry_exit_direction_and_final_view','known_content_vs_gray_gaps','reference_video_conflicts_resolved_in_prompt','no_inherited_scene_description_conflict']},'observations':['White-framed windows and trees, reddish desks and chairs, cream walls, radiators, ceiling and floor. No blackboard in this view.','Left turn moves existing windows right; large right window exits right; final view keeps windows on right.','Gray angular gaps: F0 0%, F31 7.48%, F62 24.62%, F92 36.53%, F123 40.00%.','Actual reference is Round3 F123, not archived i3 F121. It differs visibly from the Blender proxy in window framing, furniture contours, lighting and materials.','Prompt gives geometry/motion priority to the proxy and appearance priority to the reference; this resolves conflicting instructions but DOES NOT remove underlying geometric discrepancies.','RGB keep_steps=5 is soft conditioning; first-frame matching, gray filling and motion following are not guaranteed. No mask or pixel pasteback is used.'],'unresolved_prompt_conflicts':[],'remaining_modeling_and_generation_risks':['Reference/proxy mismatch may still cause boundary changes or shape drift.','Pure rotation does not validate depth; hidden chairs and unseen room geometry remain assumptions.'],'clause_review':{'subject_definitions':'Matches observed objects; gray identified as missing data.','summary_retention':'Proxy supplies motion/layout; reference supplies appearance.','motion':'Brief holds and a 30 degree left turn with no return, matching rendered trajectory.','entry_exit':'Right-side window exits right; other windows shift right.','completion':'Rear wall, ceiling and floor gaps; no inherited blackboard requirement.','sound':'Non-conflicting quiet ambience.'}}
save(A/'inputs/prompt_render_review.json',review)
require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
save(A/'provenance/lineage.json',{'proxy_path':str(V/'partial_proxy.mp4'),'proxy_sha256':sha(V/'partial_proxy.mp4'),'reference_path':str(P/'observations/0123.png'),'reference_frame':123,'reference_sha256':sha(P/'observations/0123.png'),'round3_video_sha256':sha(P/'observations/round3_generated.mp4'),'scene_sha256':sha(V/'round4_proxy.blend'),'revision_scene_sha256':sha(P/'classroom_v3_joint.blend'),'source_proxy_manifest_files_verified':len(read(V/'CHECKSUMS.json')),'original_archives_modified':False,'parent_proxy_input_png_mae':0.0,'generated_round3_to_proxy_F0_mae':38.06378421160131})
save(A/'deployment.json',{'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r4joint_{stamp}','status':'prepared_to_launch','prepared_at':now.isoformat(),'local_attempt':str(A)})
resource=(B/'check_remote_resources.py').read_text(encoding='utf-8').replace("P/f'round{n}/logs'","P/'logs'")
(C/'check_resources.py').write_text(resource,encoding='utf-8')
save(C/'STATE.json',{'status':'prepared_to_launch','attempt':'attempt_001','scope':'Independent Round4 rerun only; no continuation to Round5','deployment':read(A/'deployment.json')})
print(json.dumps(read(A/'deployment.json'),indent=2))

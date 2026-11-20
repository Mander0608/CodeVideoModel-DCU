"""One-shot freeze of reviewed original-i9 masked branch; no inference here."""
from pathlib import Path
import json,hashlib,shutil,datetime
from prompt_render_review import require_prompt_render_review
B=Path(__file__).resolve().parent; R=B/'round10'; A=R/'05_cvm/attempt_004_masked_original_i9'; T=B/'round8/05_cvm/attempt_006_masked'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (A/'deployment.json').exists()
assert read(A/'preparation_validation.json')['status']=='passed_offline'
now=datetime.datetime.now().astimezone().isoformat()
observations=[
 'Inspected full-resolution F0,F31,F62,F92,F123, original Blender comparison and repaint overlay. Initial apparent board overlap in the horizontal contact sheet was a viewing error, corrected using full frames; no source pixels were modified for that mistaken diagnosis.',
 'Greenboard enters at left, tree revealed progressively; separate dark board remains on adjoining right wall and exits right. Both remain distinct in full frames.',
 'Known-source seam has wall color discontinuity and desk-edge mismatch; 24px repaint bands address local transitions, with8px unknown borders. Inference acceptance is provisional; generated seams must be checked throughout, not only at endpoint.',
 'Supplied i0 greenboard/tree interiors protected outside mask; mask provides ceiling/floor wedges and source seams. This preservation is explicit conditioning and paste-back, not model recall.',
 'Raw original i9 reference lacks returning i0 board; prompt assigns later layout to source video. Nominal poses and shallow recess0.30 remain hypotheses, not calibration.',
 'No failed round10 outputs or historical manually corrected i9 used. Source RGB unchanged after preparation; only masks/prompt changed offline.'
]
review={'status':'passed','reviewed_by':'Codex visual inspection','reviewed_at':now,'inspected_frames':[0,31,62,92,123],
 'sha256':{'prompt':sha(A/'inputs/prompt.txt'),'video':sha(A/'inputs/partial_proxy.mp4'),'reference':sha(A/'inputs/reference.png')},
 'additional_condition_sha256':{k:sha(A/k) for k in ['inputs/source_video.mp4','inputs/repaint_mask.mp4','inputs/masks.npz','references/i9.png']},
 'checks':{k:True for k in ['visible_objects_and_returning_landmarks','entry_exit_direction_and_final_view','known_content_vs_gray_gaps','reference_video_conflicts_resolved_in_prompt','no_inherited_scene_description_conflict']},
 'observations':observations,'unresolved_prompt_conflicts':[]}
save(A/'inputs/prompt_render_review.json',review)
save(A/'offline_visual_review.json',{'accepted_for_inference':True,'accepted_for_next_round':False,'reviewed_at':now,'observations':observations,'remaining_risks':['desk-edge seam repair may fail','source exposure difference','approximate nominal alignment','masked temporal flicker']})
require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
assert sha(A/'inputs/reference.png')==sha(A/'references/i9.png')
assert (A/'prompt.txt').read_bytes()==(A/'inputs/prompt.txt').read_bytes()
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round10_masked004_{stamp}'
oldremote=read(T/'deployment.json')['remote_case']
for name in ['masked_inpaint.py','run_cvm.py','preflight.py','supervise.py','watch_and_collect.py','launch.sh']:
 code=(T/name).read_text(encoding='utf-8').replace(oldremote,remote).replace('round8_seed42','round10_seed42')
 if name=='masked_inpaint.py':
  code=code.replace('references/i7.png','references/i9.png').replace("source_references=['accepted_i7']","source_references=['raw_original_i9_F123','original_i0_via_source_video','accepted_history_via_source_video']")
 if name=='watch_and_collect.py':
  code=code.replace("r'^DENOISE (\\d+) seconds ([0-9.]+)'","r'\\bDENOISE (\\d+) seconds ([0-9.]+)'")
 (A/name).write_text(code,encoding='utf-8',newline='\n')
cfg=read(T/'cvm_config.json');cfg.update(status='prepared_to_launch',output='outputs/round10_seed42.mp4')
cfg['inputs']['reference_image']='references/i9.png'
cfg['notes']='Explicit masked method change after three rejected RGB attempts. Original i0 supplies returning historical rays; raw i9 F123 is independent reference and latest generated source elsewhere. No old manual-i9. Unknown gaps and24px source-transition bands generated; unmasked known pixels pasted back. FullH3,20actualupdates,seed42,noTurbo. Nominal-270to-300,not calibrated pose.'
cfg['source_image_reference']='raw round9 attempt001 F123 i9'
cfg['source_video_history']=['original i0 authoritative returning rays','raw i9 plus accepted history under nominal rotations']
save(A/'cvm_config.json',cfg)
shutil.copy2(R/'05_cvm/attempt_001/scene_snapshot_reference.json',A/'scene_snapshot_reference.json')
save(A/'paired_manifest.json',{'comparison_scope':'Method and representation change, not single-variable ablation','parents':['attempt_001','attempt_002_blackboard_prompt','attempt_003_keep3'],'original_CVM_baseline_passed':False,'same_seed_and_actual_updates':True,'same_conditioning_algorithm':False,'same_video':False,'same_prompt':False,'same_raw_i9_reference':True,'raw_Blender_scene_unchanged':True,'source_scene_nominal_yaw':[-270,-300],'known_regions_forced_from_source':True})
save(A/'implementation_source.json',{'baseline':'round8/05_cvm/attempt_006_masked','baseline_runner_sha256':sha(T/'masked_inpaint.py'),'changes':['reference i7 to raw i9','source provenance i0/rawi9/history','round10 case/output names','collector DENOISE regex'],'masked_core_algorithm_changed':False})
shutil.copy2(R/'provenance/check_prelaunch_resources.py',A/'check_prelaunch_resources.py')
shutil.copy2(Path(__file__),A/'stage_runner_snapshot.py')
for p in A.glob('*.py'): compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(A/'input_checksums.json',{p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r10_mask004_{stamp}','status':'prepared_to_launch','prepared_at':now};save(A/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');s.update(status='round10_masked004_prepared',current_run=str(A.resolve()),current_run_relative=A.relative_to(B).as_posix(),current_deployment=dep,updated_at=now,last_verified_phase='Masked004 offline input/prompt review passed for inference only; no submission yet. Original i0/raw i9 sources;24px seam bands preserved with prior masks.',active_control={'status':'prepared','protocol':'explicit_masked_method_change','parent':'round10/05_cvm/attempt_003_keep3','not_single_variable_control':True});s['round10']='masked004_prepared';save(B/'CONTINUOUS_LEFT_STATE.json',s)
print(json.dumps(dep,indent=2))

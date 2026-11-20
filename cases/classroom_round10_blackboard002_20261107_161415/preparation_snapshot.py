"""Prepare the user-authorized prompt-only comparison, never launch or overwrite."""
from pathlib import Path
import datetime, json, hashlib, shutil
from prompt_render_review import require_prompt_render_review
BASE=Path(__file__).resolve().parent
OLD=BASE/'round10/05_cvm/attempt_001'
NEW=BASE/'round10/05_cvm/attempt_002_blackboard_prompt'
REV=BASE/'prompt_revisions/round10_blackboard_v002'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not NEW.exists(),'Never restage an existing attempt'
require_prompt_render_review(REV/'prompt_render_review.json',prompt=REV/'prompt.txt',video=OLD/'inputs/partial_proxy.mp4',reference=OLD/'inputs/reference.png')
for rel,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items():assert sha(OLD/rel)==h,rel
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');now=datetime.datetime.now().astimezone().isoformat()
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round10_blackboard002_{stamp}'
oldremote=read(OLD/'deployment.json')['remote_case']
shutil.copytree(OLD/'inputs',NEW/'inputs')
shutil.copy2(REV/'prompt.txt',NEW/'inputs/prompt.txt')
shutil.copy2(REV/'prompt_render_review.json',NEW/'inputs/prompt_render_review.json')
save(NEW/'inputs/prompt_adaptation.json',{'base':'round10/05_cvm/attempt_001/inputs/prompt.txt','revision':'prompt_revisions/round10_blackboard_v002','prompt_changed':True,'reason':'Preserve returning left greenboard/tree and right dark board; remove inherited blank-wall description. Reviewed against actual render.','sampling_changed':False})
manifest=read(NEW/'inputs/input_manifest.json')
for name in ['reference.png','partial_proxy.mp4','prompt.txt']:
    manifest['files'][name]={'sha256':sha(NEW/'inputs'/name),'bytes':(NEW/'inputs'/name).stat().st_size}
manifest['prompt_render_review']='prompt_render_review.json';save(NEW/'inputs/input_manifest.json',manifest)
for name in ['run_cvm.py','preflight.py','supervise.py','watch_and_collect.py','launch.sh']:
    (NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(oldremote,remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg['status']='prepared_to_launch';cfg['notes']='User-authorized single prompt-only round10 comparison; preserve returning greenboard/tree. Same image, video, seed42/20steps/keep5. No explicit mask/pasteback. Audit and stop.'
save(NEW/'cvm_config.json',cfg)
shutil.copy2(OLD/'scene_snapshot_reference.json',NEW/'scene_snapshot_reference.json')
same_inputs={name:sha(NEW/'inputs'/name)==sha(OLD/'inputs'/name) for name in ['partial_proxy.mp4','reference.png','camera_trajectory.json','camera_trajectory.csv','coverage.json']}
assert all(same_inputs.values())
same_sampling={k:cfg[k]==read(OLD/'cvm_config.json')[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']};assert all(same_sampling.values())
save(NEW/'paired_manifest.json',{'parent_attempt':'round10/05_cvm/attempt_001','comparison_scope':'Prompt-only paired experiment, one seed; multiple wording edits do not isolate sentence-level causality.','only_model_condition_change':'prompt','identical_input_hashes':same_inputs,'identical_sampling':same_sampling,'parent_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'new_prompt_sha256':sha(NEW/'inputs/prompt.txt'),'runner_sha256':sha(NEW/'run_cvm.py'),'runner_identical':sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py'),'scene_snapshot_identical':sha(NEW/'scene_snapshot_reference.json')==sha(OLD/'scene_snapshot_reference.json'),'administrative_changes':'New case/screen, launch/watcher remote paths, review metadata and manifests; not model conditions.'})
require_prompt_render_review(NEW/'inputs/prompt_render_review.json',prompt=NEW/'inputs/prompt.txt',video=NEW/'inputs/partial_proxy.mp4',reference=NEW/'inputs/reference.png')
(NEW/'README.md').write_text('Round10 blackboard prompt-only comparison. User authorized one run. Same original-i9 image and Blender input video as attempt001, same seed42/20steps/keep5; only prompt changed. Preserve both outputs. Audit greenboard/tree return, gray completion, natural joins, desks and motion, then stop; no automatic retry or round11/12.\n',encoding='utf-8')
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r10_bb002_{stamp}','status':'prepared_to_launch','prepared_at':now};save(NEW/'deployment.json',dep)
state=read(BASE/'CONTINUOUS_LEFT_STATE.json')
state.update(status='round10_blackboard_prompt002_prepared',current_round=10,current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=dep,updated_at=now,continue_main_chain=False)
state['authorization']='User explicitly authorized one rerun with reviewed round10 blackboard prompt v002. Same other conditions; collect/audit/compare/report then stop. No automatic retry or round11/12.'
state['round10']='blackboard_prompt002_prepared'
state['prepared_prompt_revision'].update(status='staged_for_user_authorized_run',inference_authorized=True,attempt=NEW.relative_to(BASE).as_posix())
save(BASE/'CONTINUOUS_LEFT_STATE.json',state)
plan=BASE/'CONTINUOUS_LEFT_PLAN.md';plan.write_text(f'LATEST USER AUTHORIZATION {now}: One round10 blackboard prompt002 rerun authorized; prepared at {NEW.relative_to(BASE).as_posix()}. Same inputs and sampling as001, prompt-only. Read actual state; never duplicate submission. Audit greenboard/tree, gray gaps, structure and motion, archive/compare/report then STOP, no round11/12 or auto retry.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dep,indent=2))

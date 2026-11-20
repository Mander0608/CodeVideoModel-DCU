"""Prepare the authorized prompt-only retry. Do not overwrite completed attempts."""
from pathlib import Path
import json,hashlib,shutil,datetime,sys,py_compile
C=Path(__file__).resolve().parent;OLD=C/'attempt_002_resource_queue';A=C/'attempt_003_prompt002';REV=C/'prompt_revisions/v002_complete_gray'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
assert not A.exists(),'Refuse to restage an existing attempt'
assert read(OLD/'LIVE_STATUS.json')['status']=='completed'
assert read(OLD/'result_status.json')['visual_status']=='failed_gray_completion'
manifest=read(OLD/'input_checksums.json')
for rel,h in manifest.items():
 assert sha(OLD/rel)==h,rel
 dest=A/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/rel,dest)
revision=read(REV/'revision.json');assert sha(REV/'prompt.txt')==revision['new_prompt_sha256']
shutil.copy2(REV/'prompt.txt',A/'inputs/prompt.txt')
stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S');now=datetime.datetime.now().astimezone().isoformat()
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_r4_joint_prompt002_{stamp}'
olddep=read(OLD/'deployment.json')
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r4joint_p002_{stamp}','status':'prepared_to_launch','prepared_at':now,'local_attempt':str(A),'queue_policy':olddep['queue_policy'],'prompt_revision':'v002_complete_gray','parent_attempt':OLD.name}
save(A/'deployment.json',dep)
(A/'launch.sh').write_text((OLD/'launch.sh').read_text(encoding='utf-8').replace(olddep['remote_case'],remote),encoding='utf-8',newline='\n')
review=read(OLD/'inputs/prompt_render_review.json')
review.update(reviewed_at=now,reviewed_by='Codex: new prompt reviewed against unchanged reference and F0/31/62/92/123 proxy views',sha256={'prompt':sha(A/'inputs/prompt.txt'),'video':sha(A/'inputs/partial_proxy.mp4'),'reference':sha(A/'inputs/reference.png')})
review['observations'] += ['Prompt v002 makes complete generation of all gray unknown regions the primary task in summary and detailed description.','Retention explicitly excludes gray placeholders; left rear wall, upper ceiling, lower floor and supported furniture continuations match actual gray coverage.','Motion/layout priority and reference appearance role remain; gray completion is requested from first missing region to final frame.','This is a prompt-only RGB conditioning retry authorized by the user; actual mask_input remains false. Visual review verifies instruction/input consistency, not expected model success.']
review['clause_review'].update(summary_retention='Primary objective is complete unknown-region generation. Retention applies only to non-gray observed content.',completion='Explicit every-frame/edge/corner completion, connected geometry, texture, light and temporal consistency. No flat recoloring or cutout boundary.',motion='Same left turn, holds, rightward window motion and right window exit as before.')
save(A/'inputs/prompt_render_review.json',review)
sys.path.insert(0,str(A));from prompt_render_review import require_prompt_render_review
require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
same=['inputs/partial_proxy.mp4','inputs/reference.png','cvm_config.json','run_cvm.py','preflight.py','supervise.py','resource_probe.py','watch_and_collect.py']
for rel in same:assert sha(A/rel)==sha(OLD/rel),rel
save(A/'provenance/prompt_only_comparison.json',{'parent_attempt':str(OLD),'changed_model_condition':'Entire prompt revised as requested; not an individual-sentence ablation','old_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'new_prompt_sha256':sha(A/'inputs/prompt.txt'),'identical_file_sha256':{r:sha(A/r) for r in same},'parent_visual_outcome':'failed_gray_completion','scope':'Single-seed prompt comparison; no guaranteed improvement or general causal claim'})
for name in ['prompt.txt','revision.json','README.md']:
 dest=A/'provenance/prompt_revision_v002'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(REV/name,dest)
shutil.copy2(C/'logs/prompt002_resource_check.json',A/'provenance/current_remote_resource_check.json')
deploy=(C/'deploy_queue_once.py').read_text(encoding='utf-8').replace('attempt_002_resource_queue','attempt_003_prompt002')
deploy=deploy.replace('logs/remote_resource_check.json','logs/prompt002_resource_check.json')
deploy=deploy.replace("assert resources['exit_code']==0  # Prior source/runtime verification; not an availability claim.","assert resources['exit_code']==0\nassert (datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(resources['checked_at'])).total_seconds()<600")
(C/'deploy_prompt002_once.py').write_text(deploy,encoding='utf-8')
shutil.copy2(C/'deploy_prompt002_once.py',A/'provenance/deploy_prompt002_once.py');shutil.copy2(Path(__file__),A/'provenance/prepare_prompt002_retry.py')
for p in list(A.glob('*.py'))+[C/'deploy_prompt002_once.py']:py_compile.compile(str(p),doraise=True)
save(A/'input_checksums.json',{p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['deployment.json','input_checksums.json']})
save(C/'STATE.json',{'status':'prepared_to_launch','attempt':A.name,'deployment':dep,'scope':'Authorized new-prompt Round4 rerun only','previous_attempt':OLD.name,'previous_visual_status':'failed_gray_completion'})
print(json.dumps(dep,indent=2));print('FROZEN_FILES',len(read(A/'input_checksums.json')))

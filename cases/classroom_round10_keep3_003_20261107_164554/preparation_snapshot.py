"""One targeted video-conditioning duration control after prompt002 diagnosis."""
from pathlib import Path
import json,hashlib,shutil,datetime
from prompt_render_review import require_prompt_render_review
BASE=Path(__file__).resolve().parent;OLD=BASE/'round10/05_cvm/attempt_002_blackboard_prompt';NEW=BASE/'round10/05_cvm/attempt_003_keep3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not NEW.exists()
assert read(OLD/'timing.json')['exit_code']==0
assert read(OLD/'analysis/visual_review.json')['accepted_for_modeling'] is False
for rel,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items():assert sha(OLD/rel)==h,rel
require_prompt_render_review(OLD/'inputs/prompt_render_review.json',prompt=OLD/'inputs/prompt.txt',video=OLD/'inputs/partial_proxy.mp4',reference=OLD/'inputs/reference.png')
shutil.copytree(OLD/'inputs',NEW/'inputs')
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');now=datetime.datetime.now().astimezone().isoformat();remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round10_keep3_003_{stamp}'
oldremote=read(OLD/'deployment.json')['remote_case']
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','launch.sh','watch_and_collect.py']:(NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(oldremote,remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');assert cfg['keep_steps']==5
cfg.update(status='prepared_to_launch',keep_steps=3,keep_steps_semantics='Video-reference conditioning only for first3of20denoising updates; not spatial pixel preservation.',notes='Targetedkeep_steps5->3 control afterprompt002 returned greenboard but retained gray untilF120. Same prompt/image/video/seed42/steps20. Audit completion and landmark/motion regressions; one control then reassess.')
cfg['chain_protocol']['sampling']['keep_steps']=3
save(NEW/'cvm_config.json',cfg)
same={p.name:sha(p) for p in (OLD/'inputs').iterdir() if p.is_file()}
for name,h in same.items():assert sha(NEW/'inputs'/name)==h,name
assert sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py')
save(NEW/'paired_manifest.json',{'parent_attempt':OLD.relative_to(BASE).as_posix(),'only_changed_model_parameter':{'keep_steps':{'from':5,'to':3}},'same_input_sha256':same,'runner_identical':True,'scene_snapshot_identical':True,'same_model_and_sampling_as_round1':False,'deviation':'keep_steps3 rather than5; totalsteps20/seed42 unchanged','hypothesis':'Reducing duration of gray-bearing RGB reference may release gray-copy; may degrade known layout. Not proven.','retry_limit':'One targetedkeep3control, then consolidate evidence before further changes.'})
shutil.copy2(OLD/'analysis/conditioning_diagnosis.json',NEW/'conditioning_diagnosis.json')
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
require_prompt_render_review(NEW/'inputs/prompt_render_review.json',prompt=NEW/'inputs/prompt.txt',video=NEW/'inputs/partial_proxy.mp4',reference=NEW/'inputs/reference.png')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r10_k3_003_{stamp}','status':'prepared_to_launch','prepared_at':now};save(NEW/'deployment.json',dep)
s=read(BASE/'CONTINUOUS_LEFT_STATE.json');s.update(status='round10_keep3_control_prepared',round10='prompt002_rejected_keep3_control_prepared',current_run=str(NEW),current_run_relative=NEW.relative_to(BASE).as_posix(),current_deployment=dep,updated_at=now)
s['prepared_prompt_revision']['status']='prompt002_completed_keep3_control_prepared';s['active_control']={'parent':'round10/05_cvm/attempt_002_blackboard_prompt','change':'keep_steps5->3 only','reason':'Greenboard returned; temporal gray-copy and morphing remain. See002diagnosis.','after':'Audit; if failed consolidate before further changes; if accepted continue11/12.'};save(BASE/'CONTINUOUS_LEFT_STATE.json',s)
plan=BASE/'CONTINUOUS_LEFT_PLAN.md';plan.write_text(f'LATEST {now}: prompt002 completed16:39:57 exit0,1510.760s,124frames/SHA verified. Greenboard/tree return improved; gray persistsF27..120 and abruptly fillsF121, board/desks morph; motion chain invalid56->64, endpoint6inliers. Rejected for propagation;001/002 preserved,928filearchive verified. One keep_steps5->3 control prepared at round10/05_cvm/attempt_003_keep3, same reviewedprompt/image/video/seed42/20steps; weaker gray-bearing video condition hypothesis, potential layout regression. Read state for actual launch. No blind sweep; reassess on failure, continue11/12 only after accepted result.\n\n'+plan.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dep,indent=2))

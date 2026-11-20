"""Prepare only the newly authorized natural-joins prompt control, once."""
from pathlib import Path
import json,hashlib,shutil,datetime,difflib
B=Path(__file__).resolve().parent
OLD=B/'round9/05_cvm/attempt_005_prompt_only'
NEW=B/'round9/05_cvm/attempt_006_natural_completion'
SOURCE=B/'round9/08_user_analysis/prompt_natural_completion_v006.txt'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not NEW.exists()
assert read(OLD/'timing.json')['status']=='completed'
assert all(sha(OLD/p)==h for p,h in read(OLD/'ARCHIVE_CHECKSUMS.json').items())
shutil.copytree(OLD/'inputs',NEW/'inputs');(NEW/'baseline').mkdir()
for name in ['prompt.txt','prompt_adaptation.json','input_manifest.json']:shutil.copy2(OLD/'inputs'/name,NEW/'baseline'/name)
shutil.copy2(SOURCE,NEW/'inputs/prompt.txt')
prompt=(NEW/'inputs/prompt.txt').read_text(encoding='utf-8')
(NEW/'prompt.diff').write_text(''.join(difflib.unified_diff((OLD/'inputs/prompt.txt').read_text(encoding='utf-8').splitlines(True),prompt.splitlines(True),fromfile='attempt005/prompt.txt',tofile='attempt006/prompt.txt')),encoding='utf-8')
save(NEW/'inputs/prompt_adaptation.json',{'baseline_attempt':'attempt_005_prompt_only','changed_model_conditions':['prompt.txt'],'user_requested':True,'purpose':'Fill gray regions into one continuous classroom; preserve known objects/layout while allowing localized natural blending at joins.','source':str(SOURCE),'source_sha256':sha(SOURCE)})
m=read(NEW/'inputs/input_manifest.json');m['status']='prompt_only_control_prepared_not_run'
for name,rec in m['files'].items():rec.update(sha256=sha(NEW/'inputs'/name),bytes=(NEW/'inputs'/name).stat().st_size)
save(NEW/'inputs/input_manifest.json',m)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');now=datetime.datetime.now().astimezone().isoformat()
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round9_prompt006_{stamp}'
oldremote=read(OLD/'deployment.json')['remote_case']
for name in ['run_cvm.py','preflight.py','scene_snapshot_reference.json','check_prelaunch_resources.py']:shutil.copy2(OLD/name,NEW/name)
for name in ['supervise.py','watch_and_collect.py','launch.sh']:
 (NEW/name).write_text((OLD/name).read_text(encoding='utf-8').replace(oldremote,remote),encoding='utf-8',newline='\n')
cfg=read(OLD/'cvm_config.json');cfg.update(status='prepared_to_launch',notes='New user-authorized prompt-only control006: natural gray-region completion and localized seam blending while preserving known objects/layout. Same i8/video/seed42/20steps/keep5 as001 and005. No masks/pasteback. Only this inference authorized; main chain stays stopped.')
save(NEW/'cvm_config.json',cfg)
fixed={p.name:sha(p) for p in (OLD/'inputs').iterdir() if p.is_file() and p.name not in ['prompt.txt','prompt_adaptation.json','input_manifest.json']}
assert all(sha(NEW/'inputs'/p)==h for p,h in fixed.items())
assert sha(NEW/'run_cvm.py')==sha(OLD/'run_cvm.py')
save(NEW/'paired_manifest.json',{'baseline_attempt':'../attempt_005_prompt_only','additional_baseline':'../attempt_001','changed_model_conditions':['prompt.txt'],'all_other_input_files_identical':fixed,'runner_identical':True,'runner_sha256':sha(NEW/'run_cvm.py'),'parameters':{k:cfg[k] for k in ['seed','steps','keep_steps','frames','fps','width','height']},'independent_reference_images':1,'reference':'i8','mask_input':False,'camera_parameter_input':False,'pasteback':False,'baseline_prompt_sha256':sha(OLD/'inputs/prompt.txt'),'candidate_prompt_sha256':sha(NEW/'inputs/prompt.txt'),'limits':'One seed; multiple wording changes, only prompt input changed. No training-distribution or calibrated pose claims.'})
(NEW/'README.md').write_text('''# User-authorized round9 natural-completion prompt006

Only this additional inference is authorized. Main chain remains stopped; no round11/12, no automatic retry/reference replacement.
Same raw Blender partial video, one i8 reference, seed42/20steps/keep5 and runner as original001 and005. Only prompt changes. Source draft is prompt_natural_completion_v006.txt.
Deploy once via deploy_round9_prompt006.py after resource check. Never rerun preparation. Collector only downloads existing results.
After completion run audit_cvm_attempt.py --round 9 --attempt attempt_006_natural_completion. Adapt diagnose_round9_motion.py to006, never overwrite original001. Compare001/005/006 completion, seams, motion and structure separately, visually inspect first/intermediate/end frames, retain all input/config/actualprompt/code/environment/timing/logs/output/SHA256. Archive and report then stop. Do not rerun hardcoded005 archivers.
''',encoding='utf-8')
shutil.copy2(Path(__file__),NEW/'preparation_snapshot.py')
for p in NEW.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
save(NEW/'input_checksums.json',{p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()})
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r9prompt006_{stamp}','status':'prepared_to_launch','prepared_at':now};save(NEW/'deployment.json',dep)
s=read(B/'CONTINUOUS_LEFT_STATE.json');save(B/'STATE_BEFORE_PROMPT006.json',s)
s['previous_user_experiment']=s['pending_user_experiment'];s['pending_user_experiment']={'path':NEW.relative_to(B).as_posix(),'status':'prepared_to_launch','round':9,'baseline':'round9/05_cvm/attempt_005_prompt_only','launch_script':'deploy_round9_prompt006.py','requested_at':now,'replace_chain_reference_automatically':False,'after_completion':'Collect/audit/compare001+005/report then stop. No retry or round11/12.'}
s.update(status='user_authorized_round9_prompt006_only',updated_at=now,continue_main_chain=False)
save(B/'CONTINUOUS_LEFT_STATE.json',s)
p=B/'CONTINUOUS_LEFT_PLAN.md';p.write_text(f'LATEST USER AUTHORIZATION {now}: User approved one new run using natural-completion prompt006. Prepared round9/05_cvm/attempt_006_natural_completion; not submitted yet. Same inputs/sampling as005, prompt only. Run deploy_round9_prompt006.py once after resources checked. Main chain remains stopped; no round11/12 or automatic retries. Audit/compare with001 and005, report then stop. Read actual state for deployment.\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'prepared':str(NEW),'identical_input_files':len(fixed),'prompt_sha256':sha(NEW/'inputs/prompt.txt'),'deployment':dep},indent=2))

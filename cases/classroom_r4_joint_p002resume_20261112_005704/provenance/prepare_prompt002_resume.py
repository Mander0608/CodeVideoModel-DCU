"""Resume authorized prompt002 after the earlier resource queue expired without inference."""
from pathlib import Path
import json,hashlib,datetime,shutil,sys,py_compile
C=Path(__file__).resolve().parent;OLD=C/'attempt_003_prompt002';A=C/'attempt_004_prompt002_resume'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
assert not A.exists(),'Never overwrite an existing attempt'
oldstate=read(OLD/'LIVE_STATUS.json');t=oldstate['timing']
assert oldstate['status']=='failed' and t['inference_started_at'] is None and oldstate['denoise_steps_completed']==0
assert oldstate.get('collected_at') and oldstate['remote_output_files']==0
for rel,h in read(OLD/'input_checksums.json').items():
 assert sha(OLD/rel)==h,rel
 dest=A/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/rel,dest)
now=datetime.datetime.now().astimezone();stamp=now.strftime('%Y%m%d_%H%M%S')
remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_r4_joint_p002resume_{stamp}'
olddep=read(OLD/'deployment.json')
dep={'remote_host':'scnet2','remote_case':remote,'screen_session':f'cvm_r4joint_p002r_{stamp}','status':'prepared_to_launch','prepared_at':now.isoformat(),'local_attempt':str(A),'prompt_revision':'v002_complete_gray','parent_attempt':OLD.name,'queue_policy':olddep['queue_policy']}
save(A/'deployment.json',dep)
(A/'launch.sh').write_text((OLD/'launch.sh').read_text(encoding='utf-8').replace(olddep['remote_case'],remote),encoding='utf-8',newline='\n')
same=['inputs/prompt.txt','inputs/partial_proxy.mp4','inputs/reference.png','cvm_config.json','run_cvm.py','preflight.py','supervise.py','resource_probe.py','watch_and_collect.py']
assert sha(A/'inputs/prompt.txt')==sha(C/'prompt_revisions/v002_complete_gray/prompt.txt')
for rel in same:assert sha(A/rel)==sha(OLD/rel),rel
sys.path.insert(0,str(A));from prompt_render_review import require_prompt_render_review
require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
save(A/'provenance/resume_lineage.json',{'previous_attempt':OLD.name,'previous_timing':t,'previous_inference_updates':0,'same_file_sha256':{r:sha(A/r) for r in same},'review_reused':'Exact reviewed prompt/video/reference hashes unchanged','scope':'Resume one authorized Round4 prompt002 inference; no subsequent rounds'})
shutil.copy2(C/'logs/prompt002_resource_check.json',A/'provenance/resume_resource_check.json')
deploy=(C/'deploy_prompt002_once.py').read_text(encoding='utf-8').replace('attempt_003_prompt002','attempt_004_prompt002_resume')
(C/'deploy_prompt002_resume_once.py').write_text(deploy,encoding='utf-8')
shutil.copy2(Path(__file__),A/'provenance/prepare_prompt002_resume.py');shutil.copy2(C/'deploy_prompt002_resume_once.py',A/'provenance/deploy_prompt002_resume_once.py')
for p in list(A.glob('*.py'))+[C/'deploy_prompt002_resume_once.py']:py_compile.compile(str(p),doraise=True)
save(A/'input_checksums.json',{p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['deployment.json','input_checksums.json']})
save(C/'STATE.json',{'status':'prepared_to_launch','attempt':A.name,'deployment':dep,'prior_attempt':'attempt003 timed out without inference','scope':'Only revised Round4 prompt002'})
print(json.dumps(dep,indent=2));print('FROZEN_FILES',len(read(A/'input_checksums.json')))

"""Deploy this frozen independent attempt once; no global experiment-state writes."""
from pathlib import Path
import json,hashlib,datetime,tarfile,subprocess,sys,os,re
C=Path(__file__).resolve().parent;A=C/'attempt_005_prompt003'
sys.path.insert(0,str(A));from prompt_render_review import require_prompt_render_review
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
D=read(A/'deployment.json');R=D['remote_case'];S=D['screen_session']
assert re.fullmatch(r'/[A-Za-z0-9_/-]+',R) and re.fullmatch(r'[A-Za-z0-9_]+',S)
assert D['status']=='prepared_to_launch' and not (A/'started_at.txt').exists() and not (A/'submission_intent.json').exists()
require_prompt_render_review(A/'inputs/prompt_render_review.json',prompt=A/'inputs/prompt.txt',video=A/'inputs/partial_proxy.mp4',reference=A/'inputs/reference.png')
for rel,h in read(A/'input_checksums.json').items():assert hashlib.sha256((A/rel).read_bytes()).hexdigest()==h,rel
resources=read(C/'logs/prompt003_resource_check.json')
# Current availability is checked by the queue, then preflight, before inference.
assert resources['exit_code']==0
assert (datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(resources['checked_at'])).total_seconds()<600
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2']
def run(args,timeout=60,**kw):
 p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=FLAGS,**kw)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p
run(SSH+[f'test ! -e {R} && mkdir -p {R}'])
bundle=C/'attempt_005_prompt003_upload.tar'
with tarfile.open(bundle,'w') as tf:
 for p in A.iterdir():
  if p.name!='__pycache__':tf.add(p,arcname=p.name)
print('UPLOAD_BYTES',bundle.stat().st_size,flush=True)
run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(bundle),f'scnet2:{R}.tar'],timeout=300)
guard=f'test ! -e {R}/started_at.txt && test ! -e {R}/timing.json && test ! -e {R}/submission.once'
run(SSH+[guard+f' && tar -xf {R}.tar -C {R}'])
code=f"from pathlib import Path\nimport json,hashlib\np=Path({R!r});m=json.loads((p/'input_checksums.json').read_text())\nfor r,h in m.items(): assert hashlib.sha256((p/r).read_bytes()).hexdigest()==h,r\nprint('REMOTE_FROZEN_FILES_VERIFIED',len(m))\n"
result=run(SSH+['python3 -'],input=code);(A/'remote_upload_validation.log').write_text(result.stdout,encoding='utf-8');print(result.stdout,flush=True)
dry=run(SSH+[f'python3 {R}/run_cvm.py --project-root /public/home/lixin1/chenxy/CodeVideoModel'])
dry_record=json.loads(dry.stdout);assert dry_record['execute'] is False
assert dry_record['argv'][-1]==R+'/outputs/round4_joint_seed42.mp4'
assert R+'/inputs/partial_proxy.mp4' in dry_record['argv'] and R+'/inputs/reference.png' in dry_record['argv']
save(A/'remote_dryrun.json',dry_record)
intent={'time':datetime.datetime.now().astimezone().isoformat(),'remote_case':R,'screen':S,'rule':'If launch response is uncertain inspect actual remote state; never resubmit blindly.'}
with (A/'submission_intent.json').open('x',encoding='utf-8') as f:json.dump(intent,f,indent=2)
run(SSH+[guard+f' && (set -C; echo submitted > {R}/submission.once) && screen -dmS {S} bash {R}/launch.sh'])
D.update(status='submitted',submitted_at=intent['time']);save(A/'deployment.json',D)
with (A/'collector.log').open('w',encoding='utf-8') as out,(A/'collector_error.log').open('w',encoding='utf-8') as err:
 child=subprocess.Popen([sys.executable,'-u',str(A/'watch_and_collect.py')],stdout=out,stderr=err,creationflags=FLAGS)
D['local_collector_pid']=child.pid;save(A/'deployment.json',D);save(C/'STATE.json',{'status':'submitted','attempt':'attempt_005_prompt003','scope':'Only revised Round4; no Round5','deployment':D})
print(json.dumps(D,indent=2),flush=True)

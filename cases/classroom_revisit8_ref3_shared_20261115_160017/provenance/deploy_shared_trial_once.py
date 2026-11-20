"""Freeze-verified tar deployment with once-only submission and result collector."""
from pathlib import Path
import datetime, hashlib, json, os, re, shutil, subprocess, sys, tarfile
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'pilot_002_shared_trial'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2']
def run(args,timeout=60,**kwargs):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=FLAGS,**kwargs)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout
D=read(P/'deployment.json');R=D['remote_case'];S=D['screen_session']
assert D['status']=='prepared_to_launch' and not (P/'submission_intent.json').exists()
assert re.fullmatch(r'/[A-Za-z0-9_/-]+',R) and re.fullmatch(r'[A-Za-z0-9_]+',S)
for r,h in read(P/'frozen_manifest.json').items():assert sha(P/r)==h,r
# Resource checks are repeated while holding both locks by batch_supervise.
run(SSH+[f'test ! -e {R} && mkdir -p {R}'])
bundle=ROOT/'pilot_002_shared_trial_upload.tar'
with tarfile.open(bundle,'w') as tf:
    for f in P.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts:tf.add(f,arcname=f.relative_to(P).as_posix())
print('UPLOAD_BYTES',bundle.stat().st_size,flush=True)
run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(bundle),f'scnet2:{R}.tar'],timeout=300)
run(SSH+[f'test ! -e {R}/submission.once && tar -xf {R}.tar -C {R}'])
verify=f"from pathlib import Path\nimport hashlib,json\np=Path({R!r})\nm=json.loads((p/'frozen_manifest.json').read_text())\nfor r,h in m.items():assert hashlib.sha256((p/r).read_bytes()).hexdigest()==h,r\nprint('REMOTE_FROZEN_FILES_VERIFIED',len(m))\n"
out=run(SSH+['python3 -'],input=verify);(P/'remote_upload_validation.log').write_text(out,encoding='utf-8');print(out,flush=True)
dry=[]
for job in read(P/'batch_config.json')['jobs']:
    result=json.loads(run(SSH+[f'python3 {R}/code/run_case.py --case {R}/{job}']))
    assert result['execute'] is False and result['image_count']==(3 if job.endswith('three') else 1)
    dry.append(result)
save(P/'remote_dryruns.json',dry)
intent={'submitted_at':datetime.datetime.now().astimezone().isoformat(),'remote_case':R,'screen':S,'rule':'Inspect remote state on ambiguous response; never resubmit blindly.'}
with (P/'submission_intent.json').open('x',encoding='utf-8') as f:json.dump(intent,f,indent=2)
run(SSH+[f'test ! -e {R}/batch_started.json && (set -C; echo submitted > {R}/submission.once) && screen -dmS {S} bash {R}/launch.sh'])
D.update(status='submitted',submitted_at=intent['submitted_at']);save(P/'deployment.json',D)
with (P/'collector.log').open('w',encoding='utf-8') as out,(P/'collector_error.log').open('w',encoding='utf-8') as err:
    child=subprocess.Popen([sys.executable,'-u',str(P/'code/watch_pilot.py')],stdout=out,stderr=err,creationflags=FLAGS)
D['local_collector_pid']=child.pid;save(P/'deployment.json',D)
save(ROOT/'STATE.json',{'status':'submitted_pending_resource_verification','pilot':'pilot_002_shared_trial','deployment':D,'strategy':'exploration_contiguous3_revisit_destination_start_intermediate'})
print(json.dumps(D,indent=2),flush=True)

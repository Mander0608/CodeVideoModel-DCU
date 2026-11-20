"""One GPU pair, serial immutable jobs; three lanes run concurrently."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, re, subprocess, time, traceback

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/public/home/lixin1/chenxy');LAUNCHER=BASE/'run_cvm_dcu.sh'
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
parser=argparse.ArgumentParser();parser.add_argument('--lane',required=True);arg=parser.parse_args()
cfg=read(ROOT/'batch_config.json');lane=cfg['lanes'][arg.lane]
statepath=ROOT/'status'/(arg.lane+'.json');statepath.parent.mkdir(exist_ok=True)
with (ROOT/'status'/(arg.lane+'.once')).open('x') as f:f.write(now())
os.environ['HIP_VISIBLE_DEVICES']=','.join(map(str,lane['devices']))
state=dict(status='starting',lane=arg.lane,devices=lane['devices'],pid=os.getpid(),started_at=now(),jobs={j:dict(status='pending') for j in lane['jobs']})
save(statepath,state);locks=[];rc=1
try:
    for job in lane['jobs']:
        case=ROOT/'cases'/job;js=state['jobs'][job];queue=time.monotonic()
        js.update(status='waiting_resources',queued_at=now());state.update(status='waiting_resources',current_job=job)
        while True:
            if time.monotonic()-queue>cfg['queue_max_seconds']:raise TimeoutError('Resource wait exceeded for '+job)
            try:
                for i in lane['devices']:
                    f=(BASE/f'worldcrafter_dcu{i}.lock').open('a');locks.append(f);fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                for f in locks:f.close()
                locks=[];js['queue_reason']='device_lock_busy';save(statepath,state);time.sleep(30);continue
            probe=subprocess.run(['bash',str(LAUNCHER),str(ROOT/'code/resource_probe.py')],capture_output=True,text=True,timeout=180)
            (case/'resource_probe.log').write_text(probe.stdout+probe.stderr)
            if probe.returncode:raise RuntimeError('Resource probe failed: '+job)
            free=json.loads(next(x[10:] for x in probe.stdout.splitlines() if x.startswith('FREE_GIB: ')))
            js.update(free_GiB=free,last_resource_check=now());save(statepath,state)
            if min(free)>cfg['min_free_GiB']:break
            for f in locks:f.close()
            locks=[];time.sleep(30)
        cmd=['bash',str(LAUNCHER),str(ROOT/'code/run_case.py'),'--job',job]
        with (case/'run.log').open('x') as log:
            js.update(status='preflight',queue_seconds=time.monotonic()-queue);state['status']='preflight';save(statepath,state)
            subprocess.run(cmd+['--preflight'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
            js.update(status='running',inference_started_at=now());state['status']='running'
            start=time.monotonic();child=subprocess.Popen(cmd+['--execute'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            js['inference_pid']=child.pid;save(statepath,state)
            try:rc=child.wait(timeout=cfg['per_job_timeout_seconds'])
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=60);raise TimeoutError('Job deadline exceeded '+job)
        js.update(exit_code=rc,elapsed_seconds=time.monotonic()-start,finished_at=now());(case/'exit.code').write_text(str(rc))
        if rc:raise RuntimeError('Inference failed '+job)
        logtext=(case/'run.log').read_text();steps=re.findall(r'^DENOISE (\d+) seconds',logtext,re.M)
        assert list(map(int,steps))==list(range(1,21)),steps
        assert len(re.findall(r'^CONDITION_STEP ',logtext,re.M))==20
        assert (case/'outputs/effective_prompt.txt').read_text().strip()==(ROOT/read(case/'config.json')['prompt']).read_text().strip()
        for rel,h in read(ROOT/'frozen_manifest.json').items():assert sha(ROOT/rel)==h,rel
        save(case/'output_checksums.json',{p.relative_to(case).as_posix():sha(p) for p in (case/'outputs').rglob('*') if p.is_file()})
        js.update(status='completed',denoise_steps=20,visual_review='pending');save(case/'timing.json',js);save(statepath,state)
        for f in locks:f.close()
        locks=[]
    state['status']='completed';rc=0
except Exception as exc:
    traceback.print_exc();state.update(status='failed',error=str(exc));rc=1
    if state.get('current_job'):
        state['jobs'][state['current_job']].update(status='failed',error=str(exc));save(ROOT/'cases'/state['current_job']/'timing.json',state['jobs'][state['current_job']])
finally:
    for f in locks:f.close()
    state.update(finished_at=now(),exit_code=rc);save(statepath,state);print('LANE_FINISHED',json.dumps(state),flush=True)
raise SystemExit(rc)

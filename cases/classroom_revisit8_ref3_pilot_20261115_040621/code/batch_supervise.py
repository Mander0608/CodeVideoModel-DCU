"""Serial once-only pilot. Wait for resources, stop on any technical failure."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, subprocess, time, traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/public/home/lixin1/chenxy')
LAUNCHER = BASE / 'run_cvm_dcu.sh'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(p, x):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(x, indent=2)); tmp.replace(p)

assert not (ROOT / 'batch_started.json').exists()
for r,h in read(ROOT/'frozen_manifest.json').items(): assert sha(ROOT/r)==h,r
cfg=read(ROOT/'batch_config.json');os.environ['HIP_VISIBLE_DEVICES']='0,1'
state={'status':'waiting_resources','started_at':now(),'supervisor_pid':os.getpid(),'jobs':{j:{'status':'pending'} for j in cfg['jobs']}}
save(ROOT/'batch_started.json',state);save(ROOT/'batch_status.json',state)
locks=[];rc=1;queued_total=0.0
try:
    for job in cfg['jobs']:
        a=ROOT/job; assert not (a/'started_at.txt').exists()
        js=state['jobs'][job];js.update(status='waiting_resources',queued_at=now());state.update(status='waiting_resources',current_job=job)
        queue_start=time.monotonic()
        while True:
            save(ROOT/'batch_status.json',state)
            if queued_total+time.monotonic()-queue_start>cfg['queue_max_seconds']:raise TimeoutError('Batch resource queue budget expired; no inference launched for '+job)
            try:
                for i in cfg['devices']:
                    f=(BASE/f'worldcrafter_dcu{i}.lock').open('a');locks.append(f)
                    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                for f in locks:f.close()
                locks=[];js['queue_reason']='lock_busy';time.sleep(60);continue
            ready=True
            for k in range(2):
                p=subprocess.run(['bash',str(LAUNCHER),str(ROOT/'code/resource_probe.py')],capture_output=True,text=True,timeout=120)
                (a/'resource_probe_latest.log').write_text(p.stdout+'\n'+p.stderr)
                if p.returncode:raise RuntimeError('Resource probe failed for '+job)
                free=json.loads(next(l[10:] for l in p.stdout.splitlines() if l.startswith('FREE_GIB: ')))
                js.update(free_GiB=free,last_resource_check=now(),queue_reason='Need >60 GiB free on both DCUs');save(ROOT/'batch_status.json',state)
                if min(free)<=cfg['min_free_GiB_per_device']:ready=False;break
                if k==0:time.sleep(30)
            if ready:break
            for f in locks:f.close()
            locks=[];time.sleep(60)
        queued_total+=time.monotonic()-queue_start
        js.update(status='preflight',queue_seconds=time.monotonic()-queue_start);state.update(status='preflight',total_queue_seconds=queued_total);save(ROOT/'batch_status.json',state)
        cmd=['bash',str(LAUNCHER),str(ROOT/'code/run_case.py'),'--case',str(a)]
        with (a/'run.log').open('w') as log:
            subprocess.run(cmd+['--preflight'],stdout=log,stderr=subprocess.STDOUT,check=True)
            with (a/'started_at.txt').open('x') as f:f.write(now())
            js.update(status='running',inference_started_at=now());state['status']='running'
            began=time.monotonic();child=subprocess.Popen(cmd+['--execute'],stdout=log,stderr=subprocess.STDOUT)
            js['inference_pid']=child.pid;save(ROOT/'batch_status.json',state);rc=child.wait()
        js.update(exit_code=rc,finished_at=now(),elapsed_seconds=time.monotonic()-began)
        (a/'exit.code').write_text(str(rc))
        if rc:raise RuntimeError('Inference failed; stopping batch at '+job)
        text=(a/'run.log').read_text();steps=re.findall(r'^DENOISE (\d+) seconds',text,re.M)
        assert list(map(int,steps))==list(range(1,21)),steps
        assert len(re.findall(r'^CONDITION_STEP ',text,re.M))==20
        assert (a/'outputs/effective_prompt.txt').read_text().strip()==(a/'inputs/prompt.txt').read_text().strip()
        for r,h in read(ROOT/'frozen_manifest.json').items():assert sha(ROOT/r)==h,r
        hashes={str(p.relative_to(a)):sha(p) for p in (a/'outputs').rglob('*') if p.is_file()}
        save(a/'remote_output_checksums.json',hashes)
        # Matched target noise and shared-image row hashes must agree within a pair.
        if job.endswith('_three'):
            one=read(ROOT/job.replace('_three','_single')/'outputs/condition_audit.json');three=read(a/'outputs/condition_audit.json')
            for k in ['target_initial_noise_sha256','audio_initial_noise_sha256']:assert one['conditions'][k]==three['conditions'][k],k
            for ref in one['references']:
                shared=next(r for r in three['references'] if r['sha256']==ref['sha256'])
                assert shared['row_sha256']==ref['row_sha256'],'Shared image latent/noise mismatch'
            save(a/'paired_noise_check.json',{'status':'passed','target_noise_identical':True,'shared_reference_rows_identical':True})
        js.update(status='completed',denoise_steps=20,remote_output_files=len(hashes),visual_acceptance='pending')
        save(a/'timing.json',js);save(ROOT/'batch_status.json',state)
        for f in locks:f.close()
        locks=[]
    state['status']='completed';rc=0
except Exception as exc:
    traceback.print_exc();state.update(status='failed',error=str(exc))
    if state.get('current_job'):
        js=state['jobs'][state['current_job']];js.update(status='failed',error=str(exc))
        a=ROOT/state['current_job'];save(a/'timing.json',js)
        if (a/'outputs').exists():save(a/'remote_output_checksums.json',{str(p.relative_to(a)):sha(p) for p in (a/'outputs').rglob('*') if p.is_file()})
    rc=1
finally:
    for f in locks:f.close()
    state.update(finished_at=now(),exit_code=rc);save(ROOT/'batch_status.json',state)
    print('BATCH_FINISHED',json.dumps(state),flush=True)
raise SystemExit(rc)

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
cfg=read(ROOT/'batch_config.json');os.environ['HIP_VISIBLE_DEVICES']='1'
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
                js.update(free_GiB=free,last_resource_check=now(),queue_reason='Single-card trial: need >=55 GiB free on DCU1');save(ROOT/'batch_status.json',state)
                if not all(f>=m for f,m in zip(free,cfg['minimum_free_GiB'])):ready=False;break
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
        # Keep runtime completion distinct from strict paired-comparison validity.
        one=read(ROOT/'baseline_single/condition_audit.json');three=read(a/'outputs/condition_audit.json')
        noise_same=all(one['conditions'][k]==three['conditions'][k] for k in ['target_initial_noise_sha256','audio_initial_noise_sha256'])
        row_checks=[]
        for ref in one['references']:
            shared=next(r for r in three['references'] if r['sha256']==ref['sha256'])
            row_checks.append({'image_sha256':ref['sha256'], 'seed_identical':shared['noise_seed']==ref['noise_seed'],
                               'rows_identical':shared['row_sha256']==ref['row_sha256'],
                               'baseline_rows_sha256':ref['row_sha256'], 'trial_rows_sha256':shared['row_sha256']})
        paired=noise_same and all(x['rows_identical'] and x['seed_identical'] for x in row_checks)
        save(a/'paired_noise_check.json', {'status':'passed' if paired else 'mismatch_not_valid_strict_pair',
             'target_audio_noise_identical':noise_same,'shared_reference_checks':row_checks,
             'interpretation':'Single-card feasibility is separate from causal effect comparison.'})
        js['strict_paired_audit']='passed' if paired else 'mismatch_not_valid_strict_pair'
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

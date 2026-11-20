"""Collect the one submitted pilot; never submit, restart or alter inference."""
from pathlib import Path
import datetime, hashlib, json, os, re, subprocess, time, traceback
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'deployment.json').read_text(encoding='utf-8'));R=D['remote_case']
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2']
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args,timeout=60):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=FLAGS)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
def collect(rel,local,recursive=False):
    args=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if recursive:args.append('-r')
    run(args+[f'scnet2:{R}/{rel}',str(local)],timeout=240)

with (ROOT/'collector.lock').open('x') as f:f.write(str(os.getpid()))
seen=set();began=time.monotonic()
try:
    while time.monotonic()-began<43200:
        try:
            s=json.loads(run(SSH+[f'cat {R}/batch_status.json']))
            save(ROOT/'batch_status.json',s)
            for job,js in s['jobs'].items():
                if js['status'] in ['pending']:continue
                a=ROOT/job
                if js['status'] in ['running','completed','failed','preflight']:
                    try:collect(job+'/run.log',a)
                    except Exception:pass
                log=(a/'run.log').read_text(encoding='utf-8',errors='replace') if (a/'run.log').exists() else ''
                js['denoise_steps_observed']=len(re.findall(r'^DENOISE \d+ seconds',log,re.M))
                phases=[x for x in log.splitlines() if re.match(r'^(PIN_|MODEL_READY|ENCODE_|REFERENCE_READY|DENOISE|DECODE_|RUN_COMPLETE|Traceback)',x)]
                js['phase']=phases[-1][:500] if phases else js['status']
                if js['status'] in ['completed','failed'] and job not in seen:
                    for name in ['resource_budget.json','preflight.json','timing.json','exit.code','started_at.txt','remote_output_checksums.json','paired_noise_check.json']:
                        try:collect(job+'/'+name,a)
                        except Exception:pass
                    hashes=read(a/'remote_output_checksums.json') if (a/'remote_output_checksums.json').exists() else {}
                    if hashes:collect(job+'/outputs',a,True)
                    for rel,h in hashes.items():assert sha(a/rel)==h,rel
                    if js['status']=='completed':
                        assert hashes and js['denoise_steps_observed']==20
                        meta=read(a/'outputs/result.run.json');assert meta['sha256']==sha(a/'outputs/result.mp4')
                        assert (a/'outputs/effective_prompt.txt').read_text(encoding='utf-8').strip()==(a/'inputs/prompt.txt').read_text(encoding='utf-8').strip()
                        cfg=read(a/'config.json');aud=read(a/'outputs/condition_audit.json')
                        assert [x['sha256'] for x in aud['references']]==[x['sha256'] for x in cfg['references']]
                    report={'status':js['status'],'remote_output_files':len(hashes),'all_remote_output_hashes_match':True,
                            'collected_at':datetime.datetime.now().astimezone().isoformat(),'visual_acceptance':'pending'}
                    save(a/'COLLECTION_STATUS.json',report);seen.add(job)
            save(ROOT/'LIVE_STATUS.json',{'checked_at':datetime.datetime.now().astimezone().isoformat(),'collector_pid':os.getpid(),'remote':s,'collected_jobs':sorted(seen)})
            print(json.dumps({'status':s['status'],'current':s.get('current_job'),'collected':sorted(seen)}),flush=True)
            if s['status'] in ['completed','failed']:
                assert all(j in seen for j,x in s['jobs'].items() if x['status'] in ['completed','failed'])
                print('COLLECTION_COMPLETE',flush=True);break
        except Exception:traceback.print_exc()
        time.sleep(45)
    else:save(ROOT/'collector_timeout.json',{'reason':'12-hour collection window ended; no inference restarted or terminated.'})
finally:(ROOT/'collector.lock').unlink(missing_ok=True)

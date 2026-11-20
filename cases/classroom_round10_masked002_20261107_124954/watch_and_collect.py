"""Read-only watcher: collect logs/timing and final outputs, never launch/restart inference."""
from pathlib import Path
import datetime, hashlib, json, os, re, subprocess, time, traceback
ROOT=Path(__file__).resolve().parent
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_round10_masked002_20261107_124954'
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
def run(args,timeout=60):
    return subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=FLAGS)
def write(name,data):
    p=ROOT/name; temp=p.with_suffix(p.suffix+'.tmp'); temp.write_text(json.dumps(data,indent=2),encoding='utf-8'); temp.replace(p)
def copy(rel,recursive=False):
    args=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if recursive: args.append('-r')
    args.extend([f'scnet2:{REMOTE}/{rel}',str(ROOT)])
    p=run(args,180); assert p.returncode==0,p.stderr

lock=ROOT/'collector.lock'
with lock.open('x') as out: out.write(str(os.getpid()))
started=time.monotonic()
try:
    while time.monotonic()-started<7200:
        try:
            p=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2',f'cat {REMOTE}/timing.json'])
            assert p.returncode==0,p.stderr
            timing=json.loads(p.stdout); write('timing.json',timing)
            copy('run.log')
            log=(ROOT/'run.log').read_text(encoding='utf-8',errors='replace')
            phases=[line for line in log.splitlines() if re.match(r'^(CASE_|PIN_|MODEL_READY|ENCODE_|DENOISE|DECODE_|RUN_COMPLETE|Traceback)',line)]
            steps=re.findall(r'\bDENOISE (\d+) seconds ([0-9.]+)',log,re.M)
            state={'updated_at':datetime.datetime.now().astimezone().isoformat(),'status':timing['status'],
               'phase':phases[-1] if phases else 'loading','denoise_steps_completed':len(steps),
               'collector_pid':os.getpid(),'timing':timing}
            write('LIVE_STATUS.json',state)
            print(json.dumps({'status':state['status'],'phase':state['phase']},ensure_ascii=False),flush=True)
            if timing['status'] in ('completed','failed'):
                for rel in ['exit.code','started_at.txt','inference_started_at.txt','finished_at.txt','preflight.json','mask_preflight.log']:
                    try: copy(rel)
                    except Exception as exc: print('OPTIONAL_COLLECT',rel,str(exc),flush=True)
                copy('outputs',recursive=True)
                video=ROOT/'outputs/round10_seed42.mp4'; meta=ROOT/'outputs/round10_seed42.run.json'
                if timing['status']=='completed':
                    assert video.exists() and meta.exists()
                    expected=json.loads(meta.read_text())['sha256']; actual=hashlib.sha256(video.read_bytes()).hexdigest()
                    assert actual==expected
                    state['local_video_sha256']=actual; state['remote_local_video_hash_match']=True
                state['collected_at']=datetime.datetime.now().astimezone().isoformat(); write('LIVE_STATUS.json',state)
                print('COLLECTION_COMPLETE',flush=True); break
        except Exception:
            traceback.print_exc()
        time.sleep(45)
    else:
        write('collector_timeout.json',{'reason':'Two-hour collection window ended; remote process is not terminated.'})
finally:
    lock.unlink(missing_ok=True)

"""Read-only watcher: collect logs/timing and final outputs, never launch/restart inference."""
from pathlib import Path
import datetime, hashlib, json, os, re, subprocess, time, traceback
ROOT=Path(__file__).resolve().parent
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit8_round3_endpoint002_20261107_152830'
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
            steps=re.findall(r'^DENOISE (\d+) seconds ([0-9.]+)',log,re.M)
            state={'updated_at':datetime.datetime.now().astimezone().isoformat(),'status':timing['status'],
               'phase':phases[-1] if phases else 'loading','denoise_steps_completed':len(steps),
               'collector_pid':os.getpid(),'timing':timing}
            write('LIVE_STATUS.json',state)
            print(json.dumps({'status':state['status'],'phase':state['phase']},ensure_ascii=False),flush=True)
            if timing['status'] in ('completed','failed'):
                for rel in ['exit.code','started_at.txt','inference_started_at.txt','finished_at.txt','preflight.json']:
                    try: copy(rel)
                    except Exception as exc: print('OPTIONAL_COLLECT',rel,str(exc),flush=True)
                copy('outputs',recursive=True)
                video=ROOT/'outputs/round3_seed42.mp4'; meta=ROOT/'outputs/round3_seed42.run.json'
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


# Update only this retry's state and separate result file after collection.
if (ROOT/'LIVE_STATUS.json').exists():
    st=json.loads((ROOT/'LIVE_STATUS.json').read_text(encoding='utf-8'))
    if st.get('collected_at'):
        import shutil
        base=ROOT.parents[2]
        path=base/'EXPERIMENT_STATE.json'
        state=json.loads(path.read_text(encoding='utf-8-sig'))
        completed=st['status']=='completed'
        dep=json.loads((ROOT/'deployment.json').read_text(encoding='utf-8'))
        dep.update(status=st['status'],finished_at=st['timing']['finished_at'])
        (ROOT/'deployment.json').write_text(json.dumps(dep,ensure_ascii=False,indent=2),encoding='utf-8')
        if state.get('active_run')=='round3/02_cvm/attempt_002':
            state.update(status='round3_endpoint_retry_completed_pending_review' if completed else 'round3_endpoint_retry_failed',active_run=None,current_deployment=dep,updated_at=datetime.datetime.now().astimezone().isoformat())
            state['rounds'][2]['status']='endpoint_retry_completed_pending_review' if completed else 'endpoint_retry_failed'
            if completed:
                shutil.copy2(ROOT/'outputs/round3_seed42.mp4',base/'round3/03_result/round3_endpoint_reference_output.mp4')
                state['latest_output']='round3/03_result/round3_endpoint_reference_output.mp4'
                state['review']=None
            path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')

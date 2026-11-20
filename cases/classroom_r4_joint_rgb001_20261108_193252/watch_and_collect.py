"""Collect one existing remote run; never submit or restart inference."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time,traceback
A=Path(__file__).resolve().parent
D=json.loads((A/'deployment.json').read_text(encoding='utf-8'));REMOTE=D['remote_case']
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2']
def run(args,timeout=60,**kw):
 p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=FLAGS,**kw)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p
def save(name,data):
 dest=A/name;tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(dest)
def collect(rel,recursive=False):
 cmd=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15']
 if recursive:cmd.append('-r')
 return run(cmd+[f'scnet2:{REMOTE}/{rel}',str(A)],timeout=240)
lock=A/'collector.lock'
with lock.open('x') as f:f.write(str(os.getpid()))
started=time.monotonic()
try:
 while time.monotonic()-started<10800:
  try:
   timing=json.loads(run(SSH+[f'cat {REMOTE}/timing.json']).stdout);save('timing.json',timing);collect('run.log')
   log=(A/'run.log').read_text(encoding='utf-8',errors='replace');steps=re.findall(r'^DENOISE (\d+) seconds ([0-9.]+)',log,re.M)
   phases=[l for l in log.splitlines() if re.match(r'^(CASE_|PIN_|MODEL_READY|ENCODE_|DENOISE|DECODE_|RUN_COMPLETE|Traceback)',l)]
   state={'updated_at':datetime.datetime.now().astimezone().isoformat(),'status':timing['status'],'phase':phases[-1] if phases else 'loading','denoise_steps_completed':len(steps),'collector_pid':os.getpid(),'timing':timing}
   save('LIVE_STATUS.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
   if timing['status'] in ['completed','failed']:
    for rel in ['exit.code','started_at.txt','inference_started_at.txt','finished_at.txt','preflight.json']:
     try:collect(rel)
     except Exception as e:print('OPTIONAL_COLLECTION',rel,str(e),flush=True)
    code=f"from pathlib import Path\nimport json,hashlib\np=Path({REMOTE!r})\nprint(json.dumps({{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'outputs').rglob('*') if f.is_file()}}))\n"
    hashes=json.loads(run(SSH+['python3 -'],input=code).stdout);save('remote_output_checksums.json',hashes)
    if hashes:collect('outputs',True)
    for rel,h in hashes.items():assert hashlib.sha256((A/rel).read_bytes()).hexdigest()==h,rel
    state['all_remote_output_hashes_match']=True;state['remote_output_files']=len(hashes)
    if timing['status']=='completed':
     assert timing['exit_code']==0
     meta=json.loads((A/'outputs/round4_joint_seed42.run.json').read_text())
     assert hashes['outputs/round4_joint_seed42.mp4']==meta['sha256'];assert len(steps)==20,len(steps)
     stream=next(s for s in meta['media']['streams'] if s['codec_type']=='video')
     assert (stream['width'],stream['height'],stream['avg_frame_rate'],int(stream['nb_read_frames']))==(960,544,'24/1',124)
     assert (A/'outputs/effective_prompt.txt').read_text(encoding='utf-8').strip()==(A/'inputs/prompt.txt').read_text(encoding='utf-8').strip()
     state['technical_checks_passed']=True;state['visual_acceptance']='pending; successful execution does not establish visual quality'
    state['collected_at']=datetime.datetime.now().astimezone().isoformat();save('LIVE_STATUS.json',state)
    print('COLLECTION_COMPLETE',flush=True);break
  except Exception:traceback.print_exc()
  time.sleep(45)
 else:save('collector_timeout.json',{'reason':'Three-hour window ended; no remote process was terminated or restarted.'})
finally:lock.unlink(missing_ok=True)

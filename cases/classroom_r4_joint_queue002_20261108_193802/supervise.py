"""Wait for both DCUs before one inference; never interrupt other tasks or retry inference."""
from pathlib import Path
import datetime,fcntl,json,os,socket,subprocess,time,traceback
A=Path(__file__).resolve().parent;BASE=Path('/public/home/lixin1/chenxy');LAUNCHER=BASE/'run_cvm_dcu.sh'
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save():
 p=A/'timing.tmp';p.write_text(json.dumps(state,indent=2));p.replace(A/'timing.json')
assert not (A/'started_at.txt').exists() and not (A/'outputs/round4_joint_seed42.mp4').exists()
os.environ['HIP_VISIBLE_DEVICES']='0,1';start=time.perf_counter();infer=None;rc=1;locks=[]
state={'status':'waiting_resources','started_at':now(),'host':socket.gethostname(),'devices':[0,1],'supervisor_pid':os.getpid(),'inference_started_at':None,'finished_at':None,'total_elapsed_seconds':None,'inference_process_elapsed_seconds':None,'timing_scope':'Total includes resource queue/preflight. Inference process includes loading/sampling/decoding.','queue_max_seconds':14400}
(A/'started_at.txt').write_text(state['started_at']+'\n');save();print('CASE_QUEUED',json.dumps(state),flush=True)
try:
 while True:
  if time.perf_counter()-start>14400:raise TimeoutError('Resources did not become available within four hours; no inference started')
  try:
   for i in [0,1]:
    f=(BASE/f'worldcrafter_dcu{i}.lock').open('a');locks.append(f);fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:
   for f in locks:f.close()
   locks=[];state['queue_reason']='DCU lock held by another task';state['last_resource_check']=now();save();time.sleep(60);continue
  probe=subprocess.run(['bash',str(LAUNCHER),str(A/'resource_probe.py')],capture_output=True,text=True,timeout=90)
  (A/'resource_probe_latest.log').write_text(probe.stdout+'\n'+probe.stderr)
  if probe.returncode:raise RuntimeError('Resource probe failed: '+probe.stderr)
  free=json.loads(next(l[10:] for l in probe.stdout.splitlines() if l.startswith('FREE_GIB: ')))
  state.update(last_resource_check=now(),free_GiB=free,queue_reason='Need >60 GiB free on each DCU');save();print('RESOURCE_WAIT',json.dumps(state),flush=True)
  if min(free)>60:
   time.sleep(30)
   probe2=subprocess.run(['bash',str(LAUNCHER),str(A/'resource_probe.py')],capture_output=True,text=True,timeout=90)
   if probe2.returncode:raise RuntimeError('Second resource probe failed')
   free2=json.loads(next(l[10:] for l in probe2.stdout.splitlines() if l.startswith('FREE_GIB: ')))
   state['free_GiB']=free2;save()
   if min(free2)>60:break
  for f in locks:f.close()
  locks=[];time.sleep(60)
 state.update(status='preflight',queue_elapsed_seconds=time.perf_counter()-start,resources_ready_at=now());save()
 subprocess.run(['bash',str(LAUNCHER),str(A/'preflight.py')],check=True)
 state.update(status='running',inference_started_at=now());infer=time.perf_counter();(A/'inference_started_at.txt').write_text(state['inference_started_at']+'\n');save()
 cmd=['bash',str(LAUNCHER),str(A/'run_cvm.py'),'--project-root',str(BASE/'CodeVideoModel'),'--execute']
 child=subprocess.Popen(cmd);state.update(inference_pid=child.pid,command=cmd);save();rc=child.wait();state['status']='completed' if rc==0 else 'failed'
except Exception:
 traceback.print_exc();state['status']='failed'
finally:
 state.update(finished_at=now(),total_elapsed_seconds=time.perf_counter()-start,exit_code=rc)
 if infer is not None:state['inference_process_elapsed_seconds']=time.perf_counter()-infer
 result=A/'outputs/round4_joint_seed42.run.json'
 if result.exists():state['runner_reported_elapsed_seconds']=json.loads(result.read_text()).get('elapsed_seconds')
 (A/'exit.code').write_text(str(rc)+'\n');(A/'finished_at.txt').write_text(state['finished_at']+'\n');save();print('CASE_FINISHED',json.dumps(state),flush=True)
 for f in locks:f.close()
raise SystemExit(rc)

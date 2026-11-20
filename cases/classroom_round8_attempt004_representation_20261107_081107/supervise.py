"""Record start/end/elapsed even when inference fails; hold the usual DCU locks."""
from pathlib import Path
import datetime, fcntl, json, os, socket, subprocess, time, traceback
ROOT=Path(__file__).resolve().parent
BASE=Path('/public/home/lixin1/chenxy')
LAUNCHER=BASE/'run_cvm_dcu.sh'
def now(): return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(state):
    tmp=ROOT/'timing.tmp'; tmp.write_text(json.dumps(state,indent=2)); tmp.replace(ROOT/'timing.json')

assert not (ROOT/'started_at.txt').exists(),'Refuse to rerun in an existing run directory'
assert not (ROOT/'outputs/round8_seed42.mp4').exists()
locks=[]
for i in [0,1]:
    f=(BASE/f'worldcrafter_dcu{i}.lock').open('a')
    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(f)
os.environ['HIP_VISIBLE_DEVICES']='0,1'
tick=time.perf_counter(); inference_tick=None; rc=1
state={'status':'preflight','started_at':now(),'host':socket.gethostname(),'devices':[0,1],
    'supervisor_pid':os.getpid(),'inference_started_at':None,'finished_at':None,
    'total_elapsed_seconds':None,'inference_process_elapsed_seconds':None,
    'timing_scope':'Total includes preflight. Inference process includes model loading, conditioning, sampling, decoding and output verification.'}
(ROOT/'started_at.txt').write_text(state['started_at']+'\n'); save(state)
print('CASE_START',json.dumps(state),flush=True)
try:
    subprocess.run(['bash',str(LAUNCHER),str(ROOT/'preflight.py')],check=True)
    state['status']='running'; state['inference_started_at']=now(); inference_tick=time.perf_counter()
    (ROOT/'inference_started_at.txt').write_text(state['inference_started_at']+'\n'); save(state)
    cmd=['bash',str(LAUNCHER),str(ROOT/'run_cvm.py'),'--project-root',str(BASE/'CodeVideoModel'),'--execute']
    child=subprocess.Popen(cmd); state['inference_pid']=child.pid; state['command']=cmd; save(state)
    rc=child.wait()
    state['inference_process_elapsed_seconds']=time.perf_counter()-inference_tick
    state['status']='completed' if rc==0 else 'failed'
except Exception:
    traceback.print_exc(); state['status']='failed'
finally:
    state['finished_at']=now(); state['total_elapsed_seconds']=time.perf_counter()-tick; state['exit_code']=rc
    if inference_tick is not None and state['inference_process_elapsed_seconds'] is None:
        state['inference_process_elapsed_seconds']=time.perf_counter()-inference_tick
    result=ROOT/'outputs/round8_seed42.run.json'
    if result.exists(): state['runner_reported_elapsed_seconds']=json.loads(result.read_text()).get('elapsed_seconds')
    (ROOT/'exit.code').write_text(str(rc)+'\n'); (ROOT/'finished_at.txt').write_text(state['finished_at']+'\n'); save(state)
    print('CASE_FINISHED',json.dumps(state),flush=True)
raise SystemExit(rc)

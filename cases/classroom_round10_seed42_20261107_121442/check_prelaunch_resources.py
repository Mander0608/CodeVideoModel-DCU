"""Read-only DCU prelaunch resource/source check; no inference submitted."""
from pathlib import Path
import argparse,subprocess,json,datetime,os
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--round',type=int,required=True);n=a.parse_args().round
code='''import torch,json,fcntl,hashlib,datetime,socket
from pathlib import Path
base=Path('/public/home/lixin1/chenxy');locks=[]
for i in [0,1]:
 f=(base/f'worldcrafter_dcu{i}.lock').open('a')
 fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert torch.cuda.device_count()==2
free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in [0,1]]
assert min(free)>60,free
expected={'cvm/video.py':'98b451d987812d6f1caec3c36208fa280866bf40be803cd4474813ad0db40e4c','run_dcu.py':'a46644c0e095eff2ee9046b08c1cb1d3096b3e61d0232dd5783fbea42e6d671f'}
actual={k:hashlib.sha256((base/'CodeVideoModel'/k).read_bytes()).hexdigest() for k in expected}
assert actual==expected,actual
print('RESOURCE_JSON '+json.dumps({'remote_time':datetime.datetime.now().astimezone().isoformat(),'host':socket.gethostname(),'free_GiB':free,'locks_available_at_check':True,'torch':torch.__version__,'hip':torch.version.hip,'source_sha256':actual}))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2','bash /public/home/lixin1/chenxy/run_cvm_dcu.sh -'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
log=Path(__file__).resolve().parent/'prelaunch_resource';log.mkdir(exist_ok=True)
(log/'remote_resource_check.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
p.check_returncode()
report=json.loads(next(line[len('RESOURCE_JSON '):] for line in p.stdout.splitlines() if line.startswith('RESOURCE_JSON ')))
report.update(checked_at=datetime.datetime.now().astimezone().isoformat(),remote_host='scnet2',exit_code=p.returncode,scope='Read-only prelaunch probe; locks released after check. Supervisor reacquires atomically before launch.')
(log/'remote_resource_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

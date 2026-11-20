"""Fresh authorized single-card retry on free physical DCU1; preserve old attempts."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, py_compile, shutil

B=Path(__file__).resolve().parents[1]
OLD=B/'pilot_003_single_gpu'
P=B/'pilot_004_single_gpu'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not P.exists(),'Do not overwrite or resubmit an existing attempt'
assert read(OLD/'TRIAL_RESULT.json')['status']=='resource_wait_expired_inference_not_started'
for r,h in read(OLD/'frozen_manifest.json').items():assert sha(OLD/r)==h,r
P.mkdir()
for r in read(OLD/'frozen_manifest.json'):
    dst=P/r;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/r,dst)
for rel,old,new in [
    ('code/run_case.py',"os.environ.get('HIP_VISIBLE_DEVICES') == '0'","os.environ.get('HIP_VISIBLE_DEVICES') == '1'"),
    ('code/batch_supervise.py',"os.environ['HIP_VISIBLE_DEVICES']='0'","os.environ['HIP_VISIBLE_DEVICES']='1'"),
    ('code/batch_supervise.py','free on DCU0','free on DCU1')]:
    f=P/rel;s=f.read_text(encoding='utf-8');assert s.count(old)==1,old;f.write_text(s.replace(old,new),encoding='utf-8')
cfg=read(P/'r02_three/config.json');cfg['resource_policy']['physical_devices']=[1]
cfg['resource_policy']['authorization']='User explicitly authorized retrying the single-card experiment.'
save(P/'r02_three/config.json',cfg)
now=datetime.datetime.now().astimezone();stamp=now.strftime('%Y%m%d_%H%M%S')
batch=read(P/'batch_config.json');batch.update(devices=[1],created_at=now.isoformat(),parent_pilot=OLD.name)
save(P/'batch_config.json',batch)
dep={'status':'prepared_to_launch','remote_host':'scnet2',
     'remote_case':f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_ref3_single_gpu_retry_{stamp}',
     'screen_session':f'cvm_ref3_1gpu_retry_{stamp}','local_pilot':str(P),'prepared_at':now.isoformat(),
     'scope':'One R2 three-reference run on physical DCU1 only; no automatic OOM retries'}
save(P/'deployment.json',dep)
(P/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport HIP_VISIBLE_DEVICES=1\ncd '+dep['remote_case']+'\nexec >> batch.log 2>&1\nexec python3 -u code/batch_supervise.py\n',encoding='utf-8',newline='\n')
def load(f,name):
    spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
native=(P/'provenance/shared_cvm_video.py').read_text(encoding='utf-8')
assert load(P/'code/run_case.py','new').adapted_source(native)==load(OLD/'code/run_case.py','old').adapted_source(native)
same=['inputs/proxy.mp4','inputs/prompt.txt']+[r['path'] for r in cfg['references']]
for r in same:assert sha(P/'r02_three'/r)==sha(OLD/'r02_three'/r)
save(P/'provenance/retry_lineage.json',{'parent':str(OLD),'physical_device':1,'logical_device':'cuda:0',
    'changed':'Physical GPU binding 0 to 1 only; all generation inputs and algorithm unchanged.',
    'same_input_sha256':{r:sha(P/'r02_three'/r) for r in same},'effective_generate_unchanged':True,
    'known_limit':'Shared-reference tensor hash mismatch in pilot002 remains unresolved; runtime success is not a causal effect claim.'})
shutil.copy2(Path(__file__),P/'provenance/prepare_single_gpu_retry.py')
deploy=(B/'code/deploy_single_gpu_once.py').read_text(encoding='utf-8').replace('pilot_003_single_gpu','pilot_004_single_gpu')
(B/'code/deploy_single_gpu_retry_once.py').write_text(deploy,encoding='utf-8')
shutil.copy2(B/'code/deploy_single_gpu_retry_once.py',P/'provenance/deploy_single_gpu_retry_once.py')
for f in (P/'code').glob('*.py'):py_compile.compile(str(f),doraise=True)
manifest={f.relative_to(P).as_posix():sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['deployment.json','frozen_manifest.json']}
save(P/'frozen_manifest.json',manifest)
save(B/'STATE.json',{'status':'prepared_single_gpu_retry','pilot':P.name,'deployment':dep,'resource_policy':cfg['resource_policy']})
print(json.dumps({'deployment':dep,'frozen_files':len(manifest)},indent=2))

"""One authorized R2 three-image feasibility trial under bounded shared-card memory."""
from pathlib import Path
import ast, datetime, hashlib, importlib.util, json, py_compile, shutil

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'pilot_001';P=ROOT/'pilot_002_shared_trial'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
assert not P.exists(),'Never overwrite/restart an existing attempt'
state=read(OLD/'LIVE_STATUS.json')['remote']
assert state['status']=='failed'
assert state['jobs']['r02_single']['status']=='completed'
assert 'inference_started_at' not in state['jobs']['r02_three']
assert read(OLD/'r02_three/COLLECTION_STATUS.json')['remote_output_files']==0
for r,h in read(OLD/'frozen_manifest.json').items():assert sha(OLD/r)==h,r
P.mkdir()
for r in read(OLD/'frozen_manifest.json'):
    if r.startswith(('code/','provenance/','r02_three/')) or r=='shared_source_checksums.json':
        dst=P/r;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/r,dst)
base=P/'baseline_single';base.mkdir()
for name in ['condition_audit.json','result.run.json','launch_record.json']:
    shutil.copy2(OLD/'r02_single/outputs'/name,base/name)
save(base/'lineage.json',{'source':str(OLD/'r02_single'),'collection_status':read(OLD/'r02_single/COLLECTION_STATUS.json'),
                        'files':{p.name:sha(p) for p in base.iterdir() if p.is_file()},'purpose':'Pair audit only; do not rerun completed baseline'})
cfg=read(P/'r02_three/config.json')
policy={'physical_devices':[0,1],'minimum_free_GiB':[28.0,22.0],
        'leave_free_headroom_GiB':2.0,'allocator_absolute_max_GiB':[48.0,32.0],
        'authorization':'User explicitly asked to try running with current available card memory.',
        'scope':'One attempt only. No model/seed/steps/resolution/prompt/reference changes. No automatic OOM retry.'}
cfg['resource_policy']=policy;save(P/'r02_three/config.json',cfg)
runner=P/'code/run_case.py';s=runner.read_text(encoding='utf-8')
old='assert min(free) > 60, free'
new="""policy = cfg['resource_policy']
    assert all(f >= m for f,m in zip(free,policy['minimum_free_GiB'])), free
    total = [torch.cuda.mem_get_info(i)[1] / 2**30 for i in range(2)]
    budgets = [min(free[i]-policy['leave_free_headroom_GiB'], policy['allocator_absolute_max_GiB'][i]) for i in range(2)]
    for i, budget in enumerate(budgets):
        torch.cuda.set_per_process_memory_fraction(budget / total[i], i)
    resource_report = {'free_GiB_at_start':free,'total_GiB':total,'allocator_cap_GiB':budgets,
                       'minimum_free_GiB':policy['minimum_free_GiB'],'headroom_GiB':policy['leave_free_headroom_GiB'],
                       'scope':'PyTorch allocator cap for this process; non-PyTorch allocations and other processes are not controlled.'}
    save(a / 'resource_budget.json', resource_report)
    print('RESOURCE_BUDGET', json.dumps(resource_report), flush=True)"""
assert s.count(old)==1;s=s.replace(old,new);runner.write_text(s,encoding='utf-8')
supervisor=P/'code/batch_supervise.py';s=supervisor.read_text(encoding='utf-8')
s=s.replace("queue_reason='Need >60 GiB free on both DCUs'", "queue_reason='Authorized bounded trial: need >=28/22 GiB free on DCU0/1'")
s=s.replace("if min(free)<=cfg['min_free_GiB_per_device']:ready=False;break", "if not all(f>=m for f,m in zip(free,cfg['minimum_free_GiB'])):ready=False;break")
s=s.replace("one=read(ROOT/job.replace('_three','_single')/'outputs/condition_audit.json')", "one=read(ROOT/'baseline_single/condition_audit.json')")
assert "'minimum_free_GiB'" in s and "ROOT/'baseline_single/condition_audit.json'" in s
supervisor.write_text(s,encoding='utf-8')
watch=P/'code/watch_pilot.py';s=watch.read_text(encoding='utf-8').replace("['preflight.json','timing.json'", "['resource_budget.json','preflight.json','timing.json'")
watch.write_text(s,encoding='utf-8')
batch=read(OLD/'batch_config.json');batch.update(jobs=['r02_three'],queue_max_seconds=600,minimum_free_GiB=[28.,22.],
    status='prepared_one_authorized_memory_trial',parent_pilot='pilot_001',same_model_conditions=True)
batch.pop('min_free_GiB_per_device',None);save(P/'batch_config.json',batch)
now=datetime.datetime.now().astimezone();stamp=now.strftime('%Y%m%d_%H%M%S')
dep={'status':'prepared_to_launch','remote_host':'scnet2','remote_case':f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit8_ref3_shared_{stamp}',
     'screen_session':f'cvm_ref3_shared_{stamp}','local_pilot':str(P),'prepared_at':now.isoformat(),
     'scope':'Only unstarted R2 three-reference case; bounded allocator on occupied cards; no retries or baseline rerun'}
save(P/'deployment.json',dep)
(P/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+dep['remote_case']+'\nexec >> batch.log 2>&1\nexec python3 -u code/batch_supervise.py\n',encoding='utf-8',newline='\n')
# Prove the effective denoising function is unchanged by the resource-gate adaptation.
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
native=(P/'provenance/shared_cvm_video.py').read_text(encoding='utf-8')
assert load(runner,'new').adapted_source(native)==load(OLD/'code/run_case.py','old').adapted_source(native)
same=['inputs/proxy.mp4','inputs/prompt.txt']+[r['path'] for r in cfg['references']]
for r in same:assert sha(P/'r02_three'/r)==sha(OLD/'r02_three'/r)
save(P/'provenance/resource_trial_lineage.json',{'parent_pilot':str(OLD),'parent_case':'r02_three','previous_inference_updates':0,
    'same_input_sha256':{r:sha(P/'r02_three'/r) for r in same},'effective_generate_unchanged':True,
    'only_execution_changes':['resource gate 60/60 GiB to 28/22 GiB','per-process allocator cap free-minus-2 GiB, bounded at48/32','one job instead of four','paired noise checked against already completed baseline'],
    'policy':policy})
shutil.copy2(Path(__file__),P/'provenance/prepare_shared_trial.py')
deploy=(ROOT/'code/deploy_pilot_once.py').read_text(encoding='utf-8').replace('pilot_001','pilot_002_shared_trial')
(ROOT/'code/deploy_shared_trial_once.py').write_text(deploy,encoding='utf-8');shutil.copy2(ROOT/'code/deploy_shared_trial_once.py',P/'provenance/deploy_shared_trial_once.py')
for f in (P/'code').glob('*.py'):py_compile.compile(str(f),doraise=True)
manifest={f.relative_to(P).as_posix():sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['deployment.json','frozen_manifest.json']}
save(P/'frozen_manifest.json',manifest)
save(ROOT/'STATE.json',{'status':'prepared_bounded_shared_trial','pilot':P.name,'deployment':dep,'completed_baseline':str(OLD/'r02_single'),'resource_policy':policy})
print(json.dumps({'deployment':dep,'frozen_files':len(manifest),'policy':policy},indent=2))

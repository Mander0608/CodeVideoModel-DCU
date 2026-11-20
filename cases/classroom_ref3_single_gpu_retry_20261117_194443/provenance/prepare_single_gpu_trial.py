"""Prepare one isolated single-DCU R2 three-reference feasibility run."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, py_compile, shutil

B = Path(__file__).resolve().parents[1]
OLD = B / 'pilot_002_shared_trial'
P = B / 'pilot_003_single_gpu'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def replace(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

assert not P.exists(), 'Do not overwrite or restart a prepared attempt'
assert read(OLD/'TRIAL_RESULT.json')['status'] == 'failed_oom_first_denoise'
for name in ['pilot_001', 'pilot_002_shared_trial']:
    for rel, h in read(B/name/'frozen_manifest.json').items():
        assert sha(B/name/rel) == h, (name, rel)
P.mkdir()
for name in ['run_case.py', 'batch_supervise.py', 'resource_probe.py', 'watch_pilot.py']:
    dst=P/'code'/name; dst.parent.mkdir(exist_ok=True)
    shutil.copy2(OLD/'code'/name, dst)
for folder in ['r02_three/inputs', 'r02_three/review', 'baseline_single']:
    shutil.copytree(OLD/folder, P/folder)
shutil.copy2(OLD/'shared_source_checksums.json', P/'shared_source_checksums.json')
(P/'provenance').mkdir()
for name in ['shared_cvm_video.py','shared_run_dcu.py']:
    shutil.copy2(OLD/'provenance'/name, P/'provenance'/name)
shutil.copy2(B/'provenance/shared_cvm_video_original_bytes.py', P/'provenance/shared_cvm_video_original_bytes.py')
cfg=read(OLD/'r02_three/config.json')
policy={'physical_devices':[0], 'minimum_free_GiB':[55.0],
        'leave_free_headroom_GiB':2.0, 'allocator_absolute_max_GiB':[60.0],
        'text_device':'cuda:0', 'generation_device':'cuda:0',
        'authorization':'User explicitly requested a single-card experiment.',
        'scope':'One R2 three-reference attempt; unchanged inputs/model/seed/schedule/resolution. No automatic OOM retry.'}
cfg['resource_policy']=policy
cfg['execution_scope']='Single-card feasibility; strict effect comparison requires separate paired encoding audit.'
save(P/'r02_three/config.json',cfg)

runner=P/'code/run_case.py'; s=runner.read_text(encoding='utf-8')
s=replace(s, 'assert torch.cuda.device_count() == 2', "assert torch.cuda.device_count() == 1\n    assert os.environ.get('HIP_VISIBLE_DEVICES') == '0'")
assert s.count('range(2)') == 5
s=s.replace('range(2)', 'range(1)')
s=replace(s, "text_device='cuda:1', generation_device='cuda:0'", "text_device='cuda:0', generation_device='cuda:0'")
s=replace(s, "refaudit, condaudit = [], {}", """save(output.parent / 'runtime_environment.json', {
        'visible_devices': os.environ.get('HIP_VISIBLE_DEVICES'),
        'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0),
        'text_device': 'cuda:0', 'generation_device': 'cuda:0',
        'python': sys.executable, 'torch_file': torch.__file__,
        'torch': torch.__version__, 'hip': torch.version.hip,
        'offload': 'Existing text leaf CPU offload and generation block CPU offload unchanged'})
    refaudit, condaudit = [], {}""")
runner.write_text(s,encoding='utf-8')

probe=P/'code/resource_probe.py'; s=probe.read_text(encoding='utf-8')
s=s.replace('device_count() == 2', 'device_count() == 1').replace('range(2)', 'range(1)')
probe.write_text(s,encoding='utf-8')
sup=P/'code/batch_supervise.py'; s=sup.read_text(encoding='utf-8')
s=replace(s, "os.environ['HIP_VISIBLE_DEVICES']='0,1'", "os.environ['HIP_VISIBLE_DEVICES']='0'")
s=s.replace('Authorized bounded trial: need >=28/22 GiB free on DCU0/1', 'Single-card trial: need >=55 GiB free on DCU0')
start=s.index('        # Matched target noise')
end=s.index("        js.update(status='completed'",start)
s=s[:start]+'''        # Keep runtime completion distinct from strict paired-comparison validity.
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
'''+s[end:]
sup.write_text(s,encoding='utf-8')
now=datetime.datetime.now().astimezone(); stamp=now.strftime('%Y%m%d_%H%M%S')
batch=read(OLD/'batch_config.json')
batch.update(created_at=now.isoformat(),devices=[0],minimum_free_GiB=[55.0],queue_max_seconds=600,
             status='prepared_single_gpu_trial',parent_pilot=OLD.name)
save(P/'batch_config.json',batch)
dep={'status':'prepared_to_launch','remote_host':'scnet2',
     'remote_case':f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_ref3_single_gpu_{stamp}',
     'screen_session':f'cvm_ref3_1gpu_{stamp}','local_pilot':str(P),'prepared_at':now.isoformat(),
     'scope':'One R2 three-reference single physical DCU0 run; no automatic retries'}
save(P/'deployment.json',dep)
(P/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport HIP_VISIBLE_DEVICES=0\ncd '+dep['remote_case']+'\nexec >> batch.log 2>&1\nexec python3 -u code/batch_supervise.py\n',encoding='utf-8',newline='\n')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
native=(P/'provenance/shared_cvm_video.py').read_text(encoding='utf-8')
assert load(runner,'single').adapted_source(native)==load(OLD/'code/run_case.py','dual').adapted_source(native)
same=['inputs/proxy.mp4','inputs/prompt.txt']+[r['path'] for r in cfg['references']]
for rel in same: assert sha(P/'r02_three'/rel)==sha(OLD/'r02_three'/rel)
save(P/'provenance/single_gpu_lineage.json',{
    'parent':str(OLD),'input_sha256':{r:sha(P/'r02_three'/r) for r in same},
    'effective_generate_source_unchanged':True,'runtime_device_count_required':1,
    'changes':['Only physical GPU0 visible/locked','text_device cuda:1 to cuda:0',
               'minimum free55GiB, allocator cap min(free-minus2,60)GiB',
               'Known paired-row mismatch reported independently from inference completion',
               'Runtime paths and device binding recorded'],
    'known_limitation':'Previous dual trial shared-reference rows mismatch remains unresolved; no strict effect claim.',
    'resource_policy':policy})
shutil.copy2(Path(__file__),P/'provenance/prepare_single_gpu_trial.py')
deploy=(B/'code/deploy_shared_trial_once.py').read_text(encoding='utf-8').replace('pilot_002_shared_trial','pilot_003_single_gpu')
(B/'code/deploy_single_gpu_once.py').write_text(deploy,encoding='utf-8')
shutil.copy2(B/'code/deploy_single_gpu_once.py',P/'provenance/deploy_single_gpu_once.py')
for f in (P/'code').glob('*.py'): py_compile.compile(str(f),doraise=True)
manifest={f.relative_to(P).as_posix():sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['deployment.json','frozen_manifest.json']}
save(P/'frozen_manifest.json',manifest)
save(B/'STATE.json',{'status':'prepared_single_gpu_trial','pilot':P.name,'deployment':dep,
                   'resource_policy':policy,'previous_trial_result':str(OLD/'TRIAL_RESULT.json')})
print(json.dumps({'deployment':dep,'frozen_files':len(manifest),'policy':policy},indent=2))

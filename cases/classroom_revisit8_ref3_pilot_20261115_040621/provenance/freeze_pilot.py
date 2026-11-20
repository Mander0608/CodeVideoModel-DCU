from pathlib import Path
import ast, datetime, hashlib, importlib.util, json, py_compile, shutil

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'pilot_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
assert not (P/'frozen_manifest.json').exists()
assert read(ROOT/'logs/layout_check.json')['status']=='passed'
(P/'code').mkdir();(P/'provenance').mkdir()
for name in ['run_case.py','resource_probe.py','batch_supervise.py','watch_pilot.py','check_layout.py']:
    shutil.copy2(ROOT/'code'/name,P/'code'/name)
    py_compile.compile(str(P/'code'/name),doraise=True)
for f in (ROOT/'provenance').iterdir():
    if f.is_file():shutil.copy2(f,P/'provenance'/f.name)
for name in ['layout_check.json','preparation_probe.json']:
    shutil.copy2(ROOT/'logs'/name,P/'provenance'/name)
expected={'cvm/video.py':sha(ROOT/'provenance/shared_cvm_video_original_bytes.py'),
          'run_dcu.py':sha(ROOT/'provenance/shared_run_dcu.py')}
assert expected=={'cvm/video.py':'98b451d987812d6f1caec3c36208fa280866bf40be803cd4474813ad0db40e4c','run_dcu.py':'a46644c0e095eff2ee9046b08c1cb1d3096b3e61d0232dd5783fbea42e6d671f'}
save(P/'shared_source_checksums.json',expected)
now=datetime.datetime.now().astimezone()
for job in read(P/'batch_config.json')['jobs']:
    a=P/job;c=read(a/'config.json')
    review={'status':'passed','reviewed_at':now.isoformat(),'reviewed_by':'Codex','inspected_proxy_frames':[0,31,62,92,123],
            'reference_sha256':[r['sha256'] for r in c['references']], 'prompt_sha256':sha(a/c['prompt']),
            'proxy_sha256':sha(a/c['proxy']),'nominal_angles_are_not_calibrated_output_poses':True,
            'observations':[
                'All actual references and the actual proxy F0/31/62/92/123 reviewed in contact sheets.',
                'R2: windows/chalkboard move right while new left region appears gray. R1 terminal triplet has nearly identical composition.',
                'R3: windows/chalkboard move left during rightward revisit. Three-reference roles explicitly destination/start/intermediate.',
                'Known photo/proxy differences remain in furniture geometry, window patch boundaries, exposure and blackboard appearance. Fixed paired inputs preserve these risks.',
                'No first-frame hard lock, masks or output pasteback. Input/prompt consistency is reviewed, not a guarantee of model success.'
            ]}
    save(a/'review/input_review.json',review)
single=read(P/'r02_single/config.json')['references'][0]
three=read(P/'r02_three/config.json')['references'][-1]
assert single['sha256']==three['sha256'] and single['noise_stream']==three['noise_stream']
for n in [2,3]:
    s=read(P/f'r{n:02d}_single/config.json');t=read(P/f'r{n:02d}_three/config.json')
    assert s['source_proxy_sha256']==t['source_proxy_sha256']
    for field in ['seed','steps','keep_steps','fps','frames','width','height','reference_short_edge_override']:
        assert s[field]==t[field]
shutil.copy2(Path(__file__),P/'provenance/freeze_pilot.py')
shutil.copy2(ROOT/'code/prepare_pilot.py',P/'provenance/prepare_pilot.py')
shutil.copy2(ROOT/'code/deploy_pilot_once.py',P/'provenance/deploy_pilot_once.py')
stamp=now.strftime('%Y%m%d_%H%M%S')
dep={'status':'prepared_to_launch','remote_host':'scnet2',
     'remote_case':f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_revisit8_ref3_pilot_{stamp}',
     'screen_session':f'cvm_revisit8_ref3_{stamp}','local_pilot':str(P),'prepared_at':now.isoformat(),
     'scope':'Four independent frozen-input paired cases, serial; stop on technical failure; no later rounds without review'}
save(P/'deployment.json',dep)
(P/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+dep['remote_case']+'\nexec >> batch.log 2>&1\nexec python3 -u code/batch_supervise.py\n',encoding='utf-8',newline='\n')
manifest={f.relative_to(P).as_posix():sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['deployment.json','frozen_manifest.json']}
save(P/'frozen_manifest.json',manifest)
print(json.dumps({'deployment':dep,'frozen_files':len(manifest)},indent=2))

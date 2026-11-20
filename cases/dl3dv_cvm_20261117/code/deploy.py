from pathlib import Path
import datetime, hashlib, importlib.util, json, py_compile, shlex, shutil, subprocess, tarfile
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/public/home/lixin1/chenxy/CodeVideoModel/cases/dl3dv_cvm_20261117'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet2']
def run(command,**kw):return subprocess.run(command,check=True,capture_output=True,text=True,encoding='utf-8',errors='replace',**kw).stdout
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
assert not (ROOT/'submission_intent.json').exists()
for path in (ROOT/'code').glob('*.py'):py_compile.compile(str(path),doraise=True)
spec=importlib.util.spec_from_file_location('runner',ROOT/'code/run_case.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
effective=module.adapt((ROOT/'provenance/server_source/cvm/video.py').read_text())
(ROOT/'provenance/generate_dryrun.py').write_text(effective)
save(ROOT/'input_review.json',dict(status='passed_for_feasibility_experiment',all_620_proxy_frames_hash_checked=True,all_5_videos_fully_decoded=True,visual_review='12 fixed sampled frames per scene; known geometry/appearance mismatches retained',no_geometry_changes=True,cache_policy='A and B share canonical per-pixel VAE latents; explicit paired noise and rows checks',planned_inference_count=10))
files=[]
for folder in ['inputs','code','provenance','cases']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(ROOT/p for p in ['source_checksums.json','batch_config.json','input_review.json'])
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
save(ROOT/'frozen_manifest.json',manifest);files.append(ROOT/'frozen_manifest.json')
bundle=ROOT/'deployment_inputs.tar'
with tarfile.open(bundle,'w') as tf:
    for p in files:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
run(SSH+[f'test ! -e {REMOTE} && mkdir -p {REMOTE}'])
run(['scp','-q','-o','BatchMode=yes',str(bundle),'scnet2:'+REMOTE+'/deployment_inputs.tar'],timeout=300)
run(SSH+[f'tar -xf {REMOTE}/deployment_inputs.tar -C {REMOTE}'])
verify=f'''from pathlib import Path
import hashlib,json
r=Path({REMOTE!r})
m=json.loads((r/'frozen_manifest.json').read_text())
for p,h in m.items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
model=Path('/public/home/lixin1/chenxy/CodeVideoModel/models/MiniMax-H3-ref2va')
items=[]
for p in model.rglob('*'):
 if p.is_file():
  item=dict(path=p.relative_to(model).as_posix(),size=p.stat().st_size,mtime=p.stat().st_mtime)
  if p.suffix in ['.json','.txt']:item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  items.append(item)
(r/'provenance/model_inventory.json').write_text(json.dumps(items,indent=2))
print('REMOTE_INPUTS_VERIFIED',len(m),'MODEL_FILES',len(items))
'''
verification=run(SSH+['python3 -'],input=verify,timeout=120);print(verification,flush=True)
(ROOT/'remote_verification.log').write_text(verification)
for job in sorted((ROOT/'cases').iterdir()):
    print(run(SSH+[f'python3 {REMOTE}/code/run_case.py --job {job.name}']),flush=True)
intent=dict(status='launching',remote_root=REMOTE,created_at=datetime.datetime.now().astimezone().isoformat(),lanes=[])
save(ROOT/'submission_intent.json',intent)
for lane in ['pair12','pair34','pair56']:
    session='dl3dv_cvm_'+lane+'_20261117'
    inner=f'exec python3 {REMOTE}/code/supervise_lane.py --lane {lane} > {REMOTE}/{lane}.log 2>&1'
    run(SSH+['screen -dmS '+session+' bash -lc '+shlex.quote(inner)])
    intent['lanes'].append(dict(lane=lane,screen=session));save(ROOT/'submission_intent.json',intent)
intent['status']='submitted';save(ROOT/'submission_intent.json',intent)
deployment=json.loads((ROOT/'deployment.json').read_text());deployment.update(status='submitted',screens=intent['lanes'],uploaded_bytes=bundle.stat().st_size);save(ROOT/'deployment.json',deployment)
print(run(SSH+['screen -ls']),flush=True)
print('SUBMITTED_ALL_THREE_LANES',flush=True)

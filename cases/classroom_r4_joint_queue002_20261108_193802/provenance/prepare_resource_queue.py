from pathlib import Path
import json,hashlib,shutil,datetime,py_compile
C=Path(__file__).resolve().parent;OLD=C/'attempt_001';A=C/'attempt_002_resource_queue'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
oldstatus=read(OLD/'LIVE_STATUS.json')
assert oldstatus['status']=='failed' and oldstatus['timing']['inference_started_at'] is None and oldstatus['denoise_steps_completed']==0 and oldstatus.get('collected_at')
assert not A.exists()
oldmanifest=read(OLD/'input_checksums.json')
for rel,h in oldmanifest.items():
 src=OLD/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
 dest=A/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');remote=f'/public/home/lixin1/chenxy/CodeVideoModel/cases/classroom_r4_joint_queue002_{stamp}'
olddep=read(OLD/'deployment.json');dep={**olddep,'remote_case':remote,'screen_session':f'cvm_r4joint_queue_{stamp}','status':'prepared_to_launch','prepared_at':datetime.datetime.now().astimezone().isoformat(),'local_attempt':str(A),'queue_policy':'Wait for both locks and >60 GiB free twice; max four hours; one inference only'}
for k in ['submitted_at','local_collector_pid']:dep.pop(k,None)
save(A/'deployment.json',dep)
(A/'launch.sh').write_text((OLD/'launch.sh').read_text().replace(olddep['remote_case'],remote),encoding='utf-8',newline='\n')
shutil.copy2(C/'supervise_queued.py',A/'supervise.py')
(A/'resource_probe.py').write_text("import torch,json\nassert torch.cuda.device_count()==2\nprint('FREE_GIB: '+json.dumps([torch.cuda.mem_get_info(i)[0]/2**30 for i in [0,1]]))\n",encoding='utf-8')
collector=(OLD/'watch_and_collect.py').read_text(encoding='utf-8').replace('10800','21600').replace('Three-hour','Six-hour')
(A/'watch_and_collect.py').write_text(collector,encoding='utf-8',newline='\n')
save(A/'provenance/preflight_failure_parent.json',{'parent_attempt':'attempt_001','failure':'Other task allocated DCUs after initial resource check; preflight free GiB 43.703125/46.349609375','parent_timing':oldstatus['timing'],'inference_was_never_started':True,'input_prompt_video_reference_unchanged':True})
deploy=(C/'deploy_once.py').read_text(encoding='utf-8').replace("A=C/'attempt_001'","A=C/'attempt_002_resource_queue'")
deploy=deploy.replace("assert (datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(resources['checked_at'])).total_seconds()<600","# Current availability is checked by the queue, then preflight, before inference.")
deploy=deploy.replace("assert min(resources['free_GiB'])>60 and resources['locks_available_at_check']","assert resources['exit_code']==0  # Prior source/runtime verification; not an availability claim.")
deploy=deploy.replace("attempt_001_upload.tar","attempt_002_resource_queue_upload.tar").replace("'attempt':'attempt_001'","'attempt':'attempt_002_resource_queue'")
(C/'deploy_queue_once.py').write_text(deploy,encoding='utf-8')
shutil.copy2(C/'deploy_queue_once.py',A/'provenance/deploy_queue_once.py');shutil.copy2(Path(__file__),A/'provenance/prepare_resource_queue.py')
for p in list(A.glob('*.py'))+[C/'deploy_queue_once.py']:py_compile.compile(str(p),doraise=True)
save(A/'input_checksums.json',{p.relative_to(A).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in A.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['deployment.json','input_checksums.json']})
for rel in ['inputs/partial_proxy.mp4','inputs/reference.png','inputs/prompt.txt']:assert hashlib.sha256((A/rel).read_bytes()).hexdigest()==oldmanifest[rel]
save(OLD/'result_status.json',{'status':'failed_before_inference','reason':'GPU availability changed after the prelaunch probe','actual_updates':0,'next_attempt':'attempt_002_resource_queue','timing':oldstatus['timing']})
save(C/'STATE.json',{'status':'prepared_resource_queue','attempt':'attempt_002_resource_queue','prior_attempt_status':'failed_before_inference','deployment':dep})
print(json.dumps(dep,indent=2))

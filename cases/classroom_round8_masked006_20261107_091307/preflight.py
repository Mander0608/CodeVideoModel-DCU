from pathlib import Path
import sys,json,hashlib,subprocess
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
expected=json.loads((ROOT/'input_checksums.json').read_text())
for rel,digest in expected.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
assert torch.cuda.device_count()==2
free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)];assert min(free)>60,free
source={r:hashlib.sha256((PROJECT/r).read_bytes()).hexdigest() for r in ['cvm/video.py','run_dcu.py']}
assert source=={'cvm/video.py':'98b451d987812d6f1caec3c36208fa280866bf40be803cd4474813ad0db40e4c','run_dcu.py':'a46644c0e095eff2ee9046b08c1cb1d3096b3e61d0232dd5783fbea42e6d671f'}
p=subprocess.run([sys.executable,str(ROOT/'masked_inpaint.py'),'--case',str(ROOT),'--output',str(ROOT/'outputs/preflight_not_generated.mp4'),'--steps','20','--seed','42','--no-turbo','--preflight-only'],capture_output=True,text=True)
(ROOT/'mask_preflight.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8');p.check_returncode()
assert 'preflight_passed' in p.stdout
report={'status':'passed','input_checksums_match':True,'free_GiB':free,'torch':torch.__version__,'hip':torch.version.hip,'shared_source_sha256':source,'protocol':'explicit_masked_completion_not_original_cvm','mask_preflight':'mask_preflight.log','actual_updates_verified_from_scheduler':20,'turbo':False}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2));print('CASE_PREFLIGHT',json.dumps(report),flush=True)

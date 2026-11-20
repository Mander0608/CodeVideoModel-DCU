from pathlib import Path
import hashlib, json, sys
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
sys.path.insert(0,str(PROJECT))
from cvm import check_video

expected=json.loads((ROOT/'input_checksums.json').read_text())
for rel,digest in expected.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
check_video(ROOT/'inputs/partial_proxy.mp4')
assert (PROJECT/'models/MiniMax-H3-ref2va/modular_model_index.json').is_file()
assert torch.cuda.device_count()==2,torch.cuda.device_count()
free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)]
assert min(free)>60,free
used=[torch.cuda.get_device_properties(i).total_memory/2**30-free[i] for i in range(2)]
assert max(used)<0.5, f'Device occupied: {used} GiB'
assert __import__('os').environ.get('HIP_VISIBLE_DEVICES')=='4,5'
source=(PROJECT/'cvm/video.py').read_text()
assert source.count('prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE')==1
report={'status':'passed','input_checksums_match':True,'proxy_format_valid':True,'free_GiB':free,'physical_devices':[4, 5],'generation_device':'cuda:0','text_device':'cuda:1','used_GiB_before_preflight':used,
   'torch':torch.__version__,'hip':torch.version.hip,
   'shared_source_sha256':{rel:hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest() for rel in ['cvm/video.py','run_dcu.py']}}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2))
print('CASE_PREFLIGHT',json.dumps(report),flush=True)

assert report["shared_source_sha256"] == {'cvm/video.py': '98b451d987812d6f1caec3c36208fa280866bf40be803cd4474813ad0db40e4c', 'run_dcu.py': 'a46644c0e095eff2ee9046b08c1cb1d3096b3e61d0232dd5783fbea42e6d671f'}

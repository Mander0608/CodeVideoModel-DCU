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
source=(PROJECT/'cvm/video.py').read_text()
assert source.count('prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE')==1
report={'status':'passed','input_checksums_match':True,'proxy_format_valid':True,'free_GiB':free,
   'torch':torch.__version__,'hip':torch.version.hip,
   'shared_source_sha256':{rel:hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest() for rel in ['cvm/video.py','run_dcu.py']}}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2))
print('CASE_PREFLIGHT',json.dumps(report),flush=True)

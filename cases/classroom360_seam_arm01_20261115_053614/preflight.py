from pathlib import Path
import hashlib,json,os,sys
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/public/home/lixin1/chenxy/CodeVideoModel')
sys.path.insert(0,str(PROJECT))
from cvm import check_video
cfg=json.loads((ROOT/'config.json').read_text())
for rel,h in json.loads((ROOT/'input_checksums.json').read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
for rel,h in cfg['expected_source_sha256'].items():assert hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()==h,rel
for path,h in cfg['expected_external_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
model=PROJECT/'models/MiniMax-H3-ref2va'
for rel,v in cfg['model_snapshot'].items():
    p=model/rel;assert p.stat().st_size==v['size'] and p.stat().st_mtime==v['mtime'] and str(p.resolve())==v['resolved'],rel
for rel,h in cfg['model_metadata_sha256'].items():assert hashlib.sha256((model/rel).read_bytes()).hexdigest()==h,rel
assert os.environ['HIP_VISIBLE_DEVICES']==','.join(map(str,cfg['physical_devices']))
assert torch.cuda.device_count()==2
assert torch.__version__=='2.7.1' and torch.version.hip=='6.3.26093', 'Runtime changed from A'
assert len(cfg['reference_images'])==cfg['expected_image_count']
for ref in cfg['reference_images']:
    assert hashlib.sha256((ROOT/ref['path']).read_bytes()).hexdigest()==ref['sha256']
free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)]
used=[torch.cuda.get_device_properties(i).total_memory/2**30-free[i] for i in range(2)]
assert min(free)>60 and max(used)<0.5,(free,used)
check_video(ROOT/cfg['code_video'])
assert not os.environ.get('CVM_SMOKE_TEST')=='1'
report={'status':'passed','physical_devices':cfg['physical_devices'],'free_GiB':free,'used_GiB':used,'source_hashes_verified':True,'input_hashes_verified':True,'torch':torch.__version__,'hip':torch.version.hip}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2));print('PREFLIGHT',json.dumps(report),flush=True)

from pathlib import Path
import hashlib
import json
import sys
import torch
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[1]
sys.path.insert(0, str(PROJECT))
from cvm import check_video

manifest = json.loads((CASE / 'input_manifest.json').read_text())
for name, meta in manifest['files'].items():
    assert hashlib.sha256((CASE / 'inputs' / name).read_bytes()).hexdigest() == meta['sha256'], name
check_video(CASE / 'inputs/partial_proxy.mp4')
assert (PROJECT / 'models/MiniMax-H3-ref2va/modular_model_index.json').exists()
assert not (CASE / 'outputs/partial_seed42.mp4').exists()
sizes = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(2)]
assert min(sizes) > 60, sizes
m = torch.nn.Sequential(torch.nn.Linear(8, 8))
expected = m[0].weight.detach().clone()
with torch.inference_mode():
    cpu_offload(m, execution_device=torch.device('cuda:1'), offload_buffers=True)
    m(torch.ones(1, 8))
    remove_hook_from_module(m, recurse=True)
    m.to('cpu')
    assert torch.equal(m[0].weight, expected)
info = {'status': 'passed', 'inputs_sha256_match': True, 'proxy_format_valid': True,
        'free_GiB': sizes, 'cleanup_probe': 'passed', 'torch': torch.__version__, 'hip': torch.version.hip}
(CASE / 'preflight.json').write_text(json.dumps(info, indent=2))
print('CASE_PREFLIGHT', json.dumps(info), flush=True)

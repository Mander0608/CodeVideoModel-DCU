"""Reproduce inference-tensor cleanup failure, then exercise the actual fix."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module
from diffusers import ComponentsManager
from run_dcu import DcuH3

def prepare():
    with torch.inference_mode():
        module = torch.nn.Sequential(torch.nn.Linear(8, 8))
        for parameter in module.parameters():
            parameter.data = parameter.detach().pin_memory()
        expected = {k: v.clone() for k, v in module.state_dict().items()}
        cpu_offload(module, execution_device=torch.device('cuda:1'), offload_buffers=True)
        module(torch.ones(1, 8))
    return module, expected

bad, _ = prepare()
try:
    remove_hook_from_module(bad, recurse=True)
except RuntimeError as error:
    assert 'inference tensor outside InferenceMode' in str(error), str(error)
    reproduced = str(error)
else:
    raise AssertionError('Expected old cleanup failure was not reproduced')

model, expected = prepare()
runner = DcuH3('cuda:1', 'cuda:0')
manager = ComponentsManager()
manager.add('text_encoder', model)
runner.managers = [manager]
runner.conditioner = SimpleNamespace(text_encoder=model)
runner.transformer = runner.generation = None
runner.close()
assert not manager.get_ids()
assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
assert all(p.device.type == 'cpu' for p in model.parameters())
record = {'status': 'passed', 'old_failure_reproduced': reproduced,
          'actual_DcuH3_close': True, 'parameters_preserved': True,
          'manager_empty': True, 'scope': 'small model lifecycle; no full checkpoint rerun'}
Path('outputs/cleanup_validation.json').write_text(json.dumps(record, indent=2))
print('CLEANUP_INFERENCE_MODE_VALIDATION_PASSED', flush=True)

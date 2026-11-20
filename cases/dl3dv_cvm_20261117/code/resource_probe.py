import json, torch
assert torch.cuda.device_count()==2
print('FREE_GIB: '+json.dumps([torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)]),flush=True)

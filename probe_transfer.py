import gc, math, time
import torch
from safetensors import safe_open
from pathlib import Path
root=Path('models/MiniMax-H3-ref2va/text_encoder')
file=next(root.glob('model-00001-*.safetensors'))
with safe_open(file,framework='pt',device='cpu') as f:
    for key in f.keys():
        shape=f.get_slice(key).get_shape()
        if 1_000_000<=math.prod(shape)<=8_000_000:
            x=f.get_tensor(key); break
print(key,x.shape,x.dtype,'MB',x.numel()*x.element_size()/1e6,flush=True)
for label in ['mapped','clone','pinned']:
    start=time.monotonic()
    src=x if label=='mapped' else x.clone() if label=='clone' else x.pin_memory()
    print(label,'prepare',time.monotonic()-start,flush=True)
    torch.cuda.synchronize()
    start=time.monotonic()
    y=src.to('cuda')
    torch.cuda.synchronize()
    print(label,'H2D_seconds',time.monotonic()-start,'finite',bool(y.isfinite().all()),flush=True)
    del y,src
    gc.collect(); torch.cuda.empty_cache()

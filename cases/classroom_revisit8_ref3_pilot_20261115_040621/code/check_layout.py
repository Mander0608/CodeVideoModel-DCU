"""CPU-only checks against installed H3 presentation and packed-layout code; no model load."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep

class Tokenizer:
    def __init__(self): self.segments=[]
    def __call__(self, s, **kw): self.segments.append(s); return {'input_ids':list(range(len(s)))}
    def convert_tokens_to_ids(self, s): return 100000 + len(s)

results=[]
for n in (1,3):
    for with_proxy in (False,True):
        refs=[MiniMaxH3ImageReference(image=Image.new('RGB',(960,544))) for _ in range(n)]
        shapes=[(1,64,1,128,226)]*n
        if with_proxy:
            refs.append(MiniMaxH3VideoReference(frames=np.zeros((124,32,32,3),dtype=np.uint8),fps=24.,audio=None,sample_rate=None))
            shapes.append((1,64,37,34,60))
        tk=Tokenizer()
        ids,tags=MiniMaxH3Ref2VATextEncoderStep._build_presentation(tk,'TEST',refs,[4]*n,[4] if with_proxy else [],[[0.2,0.8]] if with_proxy else [])
        labels=[s for s in tk.segments if s.startswith('<Picture') or s.startswith('<Video')]
        assert labels==[f'<Picture {i+1}>: ' for i in range(n)]+(['<Video 1>: '] if with_proxy else []),labels
        latents=[torch.empty(s,device='meta') for s in shapes]
        pos,tag,vi,ai,ti,nv,na=MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            torch.tensor(tags),refs,latents,[],37,34,60,207,(1,2,2),2,2,0)
        expected=sum(s[2]*(s[3]//2)*(s[4]//2) for s in shapes)
        assert nv==expected and na==0
        assert vi.numel()==nv+37*17*30 and ai.numel()==207*2
        assert pos.shape[0]==len(ids)+vi.numel()+ai.numel()
        results.append({'n_images':n,'proxy':with_proxy,'labels':labels,'reference_rows':nv,'all_rows':int(pos.shape[0]),'passed':True})
print('LAYOUT_CHECK_JSON '+json.dumps({'status':'passed','scope':'CPU synthetic geometry and installed presentation/layout, not actual model inference or memory validation','checks':results}))

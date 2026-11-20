import torch
from flash_attn import flash_attn_func
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
torch.manual_seed(42)
with torch.inference_mode():
    for n,d in [(129,64),(128,64),(512,128),(129,128)]:
        q,k,v=[torch.randn(1,4,n,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH): ref=F.scaled_dot_product_attention(q,k,v)
        for contiguous in [False,True]:
            inp=[a.transpose(1,2).contiguous() if contiguous else a.transpose(1,2) for a in [q,k,v]]
            out=flash_attn_func(*inp,dropout_p=0.0,causal=False).transpose(1,2)
            print(n,d,contiguous,'max_error',(out-ref).abs().max().item(),'mean',out.float().mean().item(),flush=True)
    q=torch.randn(1,4,129,64,device='cuda',dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.MATH): ref=F.scaled_dot_product_attention(q,q,q)
    for contiguous in [False,True]:
        inp=[q.transpose(1,2).contiguous() if contiguous else q.transpose(1,2) for _ in range(3)]
        out=flash_attn_func(*inp,dropout_p=0.0,causal=False).transpose(1,2)
        print('alias',contiguous,(out-ref).abs().max().item(),flush=True)

import faulthandler, time
faulthandler.dump_traceback_later(90,repeat=True)
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from cvm.dcu_runtime import configure
configure()
for n in [129,1024]:
    q=torch.randn(1,64,n,128,device='cuda',dtype=torch.bfloat16)
    k,v=[torch.randn(1,8,n,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    for causal in [False,True]:
        print('GQA_START',n,causal,flush=True)
        with torch.inference_mode():
            start=time.monotonic()
            out=F.scaled_dot_product_attention(q,k,v,is_causal=causal,enable_gqa=True)
            torch.cuda.synchronize()
            print('GQA_DONE',time.monotonic()-start,flush=True)
            with sdpa_kernel(SDPBackend.MATH):
                # Call the saved original rather than the vendor bridge.
                ref=F.scaled_dot_product_attention.__wrapped__(q,k,v,is_causal=causal,enable_gqa=True)
            error=(out-ref).abs().max().item()
            print('ERROR',error,flush=True)
            assert error<.04,error
faulthandler.cancel_dump_traceback_later()
print('GQA_PASSED',flush=True)

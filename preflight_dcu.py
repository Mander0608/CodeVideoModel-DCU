"""Real DCU kernel/model-operator checks; random small model, not checkpoint inference."""
import json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from cvm.dcu_runtime import configure

start=time.monotonic()
torch.manual_seed(42)
device=torch.device('cuda:0')
report={'test_kind':'random_small_model_operator_smoke_not_pretrained_inference',
        'torch':torch.__version__,'hip':torch.version.hip,'device':str(torch.cuda.get_device_properties(0))}
q=torch.randn(1,4,129,64,device=device,dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.MATH): reference=F.scaled_dot_product_attention(q,q,q)
configure()
actual=F.scaled_dot_product_attention(q,q,q)
error=(reference.float()-actual.float()).abs().max().item()
assert error<0.04, error
report['attention_max_abs_error']=error

from diffusers import MiniMaxH3Transformer3DModel, ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from cvm.video import advance, unpatchify
model=MiniMaxH3Transformer3DModel(num_attention_heads=2,attention_head_dim=128,hidden_size=128,
    num_layers=2,num_refiner_layers=1,ffn_dim=256,text_dim=32,freq_dim=32,
    time_embed_hidden_dim=128,time_embed_dim=64).eval().to(device=device,dtype=torch.bfloat16)
for name in model._keep_in_fp32_modules:
    model.get_submodule(name).float()
inputs=dict(hidden_states=torch.randn(1,8,96,device=device,dtype=torch.bfloat16),
    audio_hidden_states=torch.randn(1,4,32,device=device,dtype=torch.bfloat16),
    encoder_hidden_states=torch.randn(1,4,32,device=device,dtype=torch.bfloat16),
    timestep=torch.tensor([0.1,0.5],device=device),
    timestep_indices=torch.tensor([0]*8+[1]*8,device=device),
    token_tags=torch.tensor([0]*8+[1]*4+[2]*4,device=device),
    position_ids=torch.arange(16,device=device).view(-1,1).expand(-1,3),
    video_indices=torch.arange(8,device=device),audio_indices=torch.arange(12,16,device=device),
    text_indices=torch.arange(8,12,device=device),return_dict=False)
with torch.inference_mode(): baseline=model(**inputs)
assert all(torch.isfinite(x).all() for x in baseline)
model.to('cpu')
model.enable_group_offload(onload_device=device,offload_device=torch.device('cpu'),
                          offload_type='block_level',num_blocks_per_group=1,use_stream=False)
with torch.inference_mode(): result=model(**inputs)
errors=[(a.float()-b.float()).abs().max().item() for a,b in zip(baseline,result)]
assert max(errors)<0.04, errors
report['h3_output_shapes']=[list(x.shape) for x in result]
report['group_offload_max_abs_errors']=errors
latent=torch.randn(1,24,3,4,6,device=device)
packed=patchify_video_latents(latent,(1,2,2))
assert torch.equal(latent,unpatchify(packed,latent.shape,(1,2,2)))
updated=advance(latent,torch.zeros_like(latent),.5,torch.tensor(.8),torch.tensor(.4))
assert torch.equal(updated,latent)
report['pack_round_trip']='passed'
report['integration_zero_velocity']='passed'
report['elapsed_seconds']=time.monotonic()-start
report['peak_memory_GiB']=torch.cuda.max_memory_allocated()/2**30
Path('outputs').mkdir(exist_ok=True)
Path('outputs/preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
print('DCU_OPERATOR_PREFLIGHT_PASSED',flush=True)

"""Process-local DCU adaptation; does not alter checkpoints or sampling schedule."""
import functools
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def configure():
    if torch.version.hip is None or getattr(F.scaled_dot_product_attention, '_cvm_dcu', False):
        return
    from flash_attn import flash_attn_func
    original = F.scaled_dot_product_attention
    @functools.wraps(original)
    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        q, k, v = query, key, value
        eligible = (q.is_cuda and q.ndim == k.ndim == v.ndim == 4
                    and q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
                    and attn_mask is None and q.shape[-1] <= 256
                    and q.shape[-1] == k.shape[-1] == v.shape[-1]
                    and q.shape[0] == k.shape[0] == v.shape[0]
                    and k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
                    and (q.shape[1] == k.shape[1] or enable_gqa)
                    and (not is_causal or q.shape[-2] == k.shape[-2]))
        if eligible:
            # Vendor kernel miscomputes aliased noncontiguous Q/K/V views.
            return flash_attn_func(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
                                   dropout_p=dropout_p, softmax_scale=scale, causal=is_causal).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    sdpa._cvm_dcu = True
    F.scaled_dot_product_attention = sdpa
    print('[cvm-dcu] vendor FlashAttention bridge enabled', flush=True)

"""
Memory-bounded scaled-dot-product attention for the SDPA fallback.

Without ``flash_attn``, PyTorch's ``scaled_dot_product_attention`` is the only
option — and on a ROCm build with no AOTriton kernels for the GPU it silently
selects the *math* backend, which materialises the full ``[B, H, Lq, Lkv]``
score matrix. The shape stages here run tens of thousands of sparse tokens, so
that matrix is measured in terabytes and the run dies at the first block.

Splitting the query dimension fixes it: each chunk holds only
``[B, H, chunk, Lkv]`` scores and the results concatenate exactly. When a real
fused kernel *is* available the chunking costs a few percent and nothing else,
so this is used unconditionally on the SDPA path rather than guessing which
backend torch picked.
"""

from typing import *
import os

import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa


__all__ = ['chunked_sdpa', 'query_chunk_size']


# Score-matrix elements to hold at once, ~256 MB in fp16.
_DEFAULT_BUDGET = 1 << 27
_MIN_CHUNK = 128


def query_chunk_size(num_heads: int, kv_len: int, q_len: int) -> int:
    """
    Rows of queries to process at a time.

    ``PIXAL3D_ATTN_CHUNK`` overrides it; ``0`` disables chunking entirely for
    benchmarking against the unsplit path.
    """
    override = os.environ.get('PIXAL3D_ATTN_CHUNK')
    if override is not None:
        value = int(override)
        return q_len if value <= 0 else max(1, value)

    budget = int(os.environ.get('PIXAL3D_ATTN_BUDGET', _DEFAULT_BUDGET))
    per_row = max(1, num_heads * kv_len)
    return max(_MIN_CHUNK, min(q_len, budget // per_row))


def chunked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ``scaled_dot_product_attention`` over ``[B, H, L, C]`` tensors, computed in
    query chunks. ``attn_mask`` follows torch's broadcasting rules; a mask with
    a real query dimension is sliced along with the queries.
    """
    q_len = q.shape[-2]
    chunk = query_chunk_size(q.shape[1], k.shape[-2], q_len)
    if chunk >= q_len:
        return _sdpa(q, k, v, attn_mask=attn_mask)

    outs = []
    for start in range(0, q_len, chunk):
        stop = min(start + chunk, q_len)
        mask = attn_mask
        if mask is not None and mask.dim() >= 2 and mask.shape[-2] == q_len:
            mask = mask[..., start:stop, :]
        outs.append(_sdpa(q[:, :, start:stop], k, v, attn_mask=mask))
    return torch.cat(outs, dim=-2)

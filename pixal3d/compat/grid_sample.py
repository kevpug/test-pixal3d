"""
Pure-PyTorch replacement for ``flex_gemm.ops.grid_sample.grid_sample_3d``.

Samples a sparse voxel volume (``feats`` rows addressed by ``coords``) at
arbitrary continuous positions. Voxel ``c`` covers ``[c, c+1)`` and its centre
sits at ``c + 0.5``, which is the convention the CUDA kernel uses; the weights
and the truncation-toward-zero of the neighbour coordinates are reproduced
exactly so that output matches the fast path bit-for-bit up to float ordering.

Autograd works through this implementation for free — ``index_select`` and the
weighted sum are differentiable with respect to ``feats``.
"""

from typing import *
import torch

from .hashgrid import CoordLookup, MISS


__all__ = ['grid_sample_3d']


# Offsets of the eight trilinear neighbours, in the same order as the kernel.
_CORNERS = (
    (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
    (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5),
)

# Query points per chunk. 4096^2 texels x 8 neighbours would otherwise build a
# ~1 GB index tensor; 1M points keeps every intermediate under ~100 MB.
_CHUNK = 1 << 20


def _batch_column(n: int, b: int, l: int, device, dtype=torch.int64) -> torch.Tensor:
    return torch.arange(b, device=device, dtype=dtype).reshape(b, 1).expand(b, l).reshape(-1, 1)


def _sample_chunk(
    feats: torch.Tensor,
    lookup: CoordLookup,
    pts: torch.Tensor,        # [M, 3] float, absolute voxel-space positions
    batch: torch.Tensor,      # [M, 1] int64
    mode: str,
) -> torch.Tensor:
    c = feats.shape[1]
    m = pts.shape[0]

    if mode == 'nearest':
        query = torch.cat([batch, pts.to(torch.int32).long()], dim=1)
        idx = lookup.lookup(query)
        valid = idx != MISS
        out = feats.index_select(0, idx.clamp_min(0))
        return out * valid.unsqueeze(1).to(out.dtype)

    # Trilinear: gather the eight surrounding voxels, weight by the overlap of
    # the query point with each voxel's cell, and renormalise by the weight
    # that actually landed on an occupied voxel (surface voxels are sparse, so
    # some corners are always missing).
    offs = torch.tensor(_CORNERS, device=pts.device, dtype=pts.dtype)          # [8, 3]
    neigh = (pts.unsqueeze(1) + offs.unsqueeze(0)).to(torch.int32).long()      # [M, 8, 3]
    query = torch.cat([
        batch.unsqueeze(1).expand(m, 8, 1).reshape(-1, 1),
        neigh.reshape(-1, 3),
    ], dim=1)                                                                  # [M*8, 4]

    idx = lookup.lookup(query)
    valid = idx != MISS

    centres = neigh.reshape(-1, 3).to(pts.dtype) + 0.5                         # [M*8, 3]
    dist = (centres - pts.repeat_interleave(8, dim=0)).abs()
    weight = torch.prod(torch.clamp(1.0 - dist, min=0.0), dim=-1)
    weight = torch.where(valid, weight, torch.zeros_like(weight))

    gathered = feats.index_select(0, idx.clamp_min(0))                         # [M*8, C]
    acc = (gathered * weight.unsqueeze(1).to(gathered.dtype)).reshape(m, 8, c).sum(dim=1)
    norm = weight.reshape(m, 8).sum(dim=1).clamp_min(1e-12)
    return acc / norm.unsqueeze(1).to(acc.dtype)


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    grid: torch.Tensor,
    mode: str = 'trilinear',
) -> torch.Tensor:
    """
    Sample a sparse volume at continuous positions.

    Args:
        feats: ``[N, C]`` features of the occupied voxels.
        coords: ``[N, 4]`` integer coordinates ``(b, x, y, z)`` of those voxels.
        shape: full shape ``(B, C, W, H, D)``; only the spatial tail is used.
        grid: ``[B, L, 3]`` query positions in voxel space.
        mode: ``'trilinear'`` or ``'nearest'``.

    Returns:
        ``[B, L, C]`` sampled features.
    """
    assert mode in ('nearest', 'trilinear'), f"Invalid interpolation mode: {mode}"
    assert feats.dim() == 2, f"feats must be [N, C], got {tuple(feats.shape)}"
    assert coords.dim() == 2 and coords.shape[1] == 4, f"coords must be [N, 4], got {tuple(coords.shape)}"
    assert grid.dim() == 3 and grid.shape[2] == 3, f"grid must be [B, L, 3], got {tuple(grid.shape)}"

    b, l = grid.shape[:2]
    c = feats.shape[1]
    lookup = CoordLookup(coords, shape[-3:])

    pts = grid.reshape(-1, 3).float()
    batch = _batch_column(b * l, b, l, grid.device)

    outs = []
    for i in range(0, pts.shape[0], _CHUNK):
        outs.append(_sample_chunk(feats, lookup, pts[i:i + _CHUNK], batch[i:i + _CHUNK], mode))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return out.reshape(b, l, c)

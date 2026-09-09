"""
Pure-PyTorch UV-space rasteriser — the ``nvdiffrast`` replacement.

Texture baking needs one thing from a rasteriser: for every texel of the
atlas, which triangle covers it and what 3D position does that texel map to.
``nvdiffrast`` is CUDA-only and its community ROCm port miscomputes coverage on
wave32 hardware, so this does the job with ordinary tensor ops.

Each triangle is expanded into the texels of its UV bounding box, tested with
the same barycentric predicate and ``-0.001`` tolerance the reference kernel
uses, and the survivors are scattered into the atlas. Work is chunked over the
flat (triangle, texel) candidate list rather than over triangles, so peak
memory is fixed no matter how the atlas distributes area between charts — a
single chart covering most of the texture is split across chunks like any
other.
"""

from typing import *
import torch


__all__ = ['uv_rasterize']


# Candidate (triangle, texel) pairs to materialise at once. ~4M pairs is about
# 200 MB of intermediates, which leaves room on a 12 GB card that is also
# holding the attribute volume.
_PAIR_BUDGET = 1 << 22

_BARY_EPS = -0.001
_DEGENERATE_DENOM = 1e-6


@torch.no_grad()
def uv_rasterize(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    vertices: torch.Tensor,
    texture_size: int,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterise UV triangles into texture space.

    Args:
        uvs: ``[V, 2]`` UV coordinates in ``[0, 1]``.
        faces: ``[F, 3]`` vertex indices.
        vertices: ``[V, 3]`` 3D positions.
        texture_size: side length of the square atlas.
        verbose: print coverage statistics.

    Returns:
        ``face_ids`` ``[TS, TS]`` int32 — covering face index + 1, 0 where the
        atlas is empty — and ``pos`` ``[TS, TS, 3]`` float32, the interpolated
        3D position. Row index is V, column index is U, matching the
        convention the texture-baking code downstream expects.
    """
    device = uvs.device
    ts = int(texture_size)
    scale = float(ts - 1)
    f = faces.long()
    num_faces = f.shape[0]

    face_ids = torch.zeros(ts * ts, dtype=torch.int32, device=device)
    out_pos = torch.zeros(ts * ts, 3, dtype=torch.float32, device=device)
    if num_faces == 0:
        return face_ids.reshape(ts, ts), out_pos.reshape(ts, ts, 3)

    uv = torch.nan_to_num(uvs.float(), nan=0.0, posinf=1.0, neginf=0.0) * scale
    uv0, uv1, uv2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    p0, p1, p2 = vertices[f[:, 0]].float(), vertices[f[:, 1]].float(), vertices[f[:, 2]].float()

    min_uv = torch.minimum(torch.minimum(uv0, uv1), uv2)
    max_uv = torch.maximum(torch.maximum(uv0, uv1), uv2)

    # Texel centres sit on integers, so a triangle covers the integer lattice
    # points inside its UV bounding box.
    lo = torch.ceil(min_uv).long().clamp_(0, ts - 1)
    hi = torch.floor(max_uv).long().clamp_(0, ts - 1)
    span = (hi - lo + 1).clamp_min(0)
    counts = span[:, 0] * span[:, 1]
    starts = torch.cumsum(counts, dim=0) - counts
    total = int(counts.sum().item())
    if total == 0:
        return face_ids.reshape(ts, ts), out_pos.reshape(ts, ts, 3)

    best = torch.empty(ts * ts, dtype=torch.int64, device=device)

    for g0 in range(0, total, _PAIR_BUDGET):
        g1 = min(g0 + _PAIR_BUDGET, total)
        gidx = torch.arange(g0, g1, device=device)
        # Which triangle owns each candidate. `right=True` skips zero-area
        # triangles, whose start offset ties with the next real one.
        rep = torch.searchsorted(starts, gidx, right=True) - 1
        offset = gidx - starts[rep]
        width = span[rep, 0]
        px = (lo[rep, 0] + offset % width).float()
        py = (lo[rep, 1] + offset // width).float()

        a0, a1, a2 = uv0[rep], uv1[rep], uv2[rep]
        denom = (a1[:, 1] - a2[:, 1]) * (a0[:, 0] - a2[:, 0]) + (a2[:, 0] - a1[:, 0]) * (a0[:, 1] - a2[:, 1])
        ok = denom.abs() >= _DEGENERATE_DENOM
        inv = torch.where(ok, 1.0 / torch.where(ok, denom, torch.ones_like(denom)), torch.zeros_like(denom))

        w0 = ((a1[:, 1] - a2[:, 1]) * (px - a2[:, 0]) + (a2[:, 0] - a1[:, 0]) * (py - a2[:, 1])) * inv
        w1 = ((a2[:, 1] - a0[:, 1]) * (px - a2[:, 0]) + (a0[:, 0] - a2[:, 0]) * (py - a2[:, 1])) * inv
        w2 = 1.0 - w0 - w1
        inside = ok & (w0 >= _BARY_EPS) & (w1 >= _BARY_EPS) & (w2 >= _BARY_EPS)
        if not bool(inside.any()):
            continue

        sel = inside.nonzero(as_tuple=True)[0]
        texel = (py[sel].long() * ts + px[sel].long())

        # Pick one covering triangle per texel deterministically; overlaps in a
        # packed atlas are seam-width at most, so which one wins is immaterial.
        best.fill_(-1)
        best.scatter_reduce_(0, texel, sel, reduce='amax', include_self=True)
        hit = best >= 0
        won = best[hit]

        rep_won = rep[won]
        pos = (p0[rep_won] * w0[won].unsqueeze(1)
               + p1[rep_won] * w1[won].unsqueeze(1)
               + p2[rep_won] * w2[won].unsqueeze(1))
        face_ids[hit] = (rep_won + 1).to(torch.int32)
        out_pos[hit] = pos

    if verbose:
        covered = int((face_ids > 0).sum().item())
        print(f"[uv_rasterize] coverage {covered}/{ts * ts} ({covered / (ts * ts) * 100:.1f}%)")

    return face_ids.reshape(ts, ts), out_pos.reshape(ts, ts, 3)

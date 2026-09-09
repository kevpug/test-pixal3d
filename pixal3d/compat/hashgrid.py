"""
Pure-PyTorch replacement for the GPU hash maps in ``o_voxel`` and ``flex_gemm``.

Both extensions expose the same primitive: build a map from integer voxel
coordinates ``(b, x, y, z)`` to the row index of that voxel in a sparse tensor,
then look up (possibly out-of-range) query coordinates. Sorting the linearised
keys once and answering queries with ``torch.searchsorted`` gives the same
answer with nothing but core PyTorch ops, at O(N log N) build and O(Q log N)
per query instead of O(1) — fast enough in practice because the maps are built
once per resolution and reused through the sparse-tensor spatial cache.
"""

from typing import *
import torch


__all__ = ['CoordLookup', 'encode_coords']

MISS = -1


def _dims(spatial_shape: Sequence[int]) -> Tuple[int, int, int]:
    w, h, d = (int(s) for s in spatial_shape[-3:])
    return w, h, d


def encode_coords(
    coords: torch.Tensor,
    spatial_shape: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linearise ``[N, 4]`` integer coordinates ``(b, x, y, z)`` into int64 keys.

    Returns ``(keys, valid)``. Coordinates outside ``spatial_shape`` get an
    arbitrary key and ``valid=False``; callers must treat those as misses.
    Out-of-range queries are the common case — neighbour lookups walk off the
    edge of the grid — so this is a normal path, not an error.
    """
    w, h, d = _dims(spatial_shape)
    coords = coords.long()
    b, x, y, z = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z >= 0) & (z < d) & (b >= 0)
    keys = ((b * w + x) * h + y) * d + z
    return keys, valid


class CoordLookup:
    """
    Sorted-key lookup table over the coordinates of a sparse tensor.

    Build once from the tensor's ``coords``; query with any ``[M, 4]`` integer
    coordinates. Missing entries come back as ``MISS`` (-1) rather than the
    ``0xffffffff`` sentinel the CUDA hash map uses, because -1 is a valid index
    to clamp and mask against in torch without unsigned-integer games.
    """

    def __init__(self, coords: torch.Tensor, spatial_shape: Sequence[int]):
        self.spatial_shape = tuple(_dims(spatial_shape))
        self.device = coords.device
        keys, valid = encode_coords(coords, self.spatial_shape)
        if not bool(valid.all()):
            # Coordinates of the tensor itself should always be in range; if
            # they are not, drop them so they can never be matched.
            keys = torch.where(valid, keys, torch.full_like(keys, -1))
        order = torch.argsort(keys)
        self._sorted_keys = keys[order]
        self._order = order.to(torch.int64)

    def __len__(self) -> int:
        return int(self._sorted_keys.numel())

    def lookup(self, query: torch.Tensor, chunk: int = 1 << 22) -> torch.Tensor:
        """
        Map ``[M, 4]`` query coordinates to row indices, ``MISS`` where absent.

        Chunked so that a texture-sized query (16M texels x 8 trilinear
        neighbours) does not allocate a multi-gigabyte intermediate.
        """
        if self._sorted_keys.numel() == 0:
            return torch.full((query.shape[0],), MISS, dtype=torch.int64, device=query.device)
        if query.shape[0] <= chunk:
            return self._lookup_chunk(query)
        return torch.cat([
            self._lookup_chunk(query[i:i + chunk])
            for i in range(0, query.shape[0], chunk)
        ], dim=0)

    def _lookup_chunk(self, query: torch.Tensor) -> torch.Tensor:
        keys, valid = encode_coords(query, self.spatial_shape)
        pos = torch.searchsorted(self._sorted_keys, keys).clamp_max(self._sorted_keys.numel() - 1)
        hit = valid & (self._sorted_keys[pos] == keys)
        return torch.where(hit, self._order[pos], torch.full_like(pos, MISS))

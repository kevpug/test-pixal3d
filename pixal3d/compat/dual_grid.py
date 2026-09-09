"""
Pure-PyTorch port of ``o_voxel.convert.flexible_dual_grid_to_mesh``.

The upstream function is already almost entirely torch; its only native calls
are a GPU hash map used to find, for each intersected voxel edge, the four
voxels sharing that edge. :class:`~pixal3d.compat.hashgrid.CoordLookup`
answers exactly that query, so this port is numerically identical to the
extension rather than an approximation.

This matters more than the other fallbacks: mesh extraction sits on the
critical path of every run, so without it nothing can be generated at all on a
machine that cannot build ``o_voxel``.
"""

from typing import *
import numpy as np
import torch

from .hashgrid import CoordLookup, MISS


__all__ = ['flexible_dual_grid_to_mesh']


# The four voxels adjacent to each of the three axis-aligned edges of a voxel,
# in winding order, so that the four looked-up indices form a quad.
_EDGE_NEIGHBOR_OFFSET = (
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),   # x-axis
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),   # y-axis
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),   # z-axis
)
_QUAD_SPLIT_1 = (0, 1, 2, 0, 2, 3)
_QUAD_SPLIT_2 = (0, 1, 3, 3, 1, 2)
_QUAD_SPLIT_TRAIN = (0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4)


def _as_tensor(value, dtype, device, name, length=3):
    if isinstance(value, (int, float)):
        value = [value] * length
    if isinstance(value, (list, tuple)):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = torch.tensor(value, dtype=dtype, device=device)
    assert isinstance(value, torch.Tensor), f"{name} has unsupported type {type(value)}"
    return value.to(device=device, dtype=dtype)


def _quad_split_alignment(vertices: torch.Tensor, tris: torch.Tensor) -> torch.Tensor:
    """
    Upstream's score for a candidate quad diagonal.

    Kept index-for-index identical to ``o_voxel``: it reads columns 0-3 of the
    six-element split pattern, not the two triangles, so the two ``cross``
    calls do not compare the split's halves the way the name suggests. This is
    dead code in Pixal3D — ``split_weight`` is always supplied by the decoder —
    but reproducing it exactly keeps the fallback a port rather than a rewrite.
    """
    n0 = torch.linalg.cross(
        vertices[tris[:, 1]] - vertices[tris[:, 0]],
        vertices[tris[:, 2]] - vertices[tris[:, 0]], dim=-1)
    n1 = torch.linalg.cross(
        vertices[tris[:, 2]] - vertices[tris[:, 1]],
        vertices[tris[:, 3]] - vertices[tris[:, 1]], dim=-1)
    return (n0 * n1).sum(dim=1, keepdim=True).abs()


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Optional[torch.Tensor],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    train: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a triangle mesh from a flexible dual grid.

    Drop-in replacement for the ``o_voxel`` function of the same name; see that
    docstring for argument semantics.
    """
    device = coords.device

    aabb = _as_tensor(aabb, torch.float32, device, 'aabb')
    assert aabb.dim() == 2 and aabb.shape == (2, 3), f"aabb must be [2, 3], got {tuple(aabb.shape)}"

    if voxel_size is not None:
        voxel_size = _as_tensor(voxel_size, torch.float32, device, 'voxel_size')
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        grid_size = _as_tensor(grid_size, torch.int32, device, 'grid_size')
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    n = dual_vertices.shape[0]

    # Every intersected voxel contributes one quad per axis; look up the four
    # voxels around each such edge and keep the quads whose corners all exist.
    offsets = torch.tensor(_EDGE_NEIGHBOR_OFFSET, dtype=torch.int32, device=device).unsqueeze(0)
    edge_neighbor_voxel = coords.reshape(n, 1, 1, 3) + offsets            # [N, 3, 4, 3]
    connected_voxel = edge_neighbor_voxel[intersected_flag]               # [M, 4, 3]
    m = connected_voxel.shape[0]

    lookup = CoordLookup(
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        grid_size.tolist(),
    )
    query = torch.cat([
        torch.zeros((m * 4, 1), dtype=torch.int32, device=device),
        connected_voxel.reshape(-1, 3),
    ], dim=1)
    indices = lookup.lookup(query).reshape(m, 4)
    quad_indices = indices[(indices != MISS).all(dim=1)]                  # [L, 4]
    quad_len = quad_indices.shape[0]

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    if not train:
        if split_weight is None:
            tris_a = quad_indices[:, list(_QUAD_SPLIT_1)]
            tris_b = quad_indices[:, list(_QUAD_SPLIT_2)]
            align_a = _quad_split_alignment(mesh_vertices, tris_a)
            align_b = _quad_split_alignment(mesh_vertices, tris_b)
            mesh_triangles = torch.where(align_a > align_b, tris_a, tris_b).reshape(-1, 3)
        else:
            w = split_weight[quad_indices]
            mesh_triangles = torch.where(
                (w[:, 0] * w[:, 2]) > (w[:, 1] * w[:, 3]),
                quad_indices[:, list(_QUAD_SPLIT_1)],
                quad_indices[:, list(_QUAD_SPLIT_2)],
            ).reshape(-1, 3)
    else:
        assert split_weight is not None, "split_weight must be provided in training mode"
        quad_vs = mesh_vertices[quad_indices]
        mean_v02 = (quad_vs[:, 0] + quad_vs[:, 2]) / 2
        mean_v13 = (quad_vs[:, 1] + quad_vs[:, 3]) / 2
        w = split_weight[quad_indices]
        w02 = w[:, 0] * w[:, 2]
        w13 = w[:, 1] * w[:, 3]
        mid_vertices = (w02 * mean_v02 + w13 * mean_v13) / (w02 + w13)
        mesh_vertices = torch.cat([mesh_vertices, mid_vertices], dim=0)
        quad_indices = torch.cat([
            quad_indices,
            torch.arange(n, n + quad_len, device=device).unsqueeze(1),
        ], dim=1)
        mesh_triangles = quad_indices[:, list(_QUAD_SPLIT_TRAIN)].reshape(-1, 3)

    return mesh_vertices, mesh_triangles

"""
Pure-PyTorch submanifold sparse convolution — the ``flex_gemm`` fallback.

``flex_gemm`` reaches its speed through Triton kernels, and AMD ships no Triton
build for Windows, so on that platform the fast path is simply unavailable.
This backend reproduces its semantics with ``index_select`` + ``addmm``.

Two details make it usable rather than merely correct:

* The neighbour map is built once per (coords, kernel, dilation) and kept in
  the sparse tensor's spatial cache, exactly like the ``flex_gemm`` backend, so
  a whole decoder stack pays for it once per resolution.
* Accumulation runs over blocks of voxels, so the ``[L, V*Ci]`` im2col matrix
  is never materialised in full — at 1024^3 that matrix alone would be tens of
  gigabytes — while each block is still one large GEMM rather than V small ones.

The parameter layout is byte-identical to ``flex_gemm``'s ``(Co, Kd, Kh, Kw,
Ci)``, which is what the released checkpoints store, so weights load either way
and you can switch backends without reconverting anything.
"""

import math
from typing import *

import torch
import torch.nn as nn

from .. import SparseTensor
from ....compat.hashgrid import CoordLookup, MISS


# Elements per accumulation block. Keeps the gathered ``[block, V*Ci]`` tile to
# roughly 64 MB in fp16 regardless of kernel volume or channel count.
_BLOCK_ELEMS = 1 << 25

# Voxels per neighbour-map build step, bounding the ``[step, V, 4]`` scratch.
_NEIGHBOR_STEP = 1 << 18


def _triple(value, name: str) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        assert len(value) == 3, f"{name} must have 3 elements, got {value}"
        return tuple(int(v) for v in value)
    return (int(value),) * 3


def _kernel_offsets(kernel_size: Tuple[int, int, int], dilation: Tuple[int, int, int],
                    device: torch.device) -> torch.Tensor:
    """
    ``[V, 3]`` neighbour offsets in the same order ``flex_gemm`` uses.

    The flattening order has to match the kernel dimensions of the stored
    weight tensor, or the taps get shuffled and the model produces noise.
    """
    axes = [
        torch.arange(-(k // 2) * d, (k // 2) * d + 1, d, device=device, dtype=torch.long)
        for k, d in zip(kernel_size, dilation)
    ]
    grid = torch.meshgrid(*axes, indexing='ij')
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def _build_neighbor_map(
    coords: torch.Tensor,
    spatial_shape: Sequence[int],
    kernel_size: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    """
    ``[L, V]`` int32 map from (voxel, kernel tap) to the source row to gather.

    Taps that fall outside the occupied set are written as ``L`` rather than
    ``-1``: the forward pass appends one zero row to the features, so a miss
    gathers zeros with no mask, no branch, and no device-to-host sync.
    """
    device = coords.device
    offsets = _kernel_offsets(kernel_size, dilation, device)
    volume = offsets.shape[0]
    length = coords.shape[0]

    lookup = CoordLookup(coords, spatial_shape)
    out = torch.empty((length, volume), dtype=torch.int32, device=device)

    coords_long = coords.long()
    for start in range(0, length, _NEIGHBOR_STEP):
        stop = min(start + _NEIGHBOR_STEP, length)
        block = coords_long[start:stop]
        neigh = block.unsqueeze(1).repeat(1, volume, 1)
        neigh[:, :, 1:] += offsets.unsqueeze(0)
        found = lookup.lookup(neigh.reshape(-1, 4)).reshape(stop - start, volume)
        out[start:stop] = torch.where(found == MISS, length, found).to(torch.int32)
    return out


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                       padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), \
        'The torch sparse-conv backend only supports submanifold convolution (stride=1, padding=None)'

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _triple(kernel_size, 'kernel_size')
    self.stride = _triple(stride, 'stride')
    self.dilation = _triple(dilation, 'dilation')

    weight = torch.empty((out_channels, in_channels, *self.kernel_size))
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter("bias", None)

    # (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci), matching flex_gemm and the
    # layout the published checkpoints are stored in.
    self.weight = nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    out_channels, k0, k1, k2, in_channels = self.weight.shape
    volume = k0 * k1 * k2

    cache_key = f'SubMConv3d_gather_cache_{k2}x{k1}x{k0}_dilation{self.dilation}'
    neighbor = x.get_spatial_cache(cache_key)
    if neighbor is None:
        neighbor = _build_neighbor_map(x.coords, x.spatial_shape, (k0, k1, k2), self.dilation)
        x.register_spatial_cache(cache_key, neighbor)

    feats = x.feats
    length = feats.shape[0]
    # The extra zero row is where every out-of-support tap points.
    padded = torch.cat([feats, feats.new_zeros((1, in_channels))], dim=0)
    # (Co, Kd, Kh, Kw, Ci) -> (Kd*Kh*Kw*Ci, Co), matching the gathered tile's
    # (tap, channel) flattening so the whole block is a single GEMM.
    weight = self.weight.reshape(out_channels, volume * in_channels).t().to(feats.dtype).contiguous()
    def _block(start: int, stop: int) -> torch.Tensor:
        gathered = padded.index_select(0, neighbor[start:stop].reshape(-1).long())
        return gathered.reshape(stop - start, volume * in_channels) @ weight

    block = max(1, _BLOCK_ELEMS // (volume * in_channels))
    if length <= block:
        out = _block(0, length)
    else:
        out = feats.new_empty((length, out_channels))
        for start in range(0, length, block):
            out[start:start + block] = _block(start, min(start + block, length))

    if self.bias is not None:
        out += self.bias.to(out.dtype)
    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d is not implemented for the torch backend')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d is not implemented for the torch backend')

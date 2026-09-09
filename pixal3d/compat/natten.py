"""
Pure-PyTorch stand-in for NATTEN's 2D neighbourhood attention.

The NAF feature upsampler (``valeoai/NAF``, pulled in through ``torch.hub``)
imports ``natten`` for its cross-attention. NATTEN ships Linux-only wheels
built against CUTLASS, does not support Windows, and has no ROCm backend, so
on this platform it can neither be installed nor built.

Neighbourhood attention is ordinary attention with each query restricted to a
``kernel_size`` window around its own position. The one subtlety is the
border: NATTEN keeps every window exactly ``kernel_size`` wide by sliding it
inward rather than zero-padding, so a corner query attends to the same number
of keys as a centre one. :func:`neighborhood_indices` reproduces that, and
dilation is handled the way NATTEN defines it -- the axis is split into
``dilation`` interleaved subsequences and the window rule is applied within
whichever one the query belongs to.

Installed into ``sys.modules`` by :func:`install` before NAF is imported, and
only when the real NATTEN is absent.
"""

from typing import *
import importlib.machinery
import sys
import types

import torch


__all__ = ['na2d', 'na2d_qk', 'na2d_av', 'neighborhood_indices', 'install']


# Gathering the neighbours materialises kernel_size^2 copies of the keys, so
# the query rows are processed in chunks sized to keep that intermediate near
# this many elements.
_CHUNK_BUDGET = 32 * 1024 * 1024


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    a, b = value
    return (int(a), int(b))


def neighborhood_indices(length: int, kernel: int, dilation: int,
                         device=None) -> torch.Tensor:
    """Index of each of ``kernel`` neighbours for every position on one axis.

    Returns a ``[length, kernel]`` int64 tensor. Windows are centred where
    they fit and slid inward at the borders, which is what makes every query's
    neighbourhood the same size.
    """
    if kernel < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel}")
    if dilation < 1:
        raise ValueError(f"dilation must be positive, got {dilation}")

    position = torch.arange(length, device=device)
    offset = position % dilation                 # which interleaved subsequence
    index_in_sub = position // dilation          # position within it
    # Subsequences are unequal in length when dilation does not divide length.
    sub_length = (length - offset + dilation - 1) // dilation

    start = (index_in_sub - kernel // 2).clamp(min=0)
    start = torch.minimum(start, (sub_length - kernel).clamp(min=0))

    taps = torch.arange(kernel, device=device)
    index = offset[:, None] + (start[:, None] + taps[None, :]) * dilation
    # A window wider than its subsequence has nowhere left to slide; clamping
    # repeats the edge element rather than reading out of bounds.
    return index.clamp(max=length - 1)


def _gather_neighbors(tensor: torch.Tensor, row: torch.Tensor,
                      col: torch.Tensor) -> torch.Tensor:
    """``[B, c, W, N, D]`` -> ``[B, c, W, N, kh*kw, D]`` for the given windows.

    ``tensor`` is the full ``[B, H, W, N, D]`` map; ``row`` is ``[c, kh]``
    covering just the query rows in this chunk, ``col`` is ``[W, kw]``.
    """
    batch, _, width, heads, dim = tensor.shape
    chunk, kh = row.shape
    kw = col.shape[1]

    out = tensor.index_select(1, row.reshape(-1))
    out = out.view(batch, chunk, kh, width, heads, dim)
    out = out.index_select(3, col.reshape(-1))
    out = out.view(batch, chunk, kh, width, kw, heads, dim)
    # (B, c, kh, W, kw, N, D) -> (B, c, W, N, kh, kw, D)
    out = out.permute(0, 1, 3, 5, 2, 4, 6)
    return out.reshape(batch, chunk, width, heads, kh * kw, dim)


def _row_chunk(batch: int, width: int, heads: int, taps: int, dim: int) -> int:
    per_row = max(1, batch * width * heads * taps * dim)
    return max(1, _CHUNK_BUDGET // per_row)


def na2d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
         kernel_size, dilation=1, stride=1, is_causal=False,
         scale: Optional[float] = None, backend: Optional[str] = None,
         **kwargs) -> torch.Tensor:
    """Neighbourhood attention over ``[batch, height, width, heads, dim]``.

    Matches the signature NATTEN's modern API presents; ``backend`` and the
    other kernel-selection arguments are accepted and ignored, since there is
    only one implementation here.
    """
    if query.ndim != 5:
        raise ValueError(
            f"na2d expects [batch, height, width, heads, dim], got {tuple(query.shape)}")
    if _pair(stride) != (1, 1):
        raise NotImplementedError(f"na2d fallback supports stride 1, got {stride}")
    if is_causal not in (False, None) and any(_pair(is_causal)):
        raise NotImplementedError("na2d fallback does not implement causal masking")

    batch, height, width, heads, dim = query.shape
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    scale = dim ** -0.5 if scale is None else scale

    row = neighborhood_indices(height, kh, dh, query.device)
    col = neighborhood_indices(width, kw, dw, query.device)
    taps = kh * kw

    out = torch.empty_like(query)
    step = _row_chunk(batch, width, heads, taps, dim)
    for start in range(0, height, step):
        stop = min(start + step, height)
        rows = stop - start
        keys = _gather_neighbors(key, row[start:stop], col)
        values = _gather_neighbors(value, row[start:stop], col)

        flat = batch * rows * width * heads
        q_flat = query[:, start:stop].reshape(flat, dim, 1)
        scores = torch.bmm(keys.reshape(flat, taps, dim), q_flat).squeeze(-1)
        weights = (scores * scale).softmax(dim=-1)
        chunk = torch.bmm(weights.unsqueeze(1),
                          values.reshape(flat, taps, dim)).squeeze(1)
        out[:, start:stop] = chunk.view(batch, rows, width, heads, dim)
    return out


def na2d_qk(query: torch.Tensor, key: torch.Tensor, kernel_size,
            dilation=1, **kwargs) -> torch.Tensor:
    """Unnormalised neighbourhood scores, NATTEN's legacy ``b n h w d`` layout.

    Returns ``[batch, heads, height, width, kh*kw]``. No scaling is applied:
    the legacy call sites multiply by their own scale before the softmax.
    """
    query = query.permute(0, 2, 3, 1, 4)
    key = key.permute(0, 2, 3, 1, 4)
    batch, height, width, heads, dim = query.shape
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)

    row = neighborhood_indices(height, kh, dh, query.device)
    col = neighborhood_indices(width, kw, dw, query.device)
    taps = kh * kw

    out = query.new_empty((batch, height, width, heads, taps))
    step = _row_chunk(batch, width, heads, taps, dim)
    for start in range(0, height, step):
        stop = min(start + step, height)
        rows = stop - start
        keys = _gather_neighbors(key, row[start:stop], col)
        flat = batch * rows * width * heads
        scores = torch.bmm(keys.reshape(flat, taps, dim),
                           query[:, start:stop].reshape(flat, dim, 1)).squeeze(-1)
        out[:, start:stop] = scores.view(batch, rows, width, heads, taps)
    return out.permute(0, 3, 1, 2, 4)


def na2d_av(attn: torch.Tensor, value: torch.Tensor, kernel_size,
            dilation=1, **kwargs) -> torch.Tensor:
    """Apply legacy neighbourhood weights to values, in ``b n h w d`` layout."""
    attn = attn.permute(0, 2, 3, 1, 4)
    value = value.permute(0, 2, 3, 1, 4)
    batch, height, width, heads, taps = attn.shape
    dim = value.shape[-1]
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    if taps != kh * kw:
        raise ValueError(f"attention has {taps} taps, kernel {kh}x{kw} needs {kh * kw}")

    row = neighborhood_indices(height, kh, dh, value.device)
    col = neighborhood_indices(width, kw, dw, value.device)

    out = value.new_empty((batch, height, width, heads, dim))
    step = _row_chunk(batch, width, heads, taps, dim)
    for start in range(0, height, step):
        stop = min(start + step, height)
        rows = stop - start
        values = _gather_neighbors(value, row[start:stop], col)
        flat = batch * rows * width * heads
        mixed = torch.bmm(attn[:, start:stop].reshape(flat, 1, taps),
                          values.reshape(flat, taps, dim)).squeeze(1)
        out[:, start:stop] = mixed.view(batch, rows, width, heads, dim)
    return out.permute(0, 3, 1, 2, 4)


def install(verbose: bool = True) -> bool:
    """Register this module as ``natten`` so third-party code can import it.

    A real NATTEN installation always wins. Returns True if the shim was put
    in place.
    """
    if 'natten' in sys.modules:
        return getattr(sys.modules['natten'], '__pixal3d_fallback__', False)
    try:
        import natten  # noqa: F401
        return False
    except ImportError:
        pass

    root = types.ModuleType('natten')
    # A module built by hand has __spec__ = None, and importlib.util.find_spec
    # raises ValueError on that rather than returning None. torch.hub checks a
    # hubconf's declared dependencies with exactly that call, so without a real
    # spec the shim trades one crash for another.
    root.__spec__ = importlib.machinery.ModuleSpec('natten', loader=None,
                                                   is_package=True)
    root.__path__ = []
    root.__pixal3d_fallback__ = True
    root.__version__ = '0.0.0+pixal3d'

    functional = types.ModuleType('natten.functional')
    functional.__spec__ = importlib.machinery.ModuleSpec('natten.functional',
                                                         loader=None)
    functional.__package__ = 'natten'
    functional.__pixal3d_fallback__ = True

    for module in (root, functional):
        module.na2d = na2d
        module.na2d_qk = na2d_qk
        module.na2d_av = na2d_av
    root.functional = functional

    sys.modules['natten'] = root
    sys.modules['natten.functional'] = functional
    if verbose:
        print("[pixal3d] natten: pure-PyTorch fallback "
              "(NATTEN has no Windows or ROCm build)")
    return True

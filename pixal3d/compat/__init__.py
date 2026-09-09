"""
Portable fallbacks for Pixal3D's CUDA-only dependencies.

Pixal3D inherits TRELLIS.2's native stack — ``flash_attn``, ``flex_gemm``,
``cumesh``, ``o_voxel``, ``nvdiffrast`` — none of which builds on Windows with
ROCm. Everything in this package is a pure-PyTorch (or trimesh/xatlas)
implementation of one of those pieces, chosen automatically when the fast path
is missing and never used when it is present.

Nothing here changes behaviour on a working CUDA install: the native extension
always wins if it imports.
"""

import importlib
from typing import *

__attributes = {
    'has_module': 'probe',
    'platform': 'probe',
    'is_rocm': 'probe',
    'gpu_arch': 'probe',
    'summary': 'probe',
    'CoordLookup': 'hashgrid',
    'grid_sample_3d': 'grid_sample',
    'flexible_dual_grid_to_mesh': 'dual_grid',
    'uv_rasterize': 'uv_raster',
    'to_glb': 'postprocess',
    'simplify_mesh': 'mesh_ops',
    'clean_mesh': 'mesh_ops',
    'uv_unwrap': 'mesh_ops',
}

__submodules = ['probe', 'hashgrid', 'grid_sample', 'dual_grid', 'uv_raster',
                'mesh_ops', 'postprocess', 'dispatch']

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module = importlib.import_module(f".{__attributes[name]}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            globals()[name] = importlib.import_module(f".{name}", __name__)
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]

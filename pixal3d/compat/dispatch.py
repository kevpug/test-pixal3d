"""
Chooses between a native extension and its :mod:`pixal3d.compat` fallback.

Call sites ask for a capability rather than importing ``cumesh`` or
``flex_gemm`` directly, so a machine with a partial install (say, ``cumesh``
built but ``nvdiffrast`` not) still runs, taking the fast path exactly where it
exists. Every decision is logged once so a run's output says which path it
took — otherwise a silent fallback looks like an unexplained slowdown.
"""

from typing import *
import os

from .probe import has_module


__all__ = [
    'grid_sample_3d', 'flexible_dual_grid_to_mesh', 'uv_rasterize', 'to_glb',
    'cumesh_available', 'force_fallback', 'note',
]


_announced: Set[str] = set()


def force_fallback() -> bool:
    """``PIXAL3D_FORCE_FALLBACK=1`` ignores native extensions — useful for A/B tests."""
    return os.environ.get('PIXAL3D_FORCE_FALLBACK', '') == '1'


def note(capability: str, backend: str) -> None:
    """Print the chosen backend for ``capability``, once per process."""
    if capability in _announced:
        return
    _announced.add(capability)
    print(f"[pixal3d] {capability}: {backend}")


def _use(module: str) -> bool:
    return has_module(module) and not force_fallback()


def cumesh_available() -> bool:
    return _use('cumesh')


def grid_sample_3d(*args, **kwargs):
    """Sparse trilinear volume sampling."""
    if _use('flex_gemm'):
        from flex_gemm.ops.grid_sample import grid_sample_3d as native
        note('grid_sample_3d', 'flex_gemm')
        return native(*args, **kwargs)
    from .grid_sample import grid_sample_3d as fallback
    note('grid_sample_3d', 'pytorch fallback')
    return fallback(*args, **kwargs)


def flexible_dual_grid_to_mesh(*args, **kwargs):
    """Dual-grid mesh extraction — on the critical path of every run."""
    if _use('o_voxel'):
        from o_voxel.convert import flexible_dual_grid_to_mesh as native
        note('flexible_dual_grid_to_mesh', 'o_voxel')
        return native(*args, **kwargs)
    from .dual_grid import flexible_dual_grid_to_mesh as fallback
    note('flexible_dual_grid_to_mesh', 'pytorch fallback')
    return fallback(*args, **kwargs)


def uv_rasterize(*args, **kwargs):
    """UV-space rasterisation for texture baking."""
    from .uv_raster import uv_rasterize as fallback
    note('uv_rasterize', 'pytorch fallback')
    return fallback(*args, **kwargs)


def to_glb(*args, **kwargs):
    """Mesh cleanup, UV unwrap and texture bake into a textured GLB."""
    if _use('o_voxel') and _use('cumesh') and _use('nvdiffrast') and _use('flex_gemm'):
        import o_voxel.postprocess
        note('to_glb', 'o_voxel')
        return o_voxel.postprocess.to_glb(*args, **kwargs)
    from .postprocess import to_glb as fallback
    note('to_glb', 'pytorch/trimesh fallback')
    return fallback(*args, **kwargs)

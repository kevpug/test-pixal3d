"""
Capability probing for optional native extensions.

Pixal3D's reference environment is CUDA-only: it relies on ``flash_attn``,
``flex_gemm``, ``cumesh``, ``o_voxel`` and ``nvdiffrast``, none of which have
usable builds on Windows + ROCm. Every one of those has a pure-PyTorch
replacement in :mod:`pixal3d.compat`, so the code needs a cheap, side-effect
free way to ask "is the fast path here?" before deciding what to run.

This module is deliberately dependency-free (``importlib`` only) so that the
backend ``config`` modules can call it while the package is still importing.
"""

from typing import *
import importlib.util
import os


__all__ = [
    'has_module',
    'platform',
    'is_rocm',
    'gpu_arch',
    'accelerator_note',
    'default_attn_backend',
    'default_sparse_conv_backend',
    'summary',
]


_module_cache: Dict[str, bool] = {}


def has_module(name: str) -> bool:
    """
    Whether ``name`` is importable, without importing it.

    ``find_spec`` is enough for our purposes and avoids paying the (large)
    import cost of e.g. ``flash_attn`` just to find out it is missing. Results
    are cached because the config modules probe the same names repeatedly.
    """
    if name in _module_cache:
        return _module_cache[name]
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # A broken/partially-installed package can raise from find_spec itself.
        found = False
    _module_cache[name] = found
    return found


def platform() -> str:
    """
    How torch is reaching the GPU: ``rocm``, ``zluda``, ``cuda``, ``directml``
    or ``cpu``.

    The last two matter because a lot of Windows AMD setups for image
    generation use them, and neither is what this port targets:

    * **ZLUDA** runs a *CUDA* build of torch on an AMD card through a
      translation layer. ``torch.cuda`` works, but ``torch.version.hip`` is
      unset and device properties carry no ``gcnArchName``, so anything keyed
      on the gfx target silently gets the wrong answer unless we notice.
    * **DirectML** is a separate backend entirely (``privateuseone``), not
      ``torch.cuda``, so this pipeline cannot use it at all.
    """
    try:
        import torch
    except ImportError:
        return 'cpu'
    if getattr(torch.version, 'hip', None):
        return 'rocm'
    if getattr(torch.version, 'cuda', None):
        # A CUDA build reporting an AMD device is ZLUDA (or an equivalent
        # shim); there is no such thing as a genuine AMD CUDA GPU.
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).lower()
                if 'amd' in name or 'radeon' in name or 'gfx' in name:
                    return 'zluda'
        except Exception:
            pass
        return 'cuda'
    if has_module('torch_directml'):
        return 'directml'
    return 'cpu'


def is_rocm() -> bool:
    return platform() == 'rocm'


def accelerator_note() -> Optional[str]:
    """A warning about the torch build, or ``None`` when it is a supported one."""
    kind = platform()
    if kind == 'directml':
        return ("torch-directml is installed. DirectML is a separate backend from "
                "torch.cuda, so this pipeline cannot run on it — install AMD's ROCm "
                "build of torch (scripts/install_rocm_torch.py) into a separate venv.")
    if kind == 'zluda':
        return ("This looks like ZLUDA: a CUDA build of torch driving an AMD GPU. "
                "Untested here — the gfx target is inferred from the device name "
                "rather than reported by the driver. If results look wrong, install "
                "AMD's native ROCm torch instead (scripts/install_rocm_torch.py).")
    if kind == 'cpu':
        return "No GPU backend in this torch build — generation will be unusably slow."
    return None


# Used only when the driver does not report a gfx target, which is the case
# under ZLUDA. Ordered most-specific first: an "RX 6800M" is gfx1031 while a
# plain "RX 6800" is gfx1030, so the suffixed entries have to win.
_NAME_TO_ARCH: Tuple[Tuple[str, str], ...] = (
    # RDNA2 laptop / small dies
    ('6700m', 'gfx1031'), ('6800m', 'gfx1031'), ('6850m', 'gfx1031'),
    ('6600m', 'gfx1032'), ('6650m', 'gfx1032'), ('6700s', 'gfx1032'),
    ('6800s', 'gfx1032'), ('6600s', 'gfx1032'),
    ('6500m', 'gfx1034'), ('6450m', 'gfx1034'), ('6300m', 'gfx1034'),
    ('680m', 'gfx1035'), ('660m', 'gfx1035'),
    # RDNA2 desktop
    ('6950', 'gfx1030'), ('6900', 'gfx1030'), ('6800', 'gfx1030'),
    ('6750', 'gfx1031'), ('6700', 'gfx1031'),
    ('6650', 'gfx1032'), ('6600', 'gfx1032'),
    ('6500', 'gfx1034'), ('6400', 'gfx1034'),
    # RDNA3
    ('7900', 'gfx1100'), ('7800', 'gfx1101'), ('7700s', 'gfx1102'),
    ('7700', 'gfx1101'), ('7600', 'gfx1102'), ('780m', 'gfx1103'),
    ('760m', 'gfx1103'), ('8060s', 'gfx1151'), ('8050s', 'gfx1151'),
    # RDNA4
    ('9070', 'gfx1201'), ('9060', 'gfx1200'),
)


def arch_from_device_name(name: str) -> Optional[str]:
    """Best-effort gfx target from a marketing name, for when the driver hides it."""
    lowered = name.lower().replace('-', ' ')
    for token, arch in _NAME_TO_ARCH:
        if token in lowered:
            return arch
    return None


def gpu_arch() -> Optional[str]:
    """
    The gfx target of the first visible GPU (ROCm), or the compute capability
    (CUDA). ``None`` when no accelerator is visible.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None
    arch = getattr(props, 'gcnArchName', None)
    if arch:
        # gcnArchName looks like 'gfx1031:xnack-' — keep just the target.
        return arch.split(':')[0]
    # A CUDA build has no gcnArchName. Under ZLUDA the device is still an AMD
    # part, so fall back to the marketing name; a real NVIDIA card reports its
    # compute capability as usual.
    guessed = arch_from_device_name(getattr(props, 'name', '') or '')
    if guessed:
        return guessed
    return f'sm_{props.major}{props.minor}'


def default_attn_backend() -> str:
    """
    Pick an attention backend that is actually installed.

    ``flash_attn`` has no ROCm build for RDNA2 and none at all for Windows, so
    anything but a working install falls back to PyTorch SDPA, which every
    attention entry point in this repo supports.
    """
    if has_module('flash_attn'):
        return 'flash_attn'
    if has_module('xformers'):
        return 'xformers'
    return 'sdpa'


def default_sparse_conv_backend() -> str:
    """
    Pick a sparse-convolution backend that is actually installed.

    ``flex_gemm`` needs Triton, which AMD does not ship for Windows, so the
    pure-PyTorch ``torch`` backend is the fallback rather than spconv (CUDA
    only) or torchsparse (CUDA only).
    """
    if has_module('flex_gemm'):
        return 'flex_gemm'
    if has_module('spconv'):
        return 'spconv'
    if has_module('torchsparse'):
        return 'torchsparse'
    return 'torch'


def summary() -> Dict[str, Any]:
    """A dict describing the runtime, for logging and ``scripts/check_env.py``."""
    info: Dict[str, Any] = {
        'platform': platform(),
        'gpu_arch': gpu_arch(),
        'accelerator_note': accelerator_note(),
        'attn_backend': os.environ.get('ATTN_BACKEND') or default_attn_backend(),
        'sparse_conv_backend': os.environ.get('SPARSE_CONV_BACKEND') or default_sparse_conv_backend(),
    }
    for name in ('torch', 'flash_attn', 'xformers', 'flex_gemm', 'cumesh',
                 'o_voxel', 'nvdiffrast', 'nvdiffrec_render', 'triton',
                 'xatlas', 'fast_simplification', 'trimesh'):
        info[name] = has_module(name)
    return info

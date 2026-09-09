"""
Runtime configuration: device selection, VRAM presets and block offloading.

Import and call :func:`configure` before anything touches CUDA/HIP. It sets the
allocator environment variables, picks attention and sparse-conv backends to
match what is installed, and returns a :class:`Preset` describing how hard to
push the GPU.

The presets exist because Pixal3D's defaults assume a datacentre card: the
reference pipeline wants roughly 18 GB, and the shipped ``--low_vram`` path
still wants 10-12 GB. On a 12 GB laptop that is right at the edge once Windows
has taken its cut, and below that nothing runs at all without turning knobs.
"""

from typing import *
import gc
import os
from dataclasses import dataclass, replace


__all__ = ['Preset', 'PRESETS', 'configure', 'pick_preset', 'enable_block_offload',
           'free_memory', 'get_device', 'describe', 'resolve_dtype', 'apply_dtype',
           'sdpa_backends', 'has_fused_attention', 'set_cond_dtype', 'cond_autocast', 'resolve_cond_dtype']


@dataclass
class Preset:
    """A VRAM budget expressed as concrete pipeline settings."""
    name: str
    min_vram_gb: float
    low_vram: bool
    resolution: int
    max_num_tokens: int
    texture_size: int
    decimation_target: int
    block_offload: bool
    attn_budget: Optional[int] = None
    cfg_batch: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# Ordered from largest to smallest; `auto` walks this list.
PRESETS: Dict[str, Preset] = {
    'max': Preset('max', 24.0, low_vram=False, resolution=1536, max_num_tokens=49152,
                  texture_size=4096, decimation_target=1000000, block_offload=False,
                  cfg_batch=True),
    '16gb': Preset('16gb', 15.0, low_vram=True, resolution=1536, max_num_tokens=49152,
                   texture_size=4096, decimation_target=1000000, block_offload=False,
                   cfg_batch=True),
    '12gb': Preset('12gb', 11.0, low_vram=True, resolution=1024, max_num_tokens=32768,
                   texture_size=2048, decimation_target=300000, block_offload=False,
                   attn_budget=1 << 27),
    '8gb': Preset('8gb', 7.5, low_vram=True, resolution=1024, max_num_tokens=16384,
                  texture_size=2048, decimation_target=250000, block_offload=True,
                  attn_budget=1 << 26),
    '6gb': Preset('6gb', 0.0, low_vram=True, resolution=1024, max_num_tokens=8192,
                  texture_size=1024, decimation_target=150000, block_offload=True,
                  attn_budget=1 << 25),
}


def total_vram_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return None


def pick_preset(name: str = 'auto') -> Preset:
    """
    Resolve a preset name, or choose one from the detected VRAM.

    Auto-detection deliberately leaves headroom: on Windows the display driver
    reserves part of the card and a run that fits in theory can still OOM, so
    the thresholds sit a gigabyte or so below the nominal capacity.
    """
    name = (name or 'auto').lower()
    if name in PRESETS:
        return PRESETS[name]
    if name != 'auto':
        raise ValueError(f"Unknown VRAM preset '{name}'. Choose from: {', '.join(PRESETS)}, auto")

    vram = total_vram_gb()
    if vram is None:
        return PRESETS['12gb']
    for preset in PRESETS.values():
        if vram >= preset.min_vram_gb:
            return preset
    return PRESETS['6gb']


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'          # ROCm builds expose HIP devices as 'cuda' too
    return 'cpu'


def describe() -> str:
    from .compat.probe import summary
    info = summary()
    vram = total_vram_gb()
    parts = [f"torch backend: {info['platform']}"]
    if info['gpu_arch']:
        parts.append(f"gpu: {info['gpu_arch']}")
    if vram:
        parts.append(f"vram: {vram:.1f} GB")
    attention = info['attn_backend']
    if attention == 'sdpa':
        backends = sdpa_backends()
        kind = ('flash' if backends['flash']
                else 'mem-efficient' if backends['mem_efficient']
                else 'math, no fused kernel')
        attention = f"sdpa ({kind})"
    parts.append(f"attention: {attention}")
    parts.append(f"sparse conv: {info['sparse_conv_backend']}")
    native = [n for n in ('flash_attn', 'flex_gemm', 'cumesh', 'o_voxel', 'nvdiffrast') if info[n]]
    parts.append("native extensions: " + (", ".join(native) if native else "none (pure-torch fallbacks)"))
    line = "[pixal3d] " + " | ".join(parts)
    # A ZLUDA or DirectML install is easy to mistake for a working ROCm one, so
    # say so on every run rather than only in check_env.
    if info.get('accelerator_note'):
        line += "\n[pixal3d] " + info['accelerator_note']
    return line


def _apply_backends(attn_backend: Optional[str], conv_backend: Optional[str]) -> None:
    """Update already-imported backend config modules in place."""
    import sys

    attention = sys.modules.get('pixal3d.modules.attention.config')
    if attention is not None and attn_backend:
        attention.set_backend(attn_backend)

    sparse = sys.modules.get('pixal3d.modules.sparse.config')
    if sparse is not None:
        if attn_backend and attn_backend in sparse.VALID_ATTN_BACKENDS:
            sparse.set_attn_backend(attn_backend)
        if conv_backend:
            sparse.set_conv_backend(conv_backend)


# RDNA2 has packed fp16 arithmetic but no bfloat16 instructions at all, so a
# bf16 GEMM there is emulated through fp32 and runs at roughly half the fp16
# rate. Every other target either has bf16 hardware or is not worth special
# casing.
_NO_BF16_HARDWARE = ('gfx1010', 'gfx1011', 'gfx1012', 'gfx1030', 'gfx1031',
                     'gfx1032', 'gfx1034', 'gfx1035', 'gfx1036')


def resolve_dtype(requested: str = 'auto') -> Optional['torch.dtype']:  # noqa: F821
    """
    The dtype to convert the flow models to, or ``None`` to keep the
    checkpoint's own (bfloat16).

    ``auto`` picks float16 on GPUs with no bfloat16 hardware. That is a real
    numerics change — the checkpoints were trained in bfloat16, which has far
    more exponent range — so it prints what it did and can be overridden.
    """
    import torch
    from .compat.probe import gpu_arch

    if requested in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if requested in ('fp16', 'float16', 'half'):
        return torch.float16
    if requested in ('fp32', 'float32'):
        return torch.float32
    if requested != 'auto':
        raise ValueError(f"Unknown dtype '{requested}'. Use auto, bfloat16, float16 or float32.")

    arch = gpu_arch()
    if arch and arch.lower() in _NO_BF16_HARDWARE:
        return torch.float16
    return None


def apply_dtype(pipeline, dtype, verbose: bool = True) -> int:
    """
    Convert the flow models' transformer blocks to ``dtype``.

    Returns how many models were converted. Only the DiT torsos are touched --
    that is what ``convert_to`` covers, and it is where essentially all of the
    sampling FLOPs are. The embedders, VAEs and conditioning backbones keep
    their own precision.
    """
    if dtype is None:
        return 0
    converted, was = 0, set()
    for name, model in getattr(pipeline, 'models', {}).items():
        if 'flow_model' not in name:
            continue
        convert = getattr(model, 'convert_to', None)
        if convert is None:
            continue
        was.add(str(getattr(model, 'dtype', 'unknown')).replace('torch.', ''))
        convert(dtype)
        converted += 1
    if verbose and converted:
        before = '/'.join(sorted(was)) or 'unknown'
        after = str(dtype).replace('torch.', '')
        note = '' if before != after else ' (already there, no change)'
        print(f"[pixal3d] flow model dtype: {before} -> {after} "
              f"({converted} models){note}")
    return converted


# The conditioning encoders (DINOv3 ViT-L, the NAF upsampler, the Flux VAE) are
# frozen fp32 modules that run four times per generation at up to 1024x1024.
# Autocast gives them fp16/bf16 matmuls without touching the stored weights, and
# the projection maths that genuinely needs fp32 already disables autocast
# locally.
_COND_DTYPE: Optional['torch.dtype'] = None  # noqa: F821


def resolve_cond_dtype(requested: str = 'auto'):
    """
    Autocast dtype for the conditioning encoders, or ``None`` for none.

    ``auto`` means "the fastest low-precision type this GPU has": float16 where
    there is no bfloat16 hardware, bfloat16 otherwise. ``float32`` means run
    them as upstream does, with autocast off.
    """
    import torch

    if requested == 'float32':
        return None
    if requested != 'auto':
        return resolve_dtype(requested)
    if not torch.cuda.is_available():
        return None
    return resolve_dtype('auto') or torch.bfloat16


def set_cond_dtype(dtype) -> None:
    global _COND_DTYPE
    _COND_DTYPE = dtype


def cond_autocast():
    """Autocast context for the image-conditioning encoders."""
    import contextlib
    import torch

    if _COND_DTYPE is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=_COND_DTYPE)


def sdpa_backends() -> Dict[str, bool]:
    """
    Which fused attention backends torch can actually use here.

    Without one of these, ``scaled_dot_product_attention`` falls back to the
    math kernel, which writes the whole score matrix to memory — the dominant
    cost of a run on hardware with no fused kernel.
    """
    import torch

    result = {'flash': False, 'mem_efficient': False, 'math': True}
    if not torch.cuda.is_available():
        return result

    query = torch.zeros(1, 4, 512, 64, device='cuda', dtype=torch.float16)
    try:
        from torch.backends.cuda import SDPAParams, can_use_flash_attention, can_use_efficient_attention
        params = SDPAParams(query, query, query, None, 0.0, False, False)
        result['flash'] = bool(can_use_flash_attention(params, False))
        result['mem_efficient'] = bool(can_use_efficient_attention(params, False))
    except Exception:
        # Older torch: fall back to the global enable flags, which at least say
        # whether the build has the backends compiled in.
        try:
            result['flash'] = bool(torch.backends.cuda.flash_sdp_enabled())
            result['mem_efficient'] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        except Exception:
            pass
    return result


def has_fused_attention() -> bool:
    backends = sdpa_backends()
    return backends['flash'] or backends['mem_efficient']


def configure(
    vram: str = 'auto',
    attn_backend: Optional[str] = None,
    conv_backend: Optional[str] = None,
    attn_chunk: Optional[int] = None,
    cfg_batch: Optional[bool] = None,
    verbose: bool = True,
) -> Preset:
    """
    Set process-wide environment defaults and return the chosen preset.

    Must run before torch initialises its allocator, i.e. before the first CUDA
    allocation, which in practice means before importing the pipeline.
    """
    # Fragmentation is the usual cause of OOM here: the sparse stages allocate
    # and free wildly different sizes every step. Both spellings are set because
    # ROCm builds read the HIP one and CUDA builds the other.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    # Lets PyTorch use AOTriton's prebuilt flash / memory-efficient attention on
    # RDNA3 and RDNA4. Ignored where AOTriton has no kernels for the target
    # (RDNA2 among them), so it is safe to set unconditionally.
    os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')

    if attn_backend:
        os.environ['ATTN_BACKEND'] = attn_backend
    if conv_backend:
        os.environ['SPARSE_CONV_BACKEND'] = conv_backend
    # The backend config modules read the environment once, at import. When
    # configure() runs a second time in a long-lived process — a Gradio server
    # or ComfyUI reloading a pipeline — push the change through directly.
    _apply_backends(attn_backend, conv_backend)

    preset = pick_preset(vram)
    if preset.attn_budget is not None:
        os.environ.setdefault('PIXAL3D_ATTN_BUDGET', str(preset.attn_budget))
    if cfg_batch is None:
        cfg_batch = preset.cfg_batch
    os.environ['PIXAL3D_CFG_BATCH'] = '1' if cfg_batch else '0'
    preset = replace(preset, cfg_batch=bool(cfg_batch))
    if attn_chunk is not None:
        os.environ['PIXAL3D_ATTN_CHUNK'] = str(attn_chunk)

    if verbose:
        print(describe())
        print(f"[pixal3d] vram preset: {preset.name} "
              f"(resolution {preset.resolution}, texture {preset.texture_size}, "
              f"low_vram={preset.low_vram}, block_offload={preset.block_offload}, "
              f"cfg_batch={preset.cfg_batch})")
    return preset


def free_memory() -> None:
    """Drop cached blocks — worth calling between pipeline stages on a small card."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable_block_offload(model, device: str = 'cuda') -> bool:
    """
    Stream a transformer's blocks between CPU and GPU one at a time.

    Detaches ``model.blocks`` from the module tree — keeping it reachable as a
    plain attribute — so the pipeline's own ``model.to(device)`` no longer drags
    every block onto the card, then moves each block in just for its forward
    pass. Weight residency drops from the whole 1.3B model to a single block, at
    the cost of streaming the weights over PCIe once per denoising step.

    Call this only after the weights are loaded: the detached blocks no longer
    appear in ``state_dict()``.
    """
    import torch.nn as nn

    blocks = getattr(model, 'blocks', None)
    if not isinstance(blocks, nn.Module):
        return False
    if getattr(model, '_pixal3d_block_offload', False):
        return True

    model._modules.pop('blocks', None)
    object.__setattr__(model, 'blocks', blocks)
    blocks.to('cpu')

    # Both hooks must return None: a pre-hook's return value replaces the
    # module's inputs and a forward hook's replaces its outputs, so returning
    # what `Module.to` gives back (the module itself) would corrupt the call.
    def _to_device(module, _inputs, dev=device):
        module.to(dev)

    def _to_host(module, _inputs, _output):
        module.to('cpu')

    for block in blocks:
        block.register_forward_pre_hook(_to_device)
        block.register_forward_hook(_to_host)

    object.__setattr__(model, '_pixal3d_block_offload', True)
    return True

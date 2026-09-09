"""
ComfyUI entry point.

Cloning this repository straight into ``ComfyUI/custom_nodes/`` is enough to
register the Pixal3D nodes — ComfyUI imports this file, which forwards to
``comfyui/nodes.py``. The import is guarded so a missing dependency shows up as
one warning instead of stopping ComfyUI from starting.
"""

try:
    from .comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _exc:                                   # pragma: no cover
    print(f"[Pixal3D] custom nodes unavailable: {type(_exc).__name__}: {_exc}")
    print("[Pixal3D] run scripts/check_env.py in ComfyUI's Python to diagnose")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

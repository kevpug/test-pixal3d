from typing import *
from ...compat.probe import default_attn_backend

VALID_BACKENDS = ['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa', 'naive']

BACKEND = 'flash_attn'
DEBUG = False

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in VALID_BACKENDS:
        BACKEND = env_attn_backend
    else:
        # flash-attn has no ROCm build for RDNA and none at all for Windows.
        # Fall back to whatever is installed rather than failing at import.
        BACKEND = default_attn_backend()
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# In-place initialisers that fill parameters with random numbers. Constructing
# a 1.3B DiT runs these over every weight, and the checkpoint then overwrites
# all of it — about seven seconds of pure waste per model, four models per run.
# Only the RNG-backed fills are stubbed. The deterministic ones (constant_,
# zeros_, ones_, eye_) are memsets that cost nothing, and skipping them could
# leave a non-persistent buffer holding whatever was in memory -- something
# load_state_dict would not report as a missing key.
_RANDOM_INIT_FUNCTIONS = (
    "uniform_", "normal_", "trunc_normal_", "dirac_",
    "xavier_uniform_", "xavier_normal_",
    "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_",
)


def _construct_without_random_init(name: str, args: dict, **kwargs):
    """
    Build a model with ``torch.nn.init``'s fills stubbed out.

    Everything a constructor computes rather than randomises — RoPE
    frequencies, positional tables, buffers — is still built normally; only the
    throwaway RNG is skipped. Returns ``None`` if construction fails, so the
    caller can fall back to the ordinary path.
    """
    import torch

    init = torch.nn.init
    saved = {fn: getattr(init, fn) for fn in _RANDOM_INIT_FUNCTIONS if hasattr(init, fn)}

    def noop(tensor, *args, **kwargs):
        return tensor

    try:
        for fn in saved:
            setattr(init, fn, noop)
        return __getattr__(name)(**args, **kwargs)
    except Exception:
        return None
    finally:
        for fn, original in saved.items():
            setattr(init, fn, original)


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.

    Set ``PIXAL3D_FAST_INIT=0`` to disable the skip-random-init fast path. The
    fast path is used only when the checkpoint supplies every parameter, so a
    partial checkpoint can never leave uninitialised weights behind.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    state = load_file(model_file)

    model = None
    if os.environ.get('PIXAL3D_FAST_INIT', '1') == '1':
        candidate = _construct_without_random_init(config['name'], config['args'], **kwargs)
        if candidate is not None:
            result = candidate.load_state_dict(state, strict=False)
            # A missing key means a weight the skipped RNG would have set; that
            # would leave uninitialised memory in the model, so start over.
            model = candidate if not result.missing_keys else None

    if model is None:
        model = __getattr__(config['name'])(**config['args'], **kwargs)
        model.load_state_dict(state, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder

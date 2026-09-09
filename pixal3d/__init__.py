"""
Pixal3D — pixel-aligned 3D generation from images.

Submodules are imported on first access rather than eagerly, matching the rest
of the package. That keeps ``pixal3d.runtime`` — which has to run before the
GPU is touched — importable without pulling in the model and renderer stacks.
"""

import importlib

__submodules = ['models', 'modules', 'pipelines', 'renderers', 'representations',
                'utils', 'compat', 'runtime']

__all__ = list(__submodules)


def __getattr__(name):
    if name in __submodules:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(set(globals()) | set(__submodules))


# For Pylance
if __name__ == '__main__':
    from . import models, modules, pipelines, renderers, representations, utils, compat, runtime

from typing import *
import os

import torch


def _cfg_batch_enabled() -> bool:
    """``PIXAL3D_CFG_BATCH=1`` merges the two guidance passes into one forward."""
    return os.environ.get('PIXAL3D_CFG_BATCH', '0') == '1'


def _cat_batch(a, b):
    """Concatenate two conditioning payloads along the batch dimension."""
    from ...modules.sparse import SparseTensor, sparse_cat
    from ...modules.sparse.basic import VarLenTensor, varlen_cat

    if isinstance(a, dict):
        return {key: _cat_batch(a[key], b[key]) for key in a}
    if isinstance(a, (tuple, list)):
        return type(a)(_cat_batch(x, y) for x, y in zip(a, b))
    # SparseTensor subclasses VarLenTensor, so it has to be tested first.
    if isinstance(a, SparseTensor):
        return sparse_cat([a, b], dim=0)
    if isinstance(a, VarLenTensor):
        return varlen_cat([a, b], dim=0)
    if isinstance(a, torch.Tensor):
        return torch.cat([a, b], dim=0)
    raise TypeError(f"Cannot batch conditioning of type {type(a).__name__}")


def _split_batch(prediction, batch_size: int):
    from ...modules.sparse import SparseTensor

    if isinstance(prediction, SparseTensor):
        return (prediction[list(range(batch_size))],
                prediction[list(range(batch_size, 2 * batch_size))])
    return prediction[:batch_size], prediction[batch_size:]


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _cfg_cache(self, x_t):
        """
        Per-run cache for the batched CFG inputs.

        A sampler is reused across passes (LR shape, HR shape, texture), so the
        cache holds a single slot keyed on the identity of the current run's
        coordinates. Rebuilding the batched skeleton every step would be wrong
        as well as slow: ``sparse_cat`` returns a fresh SparseTensor with an
        empty spatial cache, which would force the windowed-attention
        serialisation to be recomputed on every denoising step.
        """
        from ...modules.sparse import SparseTensor, sparse_cat

        key = x_t.coords if isinstance(x_t, SparseTensor) else x_t.shape
        cache = getattr(self, '_pixal3d_cfg_cache', None)
        if cache is None or cache['key'] is not key:
            cache = {'key': key, 'template': None, 'cond': {}}
            if isinstance(x_t, SparseTensor):
                cache['template'] = sparse_cat([x_t, x_t], dim=0)
            self._pixal3d_cfg_cache = cache
        return cache

    def _cfg_batch_cond(self, cache, name, value, other=None):
        """Batch ``value`` once per run, keeping a strong ref so ids stay valid."""
        entry = cache['cond'].get(name)
        if entry is not None and entry[0] is value and entry[1] is other:
            return entry[2]
        batched = _cat_batch(value, value if other is None else other)
        cache['cond'][name] = (value, other, batched)
        return batched

    def _cfg_batched_prediction(self, model, x_t, t, cond, neg_cond, **kwargs):
        """
        One batch-2N forward instead of two batch-N forwards: identical
        arithmetic, half the kernel launches, but roughly double the peak
        activation memory -- hence the opt-in.
        """
        from ...modules.sparse import SparseTensor
        from ...modules.sparse.basic import VarLenTensor

        batch_size = x_t.shape[0]
        cache = self._cfg_cache(x_t)
        if isinstance(x_t, SparseTensor):
            x_in = cache['template'].replace(torch.cat([x_t.feats, x_t.feats], dim=0))
        else:
            x_in = torch.cat([x_t, x_t], dim=0)

        cond_in = self._cfg_batch_cond(cache, '__cond__', cond, neg_cond)
        # Extra conditioning tensors (e.g. ``concat_cond``) are shared by both
        # guidance branches but still have to be duplicated to line up rows.
        batched_kwargs = dict(kwargs)
        for name, value in kwargs.items():
            if isinstance(value, (torch.Tensor, VarLenTensor)) and value.shape[0] == batch_size:
                batched_kwargs[name] = self._cfg_batch_cond(cache, name, value)

        batched = super()._inference_model(model, x_in, t, cond_in, **batched_kwargs)
        return _split_batch(batched, batch_size)

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            if _cfg_batch_enabled():
                pred_pos, pred_neg = self._cfg_batched_prediction(
                    model, x_t, t, cond, neg_cond, **kwargs)
            else:
                pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
                pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred

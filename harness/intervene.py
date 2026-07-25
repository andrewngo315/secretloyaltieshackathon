from __future__ import annotations
from contextlib import contextmanager

import torch


def _decoder_layers(model):
    return model.model.layers


def _apply_edit(hidden: torch.Tensor, direction: torch.Tensor, mode: str,
                alpha: float) -> torch.Tensor:
    d = direction.to(hidden.dtype).to(hidden.device)
    d = d / d.norm()
    if mode == "ablate":
        coef = (hidden * d).sum(-1, keepdim=True)
        return hidden - coef * d
    if mode == "steer":
        return hidden + alpha * d
    raise ValueError(mode)


@contextmanager
def edited(model, direction, mode: str, alpha: float = 1.0):
    dvec = torch.as_tensor(direction, dtype=torch.float32)
    handles = []

    def make_hook():
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = _apply_edit(output[0], dvec, mode, alpha)
                return (h,) + tuple(output[1:])
            return _apply_edit(output, dvec, mode, alpha)
        return hook

    try:
        for layer in _decoder_layers(model):
            handles.append(layer.register_forward_hook(make_hook()))
        yield
    finally:
        for h in handles:
            h.remove()


def norm_matched_random_direction(reference, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(len(reference))
    return (v / np.linalg.norm(v)).tolist()
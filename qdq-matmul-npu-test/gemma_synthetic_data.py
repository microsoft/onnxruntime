"""Shared synthetic distributions for Gemma model generation and evaluation."""

from __future__ import annotations

import numpy as np


WEIGHT_STD = 0.02
WEIGHT_CLIP = 0.1
NORM_WEIGHT_MEAN = 1.0
NORM_WEIGHT_CLIP = (0.9, 1.1)
HIDDEN_STATE_STD = 1.0
HIDDEN_STATE_CLIP = 4.0
DEFAULT_PADDING_FRACTION = 0.05
PADDING_MASK_VALUE = -10000.0


def bounded_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Generate deterministic finite normal values clipped to explicit bounds."""
    values = rng.normal(mean, std, shape)
    return np.clip(values, minimum, maximum).astype(np.float32)


def make_hidden_states(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    std: float = HIDDEN_STATE_STD,
    clip: float = HIDDEN_STATE_CLIP,
) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        raise ValueError("hidden-state standard deviation must be positive and finite")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("hidden-state clip must be positive and finite")
    return bounded_normal(rng, shape, 0.0, std, -clip, clip)


def make_additive_attention_mask(
    shape: tuple[int, ...],
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
) -> np.ndarray:
    """Create a finite additive mask with trailing positions marked as padding."""
    if not 0.0 <= padding_fraction < 1.0:
        raise ValueError("padding fraction must be in the range [0, 1)")
    if not shape or shape[-1] < 1:
        raise ValueError("attention-mask shape must have a non-empty sequence axis")

    mask = np.zeros(shape, dtype=np.float32)
    padded_tokens = int(np.ceil(shape[-1] * padding_fraction))
    if padded_tokens:
        mask[..., -padded_tokens:] = PADDING_MASK_VALUE
    return mask

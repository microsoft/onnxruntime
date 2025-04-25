from typing import Any

import torch
from transformers.cache_utils import DynamicCache, MambaCache

############
# MambaCache
############


# self.conv_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.conv_kernel_size,
#     device=device,
#     dtype=dtype,
# )
# self.ssm_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.ssm_state_size,
#     device=device,
#     dtype=dtype,
# )
def flatten_mamba_cache(
    mamba_cache: MambaCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`MambaCache` with python objects."""
    flat = [
        (k, getattr(mamba_cache, k))
        for k in [
            # "max_batch_size",  # new in transformers==4.47
            # "intermediate_size",
            # "ssm_state_size",
            # "conv_kernel_size",
            "conv_states",
            "ssm_states",
        ]
        if hasattr(mamba_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(
    values: list[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> MambaCache:
    """Restores a :class:`MambaCache` from python objects."""
    conv_states, ssm_states = values

    class _config:
        def __init__(self):
            if isinstance(conv_states, list):
                self.intermediate_size = conv_states[0].shape[1]
                self.state_size = ssm_states[0].shape[2]
                self.conv_kernel = conv_states[0].shape[2]
                self.num_hidden_layers = len(conv_states)
            else:
                self.intermediate_size = conv_states.shape[2]
                self.state_size = ssm_states.shape[3]
                self.conv_kernel = conv_states.shape[3]
                self.num_hidden_layers = conv_states.shape[0]

    cache = MambaCache(
        _config(),
        max_batch_size=1,
        dtype=values[-1][0].dtype,
        device="cpu" if values[-1][0].get_device() < 0 else "cuda",
    )
    values = dict(zip(context, values, strict=False))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_mamba_cache(
    d: dict[Any, Any],
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`MambaCache` with python objects."""
    values, context = flatten_mamba_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values, strict=False)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`DynamicCache` with python objects."""
    flat = [(k, getattr(dynamic_cache, k)) for k in ["key_cache", "value_cache"] if hasattr(dynamic_cache, k)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(
    d: dict[Any, Any],
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`DynamicCache` with python objects."""
    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values, strict=False)], context


def unflatten_dynamic_cache(
    values: list[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> DynamicCache:
    """Restores a :class:`DynamicCache` from python objects."""
    cache = DynamicCache()
    values = dict(zip(context, values, strict=False))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache

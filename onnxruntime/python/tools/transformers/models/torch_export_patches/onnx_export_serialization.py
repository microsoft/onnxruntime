from typing import Any, Dict, List, Tuple
import torch
import transformers

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
    mamba_cache: transformers.cache_utils.MambaCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
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
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.MambaCache:
    """Restores a :class:`transformers.cache_utils.MambaCache` from python objects."""
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

    from transformers.cache_utils import MambaCache

    cache = MambaCache(
        _config(),
        max_batch_size=1,
        dtype=values[-1][0].dtype,
        device="cpu" if values[-1][0].get_device() < 0 else "cuda",
    )
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_mamba_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    import torch

    values, context = flatten_mamba_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: transformers.cache_utils.DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    flat = [
        (k, getattr(dynamic_cache, k))
        for k in ["key_cache", "value_cache"]
        if hasattr(dynamic_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    import torch

    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_dynamic_cache(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache

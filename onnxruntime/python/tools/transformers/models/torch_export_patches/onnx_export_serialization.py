# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import pprint
from collections.abc import Callable
from typing import Any

import optree
import packaging.version as pv
import torch
import transformers
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, MambaCache
from transformers.modeling_outputs import BaseModelOutput

PATCH_OF_PATCHES: set[Any] = set()


def _register_class_serialization(
    cls,
    f_flatten: Callable,
    f_unflatten: Callable,
    f_flatten_with_keys: Callable,
    f_check: Callable | None = None,
    verbose: int = 0,
) -> bool:
    if cls is not None and cls in torch.utils._pytree.SUPPORTED_NODES:
        return False

    if verbose:
        print(f"[_register_cache_serialization] register {cls}")
    torch.utils._pytree.register_pytree_node(
        cls,
        f_flatten,
        f_unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=f_flatten_with_keys,
    )
    if pv.Version(torch.__version__) < pv.Version("2.7"):
        if verbose:
            print(f"[_register_cache_serialization] register {cls} for torch=={torch.__version__}")
        torch.fx._pytree.register_pytree_flatten_spec(cls, lambda x, _: f_flatten(x)[0])

    # check
    if f_check:
        from . import string_type

        inst = f_check()
        values, spec = torch.utils._pytree.tree_flatten(inst)
        restored = torch.utils._pytree.tree_unflatten(values, spec)
        assert string_type(inst, with_shape=True) == string_type(restored, with_shape=True), (
            f"Issue with registration of class {cls} "
            f"inst={string_type(inst, with_shape=True)}, "
            f"restored={string_type(restored, with_shape=True)}"
        )
    return True


def _register_cache_serialization(verbose: int = 0) -> dict[str, bool]:
    # DynamicCache serialization is different in transformers and does not
    # play way with torch.export.export.
    # see test test_export_dynamic_cache_cat with NOBYPASS=1
    # :: NOBYBASS=1 python _unittests/ut_torch_export_patches/test_dynamic_class.py -k e_c
    # This is caused by this line:
    # torch.fx._pytree.register_pytree_flatten_spec(
    #           DynamicCache, _flatten_dynamic_cache_for_fx)
    # so we remove it anyway
    if (
        DynamicCache in torch.utils._pytree.SUPPORTED_NODES
        and DynamicCache not in PATCH_OF_PATCHES
        # and pv.Version(torch.__version__) < pv.Version("2.7")
        and pv.Version(transformers.__version__) >= pv.Version("4.50")
    ):
        if verbose:
            print(
                f"[_fix_registration] DynamicCache is unregistered and "
                f"registered first for transformers=={transformers.__version__}"
            )
        _unregister(DynamicCache, verbose=verbose)
        _register_class_serialization(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            flatten_with_keys_dynamic_cache,
            # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            verbose=verbose,
        )
        if verbose:
            print("[_fix_registration] DynamicCache done.")
        # To avoid doing it multiple times.
        PATCH_OF_PATCHES.add(DynamicCache)

    # BaseModelOutput serialization is incomplete.
    # It does not include dynamic shapes mapping.
    if BaseModelOutput in torch.utils._pytree.SUPPORTED_NODES and BaseModelOutput not in PATCH_OF_PATCHES:
        if verbose:
            print(
                f"[_fix_registration] BaseModelOutput is unregistered and "
                f"registered first for transformers=={transformers.__version__}"
            )
        _unregister(BaseModelOutput, verbose=verbose)
        _register_class_serialization(
            BaseModelOutput,
            flatten_base_model_output,
            unflatten_base_model_output,
            flatten_with_keys_base_model_output,
            verbose=verbose,
        )
        if verbose:
            print("[_fix_registration] BaseModelOutput done.")

        # To avoid doing it multiple times.
        PATCH_OF_PATCHES.add(BaseModelOutput)

    unregistered_dynamic_cache = _register_class_serialization(
        DynamicCache,
        flatten_dynamic_cache,
        unflatten_dynamic_cache,
        flatten_with_keys_dynamic_cache,
        # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
        verbose=verbose,
    )
    unregistered_base_model_output = _register_class_serialization(
        BaseModelOutput,
        flatten_base_model_output,
        unflatten_base_model_output,
        flatten_with_keys_base_model_output,
        verbose=verbose,
    )
    unregistered_encode_decode_cache = _register_class_serialization(
        EncoderDecoderCache,
        flatten_encoder_decoder_cache,
        unflatten_encoder_decoder_cache,
        flatten_with_keys_encoder_decoder_cache,
        verbose=verbose,
    )
    unregistered_mamba_cache = _register_class_serialization(
        MambaCache,
        flatten_mamba_cache,
        unflatten_mamba_cache,
        flatten_with_keys_mamba_cache,
        verbose=verbose,
    )

    return dict(
        DynamicCache=unregistered_dynamic_cache,
        MambaCache=unregistered_mamba_cache,
        EncoderDecoderCache=unregistered_encode_decode_cache,
        BaseModelOutput=unregistered_base_model_output,
    )


def _unregister(cls: type, verbose: int = 0):
    # torch.utils._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
    else:
        if cls in torch.utils._pytree.SUPPORTED_NODES:
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    optree.unregister_pytree_node(cls, namespace="torch")
    if cls in torch.utils._pytree.SUPPORTED_NODES:
        import packaging.version as pv

        if pv.Version(torch.__version__) < pv.Version("2.7.0"):
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    assert cls not in torch.utils._pytree.SUPPORTED_NODES, (
        f"{cls} was not successful unregistered "
        f"from torch.utils._pytree.SUPPORTED_NODES="
        f"{pprint.pformat(list(torch.utils._pytree.SUPPORTED_NODES))}"
    )
    if verbose:
        print(f"[_unregister_cache_serialization] unregistered {cls.__name__}")


def _unregister_cache_serialization(undo: dict[str, bool], verbose: int = 0):
    for cls in [MambaCache, DynamicCache, EncoderDecoderCache, BaseModelOutput]:
        if undo.get(cls.__name__, False):
            _unregister(cls, verbose)


############
# MambaCache
############


def flatten_mamba_cache(mamba_cache: MambaCache) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    flat = [
        ("conv_states", mamba_cache.conv_states),
        ("ssm_states", mamba_cache.ssm_states),
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(values: list[Any], context: torch.utils._pytree.Context, output_type=None) -> MambaCache:
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
    cache: MambaCache,
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    values, context = flatten_mamba_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values, strict=False)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(dynamic_cache: DynamicCache) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_dynamic_cache"):
        return transformers.cache_utils._flatten_dynamic_cache(dynamic_cache)
    flat = [("key_cache", dynamic_cache.key_cache), ("value_cache", dynamic_cache.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_with_keys_dynamic_cache"):
        return transformers.cache_utils._flatten_with_keys_dynamic_cache(dynamic_cache)
    values, context = flatten_dynamic_cache(dynamic_cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values, strict=False)], context


def unflatten_dynamic_cache(values: list[Any], context: torch.utils._pytree.Context, output_type=None) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    if hasattr(transformers.cache_utils, "_unflatten_dynamic_cache"):
        assert output_type is None, f"output_type={output_type} not supported"
        return transformers.cache_utils._unflatten_dynamic_cache(values, context)

    cache = transformers.cache_utils.DynamicCache()
    values = dict(zip(context, values, strict=False))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


#####################
# EncoderDecoderCache
#####################


def flatten_encoder_decoder_cache(ec_cache: EncoderDecoderCache) -> tuple[list[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten(dictionary)


def flatten_with_keys_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)


def unflatten_encoder_decoder_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> EncoderDecoderCache:
    """Restores a :class:`transformers.cache_utils.EncoderDecoderCache` from python objects."""
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    return EncoderDecoderCache(**dictionary)


#################
# BaseModelOutput
#################


def flatten_base_model_output(bo: BaseModelOutput) -> tuple[list[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.modeling_outputs.BaseModelOutput`
    with python objects.
    """
    return list(bo.values()), list(bo.keys())


def flatten_with_keys_base_model_output(
    bo: BaseModelOutput,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.modeling_outputs.BaseModelOutput`
    with python objects.
    """
    values, context = flatten_base_model_output(bo)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values, strict=False)], context


def unflatten_base_model_output(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> BaseModelOutput:
    """
    Restores a :class:`transformers.modeling_outputs.BaseModelOutput`
    from python objects.
    """
    return BaseModelOutput(**dict(zip(context, values, strict=False)))

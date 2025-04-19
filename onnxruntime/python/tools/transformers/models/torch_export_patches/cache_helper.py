# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import packaging.version as pv
import torch
import transformers
import transformers.cache_utils


def is_cache_dynamic_registered(fast: bool = False) -> bool:
    """
    Tells class :class:`transformers.cache_utils.DynamicCache` can be
    serialized and deserialized. Only then, :func:`torch.export.export`
    can export a model.

    :param fast: if True, do not check the serialization is ok as well
    :return: result
    """
    if fast:
        return transformers.cache_utils.DynamicCache in torch.utils._pytree.SUPPORTED_NODES
    bsize, nheads, slen, dim = 2, 4, 3, 7
    cache = make_dynamic_cache(
        [
            (
                torch.randn(bsize, nheads, slen, dim),
                torch.randn(bsize, nheads, slen, dim),
            )
            for i in range(2)
        ]
    )
    values, spec = torch.utils._pytree.tree_flatten(cache)
    cache2 = torch.utils._pytree.tree_unflatten(values, spec)
    return len(cache2.key_cache) == len(cache.value_cache)


def flatten_unflatten_for_dynamic_shapes(obj):
    """
    Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    :param obj: object from a custom class
    :return: the serialized object
    """
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children_specs:
        end += subspec.num_leaves
        subtrees.append(subspec.unflatten(flat[start:end]))
        start = end
    return subtrees


if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`
        """
        return transformers.cache_utils.DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    """
    Creates an EncoderDecoderCache.
    """
    return transformers.cache_utils.EncoderDecoderCache(
        self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache
    )

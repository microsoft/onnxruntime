import packaging.version as pv
import torch
from transformers import __version__ as transformers_version
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


def get_dynamic_cache_key_value(
    cache: DynamicCache,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if hasattr(cache, "key_cache"):
        return cache.key_cache, cache.value_cache
    return [layer.keys for layer in cache.layers], [layer.values for layer in cache.layers]


def is_cache_dynamic_registered(fast: bool = False) -> bool:
    """
    Tells class :class:`DynamicCache` can be
    serialized and deserialized. Only then, :func:`torch.export.export`
    can export a model.

    :param fast: if True, do not check the serialization is ok as well
    :return: result
    """
    if fast:
        return DynamicCache in torch.utils._pytree.SUPPORTED_NODES
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
    cache2_keys, _ = get_dynamic_cache_key_value(cache2)
    _, cache_values = get_dynamic_cache_key_value(cache)
    return len(cache2_keys) == len(cache_values)


if pv.Version(transformers_version) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> DynamicCache:
        """
        Creates an instance of :class:`DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`DynamicCache`
        """
        return DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> DynamicCache:
        """
        Creates an instance of :class:`DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`DynamicCache`
        """
        cache = DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_encoder_decoder_cache(
    self_attention_cache: DynamicCache,
    cross_attention_cache: DynamicCache,
) -> EncoderDecoderCache:
    """
    Creates an EncoderDecoderCache.
    """
    return EncoderDecoderCache(self_attention_cache, cross_attention_cache)

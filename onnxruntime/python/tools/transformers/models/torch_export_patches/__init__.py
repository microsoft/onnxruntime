import packaging.version as pv
import torch
import transformers


def is_torchdynamo_exporting() -> bool:
    "Tells if torch is exporting a model."
    import torch

    if not hasattr(torch.compiler, "is_exporting"):
        # torch.compiler.is_exporting requires torch>=2.7
        return False

    try:
        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_exporting()  # type: ignore
        except Exception:
            return False


def string_type(anything, **args):
    # too long
    return str(anything)


if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`

        Example:

        ::

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
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

        Example:

        ::

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    "Creates an EncoderDecoderCache."
    return transformers.cache_utils.EncoderDecoderCache(
        self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache
    )


def torch_deepcopy(value):
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(torch_deepcopy(v) for v in value)
    if isinstance(value, list):
        return [torch_deepcopy(v) for v in value]
    if isinstance(value, set):
        return {torch_deepcopy(v) for v in value}
    if isinstance(value, dict):
        return {k: torch_deepcopy(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, transformers.cache_utils.DynamicCache):
        return make_dynamic_cache(torch_deepcopy(list(zip(value.key_cache, value.value_cache, strict=False))))
    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(f"torch_deepcopy not implemented for type {type(value)}")

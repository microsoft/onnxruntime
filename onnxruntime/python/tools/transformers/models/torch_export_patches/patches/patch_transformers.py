import inspect
import sys
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _patch_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int | None = None,
):
    """Patched method."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )

    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window - 1

        context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
        # In this case, the current implementation of torch fails (17/12/2024).
        # Try model Phi-3.5-Mini-Instruct.
        mask = mask.masked_fill(context_mask, torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


if sys.version_info[:2] <= (3, 11):

    @dataclass
    class patched_AttentionMaskConverter:
        """
        Patches
        ``AttentionMaskConverter._make_causal_mask``.
        """

        _PATCHES_ = ["_make_causal_mask"]
        _PATCHED_CLASS_ = AttentionMaskConverter

        @staticmethod
        def _make_causal_mask(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
            sliding_window: int | None = None,
        ):
            """Patched method."""
            return _patch_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length, sliding_window)

else:

    @dataclass
    class patched_AttentionMaskConverter:
        """
        Patches
        ``AttentionMaskConverter._make_causal_mask``.
        """

        _PATCHES_ = ["_make_causal_mask"]
        _PATCHED_CLASS_ = AttentionMaskConverter

        @staticmethod
        def _make_causal_mask(
            self,
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
            sliding_window: int | None = None,
        ):
            """Patched method."""
            return _patch_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length, sliding_window)


class patched_DynamicCache:
    """
    Applies modifications implemented in PR
    `transformers/#36652 <https://github.com/huggingface/transformers/pull/36652>`_.
    """

    _PATCHES_ = ["reorder_cache", "update", "crop", "from_batch_splits", "get_seq_length"]
    _PATCHED_CLASS_ = DynamicCache

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states.
        A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or self.key_cache[layer_idx].numel() == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states`
        and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.
                No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif not self.key_cache[layer_idx].numel():  # prefers not t.numel() to len(t) == 0 to export the model
                # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length`
        in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        This is used in assisted decoding and contrastive search.
        """
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    @classmethod
    def from_batch_splits(cls, splits: list[DynamicCache]) -> DynamicCache:
        """This is the opposite of the above `batch_split()` method.
        This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx].numel()]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx].numel()]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache


class patched_GenerationMixin:
    """
    Applies modifications implemented in PR
    `transformers/#36652 <https://github.com/huggingface/transformers/pull/36652>`_.
    """

    _PATCHES_ = [
        "_cache_dependant_input_preparation",
        "_cache_dependant_input_preparation_exporting",
        "prepare_inputs_for_generation",
    ]
    _PATCHED_CLASS_ = GenerationMixin

    def _cache_dependant_input_preparation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Generic cache-dependent input preparation
        The code is put in a separate function to allow granular unit testing
        as it needs a different implementation to be exportable.

        If we have cache: let's slice `input_ids` through `cache_position`,
        to keep only the unprocessed tokens
        - Exception 1: when passing input_embeds,
          input_ids may be missing entries
        - Exception 2: some generation methods do special slicing of input_ids,
          so we don't need to do it here
        - Exception 3: with synced GPUs cache_position may go out of bounds,
          but we only want dummy token in that case.
        - Exception 4: If input_embeds are passed then slice it through
          `cache_position`, to keep only the unprocessed tokens and
          generate the first token for each sequence.
          Later use the generated Input ids for continuation.

        The current implementation does not rely on ``self`` and could be
        a class method. It is left as a standard method to be easily rewritten.
        Original code:

        .. code-block:: python

            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1
                cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
            return inputs_embeds, input_ids
        """
        return self._cache_dependant_input_preparation_exporting(input_ids, inputs_embeds, cache_position)

    def _cache_dependant_input_preparation_exporting(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        This method implements method ``_cache_dependant_input_preparation``
        with :func:`torch.cond` to make it exportable with :func:`torch.export.export`.
        The code is put in a separate function to allow granular unit testing.
        """
        if inputs_embeds is None:
            input_ids = input_ids[:, cache_position]
        else:
            # This is the code we need to implemented with torch.cond.
            # if input_ids.shape[1] == 0:
            #     inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            # else:
            #     if cache_position[-1] >= input_ids.shape[1]:
            #         input_ids = input_ids[:, -cache_position.shape[0] :]
            #     else:
            #         if input_ids.shape[1] != cache_position.shape[0]:
            #             input_ids = input_ids[:, cache_position]
            def branch_1(inputs_embeds, cache_position):
                return inputs_embeds[:, -cache_position.shape[0] :]

            def branch_2(input_ids, cache_position):
                return input_ids[:, -cache_position.shape[0] :]

            def branch_3(input_ids, cache_position):
                return input_ids[:, cache_position]

            inputs_embeds, input_ids = torch.cond(
                input_ids.shape[1] == 0,
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        branch_1(inputs_embeds, cache_position),
                        input_ids,
                    )
                ),
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        inputs_embeds,
                        torch.cond(
                            cache_position[-1] >= input_ids.shape[1],
                            branch_2,
                            lambda input_ids, cache_position: (
                                torch.cond(
                                    input_ids.shape[1] != cache_position.shape[0],
                                    branch_3,
                                    (lambda input_ids, cache_position: input_ids),
                                    [input_ids, cache_position],
                                )
                            ),
                            [input_ids, cache_position],
                        ),
                    )
                ),
                [input_ids, inputs_embeds, cache_position],
            )
        return inputs_embeds, input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation.
        In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation
        for expected arguments (different models might have different
        requirements for e.g. `past_key_values`).
        This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support
        # (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in
        # `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`.
        # Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling
        # `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(input_ids, inputs_embeds, cache_position)

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want
        # to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input
        # that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a
        # `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance,
            # to reduce recompilations. If the function to create
            # the 4D causal mask exists,
            # it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                pass
                # logger.warning_once(
                #     f"{self.__class__.__name__} has no "
                #     "`_prepare_4d_causal_attention_mask_with_cache_position` method "
                #     "defined in its base modeling class. "
                #     "Compiled forward passes will be sub-optimal. If you're "
                #     "writing code, see Llama for an example implementation. "
                #     "If you're a user, please report this "
                #     "issue on GitHub."
                # )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

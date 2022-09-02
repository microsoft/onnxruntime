import os
from typing import List, Tuple

import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from utils import export_summarization_enc_dec_past

from onnxruntime import InferenceSession, SessionOptions


class PastKeyValuesHelper:
    """Helper functions to process past key values for encoder-decoder model"""

    @staticmethod
    def group_by_self_and_cross(present_key_values, concat=False):
        """
        Split present state from grouped by layer to grouped by self/cross attention.
        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...), (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        :param present_key_values: From past_key_values of a model
        :param concat: If concat self attention with cross attention key/value to return
        """
        present_self: List[torch.Tensor] = []
        present_cross: List[torch.Tensor] = []
        for _, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            present_key_self, present_value_self, present_key_cross, present_value_cross = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        if concat:
            return present_self + present_cross
        else:
            return present_self, present_cross

    @staticmethod
    def get_input_names(past_key_values: Tuple[Tuple[torch.Tensor]], encoder=True):
        """Process input names of model wrapper

        Args:
            past_key_values (Tuple[Tuple[torch.Tensor]]): Consider `self` and `cross` past_key_values

        Returns:
            string: input names
        """
        names = []
        num_layers = len(past_key_values) // 4 if encoder else len(past_key_values)
        prefix = "past_" if not encoder else "present_"
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_cross_{i}", f"value_cross_{i}"]])
        return names

    @staticmethod
    def get_output_names(past_key_values: Tuple[Tuple[torch.Tensor]]):
        """Process output names of model wrapper

        Args:
            past_key_values (Tuple[Tuple[torch.Tensor]]): Only consider `self` past_key_values

        Returns:
            string: output names
        """
        names = []
        num_layers = len(past_key_values)
        prefix = "present_"
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
        return names

    @staticmethod
    def back_group_by_layer(past_key_values: Tuple[Tuple[torch.Tensor]]):
        """
        Reorder past state from grouped by self/cross attention to grouped by layer.
        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ..., past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),

        :param present_key_values: From past_key_values of a model
        """
        past_tuples = ()
        half_idx = len(past_key_values) // 2
        for i in range(len(past_key_values) // 4):
            idx = 2 * i
            past_tuples += (
                (
                    past_key_values[idx],
                    past_key_values[idx + 1],
                    past_key_values[half_idx + idx],
                    past_key_values[half_idx + idx + 1],
                ),
            )
        return past_tuples


def config_initialize(args):

    model_dir = args.model_dir
    config = BartConfig.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    if args.spm_path:
        tokenizer = BartTokenizer(args.spm_path, args.input_text, args.vocab_path, config=config)

    config.use_decoder = True
    assert tokenizer.eos_token_id == config.decoder_start_token_id

    config.do_blenderbot_90_layernorm = False
    config.extra_pos_embeddings = 2
    config.force_bos_token_to_be_generated = False
    config.static_position_embeddings = False

    return config, tokenizer


def model_initialize(config, tokenizer, args):
    """
    model initialization and input data preprcessing
    """

    model_dir = args.model_dir
    device = args.device
    input_text = args.input_text

    model = BartForConditionalGeneration.from_pretrained(model_dir, config=config).eval()

    model = model.to(device)

    lang = "__en__"
    features = [tokenizer.convert_tokens_to_ids(lang)]
    features.extend(
        tokenizer.encode_plus(input_text, add_special_tokens=False, max_length=510, truncation=True)["input_ids"]
    )
    features.append(tokenizer.eos_token_id)
    input_data = torch.LongTensor(features).unsqueeze(0).to(device)

    return model, input_data

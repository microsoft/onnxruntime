import os
import time
from typing import Any, Dict, Optional

import torch
from transformers import BartForConditionalGeneration, file_utils
from utils import export_helper


class DecoderInitWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, last_hidden_state, decoder_input_ids, attention_mask):
        encoder_outputs = file_utils.ModelOutput()
        encoder_outputs["last_hidden_state"] = last_hidden_state
        encoder_outputs["hidden_states"] = None
        encoder_outputs["attentions"] = None
        return self.m(
            None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )


class EncoderDecoderInit(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoderwrapper: DecoderInitWrapper):
        super().__init__()
        self.encoder = encoder
        self.decoder_init = decoderwrapper

    def forward(
        self, encoder_input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor, decoder_input_ids: torch.Tensor
    ):
        encoder_out = self.encoder(encoder_input_ids, encoder_attention_mask)

        encoder_output = encoder_out["last_hidden_state"]
        decoder_out = self.decoder_init(encoder_output, decoder_input_ids, encoder_attention_mask)
        present_self, present_cross = export_helper.PastKeyValuesHelper.group_by_self_and_cross(
            decoder_out.past_key_values
        )
        present = present_self + present_cross

        return decoder_out.logits, encoder_output, present


def _create_encoder_export(args, config):
    """
    Exports the encoder part of BART by inserting torch.onnx.export and
    wrapper torch.nn.Module into `_prepare_encoder_decoder_kwargs_for_generation`.

    Args:
        args: User input.
        config: BART config
    """

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:

        # retrieve encoder hidden states
        # 1. get encoder
        encoder = self.get_encoder()
        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = input_ids
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        encoder_input_ids = input_ids.type(torch.int32)
        encoder_attention_mask = encoder_kwargs["attention_mask"].type(torch.int32)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.int32, device=input_ids.device)
            * config.decoder_start_token_id
        )

        wrapped_decoder = DecoderInitWrapper(self).eval()
        encdecinit = EncoderDecoderInit(encoder, wrapped_decoder)
        # use results from encdecinit here to compare with ORT.
        _, _, present = encdecinit(encoder_input_ids, encoder_attention_mask, decoder_input_ids)
        output_past_names = export_helper.PastKeyValuesHelper.get_input_names(present, encoder=True)

        # random name to use in dynamic axes
        sequence_length = "1"
        num_heads = str(config.encoder_attention_heads)
        hidden_size = str(config.d_model)
        head_size = str(config.encoder_attention_heads)

        dynamic_axes = {
            "encoder_input_ids": {0: "batch", 1: "sequence"},
            "encoder_attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence", 2: hidden_size},
            "decoder_input_ids": {0: "batch", 1: sequence_length},
            "logits": {0: "batch", 1: sequence_length},
        }

        for name in output_past_names:
            if "cross" in name:
                dynamic_axes[name] = {
                    0: "batch",
                    1: num_heads,
                    2: "sequence",
                    3: head_size,
                }
            else:  # self attention past state
                dynamic_axes[name] = {
                    0: "batch",
                    1: num_heads,
                    2: sequence_length,
                    3: head_size,
                }

        model_path = os.path.join(args.output, "edinit.onnx")
        torch.onnx.export(
            encdecinit,
            (encoder_input_ids, encoder_attention_mask, decoder_input_ids),
            model_path,
            opset_version=args.opset_version,
            do_constant_folding=False,
            input_names=["encoder_input_ids", "encoder_attention_mask", "decoder_input_ids"],
            output_names=["logits", "encoder_hidden_states"] + output_past_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            verbose=True,
        )

        return model_kwargs

    return _prepare_encoder_decoder_kwargs_for_generation


def export_encoder(args):

    beam = args.num_beams
    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = args.repetition_penalty
    no_repeat_ngram_size = args.no_repeat_ngram_size

    config, tokenizer = export_helper.config_initialize(args)

    with torch.no_grad():

        model, input_data = export_helper.model_initialize(config, tokenizer, args)
        start_time = time.time()
        model._prepare_encoder_decoder_kwargs_for_generation = _create_encoder_export(args, config).__get__(
            model, BartForConditionalGeneration
        )

        pred_ids = model.generate(
            input_data,
            decoder_start_token_id=tokenizer.eos_token_id,
            num_beams=beam,
            num_return_sequences=beam,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        print(tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))

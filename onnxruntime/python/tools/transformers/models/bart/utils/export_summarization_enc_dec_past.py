# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import os
import time

import torch
from transformers import BartConfig, BartForConditionalGeneration, file_utils
from utils import export_helper

from onnxruntime import InferenceSession, SessionOptions


def decoder_config_update(config: BartConfig):
    """
    Add parameters into decoder config to help exporting. These parameters
    are later consumed with control flow and assertion

    Args:
        config: BART config
    """

    new_attributes = {
        "is_decoder_exported": False,
        "is_decoder_with_past_exported": False,
        "during_export": False,
        "expected_args": {"use_cache": True, "return_dict": True},
        "expected_inputs": ["decoder_input_ids", "attention_mask", "encoder_outputs"],
    }
    config.update(new_attributes)
    return config


class DecoderWrapper(torch.nn.Module):
    """main part of BART Decoder wrapper.

    We split BART into three parts: Encoder plus initial part of decodoer, main part of decodoer, and Beam Search.
    This module is the main part of decodoer.

    Attributes:
        model: the BART model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *past: torch.Tensor,
    ):
        encoder_outputs = file_utils.ModelOutput()
        encoder_outputs["last_hidden_state"] = encoder_hidden_states
        encoder_outputs["hidden_states"] = None
        encoder_outputs["attentions"] = None
        if len(past) == 0:
            past_key_values = None
        else:
            past_key_values = export_helper.back_group_by_layer(past)
        decoder_out = self.model(
            None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        present_self, _ = export_helper.group_by_self_and_cross(decoder_out.past_key_values)
        return decoder_out.logits, present_self


def _decoder_forward_wrapper(model: BartForConditionalGeneration, decoder_config: BartConfig, args):
    """Wrapper function to pass in args and config.

    Exporting the decoder of BART by inserting torch.onnx.export and wrapper
    torch.nn.Module into `forward`.

    Args:
        model: BartForConditionalGeneration module.
        args: User input.
        decoder_config: BART config after `decoder_config_update`.
    """
    model._forward = model.forward

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        # attention_mask is required to show as forward parameters based on `generation_utils.py`
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py#L1200-L1203
        kwargs["attention_mask"] = attention_mask

        if decoder_config.is_decoder_exported and decoder_config.is_decoder_with_past_exported:
            outputs = self._forward(input_ids, **kwargs)
            return outputs

        inputs = []
        for k, v in decoder_config.expected_args.items():
            assert k in kwargs, f"Expecting {k} in kwargs."
            assert v == kwargs[k], f"For {k}, expected {v}, got {kwargs[k]}."

        for k in decoder_config.expected_inputs:
            assert k in kwargs
            if k == "encoder_outputs":
                inputs.append(kwargs[k]["last_hidden_state"])
            elif kwargs[k] is not None:
                inputs.append(kwargs[k].type(torch.int32))

        # delete unused key//value
        for k in list(kwargs.keys()):
            if (
                k not in decoder_config.expected_args
                and k not in decoder_config.expected_inputs
                and k != "past_key_values"
            ):
                del kwargs[k]

        encoder_outputs = kwargs["encoder_outputs"]
        for k, v in encoder_outputs.items():
            if k == "last_hidden_state":
                assert v is not None
            else:
                assert v is None

        assert "past_key_values" in kwargs
        # need to preserve past inputs, because values are altered after forward.
        past_inputs = kwargs["past_key_values"]
        past_input_list = []
        if past_inputs is not None and not decoder_config.is_decoder_with_past_exported:
            past_input_list = copy.deepcopy(export_helper.group_by_self_and_cross(past_inputs, concat=True))

        # compare `decoder_out_pt` with ORT results.
        decoder_out_pt = self._forward(input_ids, **kwargs)
        past_outputs = decoder_out_pt.past_key_values

        if past_inputs is not None and decoder_config.is_decoder_with_past_exported:
            return decoder_out_pt
        if past_inputs is None and decoder_config.is_decoder_exported:
            return decoder_out_pt

        input_past_names = []
        onnx_model_path = os.path.join(args.output, "decoder.onnx")

        if past_inputs is not None:
            decoder_config.is_decoder_with_past_exported = True
            inputs.extend(past_input_list)
            input_past_names = export_helper.get_input_names(past_inputs, encoder=False)
            onnx_model_path = os.path.join(args.output, "decoder_past.onnx")
        else:
            decoder_config.is_decoder_exported = True

        input_names = ["input_ids", "encoder_attention_mask", "encoder_hidden_states"] + input_past_names
        output_past_names = export_helper.get_output_names(past_outputs)
        output_names = ["logits"] + output_past_names

        sequence_length = "1"
        num_heads = str(decoder_config.encoder_attention_heads)
        head_size = str(decoder_config.encoder_attention_heads)
        hidden_size = str(decoder_config.d_model)

        dynamic_axes = {
            "input_ids": {0: "batch", 1: sequence_length},
            "encoder_attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence", 2: hidden_size},
            "logits": {0: "batch", 1: sequence_length},
        }

        for name in output_past_names:
            if "self" in name:
                dynamic_axes[name] = {
                    0: "batch",
                    1: num_heads,
                    2: sequence_length,
                    3: head_size,
                }

        for name in input_past_names:
            if "self" in name:
                dynamic_axes[name] = {
                    0: "batch",
                    1: num_heads,
                    2: "sequence",
                    3: head_size,
                }

        wrapped_decoder = DecoderWrapper(model).eval()
        # export decoder with past
        if decoder_config.is_decoder_with_past_exported:
            logits, present = wrapped_decoder(inputs[0], inputs[1], inputs[2], *inputs[3:])
            decoder_config.during_export = True
            torch.onnx.export(
                wrapped_decoder,
                tuple(inputs),
                onnx_model_path,
                opset_version=args.opset_version,
                do_constant_folding=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=True,
            )
            decoder_config.during_export = False

            # Test the generated model with onnxruntime
            print("========== ORT inference test on Decoder ... ==========")
            ort_inputs = {name: value.cpu().numpy() for name, value in zip(input_names, inputs)}
            # NOTE: encoder_hidden_states is not used and deleted
            ort_inputs.pop("encoder_hidden_states")
            sess_options = SessionOptions()
            sess_options.log_severity_level = 4
            sess = InferenceSession(onnx_model_path, sess_options, providers=["CPUExecutionProvider"])
            out = sess.run(None, ort_inputs)

            for ort_out, torch_out in zip(out, [logits] + present):
                torch.testing.assert_close(ort_out, torch_out.cpu().numpy(), check_dtype=True, atol=1e-4, rtol=1e-2)

            print("========== [SUCCESS] ORT inference test on Decoder ==========")

        return decoder_out_pt

    return forward


def export_decoder(args):
    """Export BART decoder to ONNX.

    By replacing the inner function of `BartForConditionalGeneration` forward,
    we export encoder model into ONNX with the usage of model.genration()

    Args:
        args: User input.

    Return:
        Decoder ONNX model in the given output directory, or under onnx_models folder.
    """
    beam = args.num_beams
    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = args.repetition_penalty
    no_repeat_ngram_size = args.no_repeat_ngram_size

    config, tokenizer = export_helper.initialize_config(args)
    config.use_decoder = True
    config = decoder_config_update(config)

    with torch.no_grad():

        model, input_data = export_helper.initialize_model(config, tokenizer, args)
        start_time = time.time()

        model.forward = _decoder_forward_wrapper(model, config, args).__get__(model, BartForConditionalGeneration)

        pred_ids = model.generate(
            input_data,
            decoder_start_token_id=tokenizer.eos_token_id,
            num_beams=beam,
            num_return_sequences=beam,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=True,
        )
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        print(tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))

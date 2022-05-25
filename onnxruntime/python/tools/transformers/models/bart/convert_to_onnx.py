"""
isort:skip_file
"""

import sys
from json import decoder

import os
import time
from typing import Any, Dict, List, OrderedDict

import argparse
import transformers
import numpy as np
import onnxruntime
import torch
from transformers.file_utils import ModelOutput

class PastKeyValuesDictHelper:
    @staticmethod
    def get_names(past_dicts: List[Dict[str, Dict[str, torch.Tensor]]], is_input=True):
        names = []
        num_layers = len(past_dicts)
        prefix = "input_" if is_input else "output_"
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}", f"key_cross_{i}", f"value_cross_{i}"]])
        return names

    def from_dict_to_list(past_dicts: List[Dict[str, Dict[str, torch.Tensor]]]):
        past_list: List[torch.Tensor] = []
        for d in past_dicts:
            for _, t in d["self"].items():
                past_list.extend([t])
            for _, t in d["encoder_decoder"].items():
                past_list.extend([t])
        return past_list

    def from_list_to_dict(past_list: List[torch.Tensor]):
        past_dicts: List[Dict[str, Dict[str, torch.Tensor]]] = []
        for i in range(len(past_list) // 4):
            idx = 4 * i
            d = {
                "self": {
                    "prev_key": past_list[idx],
                    "prev_value": past_list[idx + 1],
                },
                "encoder_decoder": {
                    "prev_key": past_list[idx + 2],
                    "prev_value": past_list[idx + 3],
                }
            }
            past_dicts.extend([d])
        return past_dicts


opset_version = 14
onnx_encoder_path = "onnx_model/encoder.onnx"
onnx_decoder_path = "onnx_model/decoder.onnx"
onnx_decoder_with_past_path = "onnx_model/decoder_past.onnx"

# Wrapper to export encoder
def _prepare_encoder_decoder_kwargs_for_generation(
    self, input_ids: torch.LongTensor, model_kwargs) -> Dict[str, Any]:

    # retrieve encoder hidden states
    encoder = self.get_encoder()
    encoder_kwargs = {
        argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
    }
    model_kwargs["encoder_outputs"] = encoder(input_ids, **encoder_kwargs)

    # export encoder
    # NOTE: This method is called only once in model.generate(),
    #       hence it is fine to add onnx export here.
    torch.onnx.export(
        encoder, (input_ids, encoder_kwargs['attention_mask']), onnx_encoder_path,
        opset_version=opset_version,
        do_constant_folding=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_outputs"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_outputs": {0: "batch", 1: "sequence"}
        },
        verbose=True)

    # Evaluate in ORT
    encoder_sess = onnxruntime.InferenceSession(onnx_encoder_path,
        providers=["CPUExecutionProvider"])
    encoder_out_ort = encoder_sess.run(None, {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": encoder_kwargs["attention_mask"].cpu().numpy(),
    })
    encoder_out_pt = model_kwargs["encoder_outputs"]["last_hidden_state"]

    np.testing.assert_allclose(encoder_out_pt.cpu().numpy(), encoder_out_ort[0],
        atol=0.2, rtol=1e-2)

    return model_kwargs


class DecoderExportConfig:
    is_decoder_exported = False
    is_decoder_with_past_exported = False

    expected_args = {
        "use_cache": True,
        "return_dict": True,
    }

    expected_inputs = [
        "decoder_input_ids",
        "attention_mask",
        "encoder_outputs",
    ]



def _decoder_forward_wrapper(model, fwd, decoder_config):
    model._forward = model.forward

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, decoder_input_ids, attention_mask, last_hidden_state, *past):
            encoder_outputs = ModelOutput()
            encoder_outputs['last_hidden_state'] = last_hidden_state
            encoder_outputs['hidden_states'] = None
            encoder_outputs['attentions'] = None

            if len(past) == 0:
                past_key_values = None
            else:
                past_key_values = PastKeyValuesDictHelper.from_list_to_dict(past)

            return self.m(
                None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True)

    def forward(self, input_ids, **kwargs):
        if decoder_config.is_decoder_exported and decoder_config.is_decoder_with_past_exported:
            return self._forward(input_ids, **kwargs)

        inputs = []

        for k, v in decoder_config.expected_args.items():
            assert k in kwargs, f"Expecting {k} in kwargs."
            assert v == kwargs[k], f"For {k}, expected {v}, got {kwargs[k]}."

        for k in decoder_config.expected_inputs:
            assert k in kwargs
            if k == "encoder_outputs":
                inputs.append(kwargs[k]["last_hidden_state"])
            else:
                inputs.append(kwargs[k])

        for k, v in kwargs.items():
            assert k in decoder_config.expected_args or k in decoder_config.expected_inputs or k == "past_key_values", \
                f"Unexpected argument {k}."

        encoder_outputs = kwargs["encoder_outputs"]
        for k, v in encoder_outputs.items():
            if k == "last_hidden_state":
                assert v is not None
            else:
                assert v is None

        assert "past_key_values" in kwargs

        # NOTE: need to preserve past inputs, because values are altered after forward.
        past_inputs = kwargs["past_key_values"]
        past_input_list = []
        if past_inputs is not None and not decoder_config.is_decoder_with_past_exported:
            import copy
            past_input_list = copy.deepcopy(PastKeyValuesDictHelper.from_dict_to_list(past_inputs))


        decoder_out_pt = self._forward(input_ids, **kwargs)
        past_outputs = decoder_out_pt.past_key_values

        if past_inputs is not None and decoder_config.is_decoder_with_past_exported:
            return decoder_out_pt
        if past_inputs is None and decoder_config.is_decoder_exported:
            return decoder_out_pt

        input_past_names = []
        onnx_model_path = onnx_decoder_path

        if past_inputs is not None:
            decoder_config.is_decoder_with_past_exported = True
            inputs.extend(past_input_list)
            input_past_names = PastKeyValuesDictHelper.get_names(past_inputs)
            onnx_model_path = onnx_decoder_with_past_path
        else:
            decoder_config.is_decoder_exported = True

        input_names = ["decoder_input_ids", "attention_mask", "last_hidden_state"] + input_past_names
        output_past_names = PastKeyValuesDictHelper.get_names(past_outputs, False)
        output_names = ["logits"] + output_past_names + ["encoder_last_hidden_state"]
        dynamic_axes = {
            "decoder_input_ids": [0, 1],
            "attention_mask": [0, 1],
            "last_hidden_state": [0, 1],
            "logits": [0],
            "encoder_last_hidden_state": [0, 1],
        }
        for name in input_past_names + output_past_names:
            dynamic_axes[name] = [0, 2]

        wrapped_decoder = DecoderWrapper(model).eval()

        # export decoder with past
        torch.onnx.export(
            wrapped_decoder,
            tuple(inputs),
            onnx_model_path,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=True,
        )

        # Evaluate in ORT
        decoder_sess = onnxruntime.InferenceSession(onnx_model_path,
            providers=["CPUExecutionProvider"])
        decoder_inputs = {
            "decoder_input_ids": inputs[0].cpu().numpy(),  # .clone().repeat(2, 1)
            "attention_mask": inputs[1].cpu().numpy(),
            "last_hidden_state": inputs[2].cpu().numpy(),
        }
        for name, past in zip(input_past_names, past_input_list):
            decoder_inputs[name] = past.cpu().numpy()
        decoder_out_ort = decoder_sess.run(None, decoder_inputs)

        np.testing.assert_allclose(decoder_out_pt["logits"].cpu().numpy(), decoder_out_ort[0],
            atol=0.2, rtol=1e-2)
        np.testing.assert_allclose(decoder_out_pt["encoder_last_hidden_state"].cpu().numpy(), decoder_out_ort[-1],
            atol=0.2, rtol=1e-2)
        print("logits:", decoder_out_pt["logits"].shape, decoder_out_pt["logits"])
        print("encoder_last_hidden_state:", decoder_out_pt["encoder_last_hidden_state"].shape)
        past_output_list = PastKeyValuesDictHelper.from_dict_to_list(past_outputs)
        for past_pt, past_ort in zip(past_output_list, decoder_out_ort[1:-1]):
            print("past_pt: ", past_pt.shape)
            np.testing.assert_allclose(past_pt.cpu().numpy(), past_ort, atol=0.2, rtol=1e-2)

        return decoder_out_pt

    return forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')

    args = parser.parse_args

    model_dir = args.model_dir
    config = transformers.BartConfig.from_pretrained(model_dir)
    tokenizer = transformers.BartTokenizer(
        os.path.join(model_dir, 'sentencepiece.bpe.model'),
        os.path.join(model_dir, 'dict.src.txt'),
        os.path.join(model_dir, 'dict.tgt.txt'),
        config=config)

    # summarization
    config.extra_config["fs_args"]["max_len_b"] = 512
    # config.extra_config["fs_args"]["fp16"] = False

    # device_name = "cpu"
    device_name = "cuda"
    config.use_decoder = True

    with torch.no_grad():
        model = transformers.BartForConditionalGeneration.from_pretrained(model_dir, config=config).eval()
        model = model.to(device_name)

        input_text = (
            "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
            "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
            "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
        )
        beam = 5

        lang = '__en__'
        features = [tokenizer.convert_tokens_to_ids(lang)]
        features.extend(tokenizer.encode_plus(input_text, add_special_tokens=False, max_length=510, truncation=True)["input_ids"])
        features.append(tokenizer.eos_token_id)
        input_data = torch.LongTensor(features).unsqueeze(0).to(device_name)

        start_time = time.time()

        model._prepare_encoder_decoder_kwargs_for_generation = _prepare_encoder_decoder_kwargs_for_generation.__get__(model, BartForConditionalGeneration)
        model.forward = _decoder_forward_wrapper(model, model.forward, DecoderExportConfig).__get__(model, BartForConditionalGeneration)

        pred_ids = model.generate(
            input_data,
            decoder_start_token_id=tokenizer.eos_token_id,
            num_beams=beam,
            num_return_sequences=beam,
            min_length=10,
            max_length=256,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            use_cache=True)
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        print(tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))


if __name__ == "__main__":
    main()

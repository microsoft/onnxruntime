# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import sys
import re
from pathlib import Path
from typing import Union
import logging
import torch
from transformers import T5ForConditionalGeneration
from onnxruntime import InferenceSession
from t5_encoder import T5Encoder, T5EncoderHelper
from t5_decoder import T5DecoderInit, T5Decoder, T5DecoderHelper
from t5_encoder_decoder_init import T5EncoderDecoderInit, T5EncoderDecoderInitHelper

logger = logging.getLogger(__name__)

PRETRAINED_T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3B", "t5-11B"]


class T5Helper:

    @staticmethod
    def get_onnx_path(output_dir: str, model_name_or_path: str, suffix: str = "", new_folder: bool = False):
        """Build onnx path

        Args:
            output_dir (str): output directory
            model_name_or_path (str): pretrained model name, or path to the model checkpoint
            suffix (str, optional): suffix like "_encoder" or "_decoder_fp16" will be appended to file name. Defaults to None.
            new_folder (bool, optional): create a new directory for the model. Defaults to False.

        Returns:
            [type]: [description]
        """
        model_name = model_name_or_path
        if model_name not in PRETRAINED_T5_MODELS and not re.match(
                r'^[\w_-]+$', model_name_or_path):  # It is not a name, shall be a path
            assert os.path.isdir(model_name_or_path)
            model_name = Path(model_name_or_path).parts[-1]

        model_name += suffix

        dir = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(dir, model_name + ".onnx")

    @staticmethod
    def load_model(model_name_or_path, cache_dir, device, merge_encoder_and_decoder_init: bool = True):
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        decoder = T5Decoder(model.decoder, model.lm_head, model.config)
        decoder.eval().to(device)

        if merge_encoder_and_decoder_init:
            encoder_decoder_init = T5EncoderDecoderInit(model.encoder,
                                                        model.decoder,
                                                        model.lm_head,
                                                        model.config,
                                                        decoder_start_token_id=None)
            return {"encoder_decoder_init": encoder_decoder_init, "decoder": decoder}
        else:
            encoder = T5Encoder(model.encoder, model.config)
            encoder.eval().to(device)
            decoder_init = T5DecoderInit(model.decoder, model.lm_head, model.config)
            decoder_init.eval().to(device)
            return {"encoder": encoder, "decoder": decoder, "decoder_init": decoder_init}

    @staticmethod
    def export_onnx(model: Union[T5Encoder, T5Decoder, T5DecoderInit, T5EncoderDecoderInit],
                    device: torch.device,
                    onnx_model_path: str,
                    verbose: bool = True,
                    use_external_data_format: bool = False,
                    use_decoder_input_ids: bool = True):
        if isinstance(model, T5Encoder):
            T5EncoderHelper.export_onnx(model, device, onnx_model_path, verbose, use_external_data_format)
        elif isinstance(model, T5EncoderDecoderInit):
            T5EncoderDecoderInitHelper.export_onnx(model, device, onnx_model_path, use_decoder_input_ids, verbose,
                                                   use_external_data_format)
        else:
            T5DecoderHelper.export_onnx(model, device, onnx_model_path, verbose, use_external_data_format)

    @staticmethod
    def optimize_onnx(onnx_model_path,
                      optimized_model_path,
                      is_float16,
                      num_attention_heads,
                      hidden_size,
                      use_external_data_format=False):
        """ Optimize ONNX model with an option to convert it to use mixed precision.
        """
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from optimizer import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='bert',
                           num_heads=num_attention_heads,
                           hidden_size=hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=False)
        if is_float16:
            m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format)

    @staticmethod
    def verify_onnx(model: Union[T5Encoder, T5Decoder, T5DecoderInit, T5EncoderDecoderInit],
                    ort_session: InferenceSession, device: torch.device):
        """ Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good.
        """
        if isinstance(model, T5Encoder):
            return T5EncoderHelper.verify_onnx(model, ort_session, device)
        elif isinstance(model, T5EncoderDecoderInit):
            return T5EncoderDecoderInitHelper.verify_onnx(model, ort_session, device)
        else:
            return T5DecoderHelper.verify_onnx(model, ort_session, device)

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
from whisper_decoder import WhisperDecoder, WhisperDecoderHelper, WhisperDecoderInit
from whisper_encoder import WhisperEncoder, WhisperEncoderHelper
from whisper_encoder_decoder_init import WhisperEncoderDecoderInit, WhisperEncoderDecoderInitHelper

from onnxruntime import InferenceSession

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from float16 import float_to_float16_max_diff  # noqa: E402
from onnx_model import OnnxModel  # noqa: E402
from optimizer import optimize_model  # noqa: E402

logger = logging.getLogger(__name__)

PRETRAINED_WHISPER_MODELS = [
    "whisper-tiny",
    "whisper-tiny.en",
    "whisper-small",
    "whisper-small.en",
    "whisper-medium",
    "whisper-medium.en",
    "whisper-base",
    "whisper-base.en",
    "whisper-large",
    "whisper-large-v2",
]


class WhisperHelper:
    @staticmethod
    def get_onnx_path(
        output_dir: str,
        model_name_or_path: str,
        suffix: str = "",
        new_folder: bool = False,
    ) -> str:
        """Build onnx path

        Args:
            output_dir (str): output directory
            model_name_or_path (str): pretrained model name, or path to the model checkpoint
            suffix (str, optional): suffix like "_encoder" or "_decoder_fp16" will be appended to file name. Defaults to None.
            new_folder (bool, optional): create a new directory for the model. Defaults to False.

        Returns:
            str: path of onnx model
        """
        model_name = model_name_or_path
        if os.path.isdir(model_name_or_path):
            model_name = Path(model_name_or_path).parts[-1]
        else:
            model_name = model_name.split("/")[-1]

        model_name += suffix

        directory = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(directory, model_name + ".onnx")

    @staticmethod
    def load_model(
        model_name_or_path: str,
        cache_dir: str,
        device: torch.device,
        merge_encoder_and_decoder_init: bool = True,
        state_dict_path: str = "",
    ) -> Dict[str, torch.nn.Module]:
        """Load model given a pretrained name or path, then build models for ONNX conversion.

        Args:
            model_name_or_path (str): pretrained model name or path
            cache_dir (str): cache directory
            device (torch.device): device to run the model
            merge_encoder_and_decoder_init (bool, optional): Whether merge encoder and decoder initialization into one ONNX model. Defaults to True.
        Returns:
            Dict[str, torch.nn.Module]: mapping from name to modules for ONNX conversion.
        """
        model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if state_dict_path:
            model.load_state_dict(torch.load(state_dict_path), strict=False)

        decoder = WhisperDecoder(model, model.config)
        decoder.eval().to(device)

        if merge_encoder_and_decoder_init:
            encoder_decoder_init = WhisperEncoderDecoderInit(
                model,
                model,
                model.config,
                decoder_start_token_id=None,
            )
            return {"encoder_decoder_init": encoder_decoder_init, "decoder": decoder}
        else:
            encoder = WhisperEncoder(model.model.encoder, model.config)
            encoder.eval().to(device)
            decoder_init = WhisperDecoderInit(model.decoder, model.config)
            decoder_init.eval().to(device)
            return {
                "encoder": encoder,
                "decoder": decoder,
                "decoder_init": decoder_init,
            }

    @staticmethod
    def export_onnx(
        model: Union[WhisperEncoder, WhisperDecoder, WhisperDecoderInit, WhisperEncoderDecoderInit],
        device: torch.device,
        onnx_model_path: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_decoder_input_ids: bool = True,
        use_int32_inputs: bool = False,
    ):
        if isinstance(model, WhisperEncoder):
            WhisperEncoderHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                verbose,
                use_external_data_format,
            )
        elif isinstance(model, WhisperEncoderDecoderInit):
            WhisperEncoderDecoderInitHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                use_decoder_input_ids,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )
        else:
            WhisperDecoderHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )

    @staticmethod
    def auto_mixed_precision(
        onnx_model: OnnxModel,
        op_block_list: Tuple[str] = (
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "Relu",
            "Add",
        ),
    ):
        """Convert model to mixed precision.
           It detects whether original model has fp16 precision weights, and set parameters for float16 conversion automatically.
        Args:
            onnx_model (OnnxModel): optimized ONNX model
            op_block_list (List[str], optional): . Defaults to ["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Relu", "Add"]
        Returns:
            parameters(dict): a dictionary of parameters used in float16 conversion
        """
        op_full_set = set([node.op_type for node in onnx_model.nodes()])
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.info(f"fp32 op: {fp32_op_set} fp16 op: {fp16_op_set}")

        # logits is the first output
        logits_output_name = onnx_model.graph().output[0].name

        # We use the weight in last MatMul node to detect whether the model is stored with float16 weights from training.
        is_weight_fp16_precision = False
        output_name_to_node = onnx_model.output_name_to_node()
        assert logits_output_name in output_name_to_node
        node = output_name_to_node[logits_output_name]
        last_matmul_node = None
        if node.op_type == "MatMul":
            last_matmul_node = node
            logger.info(f"Found last MatMul node for logits: {node.name}")
            initializer = None
            for input in node.input:
                initializer = onnx_model.get_initializer(input)
                if initializer is not None:
                    break

            # when the max difference of value after converting float to float16 is lower than a threshold (1e-6),
            # we can deduce that the weights are stored in float16 precision.
            max_diff = float_to_float16_max_diff(initializer)
            logger.debug(f"max diff of converting weights in last MatMul node {node.name}: {max_diff}")
            is_weight_fp16_precision = max_diff < 1e-6
        else:
            logger.warning(f"Failed to find MatMul node for logits. Found {node.op_type} of node {node.name}")

        keep_io_types = []
        node_block_list = []
        if (not is_weight_fp16_precision) and (last_matmul_node is not None):
            # When original weight is float32 precision, keep logits and last MatMul in float32 could get better precision.
            keep_io_types = [logits_output_name]
            node_block_list = [last_matmul_node.name]

        parameters = {
            "keep_io_types": keep_io_types,
            "op_block_list": list(op_block_list),
            "node_block_list": node_block_list,
            "force_fp16_initializers": is_weight_fp16_precision,
        }

        logger.info(f"auto_mixed_precision parameters: {parameters}")
        onnx_model.convert_float_to_float16(use_symbolic_shape_infer=True, **parameters)

        return parameters

    @staticmethod
    def optimize_onnx(
        onnx_model_path: str,
        optimized_model_path: str,
        is_float16: bool,
        num_attention_heads: int,
        hidden_size: int,
        use_external_data_format: bool = False,
        auto_mixed_precision: bool = True,
        use_gpu: bool = False,
        provider: str = "cpu",
    ):
        """Optimize ONNX model with an option to convert it to use mixed precision."""

        from fusion_options import FusionOptions

        optimization_options = FusionOptions("bart")
        optimization_options.use_multi_head_attention = True
        optimization_options.disable_multi_head_attention_bias = provider == "rocm"

        m = optimize_model(
            onnx_model_path,
            model_type="bart",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=2 if not use_external_data_format else None,
            optimization_options=optimization_options,
            use_gpu=use_gpu,
            only_onnxruntime=False,
        )

        if is_float16:
            if auto_mixed_precision:
                WhisperHelper.auto_mixed_precision(m)
            else:
                m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format, all_tensors_to_one_file=True)

    @staticmethod
    def verify_onnx(
        model_name_or_path: str,
        ort_session: InferenceSession,
        device: torch.device,
    ):
        """Compare the result from PyTorch and ONNX Runtime to verify the ONNX model is good."""
        pt_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        processor = WhisperProcessor.from_pretrained(model_name_or_path)
        config = WhisperConfig.from_pretrained(model_name_or_path)

        # Try to import `datasets` pip package
        try:
            from datasets import load_dataset
        except Exception as e:
            logger.error(f"An error occurred while importing `datasets`: {e}", exc_info=True)
            install_cmd = "pip install datasets"
            logger.warning(f"Could not import `datasets`. Attempting to install `datasets` via `{install_cmd}`.")
            os.system(install_cmd)

        from datasets import load_dataset  # noqa: F811

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_features = processor([ds[0]["audio"]["array"]], return_tensors="pt").input_features

        batch_size, max_length, min_length, num_beams, num_return_sequences = 1, 26, 0, 5, 1
        length_penalty, repetition_penalty = 1.0, 1.0
        inputs = {
            "input_features": input_features.to(device),
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "early_stopping": True,
            "use_cache": True,
        }
        pt_outputs = pt_model.generate(**inputs).detach().cpu().numpy()

        del inputs["early_stopping"]
        del inputs["use_cache"]
        ort_names = list(map(lambda entry: entry.name, ort_session.get_inputs()))
        ort_dtypes = list(map(lambda entry: entry.type, ort_session.get_inputs()))
        ort_to_np = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
        }

        for name, dtype in zip(ort_names, ort_dtypes):
            if name == "input_features":
                inputs[name] = inputs[name].detach().cpu().numpy()
            elif name == "vocab_mask":
                inputs[name] = np.ones(config.vocab_size, dtype=ort_to_np[dtype])
            elif name == "prefix_vocab_mask":
                inputs[name] = np.ones((batch_size, config.vocab_size), dtype=ort_to_np[dtype])
            elif name == "decoder_input_ids":
                inputs[name] = np.array([[config.decoder_start_token_id, 50259, 50359, 50363]], dtype=ort_to_np[dtype])
            elif name == "logits_processor":
                inputs[name] = np.array([1], dtype=ort_to_np[dtype])
            else:
                inputs[name] = np.array([inputs[name]], dtype=ort_to_np[dtype])
        ort_outputs = ort_session.run(None, inputs)[0][0]

        if pt_outputs.shape != ort_outputs.shape:
            logger.warning("PyTorch and ONNX Runtime outputs do not have the same shape")

        diff = pt_outputs - ort_outputs
        max_diff = max(diff.min(), diff.max(), key=abs)

        if max_diff > 0:
            # For ONNX Runtime INT8 model
            pt_expected_transcription = (
                " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
            )
            pt_transcription = processor.batch_decode(pt_outputs, skip_special_tokens=True)
            ort_expected_transcription = (
                " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
            )
            ort_transcription = processor.batch_decode(ort_outputs, skip_special_tokens=True)

            parity = (
                pt_expected_transcription == pt_transcription[0] and ort_expected_transcription == ort_transcription[0]
            )
            if parity:
                max_diff = 0

        return max_diff

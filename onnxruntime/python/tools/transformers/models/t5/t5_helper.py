# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
from t5_decoder import T5Decoder, T5DecoderHelper, T5DecoderInit
from t5_encoder import T5Encoder, T5EncoderHelper
from t5_encoder_decoder_init import T5EncoderDecoderInit, T5EncoderDecoderInitHelper
from transformers import T5ForConditionalGeneration

from onnxruntime import InferenceSession

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from float16 import float_to_float16_max_diff
from fusion_utils import FusionUtils
from onnx_model import OnnxModel
from optimizer import optimize_model

logger = logging.getLogger(__name__)

PRETRAINED_T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]


class T5Helper:
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
            model_name.split("/")[-1]

        model_name += suffix

        dir = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(dir, model_name + ".onnx")

    @staticmethod
    def load_model(
        model_name_or_path: str,
        cache_dir: str,
        device: torch.device,
        merge_encoder_and_decoder_init: bool = True,
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
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        decoder = T5Decoder(model.decoder, model.lm_head, model.config)
        decoder.eval().to(device)

        if merge_encoder_and_decoder_init:
            encoder_decoder_init = T5EncoderDecoderInit(
                model.encoder,
                model.decoder,
                model.lm_head,
                model.config,
                decoder_start_token_id=None,
            )
            return {"encoder_decoder_init": encoder_decoder_init, "decoder": decoder}
        else:
            encoder = T5Encoder(model.encoder, model.config)
            encoder.eval().to(device)
            decoder_init = T5DecoderInit(model.decoder, model.lm_head, model.config)
            decoder_init.eval().to(device)
            return {
                "encoder": encoder,
                "decoder": decoder,
                "decoder_init": decoder_init,
            }

    @staticmethod
    def export_onnx(
        model: Union[T5Encoder, T5Decoder, T5DecoderInit, T5EncoderDecoderInit],
        device: torch.device,
        onnx_model_path: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_decoder_input_ids: bool = True,
        use_int32_inputs: bool = False,
    ):
        if isinstance(model, T5Encoder):
            T5EncoderHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )
        elif isinstance(model, T5EncoderDecoderInit):
            T5EncoderDecoderInitHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                use_decoder_input_ids,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )
        else:
            T5DecoderHelper.export_onnx(
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
        op_block_list: List[str] = [
            "Pow",
            "ReduceMean",
            "Add",
            "Sqrt",
            "Div",
            "Mul",
            "Softmax",
            "Relu",
        ],
    ):
        """Convert model to mixed precision.
           It detects whether original model has fp16 precision weights, and set parameters for float16 conversion automatically.
        Args:
            onnx_model (OnnxModel): optimized ONNX model
            op_block_list (List[str], optional): . Defaults to ["Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Softmax", "Relu"]
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
            "op_block_list": op_block_list,
            "node_block_list": node_block_list,
            "force_fp16_initializers": is_weight_fp16_precision,
        }

        logger.info(f"auto_mixed_precision parameters: {parameters}")
        onnx_model.convert_float_to_float16(use_symbolic_shape_infer=True, **parameters)

        fusion_utils = FusionUtils(onnx_model)
        fusion_utils.remove_cascaded_cast_nodes()
        fusion_utils.remove_useless_cast_nodes()

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
    ):
        """Optimize ONNX model with an option to convert it to use mixed precision."""
        m = optimize_model(
            onnx_model_path,
            model_type="bert",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=0,
            optimization_options=None,
            use_gpu=False,
        )
        if is_float16:
            if auto_mixed_precision:
                T5Helper.auto_mixed_precision(m)
            else:
                m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format)

    @staticmethod
    def verify_onnx(
        model: Union[T5Encoder, T5Decoder, T5DecoderInit, T5EncoderDecoderInit],
        ort_session: InferenceSession,
        device: torch.device,
        use_int32_inputs: bool,
    ):
        """Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good."""
        if isinstance(model, T5Encoder):
            return T5EncoderHelper.verify_onnx(model, ort_session, device, use_int32_inputs)

        if isinstance(model, T5EncoderDecoderInit):
            return T5EncoderDecoderInitHelper.verify_onnx(model, ort_session, device, use_int32_inputs)

        return T5DecoderHelper.verify_onnx(model, ort_session, device, use_int32_inputs)

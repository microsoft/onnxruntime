# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
This converts GPT2 or T5 model to onnx with beam search operator.

Example 1: convert gpt2 model with beam search:
    python convert_generation.py -m gpt2 --output gpt2_beam_search.onnx

Example 2: convert T5 model with beam search in two steps:
    cd ./models/t5
    python convert_to_onnx.py -m t5-small
    cd ../..
    python convert_generation.py -m t5-small --model_type t5                                   \
        --decoder_onnx ./models/t5/onnx_models/t5-small_decoder.onnx                            \
        --encoder_decoder_init_onnx ./models/t5/onnx_models/t5-small_encoder_decoder_init.onnx  \
        --output ./models/t5/onnx_models/t5_small_beam_search.onnx

Example 3: convert T5 model with beam search. All in one step:
    python convert_generation.py -m t5-small --model_type t5 --output ./models/t5/onnx_models/t5_small_beam_search.onnx

Example 4: convert MT5 model with external data file like mt5-base-beamsearch.onnx.data in below example.
    python convert_generation.py -m google/mt5-base --model_type mt5 --output mt5-base-beamsearch.onnx -e

Example 5: convert gpt2 model with greedy search:
    python convert_generation.py -m gpt2 --output gpt2_greedy_search.onnx --num_beams 1 --num_return_sequences 1
"""

import argparse
import logging
import math
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx
import torch
from benchmark_helper import Precision
from fusion_utils import NumpyHelper
from onnx import GraphProto, ModelProto, TensorProto
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    MT5Config,
    MT5ForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_available_providers

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "gpt2"))
from gpt2_helper import PRETRAINED_GPT2_MODELS  # noqa: E402
from models.gpt2.convert_to_onnx import main as convert_gpt2_to_onnx  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "t5"))
from benchmark_helper import setup_logger
from models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models  # noqa: E402
from models.t5.t5_helper import PRETRAINED_MT5_MODELS, PRETRAINED_T5_MODELS  # noqa: E402
from onnx_model import OnnxModel

logger = logging.getLogger("")


class GenerationType(Enum):
    BEAMSEARCH = "beam_search"
    GREEDYSEARCH = "greedy_search"

    def __str__(self):
        return self.value


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments

    Args:
        argv (Optional[List[str]], optional): _description_. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    input_group = parser.add_argument_group("Input options")

    input_group.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        type=str,
        help="Pytorch model checkpoint path, or pretrained model name in the list: "
        + ", ".join(PRETRAINED_GPT2_MODELS + PRETRAINED_T5_MODELS + PRETRAINED_MT5_MODELS),
    )

    input_group.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5", "mt5"],
        help="Model type (default is gpt2) in the list: " + ", ".join(["gpt2", "t5", "mt5"]),
    )

    input_group.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    input_group.add_argument(
        "--decoder_onnx",
        required=False,
        type=str,
        default="",
        help="Path of onnx model for decoder. Specify it when you have exported the model.",
    )

    input_group.add_argument(
        "--encoder_decoder_init_onnx",
        required=False,
        type=str,
        default="",
        help="Path of ONNX model for encoder and decoder initialization. Specify it when you have exported the model.",
    )

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Print more information",
    )
    parser.set_defaults(verbose=False)

    output_group = parser.add_argument_group("Output options")

    output_group.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output path for onnx model with beam search.",
    )

    output_group.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16],
        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision",
    )

    output_group.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="save external data for model > 2G",
    )
    output_group.set_defaults(use_external_data_format=False)

    output_group.add_argument(
        "-s", "--run_shape_inference", required=False, action="store_true", help="run shape inference"
    )
    output_group.set_defaults(run_shape_inference=False)

    output_group.add_argument(
        "-pvs",
        "--pad_vocab_size",
        required=False,
        action="store_true",
        help="Pad logits MatMul weight to be a multiple of 8 along the dimension where dim value is the vocab size",
    )
    output_group.set_defaults(pad_vocab_size=True)

    output_group.add_argument(
        "-i",
        "--disable_shared_initializers",
        required=False,
        action="store_true",
        help="do not share initializers in encoder and decoder. It will increase memory usage of t5/mt5 models.",
    )
    output_group.set_defaults(disable_shared_initializers=False)

    model_group = parser.add_argument_group("Beam search parameters that stored in the output model")

    model_group.add_argument(
        "--output_sequences_scores",
        required=False,
        action="store_true",
        help="output sequences scores",
    )
    model_group.set_defaults(output_sequences_scores=False)

    model_group.add_argument(
        "--output_token_scores",
        required=False,
        action="store_true",
        help="output token scores",
    )
    model_group.set_defaults(output_token_scores=False)

    model_group.add_argument("--early_stopping", required=False, action="store_true")
    model_group.set_defaults(early_stopping=False)

    model_group.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        required=False,
        default=0,
        help="No repeat ngram size",
    )

    model_group.add_argument(
        "--vocab_mask",
        required=False,
        action="store_true",
        help="Enable vocab_mask. This mask applies only to every generated token to filter some bad words.",
    )
    model_group.set_defaults(vocab_mask=False)

    model_group.add_argument(
        "--prefix_vocab_mask",
        required=False,
        action="store_true",
        help="Enable prefix_vocab_mask. This mask can be used to filter bad words in the first generated token only",
    )
    model_group.set_defaults(prefix_vocab_mask=False)

    model_group.add_argument(
        "--custom_attention_mask",
        required=False,
        action="store_true",
        help="Enable custom_attention_mask. This mask can be used to replace default encoder attention mask",
    )
    model_group.set_defaults(custom_attention_mask=False)

    beam_parameters_group = parser.add_argument_group(
        "Beam search parameters not stored in the output model, for testing parity and performance"
    )

    beam_parameters_group.add_argument("--min_length", type=int, required=False, default=1, help="Min sequence length")

    beam_parameters_group.add_argument("--max_length", type=int, required=False, default=50, help="Max sequence length")

    beam_parameters_group.add_argument("--num_beams", type=int, required=False, default=4, help="Beam size")

    beam_parameters_group.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=1,
        help="Number of return sequence <= num_beams",
    )

    beam_parameters_group.add_argument(
        "--length_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage short sentence.",
    )

    beam_parameters_group.add_argument(
        "--repetition_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage.",
    )

    beam_parameters_group.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=-1,
        help="Vocab_size of the underlying model used to decide the shape of vocab mask",
    )

    test_group = parser.add_argument_group("Other options for testing parity and performance")

    test_group.add_argument(
        "--use_gpu", required=False, action="store_true", help="use GPU for inference. Required for fp16."
    )
    test_group.set_defaults(use_gpu=False)

    test_group.add_argument(
        "--disable_parity",
        required=False,
        action="store_true",
        help="do not run parity test",
    )
    test_group.set_defaults(disable_parity=False)

    test_group.add_argument(
        "--torch_performance",
        required=False,
        action="store_true",
        help="test PyTorch performance",
    )
    test_group.set_defaults(torch_performance=False)

    test_group.add_argument(
        "--total_runs",
        required=False,
        type=int,
        default=1,
        help="Number of times of inference for latency measurement",
    )

    test_group.add_argument(
        "--save_test_data",
        required=False,
        action="store_true",
        help="save test data for onnxruntimer_perf_test tool",
    )
    test_group.set_defaults(save_test_data=False)

    args = parser.parse_args(argv)

    return args


def gpt2_to_onnx(args: argparse.Namespace):
    """Convert GPT-2 model to onnx

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    model_name = args.model_name_or_path

    arguments = [
        "--model_name_or_path",
        model_name,
        "--output",
        args.decoder_onnx,
        "--optimize_onnx",
        "--precision",
        "fp32" if args.precision == Precision.FLOAT32 else "fp16",
        "--test_runs",
        "1",
        "--test_cases",
        "10",
        "--use_int32_inputs",  # BeamSearch requires to use int32 for input_ids, position_ids and attention_mask
        "--overwrite",  # Overwrite onnx file if existed
    ]
    if args.use_gpu:
        arguments.append("--use_gpu")
    if args.use_external_data_format:
        arguments.append("--use_external_data_format")

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 or mixed precision model cannot run in CPU. Please add --use_gpu"
        # TODO(tianleiwu): Use auto mixed precision for fp16 conversion: arguments.append('--auto_mixed_precision')
        #       Need change cuda kernel to support a combination of fp32 logits and fp16 past state.
        #       Currently logits and past state shall be same data type.
        arguments.extend(["--op_block_list", "Add", "LayerNormalization", "FastGelu"])

    if args.verbose:
        logger.info(f"arguments for convert_to_onnx:{arguments}")

    convert_gpt2_to_onnx(argv=arguments)


def t5_to_onnx(args: argparse.Namespace):
    """Convert T5 model to onnx

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    paths = export_t5_onnx_models(
        args.model_name_or_path,
        args.cache_dir,
        Path(args.output).parent,
        use_gpu=args.use_gpu,
        use_external_data_format=args.use_external_data_format,
        optimize_onnx=False,
        precision=args.precision,
        verbose=False,
        use_decoder_start_token=False,
        merge_encoder_and_decoder_init=True,
        overwrite=True,
        disable_auto_mixed_precision=False,
        use_int32_inputs=True,
        model_type=args.model_type,
    )

    logger.debug(f"onnx model for encoder: {paths[0]}")
    logger.debug(f"onnx model for decoder: {paths[1]}")
    args.encoder_decoder_init_onnx = paths[0]
    args.decoder_onnx = paths[1]


def shape_inference(onnx_path: str, use_external_data_format: bool = True):
    """Shape inference on an onnx file, which will be overwritten.

    Args:
        onnx_path (str): Path of onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """
    # Run symbolic shape inference to walk around ORT shape inference issue for subgraph.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    model = onnx.load_model(onnx_path, load_external_data=True)
    out = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=False)
    if out:
        OnnxModel.save(out, onnx_path, save_as_external_data=use_external_data_format)
    else:
        logger.warning("Failed to run symbolic shape inference on the model.")


def pad_weights_of_logits_matmul(onnx_path: str, use_external_data_format: bool = True) -> bool:
    """Pad the logits MatMul weight in the provided decoder model, which will be overwritten.

    Args:
        onnx_path (str): Path of onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """
    decoder_model_proto = onnx.load_model(onnx_path, load_external_data=True)

    logits_output_name = decoder_model_proto.graph.output[0].name

    decoder_model = OnnxModel(decoder_model_proto)

    output_name_to_node = decoder_model.output_name_to_node()
    assert logits_output_name in output_name_to_node

    matmul_node = output_name_to_node[logits_output_name]
    # Sanity check - the logits need to be produced by a MatMul node
    if matmul_node.op_type != "MatMul":
        return False

    # The logits MatMul weight MUST be an initializer (or)
    # it MUST be flowing through a Transpose whose input is
    # an initializer
    pad_along_axis_1 = True
    logits_weight = decoder_model.get_initializer(matmul_node.input[1])
    if logits_weight is None:
        transpose_before_matmul = decoder_model.match_parent(matmul_node, "Transpose", 1)

        if transpose_before_matmul is None:
            return False

        logits_weight = decoder_model.get_initializer(transpose_before_matmul.input[0])

        if logits_weight is None:
            return False

        pad_along_axis_1 = False

    # The logits MatMul weight MUST be fp16
    if logits_weight.data_type != TensorProto.DataType.FLOAT16:
        return False

    # The logits MatMul weight MUST be 2-dimensional
    if len(logits_weight.dims) != 2:
        return False

    # Pad and over-write the initializer (if needed)
    actual_vocab_size = logits_weight.dims[1]

    if (actual_vocab_size % 8) == 0:
        # Already "padded"
        return True

    padded_vocab_size = math.ceil(actual_vocab_size / 8) * 8
    padding = padded_vocab_size - actual_vocab_size

    # TODO(hasesh): Handle cases where the fp16 data is stored in the
    # non-raw data field
    if logits_weight.raw_data:
        if pad_along_axis_1:
            padding_data = np.zeros((logits_weight.dims[0], padding), dtype=np.float16)
            weight_with_padding = np.concatenate((NumpyHelper.to_array(logits_weight), padding_data), axis=1)
            logits_weight.dims[1] = padded_vocab_size
        else:
            padding_data = np.zeros((padding, logits_weight.dims[1]), dtype=np.float16)
            weight_with_padding = np.concatenate((NumpyHelper.to_array(logits_weight), padding_data), axis=0)
            logits_weight.dims[0] = padded_vocab_size

        logits_weight.raw_data = weight_with_padding.tobytes()
    else:
        return False

    # Save the model
    OnnxModel.save(decoder_model_proto, onnx_path, save_as_external_data=use_external_data_format)
    return True


def create_ort_session(model_path: str, use_gpu: bool) -> InferenceSession:
    """Create OnnxRuntime session.

    Args:
        model_path (str): onnx model path
        use_gpu (bool): use GPU or not

    Raises:
        RuntimeError: CUDAExecutionProvider is not available when --use_gpu is specified.

    Returns:
        onnxruntime.InferenceSession: The created session.
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    if use_gpu:
        if "CUDAExecutionProvider" not in get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available for --use_gpu!")
        else:
            logger.info("use CUDAExecutionProvider")

    ort_session = InferenceSession(model_path, sess_options, providers=execution_providers)
    return ort_session


def verify_gpt2_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify GPT-2 subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of GPT-2
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = Precision.FLOAT16 == precision

    input_count = len(graph.input)
    layer_count = input_count - 3
    assert layer_count >= 1

    expected_inputs = ["input_ids", "position_ids", "attention_mask"] + [f"past_{i}" for i in range(layer_count)]
    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32
        if i >= 3:
            expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT

        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")
    logger.info("Verifying GPT-2 graph inputs: name and data type are good.")

    expected_outputs = ["logits"] + [f"present_{i}" for i in range(layer_count)]
    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {output_type}")
    logger.info("Verifying GPT-2 graph outputs: name and data type are good.")

    # TODO(tianleiwu): verify shapes of inputs and outputs.
    return


def verify_t5_decoder_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify T5 decoder subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of T5 decoder
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = Precision.FLOAT16 == precision
    float_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT

    input_count = len(graph.input)
    layer_count = (input_count - 3) // 4
    assert layer_count >= 1

    # Expect inputs:
    #   input_ids: int32 (B, 1)
    #   encoder_attention_mask: int32 (B, encode_sequence_length)
    #   encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)

    #   past_key_self_0: (B, num_heads, past_decode_sequence_length, head_size)
    #   past_value_self_0: (B, num_heads, past_decode_sequence_length, head_size)
    #   ... (for each self attention layer)

    #   past_key_cross_0: (B, num_heads, encode_sequence_length, head_size)
    #   past_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
    #   ... (for each cross attention layer)

    # TODO: encoder_hidden_states is optional
    expected_inputs = ["input_ids", "encoder_attention_mask", "encoder_hidden_states"]
    for i in range(layer_count):
        expected_inputs.append(f"past_key_self_{i}")
        expected_inputs.append(f"past_value_self_{i}")
    for i in range(layer_count):
        expected_inputs.append(f"past_key_cross_{i}")
        expected_inputs.append(f"past_value_cross_{i}")

    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32 if i < 2 else float_type
        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")

    # Expect outputs:
    #   logits:               (B, 1, vocab_size)
    #   present_key_self_0:   (B, num_heads, past_decode_sequence_length + 1, head_size)
    #   present_value_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
    #                     ... (for each self attention layer)
    expected_outputs = ["logits"]
    for i in range(layer_count):
        expected_outputs.append(f"present_key_self_{i}")
        expected_outputs.append(f"present_value_self_{i}")

    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != float_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {float_type}. Got {output_type}")


def verify_t5_encoder_decoder_init_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify T5 decoder subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of T5 decoder
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = Precision.FLOAT16 == precision
    layer_count = (len(graph.output) - 2) // 4
    assert layer_count >= 1

    # Expect 3 inputs:
    #   encoder_input_ids:      int32 (B, encode_sequence_length)
    #   encoder_attention_mask: int32 (B, encode_sequence_length)
    #   decoder_input_ids:      int32 (B, 1)
    expected_inputs = ["encoder_input_ids", "encoder_attention_mask", "decoder_input_ids"]
    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32
        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")

    # Expected outputs:
    #   logits:                (B, 1, vocab_size)
    #   encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)
    #   present_key_self_0:    (B, num_heads, 1, head_size)
    #   present_value_self_0:  (B, num_heads, 1, head_size)
    #                      ... (for each self attention layer)
    #   present_key_cross_0:   (B, num_heads, encode_sequence_length, head_size)
    #   present_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
    #                      ... (for each cross attention layer)
    expected_outputs = ["logits", "encoder_hidden_states"]
    for i in range(layer_count):
        expected_outputs.append(f"present_key_self_{i}")
        expected_outputs.append(f"present_value_self_{i}")
    for i in range(layer_count):
        expected_outputs.append(f"present_key_cross_{i}")
        expected_outputs.append(f"present_value_cross_{i}")

    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {expected_type}. Got {output_type}")

    logger.info("T5 encoder graph verified: name and data type of inputs and outputs are good.")


def remove_shared_initializers(
    graph1: GraphProto,
    graph2: GraphProto,
    shared_prefix: str = "shared_",
    min_elements: int = 1024,
):
    """Remove initializers with same value from two graphs.

    Args:
        graph1 (GraphProto): the first graph to process
        graph2 (GraphProto): the second graph to process
        shared_prefix (str): add prefix to the shared initializers among two graphs
        min_elements (int, optional): minimal number of elements for initializers to be considered. Defaults to 1024.
    """

    mapping_initializers_1 = {}
    mapping_initializers_2 = {}
    shared_initializers_1 = []
    shared_initializers_2 = []
    shared_initializers_names = []

    for initializer1 in graph1.initializer:
        if not (initializer1.dims and sum(initializer1.dims) >= min_elements):
            continue

        for initializer2 in graph2.initializer:
            if not (initializer2.dims and sum(initializer2.dims) >= min_elements):
                continue

            if OnnxModel.has_same_value(initializer1, initializer2):
                mapping_initializers_1[initializer1.name] = shared_prefix + initializer2.name
                shared_initializers_1.append(initializer1)

                if initializer2.name not in mapping_initializers_2:
                    shared_name = shared_prefix + initializer2.name
                    mapping_initializers_2[initializer2.name] = shared_name
                    shared_initializers_2.append(initializer2)
                    shared_initializers_names.append(shared_name)
                break

    logger.debug(f"shared initializers:{shared_initializers_names}")

    # Make sure new name does not exist in graph 1
    for node in graph1.node:
        for j in range(len(node.input)):
            if node.input[j] in shared_initializers_names:
                raise RuntimeError(f"name is found in graph 1: {node.input[j]}")

    # Make sure new name does not exist in graph 2
    for node in graph2.node:
        for j in range(len(node.input)):
            if node.input[j] in shared_initializers_names:
                raise RuntimeError(f"name is found in graph 2: {node.input[j]}")

    # Remove shared initializers from graph 2
    for initializer in shared_initializers_2:
        graph2.initializer.remove(initializer)

    # Rename value info for old names in graph 2
    for value_info in graph2.value_info:
        if value_info.name in mapping_initializers_2:
            value_info.name = mapping_initializers_2[value_info.name]

    # Rename nodes inputs in graph 2:
    for node in graph2.node:
        for j in range(len(node.input)):
            if node.input[j] in mapping_initializers_2:
                new_name = mapping_initializers_2[node.input[j]]
                logger.debug(f"graph 2 rename node {node.name} input {j} from {node.input[j]} to {new_name}")
                node.input[j] = new_name

    #  Remove shared initializers from graph 1
    for initializer in shared_initializers_1:
        graph1.initializer.remove(initializer)

    # Rename value info for old names in graph 1
    for value_info in graph1.value_info:
        if value_info.name in mapping_initializers_1:
            value_info.name = mapping_initializers_1[value_info.name]

    # Rename nodes inputs in graph 1:
    for node in graph1.node:
        for j in range(len(node.input)):
            if node.input[j] in mapping_initializers_1:
                new_name = mapping_initializers_1[node.input[j]]
                logger.debug(f"graph 1 rename node {node.name} input {j} from {node.input[j]} to {new_name}")
                node.input[j] = new_name

    # Rename shared initializers in graph 2
    for initializer in shared_initializers_2:
        initializer.name = mapping_initializers_2[initializer.name]

    for initializer in shared_initializers_2:
        shape = onnx.numpy_helper.to_array(initializer).shape
        value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
        # Need add value_info for initializers moved to parent graph. Otherwise, ORT will fail.
        graph1.value_info.append(value_info)
        graph2.value_info.append(value_info)

    return shared_initializers_2


def get_shared_initializers(encoder_model: ModelProto, decoder_model: ModelProto):
    encoder = OnnxModel(encoder_model)
    decoder = OnnxModel(decoder_model)
    encoder.add_prefix_to_names("e_")
    decoder.add_prefix_to_names("d_")
    encoder.remove_duplicated_initializer()
    decoder.remove_duplicated_initializer()
    initializers = remove_shared_initializers(encoder.model.graph, decoder.model.graph, "s_")
    return initializers


def move_initializers(
    graph: GraphProto,
    min_elements: int = 1024,
) -> List[TensorProto]:
    """Remove initializers of a graph, when they have number of elements larger than a threshold.

    Args:
        graph (GraphProto): the graph.
        min_elements (int, optional): minimal number of elements for initializers to be considered. Defaults to 1024.

    Returns:
        List[TensorProto]: initializers that are removed from the graph.
    """
    moved_initializers = []
    for tensor in graph.initializer:
        if not (tensor.dims and sum(tensor.dims) >= min_elements):
            continue
        moved_initializers.append(tensor)

    for initializer in moved_initializers:
        graph.initializer.remove(initializer)

    # Add type info, otherwise ORT will raise error: "input arg (*) does not have type information set by parent node."
    for initializer in moved_initializers:
        shape = onnx.numpy_helper.to_array(initializer).shape
        value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
        graph.value_info.append(value_info)

    return moved_initializers


def convert_generation_model(args: argparse.Namespace, generation_type: GenerationType = GenerationType.BEAMSEARCH):
    """Convert model according to command line arguments.

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    is_gpt2: bool = args.model_type == "gpt2"
    is_greedysearch: bool = generation_type == GenerationType.GREEDYSEARCH

    if is_greedysearch:
        if not is_gpt2:
            raise NotImplementedError("Currently only gpt2 with greedy search is supported")
        if args.output_sequences_scores:
            raise NotImplementedError("output_sequences_scores currently is not supported in greedy search")
        if args.output_token_scores:
            raise NotImplementedError("output_token_scores currently is not supported in greedy search")

    if is_gpt2:
        if args.decoder_onnx and os.path.exists(args.decoder_onnx):
            logger.info(f"skip convert_to_onnx since path existed: {args.decoder_onnx}")
        else:
            if not args.decoder_onnx:
                onnx_filename = "gpt2_past_{}.onnx".format("fp16" if args.precision == Precision.FLOAT16 else "fp32")
                args.decoder_onnx = Path(Path(args.output).parent, onnx_filename).as_posix()

            logger.info(f"Convert GPT model {args.model_name_or_path} to onnx {args.decoder_onnx} ...")
            gpt2_to_onnx(args)
    else:  # t5 or mt5
        if args.decoder_onnx and args.encoder_decoder_init_onnx:
            logger.info(
                f"skip convert_to_onnx since paths specified: {args.decoder_onnx} and {args.encoder_decoder_init_onnx}"
            )
        else:
            logger.info(f"Convert model {args.model_name_or_path} to onnx ...")
            t5_to_onnx(args)

    # We only want to pad the logits MatMul weight in the decoder for fp16 models.
    # The inherent assumption is that fp16 models run on GPU for which all
    # dims need to be a multiple of 8 to leverage tensor cores.
    # NOTE: We currently only support padding the MatMul logits weight for GPT2 BeamSearch.
    # This can be expanded to other models/decoding strategies later
    logits_matmul_weight_padded = False
    if (
        args.pad_vocab_size
        and args.precision == Precision.FLOAT16
        and is_gpt2
        and generation_type == GenerationType.BEAMSEARCH
    ):
        logger.info(
            f"Pad logits MatMul weights for optimal MatMul perf in fp16 on {args.decoder_onnx}. "
            "The file will be overwritten."
        )
        logits_matmul_weight_padded = pad_weights_of_logits_matmul(args.decoder_onnx, args.use_external_data_format)
        if not logits_matmul_weight_padded:
            logger.warning(
                "Tried and failed to pad logits MatMul weights. " "Performance may be sub-optimal for this MatMul"
            )

    # If the user explicitly requests running shape inference or if we padded/mutated
    # weight(s) in the decoder, we want to run shape inference to capture the new
    # shapes
    if logits_matmul_weight_padded or args.run_shape_inference:
        logger.info(f"Run symbolic shape inference on {args.decoder_onnx}. The file will be overwritten.")
        shape_inference(args.decoder_onnx, args.use_external_data_format)

    if is_gpt2:
        config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_type == "t5":
        config = T5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = MT5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.verbose:
        logger.info(f"Config={config}")

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id if is_gpt2 else config.pad_token_id
    vocab_size = config.vocab_size

    # if vocab_size is given in parameters use that.
    if args.vocab_size != -1:
        vocab_size = args.vocab_size

    decoder_model = onnx.load_model(args.decoder_onnx, load_external_data=True)
    decoder_model.graph.name = f"{args.model_type} decoder"

    if args.model_type == "gpt2":
        verify_gpt2_subgraph(decoder_model.graph, args.precision)
    else:
        verify_t5_decoder_subgraph(decoder_model.graph, args.precision)

    inputs = (
        [
            "input_ids",
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty",
            "repetition_penalty",
        ]
        if not is_greedysearch
        else [
            "input_ids",
            "max_length",
            "min_length",
            "repetition_penalty",
        ]
    )

    if args.vocab_mask:
        inputs.append("vocab_mask")
    else:
        inputs.append("")

    if args.prefix_vocab_mask:
        inputs.append("prefix_vocab_mask")
    else:
        inputs.append("")

    if args.custom_attention_mask:
        inputs.append("attention_mask")
    else:
        inputs.append("")

    outputs = ["sequences"]
    if args.output_sequences_scores:
        outputs.append("sequences_scores")

    if args.output_token_scores:
        assert args.output_sequences_scores, "--output_token_scores requires --output_sequences_scores"
        outputs.append("scores")

    node = (
        onnx.helper.make_node(
            "BeamSearch",
            inputs=inputs,
            outputs=outputs,
            name=f"BeamSearch_{args.model_type}",
        )
        if not is_greedysearch
        else onnx.helper.make_node(
            "GreedySearch",
            inputs=inputs,
            outputs=outputs,
            name=f"GreedySearch_{args.model_type}",
        )
    )

    node.domain = "com.microsoft"

    attr_to_extend = (
        [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            onnx.helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
        ]
        if not is_greedysearch
        else [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
        ]
    )

    # Explicitly pass in the vocab size via an attribute
    if logits_matmul_weight_padded:
        attr_to_extend.extend([onnx.helper.make_attribute("vocab_size", vocab_size)])

    node.attribute.extend(attr_to_extend)

    initializers = []
    if args.model_type in ["t5", "mt5"]:
        if args.run_shape_inference:
            logger.info(f"Symbolic shape inference on {args.encoder_decoder_init_onnx}. The file will be overwritten.")
            shape_inference(args.encoder_decoder_init_onnx, args.use_external_data_format)
        encoder_model = onnx.load_model(args.encoder_decoder_init_onnx, load_external_data=True)
        encoder_model.graph.name = f"{args.model_type} encoder and decoder init"
        verify_t5_encoder_decoder_init_subgraph(encoder_model.graph, args.precision)

        if not args.disable_shared_initializers:
            # Unique shared initializers from the decoder and decoder_init could reduce memory usage in inference.
            initializers = get_shared_initializers(encoder_model, decoder_model)
            logger.info(
                f"{len(initializers)} shared initializers ({[i.name for i in initializers]}) in subgraphs are moved to the main graph"
            )

            # TODO(tianleiwu): investigate the following which causes error in inference
            # Move initializer from subgraph to main graph could reduce memory usage in inference.
            # moved_initializers = move_initializers(encoder_model.graph)
            # logger.info(
            #     f"{len(moved_initializers)} initializers ({[i.name for i in moved_initializers]}) from the encoder are moved to the main graph"
            # )
            # initializers.extend(moved_initializers)

        node.attribute.extend(
            [
                onnx.helper.make_attribute("encoder", encoder_model.graph),
                onnx.helper.make_attribute("decoder", decoder_model.graph),
                onnx.helper.make_attribute(
                    "decoder_start_token_id",
                    config.decoder_start_token_id if len(encoder_model.graph.input) == 3 else -1,
                ),
            ]
        )
    else:
        # Move initializer from subgraph to main graph could reduce memory usage in inference.
        initializers = move_initializers(decoder_model.graph)
        logger.info(f"{len(initializers)} initializers from the decoder are moved to the main graph")

        node.attribute.append(onnx.helper.make_attribute("decoder", decoder_model.graph))

    # graph inputs
    input_ids = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    max_length = onnx.helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = onnx.helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = onnx.helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = onnx.helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    length_penalty = onnx.helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = onnx.helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])

    graph_inputs = (
        [
            input_ids,
            max_length,
            min_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            repetition_penalty,
        ]
        if not is_greedysearch
        else [
            input_ids,
            max_length,
            min_length,
            repetition_penalty,
        ]
    )

    if args.vocab_mask:
        vocab_mask = onnx.helper.make_tensor_value_info("vocab_mask", TensorProto.INT32, [vocab_size])
        graph_inputs.append(vocab_mask)

    if args.prefix_vocab_mask:
        prefix_vocab_mask = onnx.helper.make_tensor_value_info(
            "prefix_vocab_mask", TensorProto.INT32, ["batch_size", vocab_size]
        )
        graph_inputs.append(prefix_vocab_mask)

    if args.custom_attention_mask:
        attention_mask = onnx.helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        graph_inputs.append(attention_mask)

    # graph outputs
    sequences = (
        onnx.helper.make_tensor_value_info(
            "sequences",
            TensorProto.INT32,
            ["batch_size", "num_return_sequences", "max_length"],
        )
        if not is_greedysearch
        else onnx.helper.make_tensor_value_info(
            "sequences",
            TensorProto.INT32,
            ["batch_size", "max_length"],
        )
    )

    sequences_scores = onnx.helper.make_tensor_value_info(
        "sequences_scores", TensorProto.FLOAT, ["batch_size", "num_return_sequences"]
    )

    scores = onnx.helper.make_tensor_value_info(
        "scores",
        TensorProto.FLOAT,
        ["max_length - sequence_length", "batch_size", "num_beams", vocab_size],
    )

    graph_outputs = [sequences]

    if args.output_sequences_scores:
        graph_outputs.append(sequences_scores)

    if args.output_token_scores:
        graph_outputs.append(scores)

    new_graph = onnx.helper.make_graph(
        [node],
        f"{args.model_type} beam search" if not is_greedysearch else f"{args.model_type} greedy search",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    # Create the model
    new_model = onnx.helper.make_model(
        new_graph,
        producer_name="onnxruntime.transformers",
        opset_imports=decoder_model.opset_import,
    )

    # TODO(tianleiwu): move shared initializers from T5 encoder and decoder subgraphs to parent graph to save memory.
    if args.use_external_data_format:
        from packaging import version

        if version.parse(onnx.__version__) < version.parse("1.12.0"):
            logger.warning("Require onnx >= 1.12 to save large (>2GB) model!")

        OnnxModel.save(
            new_model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )
    else:
        onnx.save(new_model, args.output)
    logger.info(f"model save to {args.output}")


def test_torch_performance(
    args: argparse.Namespace,
    model: Union[GPT2LMHeadModel, T5ForConditionalGeneration],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
    bad_words_ids: List[List[int]],
) -> Dict[str, Any]:
    """Test PyTorch performance of text generation.

    Args:
        args (argparse.Namespace): arguments parsed from command line
        model (Union[GPT2LMHeadModel, T5ForConditionalGeneration]): PyTorch model
        input_ids (torch.Tensor): input_ids
        attention_mask (torch.Tensor): Attention mask
        eos_token_id (int): EOS token ID
        pad_token_id (int): Padding token ID
        bad_words_ids (List[List[int]]): Words shall not be generated.

    Raises:
        RuntimeError: PyTorch with CUDA is not available for --use_gpu

    Returns:
        Dict[str, Any]: A dictionary with string with metric name, and value can be integer or string.
    """
    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda for testing gpu performance.")

    if args.precision == Precision.FLOAT16:
        model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)

    torch.set_grad_enabled(False)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch_latency = []
    for _ in range(args.total_runs):
        start = time.time()
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )
        torch_latency.append(time.time() - start)
    batch_size = input_ids.shape[0]
    from benchmark_helper import get_latency_result

    return get_latency_result(torch_latency, batch_size)


def create_attention_mask(input_ids, pad_token_id):
    attention_mask = np.ones(input_ids.shape, dtype=np.int32)
    for i in range(input_ids.shape[0]):
        abs_pos = 0
        for j in range(input_ids.shape[1]):
            if input_ids[i][j] == pad_token_id and abs_pos == 0:
                attention_mask[i][j] = 0
            else:
                abs_pos += 1
    return attention_mask


def test_gpt_model(args: argparse.Namespace, sentences: Optional[List[str]] = None, is_greedy: bool = False):
    """Test GPT-2 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type == "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Use different length sentences to test batching
    if sentences is None:
        sentences = [
            "The product is released",
            "I enjoy walking in the park",
            "Test best way to invest",
        ]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    bad_words = "walk in park"
    bad_words_ids = tokenizer.encode(bad_words, add_prefix_space=True)
    bad_words_ids = [[word_id] for word_id in bad_words_ids]  # Convert to list of list
    if args.vocab_mask:
        logger.debug("bad_words_ids", bad_words_ids)
    else:
        bad_words_ids = []

    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    torch_decoded_sequences = []
    beam_outputs = None
    if not args.disable_parity:
        print("-" * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )
        print("input_ids", input_ids)
        print("huggingface transformers outputs:")
        print("sequences", beam_outputs.sequences)
        if args.output_sequences_scores:
            print("sequences_scores", beam_outputs.sequences_scores)
        if args.output_token_scores:
            print("scores", beam_outputs.scores)
        for i, sequence in enumerate(beam_outputs.sequences):
            decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
            torch_decoded_sequences.append(decoded_sequence)
            print(f"{i}: {decoded_sequence}")

    print("-" * 50)
    print("Testing beam search with onnxruntime...")

    ort_session = create_ort_session(args.output, args.use_gpu)

    if is_greedy:
        inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        }
    else:
        inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        }

    if args.vocab_mask:
        vocab_mask = np.ones((vocab_size), dtype=np.int32)
        if args.vocab_mask:
            for bad_word_id in bad_words_ids:
                vocab_mask[bad_word_id] = 0
        inputs["vocab_mask"] = vocab_mask

    if args.custom_attention_mask:
        inputs["attention_mask"] = create_attention_mask(input_ids, pad_token_id)

    batch_size = input_ids.shape[0]
    if args.prefix_vocab_mask:
        logger.info("Use prefix vocab mask with all ones in ORT, but no corresponding setting for Torch model.")
        prefix_vocab_mask = np.ones((batch_size, vocab_size), dtype=np.int32)
        inputs["prefix_vocab_mask"] = prefix_vocab_mask

    logger.debug("ORT inputs", inputs)
    result = ort_session.run(None, inputs)

    if args.save_test_data:
        test_data_dir = Path(args.output).parent.as_posix()
        logger.debug("test_data_dir", test_data_dir)
        from bert_test_data import output_test_data

        all_inputs = [inputs]
        for i, inputs in enumerate(all_inputs):
            dir = os.path.join(test_data_dir, "test_data_set_" + str(i))
            output_test_data(dir, inputs)

    # Test performance
    latency = []
    for _ in range(args.total_runs):
        start = time.time()
        _ = ort_session.run(None, inputs)
        latency.append(time.time() - start)

    from benchmark_helper import get_latency_result

    batch_size = input_ids.shape[0]
    output = get_latency_result(latency, batch_size)

    print("ORT outputs:")
    sequences = result[0]
    print("sequences", sequences)
    if args.output_sequences_scores:
        print("sequences_scores", result[1])
    if args.output_token_scores:
        print("scores", result[2])

    if is_greedy:
        (batch_size, max_length) = sequences.shape
        ort_decoded_sequences = []
        for i in range(batch_size):
            decoded_sequence = tokenizer.decode(sequences[i], skip_special_tokens=True)
            ort_decoded_sequences.append(decoded_sequence)
            print(f"batch {i} sequence: {decoded_sequence}")
    else:
        (batch_size, num_sequences, max_length) = sequences.shape
        ort_decoded_sequences = []
        for i in range(batch_size):
            for j in range(num_sequences):
                decoded_sequence = tokenizer.decode(sequences[i][j], skip_special_tokens=True)
                ort_decoded_sequences.append(decoded_sequence)
                print(f"batch {i} sequence {j}: {decoded_sequence}")

    if beam_outputs:
        torch_sequences = beam_outputs.sequences.reshape(batch_size, args.num_return_sequences, -1)
        ort_sequences = torch.LongTensor(sequences)
        print("-" * 50)
        print("Torch Sequences:")
        print(torch_sequences)
        print(torch_decoded_sequences)
        print("-" * 50)
        print("ORT Sequences:")
        print(ort_sequences)
        print(ort_decoded_sequences)
        print("-" * 50)
        # Compare the generated text instead of word IDs since ORT pads to max sequence length but Torch not.
        is_same = torch_decoded_sequences == ort_decoded_sequences
        print("Torch and ORT result is ", "same" if is_same else "different")
        output["parity"] = is_same

    if args.torch_performance:
        torch_latency_output = test_torch_performance(
            args,
            model,
            input_ids,
            attention_mask,
            eos_token_id,
            pad_token_id,
            bad_words_ids,
        )
        print("Torch Latency", torch_latency_output)

    print("ORT", output)

    return output


def test_t5_model(args: argparse.Namespace, sentences: Optional[List[str]] = None):
    """Test T5 or MT5 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type in ["t5", "mt5"]

    if args.prefix_vocab_mask:
        logger.debug("Skipping parity test as prefix vocab mask is not implemented by Hugging Face")
        return None

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"

    if args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    else:
        model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

    # Use different length sentences to test batching
    if sentences is None:
        sentences = [
            "translate English to French: The product is released",
            "summarize: research continues to show that pets bring real health benefits to their owners."
            + "Having a dog around can lead to lower levels of stress for both adults and kids.",
            # "summarize: I enjoy walking in the park. It makes my mind feel calm and refreshed. "
            # + "I enjoy looking at the trees, flowers, and wildlife around me, and listening to sound from natural.",
        ]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    bad_words = "walk in park"
    bad_words_ids = tokenizer.encode(bad_words)[:-1]  # exclude the last token (EOS)
    bad_words_ids = [[word_id] for word_id in bad_words_ids]  # Convert to list of list
    if args.vocab_mask:
        logger.debug("bad_words_ids", bad_words_ids)
    else:
        bad_words_ids = []

    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id
    vocab_size = config.vocab_size
    logger.debug(f"eos_token_id:{eos_token_id}, pad_token_id:{pad_token_id}, vocab_size:{vocab_size}")

    torch_decoded_sequences = []
    if not args.disable_parity:
        print("-" * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )

        print("input_ids", input_ids)
        print("huggingface transformers outputs:")
        print("sequences", beam_outputs.sequences)
        if args.output_sequences_scores:
            print("sequences_scores", beam_outputs.sequences_scores)
        if args.output_token_scores:
            print("scores", beam_outputs.scores)
        for i, sequence in enumerate(beam_outputs.sequences):
            decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
            torch_decoded_sequences.append(decoded_sequence)
            print("{}: {}".format(i, decoded_sequence))

    print("-" * 50)
    print("Testing beam search with onnxruntime...")

    ort_session = create_ort_session(args.output, args.use_gpu)

    vocab_mask = np.ones((vocab_size), dtype=np.int32)
    if args.vocab_mask:
        for bad_word_id in bad_words_ids:
            vocab_mask[bad_word_id] = 0

    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([args.max_length], dtype=np.int32),
        "min_length": np.array([args.min_length], dtype=np.int32),
        "num_beams": np.array([args.num_beams], dtype=np.int32),
        "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
        "length_penalty": np.array([args.length_penalty], dtype=np.float32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
    }

    if args.vocab_mask:
        inputs["vocab_mask"] = vocab_mask

    if args.custom_attention_mask:
        inputs["attention_mask"] = create_attention_mask(input_ids, pad_token_id)

    if args.save_test_data:
        test_data_dir = Path(args.output).parent.as_posix()
        logger.debug("test_data_dir", test_data_dir)
        from bert_test_data import output_test_data

        all_inputs = [inputs]
        for i, inputs in enumerate(all_inputs):
            dir = os.path.join(test_data_dir, "test_data_set_" + str(i))
            output_test_data(dir, inputs)

    logger.debug("ORT inputs", inputs)

    # Test performance
    latency = []
    for _ in range(args.total_runs):
        start = time.time()
        result = ort_session.run(None, inputs)
        latency.append(time.time() - start)
    batch_size = input_ids.shape[0]
    from benchmark_helper import get_latency_result

    output = get_latency_result(latency, batch_size)

    print("ORT outputs:")
    sequences = result[0]
    print("sequences", sequences)
    if args.output_sequences_scores:
        print("sequences_scores", result[1])
    if args.output_token_scores:
        print("scores", result[2])

    (batch_size, num_sequences, max_length) = sequences.shape
    ort_decoded_sequences = []
    for i in range(batch_size):
        for j in range(num_sequences):
            decoded_sequence = tokenizer.decode(sequences[i][j], skip_special_tokens=True)
            ort_decoded_sequences.append(decoded_sequence)
            print(f"batch {i} sequence {j}: {decoded_sequence}")

    if not args.disable_parity:
        torch_sequences = beam_outputs.sequences.reshape(batch_size, args.num_return_sequences, -1)
        ort_sequences = torch.LongTensor(sequences)
        print("-" * 50)
        print("Torch Sequences:")
        print(torch_sequences)
        print(torch_decoded_sequences)
        print("-" * 50)
        print("ORT Sequences:")
        print(ort_sequences)
        print(ort_decoded_sequences)
        print("-" * 50)
        # Compare the generated text instead of word IDs since ORT pads to max sequence length but Torch not.
        is_same = torch_decoded_sequences == ort_decoded_sequences
        print("Torch and ORT result is ", "same" if is_same else "different")
        output["parity"] = is_same

    if args.torch_performance:
        torch_latency_output = test_torch_performance(
            args,
            model,
            input_ids,
            attention_mask,
            eos_token_id,
            pad_token_id,
            bad_words_ids,
        )
        print("Torch Latency", torch_latency_output)

    print("ORT", output)
    return output


def main(argv: Optional[List[str]] = None, sentences: Optional[List[str]] = None):
    """Main entry function

    Args:
        argv (Optional[List[str]], optional): _description_. Defaults to None.
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Raises:
        ValueError: Path does not exist: --encoder_decoder_init_onnx
        ValueError: Path does not exist: --decoder_onnx
        ValueError: --decoder_onnx and --encoder_decoder_init_onnx are not used together for T5

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """

    args = parse_arguments(argv)
    setup_logger(args.verbose)

    if args.model_type in ["t5", "mt5"]:
        if args.encoder_decoder_init_onnx and not os.path.exists(args.encoder_decoder_init_onnx):
            raise ValueError(f"Path does not exist: --encoder_decoder_init_onnx {args.encoder_decoder_init_onnx}")
        if args.decoder_onnx and not os.path.exists(args.decoder_onnx):
            raise ValueError(f"Path does not exist: --decoder_onnx {args.decoder_onnx}")
        if (args.encoder_decoder_init_onnx and not args.decoder_onnx) or (
            args.decoder_onnx and not args.encoder_decoder_init_onnx
        ):
            raise ValueError("--decoder_onnx shall use together with --encoder_decoder_init_onnx")

    is_greedy = args.num_beams == 1 and args.num_return_sequences == 1

    if args.model_type == "gpt2" and is_greedy:
        convert_generation_model(args, GenerationType.GREEDYSEARCH)
    else:
        convert_generation_model(args)

    logger.info("start testing model...")
    if args.model_type in ["t5", "mt5"]:
        result = test_t5_model(args, sentences=sentences)
    else:
        result = test_gpt_model(args, sentences=sentences, is_greedy=is_greedy)

    if result:
        if args.use_external_data_format:
            logger.info(f"Output files: {args.output}, {args.output}.data")
        else:
            logger.info(f"Output file: {args.output}")

    return result


if __name__ == "__main__":
    main()

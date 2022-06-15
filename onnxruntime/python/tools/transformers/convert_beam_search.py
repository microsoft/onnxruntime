# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
This converts GPT2 or T5 model to onnx with beam search operator.

Example 1: convert gpt2 model with beam search:
    python convert_beam_search.py -m gpt2 --decoder_onnx ./onnx_models/gpt2_past_fp32.onnx     \
        --output ./onnx_models/gpt2_beam_search.onnx --output_sequences_scores

Example 2: convert T5 model with beam search in two steps:
    cd ./models/t5
    python convert_to_onnx.py -m t5-small
    cd ../..
    python convert_beam_search.py -m t5-small --model_type t5                                   \
        --decoder_onnx ./models/t5/onnx_models/t5-small_decoder.onnx                            \
        --encoder_decoder_init_onnx ./models/t5/onnx_models/t5-small_encoder_decoder_init.onnx  \
        --output ./models/t5/onnx_models/t5_small_beam_search.onnx

Example 3: convert T5 model with beam search. All in one step:
    python convert_beam_search.py -m t5-small --model_type t5 --output ./models/t5/onnx_models/t5_small_beam_search.onnx
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx
import torch
from benchmark_helper import Precision
from onnx import onnx_pb as onnx_proto
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_available_providers

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "gpt2"))
from gpt2_helper import PRETRAINED_GPT2_MODELS  # noqa: E402
from models.gpt2.convert_to_onnx import main as convert_gpt2_to_onnx  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "t5"))
from models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models  # noqa: E402

logger = logging.getLogger("")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments

    Args:
        argv (Optional[List[str]], optional): _description_. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(PRETRAINED_GPT2_MODELS),
    )

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5"],
        help="Model type in the list: " + ", ".join(["gpt2", "t5"]),
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--decoder_onnx",
        required=False,
        type=str,
        default="",
        help="Path of onnx model for decoder. Required for gpt2 model type.",
    )

    parser.add_argument(
        "--encoder_decoder_init_onnx",
        required=False,
        type=str,
        default="",
        help="Path of ONNX model for encoder and decoder initialization. For t5 model type.",
    )

    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Output path for onnx model with beam search.",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16],
        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision",
    )

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument(
        "--disable_parity",
        required=False,
        action="store_true",
        help="do not run parity test",
    )
    parser.set_defaults(disable_parity=False)

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Print more information",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--torch_performance",
        required=False,
        action="store_true",
        help="test PyTorch performance",
    )
    parser.set_defaults(torch_performance=False)

    parser.add_argument(
        "--total_runs",
        required=False,
        type=int,
        default=1,
        help="Number of times of inference for latency measurement",
    )

    beam_search_group = parser.add_argument_group("beam search options")

    beam_search_group.add_argument(
        "--output_sequences_scores",
        required=False,
        action="store_true",
        help="output sequences scores",
    )
    beam_search_group.set_defaults(output_sequences_scores=False)

    beam_search_group.add_argument(
        "--output_token_scores",
        required=False,
        action="store_true",
        help="output token scores",
    )
    beam_search_group.set_defaults(output_token_scores=False)

    beam_search_group.add_argument("--early_stopping", required=False, action="store_true")
    beam_search_group.set_defaults(early_stopping=False)

    beam_search_group.add_argument("--min_length", type=int, required=False, default=1, help="Min sequence length")

    beam_search_group.add_argument("--max_length", type=int, required=False, default=50, help="Max sequence length")

    beam_search_group.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        required=False,
        default=0,
        help="No repeat ngram size",
    )

    beam_search_group.add_argument("--num_beams", type=int, required=False, default=4, help="Beam size")

    beam_search_group.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=1,
        help="Number of return sequence <= num_beams",
    )

    beam_search_group.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1,
        help="Softmax temperature for output logits.",
    )

    beam_search_group.add_argument(
        "--length_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage short sentence.",
    )

    beam_search_group.add_argument(
        "--repetition_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage.",
    )

    beam_search_group.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=-1,
        help="Vocab_size of the underlying model",
    )

    beam_search_group.add_argument(
        "--prefix_vocab_mask",
        required=False,
        action="store_true",
        help="This vocab mask applies only to first iteration, enable if last word in query might need auto complete",
    )
    beam_search_group.set_defaults(prefix_vocab_mask=False)

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
        print(f"arguments for convert_to_onnx:{arguments}")

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
    )
    args.encoder_decoder_init_onnx = paths[0]
    args.decoder_onnx = paths[1]


def shape_inference(onnx_path: str):
    """Shape inference on an onnx file, which will be overwritten.

    Args:
        onnx_path (str): Path of onnx model
    """
    # Run symbolic shape inference to walk around ORT shape inference issue for subgraph.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    out = SymbolicShapeInference.infer_shapes(onnx.load(onnx_path), auto_merge=True, guess_output_rank=False)
    if out:
        # TODO(tianleiwu): Use external format if input has extra data.
        onnx.save(out, onnx_path)
    else:
        print("Failed to run symbolic shape inference on the model.")


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
            print("use CUDAExecutionProvider")

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

        expected_type = onnx_proto.TensorProto.INT32
        if i >= 3:
            expected_type = onnx_proto.TensorProto.FLOAT16 if is_float16 else onnx_proto.TensorProto.FLOAT

        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")
    print("Verifying GPT-2 graph inputs: name and data type are good.")

    expected_outputs = ["logits"] + [f"present_{i}" for i in range(layer_count)]
    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        expected_type = onnx_proto.TensorProto.FLOAT16 if is_float16 else onnx_proto.TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {output_type}")
    print("Verifying GPT-2 graph outputs: name and data type are good.")

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
    float_type = onnx_proto.TensorProto.FLOAT16 if is_float16 else onnx_proto.TensorProto.FLOAT

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

        expected_type = onnx_proto.TensorProto.INT32 if i < 2 else float_type
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

        expected_type = onnx_proto.TensorProto.INT32
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

        expected_type = onnx_proto.TensorProto.FLOAT16 if is_float16 else onnx_proto.TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {expected_type}. Got {output_type}")

    print("T5 encoder graph verified: name and data type of inputs and outputs are good.")


def convert_model(args: argparse.Namespace):
    """Convert model according to command line arguments.

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    is_gpt2: bool = args.model_type == "gpt2"
    if is_gpt2:
        if os.path.exists(args.decoder_onnx):
            print(f"skip convert_to_onnx since path existed: {args.decoder_onnx}")
        else:
            print(f"Convert GPT model {args.model_name_or_path} to onnx {args.decoder_onnx} ...")
            gpt2_to_onnx(args)
    else:  # t5
        if args.decoder_onnx and args.encoder_decoder_init_onnx:
            print(
                f"skip convert_to_onnx since paths specified: {args.decoder_onnx} and {args.encoder_decoder_init_onnx}"
            )
        else:
            print(f"Convert T5 model {args.model_name_or_path} to onnx ...")
            t5_to_onnx(args)

    # TODO(tianleiwu): fix shape inference for T5. Currently symbolic shape inference on T5 is broken.
    enable_shape_inference = args.model_type == "gpt2"

    if enable_shape_inference:
        print(f"Run symbolic shape inference on {args.decoder_onnx}. The file will be overwritten.")
        shape_inference(args.decoder_onnx)

    if is_gpt2:
        config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = T5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.verbose:
        print(config)

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id if is_gpt2 else config.pad_token_id
    vocab_size = config.vocab_size

    # if vocab_size is given in parameters use that.
    if args.vocab_size != -1:
        vocab_size = args.vocab_size

    model = onnx.load(args.decoder_onnx)
    model.graph.name = f"{args.model_type} decoder subgraph"

    if args.model_type == "gpt2":
        verify_gpt2_subgraph(model.graph, args.precision)
    else:
        verify_t5_decoder_subgraph(model.graph, args.precision)

    inputs = [
        "input_ids",
        "max_length",
        "min_length",
        "num_beams",
        "num_return_sequences",
        "temperature",
        "length_penalty",
        "repetition_penalty",
        "vocab_mask",
    ]
    if args.prefix_vocab_mask:
        inputs.append("prefix_vocab_mask")

    outputs = ["sequences"]
    if args.output_sequences_scores:
        outputs.append("sequences_scores")

    if args.output_token_scores:
        assert args.output_sequences_scores, "--output_token_scores requires --output_sequences_scores"
        outputs.append("scores")

    node = onnx.helper.make_node(
        "BeamSearch",
        inputs=inputs,
        outputs=outputs,
        name=f"BeamSearch_{args.model_type}",
    )
    node.domain = "com.microsoft"
    node.attribute.extend(
        [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            onnx.helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
            onnx.helper.make_attribute("decoder", model.graph),
        ]
    )

    if args.model_type == "t5":
        if enable_shape_inference:
            print(f"Run symbolic shape inference on {args.encoder_decoder_init_onnx}. The file will be overwritten.")
            shape_inference(args.encoder_decoder_init_onnx)
        init_model = onnx.load(args.encoder_decoder_init_onnx)
        init_model.graph.name = f"{args.model_type} encoder decoder init subgraph"
        verify_t5_encoder_decoder_init_subgraph(init_model.graph, args.precision)
        node.attribute.extend(
            [
                onnx.helper.make_attribute("encoder", init_model.graph),
                onnx.helper.make_attribute(
                    "decoder_start_token_id", config.decoder_start_token_id if len(init_model.graph.input) == 3 else -1
                ),
            ]
        )

    from onnx import TensorProto

    # graph inputs
    input_ids = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    max_length = onnx.helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = onnx.helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = onnx.helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = onnx.helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    temperature = onnx.helper.make_tensor_value_info("temperature", TensorProto.FLOAT, [1])
    length_penalty = onnx.helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = onnx.helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])
    vocab_mask = onnx.helper.make_tensor_value_info("vocab_mask", TensorProto.INT32, [vocab_size])

    graph_inputs = [
        input_ids,
        max_length,
        min_length,
        num_beams,
        num_return_sequences,
        temperature,
        length_penalty,
        repetition_penalty,
        vocab_mask,
    ]

    if args.prefix_vocab_mask:
        prefix_vocab_mask = onnx.helper.make_tensor_value_info(
            "prefix_vocab_mask", TensorProto.INT32, ["batch_size", vocab_size]
        )
        graph_inputs.append(prefix_vocab_mask)

    # graph outputs
    sequences = onnx.helper.make_tensor_value_info(
        "sequences",
        TensorProto.INT32,
        ["batch_size", "num_return_sequences", "max_length"],
    )

    sequences_scores = onnx.helper.make_tensor_value_info(
        "sequences_scores", TensorProto.FLOAT, ["batch_size", "num_return_sequences"]
    )

    scores = onnx.helper.make_tensor_value_info(
        "scores",
        TensorProto.FLOAT,
        ["max_length - sequence_length", "batch_size", "num_beams", vocab_size],
    )

    initializers = []

    graph_outputs = [sequences]

    if args.output_sequences_scores:
        graph_outputs.append(sequences_scores)

    if args.output_token_scores:
        graph_outputs.append(scores)

    new_graph = onnx.helper.make_graph(
        [node],
        f"{args.model_type}-beam-search",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    # Create the model
    new_model = onnx.helper.make_model(
        new_graph,
        producer_name="onnxruntime.transformers",
        opset_imports=model.opset_import,
    )

    # TODO(tianleiwu): move shared initializers from T5 encoder and decoder subgraphs to parent graph to save memory.
    onnx.save(new_model, args.output)


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
            temperature=args.temperature,
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


def test_gpt_model(args: argparse.Namespace, use_vocab_mask: bool = False, sentences: Optional[List[str]] = None):
    """Test GPT-2 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        use_vocab_mask (bool, optional): use vocabulary mask. Defaults to False.
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type == "gpt2"

    if args.temperature != 1.0:
        # TODO(tianleiwu): implement temperature in BeamSearch operator.
        print("Skipping parity test as temperature is not implemented in BeamSearch operator")
        return None

    if args.prefix_vocab_mask:
        print("Skipping parity test as prefix vocab mask is not implemented by Hugging Face")
        return None

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
    if use_vocab_mask:
        print("bad_words_ids", bad_words_ids)
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
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids,
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
    print("Test ONNX model and bream search with onnxruntime...")

    ort_session = create_ort_session(args.output, args.use_gpu)

    vocab_mask = np.ones((vocab_size), dtype=np.int32)
    if use_vocab_mask:
        for bad_word_id in bad_words_ids:
            vocab_mask[bad_word_id] = 0

    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([args.max_length], dtype=np.int32),
        "min_length": np.array([args.min_length], dtype=np.int32),
        "num_beams": np.array([args.num_beams], dtype=np.int32),
        "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
        "temperature": np.array([args.temperature], dtype=np.float32),
        "length_penalty": np.array([args.length_penalty], dtype=np.float32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        "vocab_mask": vocab_mask,
    }

    print("inputs", inputs)
    result = ort_session.run(None, inputs)

    test_data_dir = Path(args.output).parent.as_posix()
    print("test_data_dir", test_data_dir)
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


def test_t5_model(args: argparse.Namespace, use_vocab_mask: bool = False, sentences: Optional[List[str]] = None):
    """Test T5 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        use_vocab_mask (bool, optional): use vocabulary mask. Defaults to False.
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type == "t5"

    if args.temperature != 1.0:
        # TODO(tianleiwu): implement temperature in BeamSearch operator.
        print("Skipping parity test as temperature is not implemented in BeamSearch operator")
        return None

    if args.prefix_vocab_mask:
        print("Skipping parity test as prefix vocab mask is not implemented by Hugging Face")
        return None

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"

    model = T5ForConditionalGeneration.from_pretrained(
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
    if use_vocab_mask:
        print("bad_words_ids", bad_words_ids)
    else:
        bad_words_ids = []

    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id
    vocab_size = config.vocab_size
    print(f"eos_token_id:{eos_token_id}, pad_token_id:{pad_token_id}, vocab_size:{vocab_size}")

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
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids,
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
    print("Test ONNX model and bream search with onnxruntime...")

    ort_session = create_ort_session(args.output, args.use_gpu)

    vocab_mask = np.ones((vocab_size), dtype=np.int32)
    if use_vocab_mask:
        for bad_word_id in bad_words_ids:
            vocab_mask[bad_word_id] = 0

    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([args.max_length], dtype=np.int32),
        "min_length": np.array([args.min_length], dtype=np.int32),
        "num_beams": np.array([args.num_beams], dtype=np.int32),
        "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
        "temperature": np.array([args.temperature], dtype=np.float32),
        "length_penalty": np.array([args.length_penalty], dtype=np.float32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        "vocab_mask": vocab_mask,
    }

    test_data_dir = Path(args.output).parent.as_posix()
    print("test_data_dir", test_data_dir)
    from bert_test_data import output_test_data

    all_inputs = [inputs]
    for i, inputs in enumerate(all_inputs):
        dir = os.path.join(test_data_dir, "test_data_set_" + str(i))
        output_test_data(dir, inputs)

    print("inputs", inputs)

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
        ValueError: --decoder_onnx is not specified for GPT2 model
        ValueError: Path does not exist: --encoder_decoder_init_onnx
        ValueError: Path does not exist: --decoder_onnx
        ValueError: --decoder_onnx and --encoder_decoder_init_onnx are not used together for T5

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """

    args = parse_arguments(argv)

    if args.model_type == "gpt2":
        if not args.decoder_onnx:
            raise ValueError("--decoder_onnx shall be specified for gpt2 model")
    elif args.model_type == "t5":
        if args.encoder_decoder_init_onnx and not os.path.exists(args.encoder_decoder_init_onnx):
            raise ValueError(f"Path does not exist: --encoder_decoder_init_onnx {args.encoder_decoder_init_onnx}")
        if args.decoder_onnx and not os.path.exists(args.decoder_onnx):
            raise ValueError(f"Path does not exist: --decoder_onnx {args.decoder_onnx}")
        if (args.encoder_decoder_init_onnx and not args.decoder_onnx) or (
            args.decoder_onnx and not args.encoder_decoder_init_onnx
        ):
            raise ValueError("--decoder_onnx shall use together with --encoder_decoder_init_onnx")

    convert_model(args)

    if args.model_type == "t5":
        return test_t5_model(args, use_vocab_mask=True, sentences=sentences)
    else:
        return test_gpt_model(args, use_vocab_mask=True, sentences=sentences)


if __name__ == "__main__":
    main()

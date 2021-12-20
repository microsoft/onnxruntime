# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
import onnx
import logging
import argparse
from pathlib import Path
from onnx import helper
import numpy as np
from typing import List
import torch
from transformers import GPT2Config
from gpt2_helper import PRETRAINED_GPT2_MODELS
from convert_to_onnx import main as convert_gpt2_to_onnx
from benchmark_helper import Precision
"""
This converts GPT2 model to onnx with beam search operator.

Examples:
   python convert_beam_search.py -m gpt2 --gpt2_onnx .\onnx_models\gpt2_past_fp32.onnx --output .\onnx_models\gpt2_beam_search.onnx --output_sequences_scores
"""

config: GPT2Config = None

logger = logging.getLogger('')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(PRETRAINED_GPT2_MODELS))

    parser.add_argument('--cache_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help='Directory to cache pre-trained models')

    parser.add_argument('--gpt2_onnx',
                        required=True,
                        type=str,
                        help='Output directory for GPT-2 onnx model, or model path ends with .onnx')

    parser.add_argument('--output',
                        required=False,
                        type=str,
                        help='Output directory for beam search model, or model path ends with .onnx')

    parser.add_argument("-p",
                        "--precision",
                        required=False,
                        type=Precision,
                        default=Precision.FLOAT32,
                        choices=[Precision.FLOAT32, Precision.FLOAT16],
                        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision")

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument('--disable_parity', required=False, action='store_true', help="do not run parity test")
    parser.set_defaults(disable_parity=False)

    parser.add_argument('--total_runs',
                        required=False,
                        type=int,
                        default=1,
                        help='Number of times of inference for latency measurement')

    beam_search_group = parser.add_argument_group("beam search options")

    beam_search_group.add_argument('--output_sequences_scores',
                                   required=False,
                                   action='store_true',
                                   help="output sequences scores")
    beam_search_group.set_defaults(output_sequences_scores=False)

    beam_search_group.add_argument('--output_token_scores',
                                   required=False,
                                   action='store_true',
                                   help="output token scores")
    beam_search_group.set_defaults(output_token_scores=False)

    beam_search_group.add_argument('--early_stopping', required=False, action='store_true')
    beam_search_group.set_defaults(early_stopping=False)

    beam_search_group.add_argument('--min_length', type=int, required=False, default=1, help='Min sequence length')

    beam_search_group.add_argument('--max_length', type=int, required=False, default=50, help='Max sequence length')

    beam_search_group.add_argument('--no_repeat_ngram_size',
                                   type=int,
                                   required=False,
                                   default=0,
                                   help='No repeat ngram size')

    beam_search_group.add_argument('--num_beams', type=int, required=False, default=4, help='Beam size')

    beam_search_group.add_argument('--num_return_sequences',
                                   type=int,
                                   required=False,
                                   default=1,
                                   help='Number of return sequence <= num_beams')

    beam_search_group.add_argument('--temperature',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Softmax temperature for output logits.')

    beam_search_group.add_argument('--length_penalty',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Positive. >1 to penalize and <1 to encorage short sentence.')

    beam_search_group.add_argument('--repetition_penalty',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Positive. >1 to penalize and <1 to encorage.')

    mixed_precision_option_group = parser.add_argument_group(
        "mixed precision conversion parameters that works when \"--precision fp16\" is specified")

    mixed_precision_option_group.add_argument('--io_block_list',
                                              nargs='+',
                                              required=False,
                                              default=[],
                                              help='List of inputs or outputs in float32')

    mixed_precision_option_group.add_argument(
        '--op_block_list',
        nargs='+',
        required=False,
        default=[],
        help='List of operators (like Add LayerNormalization FastGelu) to compute in float32.')

    mixed_precision_option_group.add_argument('--node_block_list',
                                              nargs='+',
                                              required=False,
                                              default=[],
                                              help='List of node names to compute in float32.')

    mixed_precision_option_group.add_argument('--force_fp16_initializers',
                                              required=False,
                                              action='store_true',
                                              help='Convert all float initializers to float16.')
    mixed_precision_option_group.set_defaults(force_fp16_initializers=False)

    args = parser.parse_args(argv)

    return args


def gpt2_to_onnx(args):
    model_name = args.model_name_or_path

    print(f"use convert_to_onnx.py to convert model {model_name} to onnx {args.gpt2_onnx} ...")
    arguments = [
        '--model_name_or_path', model_name, '--output', args.gpt2_onnx, '--optimize_onnx', '--precision',
        'fp32' if args.precision == Precision.FLOAT32 else 'fp16', '--test_runs', '1', '--test_cases', '10'
    ]
    if args.use_gpu:
        arguments.append('--use_gpu')
    if args.use_external_data_format:
        arguments.append('--use_external_data_format')

    # mixed precision conversion options
    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 or mixed precision model cannot run in CPU. Please add --use_gpu"
        if args.io_block_list:
            arguments.append('--io_block_list')
            arguments.extend(args.io_block_list)
        if args.op_block_list:
            arguments.append('--op_block_list')
            arguments.extend(args.op_block_list)
        if args.node_block_list:
            arguments.append('--node_block_list')
            arguments.extend(args.node_block_list)
        if args.force_fp16_initializers:
            arguments.append('--force_fp16_initializers')

    convert_gpt2_to_onnx(arguments)


def shape_inference(gpt2_onnx_path):
    # Run symbolic shape inference to walk around ORT shape inference issue for subgraph.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    out = SymbolicShapeInference.infer_shapes(onnx.load(gpt2_onnx_path), auto_merge=True, guess_output_rank=False)
    if out:
        # TODO: Use external format if input has extra data.
        onnx.save(out, gpt2_onnx_path)
    else:
        print("Failed to run symbolic shape inference on the model.")


def create_ort_session(model_path, use_gpu):
    from onnxruntime import SessionOptions, InferenceSession, __version__ as ort_version, GraphOptimizationLevel
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    ort_session = InferenceSession(model_path, sess_options, providers=execution_providers)
    return ort_session


def convert_model(args):
    if os.path.exists(args.gpt2_onnx):
        print(f"skip convert_to_onnx since path existed: {args.gpt2_onnx}")
    else:
        gpt2_to_onnx(args)

    print(f"Run symbolic shape inference on {args.gpt2_onnx}. The file will be overwritten.")
    shape_inference(args.gpt2_onnx)

    global config
    config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    print(config)

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    model = onnx.load(args.gpt2_onnx)
    model.graph.name = "gpt2 subgraph"
    inputs = [
        "input_ids", "max_length", "min_length", "num_beams", "num_return_sequences", "temperature", "length_penalty",
        "repetition_penalty", "vocab_mask"
    ]

    outputs = ["sequences"]
    if args.output_sequences_scores:
        outputs.append("sequences_scores")

    if args.output_token_scores:
        assert args.output_sequences_scores, "--output_token_scores requires --output_sequences_scores"
        outputs.append("scores")

    node = helper.make_node('BeamSearch', inputs=inputs, outputs=outputs, name='BeamSearch_GPT2')
    node.domain = "com.microsoft"
    node.attribute.extend([
        helper.make_attribute("eos_token_id", eos_token_id),
        helper.make_attribute("pad_token_id", pad_token_id),
        helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
        helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
        helper.make_attribute("body", model.graph),
    ])

    from onnx import TensorProto

    # graph inputs
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.INT32, ['batch_size', 'sequence_length'])
    max_length = helper.make_tensor_value_info('max_length', TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info('min_length', TensorProto.INT32, [1])
    num_beams = helper.make_tensor_value_info('num_beams', TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info('num_return_sequences', TensorProto.INT32, [1])
    temperature = helper.make_tensor_value_info('temperature', TensorProto.FLOAT, [1])
    length_penalty = helper.make_tensor_value_info('length_penalty', TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info('repetition_penalty', TensorProto.FLOAT, [1])
    vocab_mask = helper.make_tensor_value_info('vocab_mask', TensorProto.INT32, [vocab_size])

    graph_inputs = [
        input_ids, max_length, min_length, num_beams, num_return_sequences, temperature, length_penalty,
        repetition_penalty, vocab_mask
    ]

    # graph outputs
    sequences = helper.make_tensor_value_info('sequences', TensorProto.INT32,
                                              ['batch_size', 'num_return_sequences', 'max_length'])

    sequences_scores = helper.make_tensor_value_info('sequences_scores', TensorProto.FLOAT,
                                                     ['batch_size', 'num_return_sequences'])

    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT,
                                           ['max_length - sequence_length', 'batch_size', 'num_beams', vocab_size])

    initializers = []

    graph_outputs = [sequences]

    if args.output_sequences_scores:
        graph_outputs.append(sequences_scores)

    if args.output_token_scores:
        graph_outputs.append(scores)

    new_graph = helper.make_graph([node], 'gpt2-beam-search', graph_inputs, graph_outputs, initializers)

    # Create the model
    new_model = helper.make_model(new_graph, producer_name='onnxruntime.transformers', opset_imports=model.opset_import)
    onnx.save(new_model, args.output)


def test_model(args, use_vocab_mask: bool = False, sentences: List[str] = None):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir,
                                            pad_token_id=tokenizer.eos_token_id)

    # Use different length sentences to test batching
    if sentences is None:
        sentences = ["The product is released", "I enjoy walking in the park", "Test best way to invest"]

    inputs = tokenizer(sentences, return_tensors='pt', padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    bad_words = "walk in park"
    bad_words_ids = tokenizer.encode(bad_words, add_prefix_space=True)
    bad_words_ids = [[word_id] for word_id in bad_words_ids]  # Convert to list of list
    if use_vocab_mask:
        print("bad_words_ids", bad_words_ids)
    else:
        bad_words_ids = None

    global config
    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    torch_decoded_sequences = []
    if not args.disable_parity:
        print('-' * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(input_ids=input_ids,
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
                                      output_scores=True)
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

    print('-' * 50)
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
        "vocab_mask": vocab_mask
    }

    test_data_dir = Path(args.output).parent.as_posix()
    print("test_data_dir", test_data_dir)
    from bert_test_data import output_test_data
    all_inputs = [inputs]
    for i, inputs in enumerate(all_inputs):
        dir = os.path.join(test_data_dir, 'test_data_set_' + str(i))
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
        is_same = (torch_decoded_sequences == ort_decoded_sequences)
        print("Torch and ORT result is ", "same" if is_same else "different")
        output["parity"] = is_same

    print(output)
    return output


def main(argv=None, sentences=None):
    args = parse_arguments(argv)

    if os.path.exists(args.output):
        print(f"skip conversion since path existed: {args.output}")
    else:
        convert_model(args)

    return test_model(args, use_vocab_mask=True, sentences=sentences)


if __name__ == '__main__':
    main()

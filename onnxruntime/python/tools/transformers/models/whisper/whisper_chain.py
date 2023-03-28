import os

import re

import onnx
import pdb
from onnx import TensorProto, helper

from transformers import WhisperConfig

import argparse
import logging
import sys

def add_attention_mask(model):
    # Add attention mask - required by BeamSearch but unused in Pytorch
    mask = helper.make_tensor_value_info('encoder_attention_mask', TensorProto.INT32, shape=['batch', 'feature_size', 'sequence'])
    model.graph.input.insert(1, mask)




def chain_model(args):

    # Load encoder/decoder and insert necessary (but unused) graph inputs expected by BeamSearch op
    encoder_model = onnx.load(args.encoder_path, load_external_data=True)
    encoder_model.graph.name = "encoderdecoderinit subgraph"
    add_attention_mask(encoder_model)

    decoder_model = onnx.load(args.decoder_path, load_external_data=True)
    decoder_model.graph.name = "decoder subgraph"
    add_attention_mask(decoder_model)

    config = WhisperConfig.from_pretrained(args.model)
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id
    decoder_start_token_id = config.decoder_start_token_id

    beam_inputs = [
        "input_features",
        "max_length",
        "min_length",
        "num_beams",
        "num_return_sequences",
        "length_penalty",
        "repetition_penalty",
        "",
        "",
        "attention_mask",
    ]
    beam_outputs = ["sequences"]

    node = helper.make_node("BeamSearch", inputs=beam_inputs, outputs=beam_outputs, name=f"BeamSearch_zcode")
    node.domain = "com.microsoft"
    node.attribute.extend(
        [
            helper.make_attribute("eos_token_id", 50256),
            helper.make_attribute("pad_token_id", pad_token_id),
            helper.make_attribute("decoder_start_token_id", 50257),
            helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            helper.make_attribute("early_stopping", True),
            helper.make_attribute("model_type", 2),
            helper.make_attribute("decoder", decoder_model.graph),
            helper.make_attribute("encoder", encoder_model.graph),
        ]
    )

    # beam graph inputs
    input_features = helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch_size", "feature_size", "sequence_length"])
    max_length = helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    length_penalty = helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, ["batch_size", "feature_size", "sequence_length"]
    )

    graph_inputs = [
        input_features,
        max_length,
        min_length,
        num_beams,
        num_return_sequences,
        length_penalty,
        repetition_penalty,
        attention_mask,
    ]

    # graph outputs
    sequences = helper.make_tensor_value_info(
        "sequences", TensorProto.INT32, ["batch_size", "num_return_sequences", "max_length"]
    )
    graph_outputs = [sequences]

    #Initializers/opsets
    initializers = []
    opset_import = [helper.make_opsetid(domain="com.microsoft", version=1), helper.make_opsetid(domain="", version=17)]


    beam_graph = helper.make_graph([node], "beam-search-test", graph_inputs, graph_outputs, initializers)
    beam_model = helper.make_model(
        beam_graph, producer_name="pytorch", opset_imports=opset_import
    )

    final_path = os.path.join(args.output_dir, args.output_model)
    onnx.save(beam_model, final_path, save_as_external_data=True, all_tensors_to_one_file=False, convert_attribute=True)
    onnx.checker.check_model(final_path, full_check=True)

def print_args(args):
    logger = logging.getLogger("generate")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


def user_command():

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--max_length", type=int, default=20, help="default to 20")
    parent_parser.add_argument("--min_length", type=int, default=0, help="default to 0")
    parent_parser.add_argument("-o", "--output_dir", type=str, default="beamsearch_model", help="default name is beamsearch_model.")
    parent_parser.add_argument("--output_model", type=str, default="whisper_beamsearch.onnx", help="default name is whisper_beamsearch.onnx.")

    parent_parser.add_argument("-b", "--num_beams", type=int, default=5, help="default to 5")
    parent_parser.add_argument("--repetition_penalty", type=float, default=1.0, help="default to 1.0")
    parent_parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="default to 3")


    required_args = parent_parser.add_argument_group("required input arguments")
    required_args.add_argument(
        "-e",
        "--encoder_path",
        type=str,
        required=True,
        help="Path to encoder subgraph",
    )
    required_args.add_argument(
        "-d",
        "--decoder_path",
        type=str,
        required=True,
        help="Path to decoder subgraph",
    )
    required_args.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model being exported (e.g. openai/whisper-large) for scraping config values",
    )

    print_args(parent_parser.parse_args())
    return parent_parser.parse_args()


if __name__ == "__main__":

    args = user_command()

    isExist = os.path.exists(args.output_dir)
    if not isExist:
        os.makedirs(args.output_dir)

    chain_model(args)

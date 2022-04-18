#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#-------------------------------------------------------------------------

import os
import time
import onnx
from pathlib import Path
from onnx import helper
import numpy as np
from typing import List, Union
import torch
from onnx import onnx_pb as onnx_proto

def convert_model():
    eos_token_id = 2
    pad_token_id = 1

    encoder_model = onnx.load('/home/wy/Zcode/export/zcode_encoder/encoder.onnx', load_external_data=True)
    encoder_model.graph.name = "encoder subgraph"
    decoder_model = onnx.load('/home/wy/Zcode/export/zcode_decoder/decoder.onnx', load_external_data=True)
    decoder_model.graph.name = "decoder subgraph"

    inputs = [
        "input_ids", "max_length", "min_length", "num_beams", "num_return_sequences", "temperature", "length_penalty",
        "repetition_penalty"
    ]

    outputs = ["sequences"]

    node = helper.make_node('BeamSearch', inputs=inputs, outputs=outputs, name=f'BeamSearch_zcode')
    node.domain = "com.microsoft"
    node.attribute.extend([
        helper.make_attribute("eos_token_id", eos_token_id),
        helper.make_attribute("pad_token_id", pad_token_id),
        helper.make_attribute("decoder_start_token_id", 2),
        helper.make_attribute("no_repeat_ngram_size", 3),
        helper.make_attribute("early_stopping", 0),
        helper.make_attribute("model_type", 1),
        helper.make_attribute("decoder", decoder_model.graph),
        helper.make_attribute("encoder", encoder_model.graph),
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

    graph_inputs = [
        input_ids, max_length, min_length, num_beams, num_return_sequences, temperature, length_penalty,
        repetition_penalty
    ]

    # graph outputs
    sequences = helper.make_tensor_value_info('sequences', TensorProto.INT32,
                                              ['batch_size', 'num_return_sequences', 'max_length'])

    initializers = []

    graph_outputs = [sequences]

    new_graph = helper.make_graph([node], 'beam-search-test', graph_inputs, graph_outputs, initializers)

    # Create the model
    new_model = helper.make_model(new_graph, producer_name='onnxruntime.transformers', opset_imports=decoder_model.opset_import)
    onnx.save(new_model, '/home/wy/Zcode/export/zcode_beamsearch/beam_search_zcode.onnx', save_as_external_data=True, all_tensors_to_one_file=True, convert_attribute=True)


convert_model()
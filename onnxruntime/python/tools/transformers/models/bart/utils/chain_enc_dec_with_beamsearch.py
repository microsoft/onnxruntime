# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os

import onnx
from onnx import TensorProto, helper
from utils import export_helper


def make_dim_proto_numeric(model, config):
    """Make dim_proto numeric.

    Args:
        model (BartForConditionalGeneration): Bart model.
        config: Bart config.
    """
    sequence_length = str(1)
    num_heads = str(config.encoder_attention_heads)
    hidden_size = str(config.d_model)
    head_size = str(config.encoder_attention_heads)

    for tensor in model.graph.output:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                sequence_length,
                num_heads,
                hidden_size,
                head_size,
            ]:
                dim_value = int(dim_proto.dim_param)
                dim_proto.Clear()
                dim_proto.dim_value = dim_value

    for tensor in model.graph.input:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                sequence_length,
                num_heads,
                hidden_size,
                head_size,
            ]:
                dim_value = int(dim_proto.dim_param)
                dim_proto.Clear()
                dim_proto.dim_value = dim_value


def convert_model(args):
    """Combine encoder, decoder, and beam search op to convert ONNX model.

    Using beam search op to connect encoder and decoder of the model, and convert it into one ONNX model.

    Args:
        args: User input.
    """
    config, _ = export_helper.initialize_config(args)

    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id
    decoder_start_token_id = config.decoder_start_token_id

    encoder_path = os.path.join(args.output, "edinit.onnx")
    decoder_path = os.path.join(args.output, "decoder_past.onnx")
    final_path = os.path.join(args.output, "model_final.onnx")

    encoder_model = onnx.load(encoder_path, load_external_data=True)
    encoder_model.graph.name = "encoderdecoderinit subgraph"
    make_dim_proto_numeric(encoder_model, config)

    decoder_model = onnx.load(decoder_path, load_external_data=True)
    decoder_model.graph.name = "decoder subgraph"
    make_dim_proto_numeric(decoder_model, config)

    inputs = [
        "input_ids",
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
    outputs = ["sequences"]

    node = helper.make_node("BeamSearch", inputs=inputs, outputs=outputs, name=f"BeamSearch_zcode")
    node.domain = "com.microsoft"
    # NOTE: take value from args or config
    node.attribute.extend(
        [
            helper.make_attribute("eos_token_id", eos_token_id),
            helper.make_attribute("pad_token_id", pad_token_id),
            helper.make_attribute("decoder_start_token_id", decoder_start_token_id),
            helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            helper.make_attribute("early_stopping", args.early_stopping),
            helper.make_attribute("model_type", 1),
            helper.make_attribute("decoder", decoder_model.graph),
            helper.make_attribute("encoder", encoder_model.graph),
        ]
    )

    # graph inputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    max_length = helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    length_penalty = helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, ["batch_size", "sequence_length"]
    )

    graph_inputs = [
        input_ids,
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
    initializers = []
    graph_outputs = [sequences]
    new_graph = helper.make_graph([node], "beam-search-test", graph_inputs, graph_outputs, initializers)

    opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
    # Create the model
    decoder_model.opset_import.append(opset_import)
    new_model = helper.make_model(
        new_graph, producer_name="onnxruntime.transformers", opset_imports=decoder_model.opset_import
    )
    # https://github.com/onnx/onnx/blob/main/onnx/helper.py
    onnx.save(new_model, final_path, save_as_external_data=True, all_tensors_to_one_file=False, convert_attribute=True)
    # check model > 2GB
    print(f"--- Check the model with path: {final_path} ---")
    onnx.checker.check_model(final_path, full_check=True)
    onnx.shape_inference.infer_shapes_path(final_path, strict_mode=True)

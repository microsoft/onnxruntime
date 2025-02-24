"""Script to generate a dummy ONNX model emulating T5 model with BeamSearch op."""

import argparse

import numpy as np
import onnx

import onnxruntime as ort
from onnxruntime.transformers.convert_generation import move_initializers


def create_model(
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    head_size: int,
    beam_size: int,
    min_length: int,
    max_length: int,
    length_penalty: float,
    sequence_as_input: bool,
    decoder_needs_input_ids: bool,
) -> onnx.ModelProto:
    encoder_graph = create_encoder(vocab_size, embed_dim, num_heads, head_size)
    decoder_graph = create_decoder(
        vocab_size, embed_dim, num_heads, head_size, sequence_as_input, decoder_needs_input_ids
    )

    # Inputs: encoder_input_ids
    encoder_input_ids = onnx.helper.make_tensor_value_info(
        "encoder_input_ids", onnx.TensorProto.INT32, ["batch_size", "encode_sequence_length"]
    )

    # Outputs: sequences, scores
    sequences = onnx.helper.make_tensor_value_info(
        "sequences", onnx.TensorProto.INT32, ["batch_size", beam_size, "decode_sequence_length"]
    )
    scores = onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, ["batch_size", beam_size])

    # Tensors
    max_length_t = onnx.numpy_helper.from_array(np.array(max_length, dtype=np.int32), name="max_length")
    min_length_t = onnx.numpy_helper.from_array(np.array(min_length, dtype=np.int32), name="min_length")
    num_beams_t = onnx.numpy_helper.from_array(np.array(beam_size, dtype=np.int32), name="num_beams")
    length_penalty_t = onnx.numpy_helper.from_array(
        np.array(length_penalty, dtype=np.float32), name="length_penalty_as_tensor"
    )

    # Nodes
    beam_search = onnx.helper.make_node(
        "BeamSearch",
        ["encoder_input_ids", "max_length", "min_length", "num_beams", "num_beams", "length_penalty_as_tensor"],
        ["sequences", "scores"],
        decoder_start_token_id=2,
        eos_token_id=2,
        early_stopping=0,
        model_type=1,
        pad_token_id=1,
        decoder=decoder_graph,
        encoder=encoder_graph,
        domain="com.microsoft",
    )

    # Graph
    graph = onnx.helper.make_graph(
        [beam_search],
        "model",
        [encoder_input_ids],
        [sequences, scores],
        [max_length_t, min_length_t, num_beams_t, length_penalty_t],
    )

    # Model
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("com.microsoft", 1)]
    )

    return model


def create_encoder(vocab_size, embed_dim, num_heads, head_size) -> onnx.GraphProto:
    # Inputs: encoder_input_ids, encoder_attention_mask, decoder_input_ids
    encoder_input_ids = onnx.helper.make_tensor_value_info(
        "encoder_input_ids", onnx.TensorProto.INT32, ["batch_size", "encode_sequence_length"]
    )
    encoder_attention_mask = onnx.helper.make_tensor_value_info(
        "encoder_attention_mask", onnx.TensorProto.INT32, ["batch_size", "encode_sequence_length"]
    )
    decoder_input_ids = onnx.helper.make_tensor_value_info(
        "decoder_input_ids", onnx.TensorProto.INT32, ["batch_size", 1]
    )

    # Outputs: logits, present_key_self_0, present_value_self_0, present_key_cross_0, present_value_cross_0, encoder_hidden_states
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, ["batch_size", "decode_sequence_length", vocab_size]
    )
    present_key_self_0 = onnx.helper.make_tensor_value_info(
        "present_key_self_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, 1, head_size]
    )
    present_value_self_0 = onnx.helper.make_tensor_value_info(
        "present_value_self_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, 1, head_size]
    )
    present_key_cross_0 = onnx.helper.make_tensor_value_info(
        "present_key_cross_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
    )
    present_value_cross_0 = onnx.helper.make_tensor_value_info(
        "present_value_cross_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
    )
    encoder_hidden_states = onnx.helper.make_tensor_value_info(
        "encoder_hidden_states", onnx.TensorProto.FLOAT, ["batch_size", "encode_sequence_length", embed_dim]
    )

    # Tensors
    encoder_embeddings_tensor = onnx.numpy_helper.from_array(
        np.random.randn(vocab_size, embed_dim).astype(np.float32), name="encoder_embeddings"
    )
    num_heads_and_size_tensor = onnx.numpy_helper.from_array(
        np.array([num_heads, head_size], dtype=np.int64), name="num_heads_and_size"
    )
    final_proj_tensor = onnx.numpy_helper.from_array(
        np.random.randn(embed_dim, vocab_size).astype(np.float32), name="init_final_proj"
    )
    self_state_before_tranpose_shape_tensor = onnx.numpy_helper.from_array(
        np.array([-1, 1, num_heads, head_size], dtype=np.int64), name="self_state_before_tranpose_shape"
    )

    # Nodes
    nodes = [
        onnx.helper.make_node("Gather", ["encoder_embeddings", "encoder_input_ids"], ["encoder_hidden_states"]),
        onnx.helper.make_node("Shape", ["encoder_hidden_states"], ["encoder_batch_seq_len"], end=2),
        onnx.helper.make_node(
            "Concat", ["encoder_batch_seq_len", "num_heads_and_size"], ["encoder_final_shape"], axis=0
        ),
        onnx.helper.make_node(
            "Reshape", ["encoder_hidden_states", "encoder_final_shape"], ["encoder_hidden_states_reshaped"]
        ),
        onnx.helper.make_node(
            "Transpose", ["encoder_hidden_states_reshaped"], ["present_key_cross_0"], perm=[0, 2, 1, 3]
        ),
        onnx.helper.make_node(
            "Transpose", ["encoder_hidden_states_reshaped"], ["present_value_cross_0"], perm=[0, 2, 1, 3]
        ),
        onnx.helper.make_node("Gather", ["encoder_embeddings", "decoder_input_ids"], ["decoder_hidden_states"]),
        onnx.helper.make_node("ReduceMean", ["encoder_hidden_states"], ["encoder_hidden_states_mean"], axes=[1]),
        onnx.helper.make_node("Add", ["decoder_hidden_states", "encoder_hidden_states_mean"], ["encoder_decoder_sum"]),
        onnx.helper.make_node("MatMul", ["encoder_decoder_sum", "init_final_proj"], ["logits"]),
        onnx.helper.make_node(
            "Reshape", ["encoder_decoder_sum", "self_state_before_tranpose_shape"], ["self_state_before_tranpose"]
        ),
        onnx.helper.make_node("Transpose", ["self_state_before_tranpose"], ["present_key_self_0"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Transpose", ["self_state_before_tranpose"], ["present_value_self_0"], perm=[0, 2, 1, 3]),
    ]

    # Graph
    graph = onnx.helper.make_graph(
        nodes,
        "encoder",
        [encoder_input_ids, encoder_attention_mask, decoder_input_ids],
        [
            logits,
            encoder_hidden_states,
            present_key_self_0,
            present_value_self_0,
            present_key_cross_0,
            present_value_cross_0,
        ],
        [
            encoder_embeddings_tensor,
            num_heads_and_size_tensor,
            final_proj_tensor,
            self_state_before_tranpose_shape_tensor,
        ],
    )
    return graph


def create_decoder(
    vocab_size, embed_dim, num_heads, head_size, sequence_as_input, decoder_needs_input_ids
) -> onnx.GraphProto:
    # Inputs: input_ids, encoder_input_ids (optional), encoder_attention_mask, past_self_key_0, past_self_value_0, past_cross_key_0, past_cross_value_0
    inputs = []
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "input_ids", onnx.TensorProto.INT32, ["batch_size", "decode_sequence_length" if sequence_as_input else 1]
        )
    )
    if decoder_needs_input_ids:
        inputs.append(
            onnx.helper.make_tensor_value_info(
                "encoder_input_ids", onnx.TensorProto.INT32, ["batch_size", "encode_sequence_length"]
            )
        )
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "encoder_attention_mask", onnx.TensorProto.INT32, ["batch_size", "encode_sequence_length"]
        )
    )
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "past_self_key_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "decode_sequence_length", head_size]
        )
    )
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "past_self_value_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "decode_sequence_length", head_size]
        )
    )
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "past_cross_key_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
        )
    )
    inputs.append(
        onnx.helper.make_tensor_value_info(
            "past_cross_value_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
        )
    )

    # Outputs: logits, present_key_self_0, present_value_self_0
    outputs = [
        onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["batch_size", 1, vocab_size]),
        onnx.helper.make_tensor_value_info(
            "present_key_self_0",
            onnx.TensorProto.FLOAT,
            ["batch_size", num_heads, "present_decode_sequence_length", head_size],
        ),
        onnx.helper.make_tensor_value_info(
            "present_value_self_0",
            onnx.TensorProto.FLOAT,
            ["batch_size", num_heads, "present_decode_sequence_length", head_size],
        ),
    ]

    # Tensors: decoder_embeddings, final_proj, self_state_before_tranpose_shape_no_batch, hidden_states_mean
    initializers = [
        onnx.numpy_helper.from_array(
            np.random.randn(vocab_size, embed_dim).astype(np.float32), name="decoder_embeddings"
        ),
        onnx.numpy_helper.from_array(np.random.randn(embed_dim, vocab_size).astype(np.float32), name="final_proj"),
        onnx.numpy_helper.from_array(
            np.array([-1, num_heads, head_size], dtype=np.int64), name="self_state_before_tranpose_shape_no_batch"
        ),
        onnx.numpy_helper.from_array(np.array([-1, 1, embed_dim], dtype=np.int64), name="hidden_states_mean_shape"),
    ]

    # Nodes
    nodes = []
    nodes.append(onnx.helper.make_node("Gather", ["decoder_embeddings", "input_ids"], ["decoder_hidden_states"]))
    if decoder_needs_input_ids:
        nodes.append(
            onnx.helper.make_node("Gather", ["decoder_embeddings", "encoder_input_ids"], ["encoder_input_embeddings"])
        )
        nodes.append(
            onnx.helper.make_node(
                "ReduceMean", ["encoder_input_embeddings"], ["encoder_input_embeddings_mean"], axes=[1]
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Mul", ["decoder_hidden_states", "encoder_input_embeddings_mean"], ["combined_hidden_states"]
            )
        )
    else:
        nodes.append(onnx.helper.make_node("Identity", ["decoder_hidden_states"], ["combined_hidden_states"]))
    nodes.append(onnx.helper.make_node("ReduceMean", ["past_cross_key_0"], ["encoder_hidden_states_mean"], axes=[2]))
    nodes.append(
        onnx.helper.make_node(
            "Reshape",
            ["encoder_hidden_states_mean", "hidden_states_mean_shape"],
            ["encoder_hidden_states_mean_reshaped"],
        )
    )
    if sequence_as_input:
        nodes.append(
            onnx.helper.make_node("ReduceMean", ["combined_hidden_states"], ["decoder_hidden_states_mean"], axes=[1])
        )
        nodes.append(
            onnx.helper.make_node(
                "Add", ["decoder_hidden_states_mean", "encoder_hidden_states_mean_reshaped"], ["encoder_decoder_sum"]
            )
        )
    else:
        nodes.append(
            onnx.helper.make_node(
                "Add", ["combined_hidden_states", "encoder_hidden_states_mean_reshaped"], ["encoder_decoder_sum"]
            )
        )
    nodes.append(onnx.helper.make_node("Shape", ["combined_hidden_states"], ["decoder_batch"], end=1))
    nodes.append(
        onnx.helper.make_node(
            "Concat",
            ["decoder_batch", "self_state_before_tranpose_shape_no_batch"],
            ["self_state_before_tranpose_shape_dec"],
            axis=0,
        )
    )
    nodes.append(onnx.helper.make_node("MatMul", ["encoder_decoder_sum", "final_proj"], ["logits"]))
    nodes.append(
        onnx.helper.make_node(
            "Reshape", ["encoder_decoder_sum", "self_state_before_tranpose_shape_dec"], ["self_state_before_tranpose"]
        )
    )
    nodes.append(
        onnx.helper.make_node("Transpose", ["self_state_before_tranpose"], ["single_self_key_0"], perm=[0, 2, 1, 3])
    )
    nodes.append(
        onnx.helper.make_node("Transpose", ["self_state_before_tranpose"], ["single_self_value_0"], perm=[0, 2, 1, 3])
    )
    nodes.append(
        onnx.helper.make_node("Concat", ["past_self_key_0", "single_self_key_0"], ["present_key_self_0"], axis=2)
    )
    nodes.append(
        onnx.helper.make_node("Concat", ["past_self_value_0", "single_self_value_0"], ["present_value_self_0"], axis=2)
    )

    # Graph
    graph = onnx.helper.make_graph(nodes, "decoder", inputs, outputs, initializers)
    return graph


def run_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    encoder_input_ids = np.array([[14, 6, 13, 9, 7]]).astype(np.int32)
    print("encoder_input_ids: ", encoder_input_ids)
    sequence, scores = ort_session.run(None, {"encoder_input_ids": encoder_input_ids})
    print("sequence: ", sequence)
    print("scores: ", scores)


def move_initializers_on_outer_scope(model) -> None:
    main_graph = model.graph
    beam_search_node = model.graph.node[0]
    decoder_graph = next(attr for attr in beam_search_node.attribute if attr.name == "decoder").g
    encoder_graph = next(attr for attr in beam_search_node.attribute if attr.name == "encoder").g
    main_graph.initializer.extend(move_initializers(decoder_graph, min_elements=10))
    main_graph.initializer.extend(move_initializers(encoder_graph, min_elements=10))


def arg_parser():
    parser = argparse.ArgumentParser(description="Generate a dummy ONNX model emulating T5 model with BeamSearch op.")
    parser.add_argument("--output-path", type=str, default="model.onnx", help="Model output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vocab-size", type=int, default=20, help="Vocab size")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of heads")
    parser.add_argument("--head-size", type=int, default=4, help="Head size")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size")
    parser.add_argument("--min-length", type=int, default=1, help="Min length")
    parser.add_argument("--max-length", type=int, default=10, help="Max length")
    parser.add_argument("--length-penalty", type=float, default=1.1, help="Length penalty")
    parser.add_argument("--move-initializers", action="store_true", help="Move initializers to outer scope")
    parser.add_argument("--sequence-as-input", action="store_true", help="Use sequence as input")
    parser.add_argument("--decoder-needs-input-ids", action="store_true", help="Decoder needs model/encoder input ids")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    np.random.seed(args.seed)

    model = create_model(
        args.vocab_size,
        args.embed_dim,
        args.num_heads,
        args.head_size,
        args.beam_size,
        args.min_length,
        args.max_length,
        args.length_penalty,
        args.sequence_as_input,
        args.decoder_needs_input_ids,
    )
    if args.move_initializers:
        move_initializers_on_outer_scope(model)
    onnx.save(model, args.output_path)

    run_model(args.output_path)

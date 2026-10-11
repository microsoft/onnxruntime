"""Script to generate a dummy ONNX model emulating a Whisper model with BeamSearch op.

The model is intentionally tiny and produces deterministic (but meaningless) outputs.
Its only purpose is to exercise the WhisperBeamSearch encoder/decoder subgraph plumbing,
in particular the decoder "use sequence as input ids" path that builds the initial decoder
feeds from the full running sequences.
"""

import argparse

import numpy as np
import onnx


def create_model(
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    head_size: int,
    feature_size: int,
    beam_size: int,
    min_length: int,
    max_length: int,
    length_penalty: float,
    sequence_as_input: bool,
) -> onnx.ModelProto:
    encoder_graph = create_encoder(vocab_size, embed_dim, num_heads, head_size, feature_size)
    decoder_graph = create_decoder(vocab_size, embed_dim, num_heads, head_size, sequence_as_input)

    # Top-level inputs: input_features (audio) and decoder_input_ids (initial transcript tokens).
    input_features = onnx.helper.make_tensor_value_info(
        "input_features", onnx.TensorProto.FLOAT, ["batch_size", feature_size, "encode_sequence_length"]
    )
    decoder_input_ids = onnx.helper.make_tensor_value_info(
        "decoder_input_ids", onnx.TensorProto.INT32, ["batch_size", "initial_decode_sequence_length"]
    )

    # Outputs: sequences, scores
    sequences = onnx.helper.make_tensor_value_info(
        "sequences", onnx.TensorProto.INT32, ["batch_size", "num_return_sequences", "decode_sequence_length"]
    )
    scores = onnx.helper.make_tensor_value_info(
        "scores", onnx.TensorProto.FLOAT, ["batch_size", "num_return_sequences"]
    )

    # Initializers for the BeamSearch parameters.
    max_length_t = onnx.numpy_helper.from_array(np.array(max_length, dtype=np.int32), name="max_length")
    min_length_t = onnx.numpy_helper.from_array(np.array(min_length, dtype=np.int32), name="min_length")
    num_beams_t = onnx.numpy_helper.from_array(np.array(beam_size, dtype=np.int32), name="num_beams")
    num_return_sequences_t = onnx.numpy_helper.from_array(np.array(1, dtype=np.int32), name="num_return_sequences")
    length_penalty_t = onnx.numpy_helper.from_array(
        np.array(length_penalty, dtype=np.float32), name="length_penalty_as_tensor"
    )

    # The Whisper BeamSearch op expects decoder_input_ids at input index 10. The intervening
    # optional inputs (repetition_penalty, vocab_mask, prefix_vocab_mask, attention_mask) are
    # left empty.
    beam_search = onnx.helper.make_node(
        "BeamSearch",
        [
            "input_features",
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty_as_tensor",
            "",
            "",
            "",
            "",
            "decoder_input_ids",
        ],
        ["sequences", "scores"],
        decoder_start_token_id=2,
        eos_token_id=2,
        early_stopping=0,
        model_type=2,
        pad_token_id=1,
        decoder=decoder_graph,
        encoder=encoder_graph,
        domain="com.microsoft",
    )

    graph = onnx.helper.make_graph(
        [beam_search],
        "model",
        [input_features, decoder_input_ids],
        [sequences, scores],
        [max_length_t, min_length_t, num_beams_t, num_return_sequences_t, length_penalty_t],
    )

    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("com.microsoft", 1)]
    )

    return model


def create_encoder(vocab_size, embed_dim, num_heads, head_size, feature_size) -> onnx.GraphProto:
    # Inputs: encoder_input_ids (audio features, float), decoder_input_ids (int32)
    encoder_input_ids = onnx.helper.make_tensor_value_info(
        "encoder_input_ids", onnx.TensorProto.FLOAT, ["batch_size", feature_size, "encode_sequence_length"]
    )
    decoder_input_ids = onnx.helper.make_tensor_value_info(
        "decoder_input_ids", onnx.TensorProto.INT32, ["batch_size", "initial_decode_sequence_length"]
    )

    # Outputs: logits, encoder_hidden_states, present_key_self_0, present_value_self_0,
    #          present_key_cross_0, present_value_cross_0
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, ["batch_size", "initial_decode_sequence_length", vocab_size]
    )
    encoder_hidden_states = onnx.helper.make_tensor_value_info(
        "encoder_hidden_states", onnx.TensorProto.FLOAT, ["batch_size", "encode_sequence_length", embed_dim]
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

    # Initializers
    feature_proj = onnx.numpy_helper.from_array(
        np.random.randn(feature_size, embed_dim).astype(np.float32), name="feature_proj"
    )
    decoder_embeddings = onnx.numpy_helper.from_array(
        np.random.randn(vocab_size, embed_dim).astype(np.float32), name="encoder_decoder_embeddings"
    )
    final_proj = onnx.numpy_helper.from_array(
        np.random.randn(embed_dim, vocab_size).astype(np.float32), name="encoder_final_proj"
    )
    num_heads_and_size = onnx.numpy_helper.from_array(
        np.array([num_heads, head_size], dtype=np.int64), name="num_heads_and_size"
    )
    self_state_shape = onnx.numpy_helper.from_array(
        np.array([-1, 1, num_heads, head_size], dtype=np.int64), name="self_state_shape"
    )

    nodes = [
        # encoder_hidden_states = transpose(features)[B, Es, Fs] @ feature_proj[Fs, E] -> [B, Es, E]
        onnx.helper.make_node("Transpose", ["encoder_input_ids"], ["features_t"], perm=[0, 2, 1]),
        onnx.helper.make_node("MatMul", ["features_t", "feature_proj"], ["encoder_hidden_states"]),
        # cross KV: reshape [B, Es, E] -> [B, Es, num_heads, head_size] -> transpose [B, num_heads, Es, head_size]
        onnx.helper.make_node("Shape", ["encoder_hidden_states"], ["enc_batch_seq"], end=2),
        onnx.helper.make_node("Concat", ["enc_batch_seq", "num_heads_and_size"], ["enc_cross_shape"], axis=0),
        onnx.helper.make_node("Reshape", ["encoder_hidden_states", "enc_cross_shape"], ["enc_cross_reshaped"]),
        onnx.helper.make_node("Transpose", ["enc_cross_reshaped"], ["present_key_cross_0"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Transpose", ["enc_cross_reshaped"], ["present_value_cross_0"], perm=[0, 2, 1, 3]),
        # decoder hidden states from decoder_input_ids
        onnx.helper.make_node("Gather", ["encoder_decoder_embeddings", "decoder_input_ids"], ["decoder_hidden_states"]),
        # logits = decoder_hidden_states[B, Ds, E] @ final_proj[E, V] -> [B, Ds, V]
        onnx.helper.make_node("ReduceMean", ["encoder_hidden_states"], ["enc_hidden_mean"], axes=[1]),
        onnx.helper.make_node("Add", ["decoder_hidden_states", "enc_hidden_mean"], ["decoder_sum"]),
        onnx.helper.make_node("MatMul", ["decoder_sum", "encoder_final_proj"], ["logits"]),
        # self KV (length 1): reduce decoder hidden over Ds -> [B, 1, E] -> [B, 1, Hn, Hs] -> [B, Hn, 1, Hs]
        onnx.helper.make_node("ReduceMean", ["decoder_sum"], ["self_hidden_mean"], axes=[1]),
        onnx.helper.make_node("Reshape", ["self_hidden_mean", "self_state_shape"], ["self_state"]),
        onnx.helper.make_node("Transpose", ["self_state"], ["present_key_self_0"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Transpose", ["self_state"], ["present_value_self_0"], perm=[0, 2, 1, 3]),
    ]

    graph = onnx.helper.make_graph(
        nodes,
        "encoder",
        [encoder_input_ids, decoder_input_ids],
        [
            logits,
            encoder_hidden_states,
            present_key_self_0,
            present_value_self_0,
            present_key_cross_0,
            present_value_cross_0,
        ],
        [feature_proj, decoder_embeddings, final_proj, num_heads_and_size, self_state_shape],
    )
    return graph


def create_decoder(vocab_size, embed_dim, num_heads, head_size, sequence_as_input) -> onnx.GraphProto:
    # Inputs: input_ids, encoder_hidden_states, past_key_self_0, past_value_self_0,
    #         past_key_cross_0, past_value_cross_0
    inputs = [
        onnx.helper.make_tensor_value_info(
            "input_ids", onnx.TensorProto.INT32, ["batch_size", "decode_sequence_length" if sequence_as_input else 1]
        ),
        onnx.helper.make_tensor_value_info(
            "encoder_hidden_states", onnx.TensorProto.FLOAT, ["batch_size", "encode_sequence_length", embed_dim]
        ),
        onnx.helper.make_tensor_value_info(
            "past_key_self_0",
            onnx.TensorProto.FLOAT,
            ["batch_size", num_heads, "past_decode_sequence_length", head_size],
        ),
        onnx.helper.make_tensor_value_info(
            "past_value_self_0",
            onnx.TensorProto.FLOAT,
            ["batch_size", num_heads, "past_decode_sequence_length", head_size],
        ),
        onnx.helper.make_tensor_value_info(
            "past_key_cross_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
        ),
        onnx.helper.make_tensor_value_info(
            "past_value_cross_0", onnx.TensorProto.FLOAT, ["batch_size", num_heads, "encode_sequence_length", head_size]
        ),
    ]

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

    initializers = [
        onnx.numpy_helper.from_array(
            np.random.randn(vocab_size, embed_dim).astype(np.float32), name="decoder_embeddings"
        ),
        onnx.numpy_helper.from_array(np.random.randn(embed_dim, vocab_size).astype(np.float32), name="final_proj"),
        onnx.numpy_helper.from_array(
            np.array([-1, num_heads, head_size], dtype=np.int64), name="self_state_shape_no_batch"
        ),
        onnx.numpy_helper.from_array(np.array([-1, 1, embed_dim], dtype=np.int64), name="hidden_mean_shape"),
    ]

    nodes = [
        onnx.helper.make_node("Gather", ["decoder_embeddings", "input_ids"], ["decoder_hidden_states"]),
        # encoder signal from encoder_hidden_states mean -> [B, 1, E]
        onnx.helper.make_node("ReduceMean", ["encoder_hidden_states"], ["enc_hidden_mean"], axes=[1]),
        onnx.helper.make_node("Reshape", ["enc_hidden_mean", "hidden_mean_shape"], ["enc_hidden_mean_reshaped"]),
        # reduce decoder hidden over the sequence dim -> [B, 1, E]
        onnx.helper.make_node("ReduceMean", ["decoder_hidden_states"], ["decoder_hidden_mean"], axes=[1]),
        onnx.helper.make_node("Add", ["decoder_hidden_mean", "enc_hidden_mean_reshaped"], ["decoder_sum"]),
        onnx.helper.make_node("MatMul", ["decoder_sum", "final_proj"], ["logits"]),
        # self KV for this step (length 1) concatenated with the running past
        onnx.helper.make_node("Shape", ["decoder_sum"], ["decoder_batch"], end=1),
        onnx.helper.make_node(
            "Concat", ["decoder_batch", "self_state_shape_no_batch"], ["self_state_shape_dec"], axis=0
        ),
        onnx.helper.make_node("Reshape", ["decoder_sum", "self_state_shape_dec"], ["self_state"]),
        onnx.helper.make_node("Transpose", ["self_state"], ["single_key_self_0"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Transpose", ["self_state"], ["single_value_self_0"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Concat", ["past_key_self_0", "single_key_self_0"], ["present_key_self_0"], axis=2),
        onnx.helper.make_node("Concat", ["past_value_self_0", "single_value_self_0"], ["present_value_self_0"], axis=2),
    ]

    graph = onnx.helper.make_graph(nodes, "decoder", inputs, outputs, initializers)
    return graph


def run_model(model_path, feature_size):
    # Imported lazily so model *generation* only depends on `onnx`; running needs `onnxruntime`.
    import onnxruntime as ort  # noqa: PLC0415

    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    encode_length = 5
    # Fixed, deterministic inputs so a C++ regression test can reproduce the exact golden outputs.
    input_features = (((np.arange(feature_size * encode_length, dtype=np.float32) % 7) - 3.0) * 0.1).reshape(
        1, feature_size, encode_length
    )
    decoder_input_ids = np.array([[2, 5]], dtype=np.int32)
    sequences, scores = ort_session.run(
        None, {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
    )
    print("input_features (flat):", input_features.flatten().tolist())
    print("decoder_input_ids:", decoder_input_ids.tolist())
    print("sequences shape:", sequences.shape)
    print("sequences:", sequences.tolist())
    print("scores:", scores.tolist())
    return sequences, scores


def arg_parser():
    parser = argparse.ArgumentParser(description="Generate a dummy ONNX model emulating Whisper with BeamSearch op.")
    parser.add_argument("--output-path", type=str, default="dummy_whisper.onnx", help="Model output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vocab-size", type=int, default=20, help="Vocab size")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of heads")
    parser.add_argument("--head-size", type=int, default=4, help="Head size")
    parser.add_argument("--feature-size", type=int, default=8, help="Encoder input feature size")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size")
    parser.add_argument("--min-length", type=int, default=1, help="Min length")
    parser.add_argument("--max-length", type=int, default=10, help="Max length")
    parser.add_argument("--length-penalty", type=float, default=1.1, help="Length penalty")
    parser.add_argument("--sequence-as-input", action="store_true", help="Use sequence as input ids")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only generate and save the model; skip running it (avoids needing an onnxruntime install)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    np.random.seed(args.seed)

    model = create_model(
        args.vocab_size,
        args.embed_dim,
        args.num_heads,
        args.head_size,
        args.feature_size,
        args.beam_size,
        args.min_length,
        args.max_length,
        args.length_penalty,
        args.sequence_as_input,
    )
    onnx.save(model, args.output_path)

    if not args.no_run:
        run_model(args.output_path, args.feature_size)

from onnx import TensorProto, helper, save_model


# Whisper decoder self attention with past_kv, present_kv, buffer sharing enabled, mask, and bias
# Used in decoder-with-past's self-attention layers
def dmmha_inside_mha_self_attn():
    num_heads, head_size = 2, 32
    hidden_size = num_heads * head_size

    # Inputs
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [hidden_size * 3])
    past_k = helper.make_tensor_value_info(
        "past_k", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    past_v = helper.make_tensor_value_info(
        "past_v", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    past_seq_len = helper.make_tensor_value_info("past_seq_len", TensorProto.INT32, [1])
    cache_indir = helper.make_tensor_value_info(
        "cache_indir", TensorProto.INT32, ["batch_size", "num_beams", "max_sequence_length"]
    )
    inputs = [q, k, v, b, past_k, past_v, past_seq_len, cache_indir]

    # Outputs
    o = helper.make_tensor_value_info("o", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    present_k = helper.make_tensor_value_info(
        "present_k", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    present_v = helper.make_tensor_value_info(
        "present_v", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    outputs = [o, present_k, present_v]

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MultiHeadAttention",
                    inputs=["q", "k", "v", "b", "", "", "past_k", "past_v", "past_seq_len", "cache_indir"],
                    outputs=["o", "present_k", "present_v"],
                    name="MultiHeadAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                    unidirectional=1,
                )
            ],
            "dmmha-inside-mha-self-attn-graph",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    save_model(model, "dmmha_inside_mha_self_attn.onnx")


# Whisper decoder self attention with past_kv, present_kv, buffer sharing enabled, mask, and bias
# Used in decoder-with-past's self-attention layers
def dmmha_self_attn():
    num_heads, head_size = 2, 32
    hidden_size = num_heads * head_size

    # Inputs
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [hidden_size * 3])
    past_k = helper.make_tensor_value_info(
        "past_k", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    past_v = helper.make_tensor_value_info(
        "past_v", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    past_seq_len = helper.make_tensor_value_info("past_seq_len", TensorProto.INT32, [1])
    beam_width = helper.make_tensor_value_info("beam_width", TensorProto.INT32, [1])
    cache_indir = helper.make_tensor_value_info(
        "cache_indir", TensorProto.INT32, ["batch_size", "num_beams", "max_sequence_length"]
    )
    inputs = [q, k, v, b, past_k, past_v, past_seq_len, beam_width, cache_indir]

    # Outputs
    o = helper.make_tensor_value_info("o", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    present_k = helper.make_tensor_value_info(
        "present_k", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    present_v = helper.make_tensor_value_info(
        "present_v", TensorProto.FLOAT, ["batch_size", num_heads, "max_sequence_length", head_size]
    )
    outputs = [o, present_k, present_v]

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DecoderMaskedMultiHeadAttention",
                    inputs=[
                        "q",
                        "k",
                        "v",
                        "",
                        "",
                        "past_k",
                        "past_v",
                        "past_seq_len",
                        "beam_width",
                        "cache_indir",
                        "b",
                    ],
                    outputs=["o", "present_k", "present_v"],
                    name="DecoderMaskedMultiHeadAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                    past_present_share_buffer=1,
                )
            ],
            "dmmha-self-attn-graph",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    save_model(model, "dmmha_self_attn.onnx")


# Whisper decoder cross attention with past_kv used directly as K and V, no mask, and bias
# Used in decoder-with-past's cross-attention layers
def dmmha_inside_mha_cross_attn():
    num_heads, head_size = 2, 32
    hidden_size = num_heads * head_size
    encoder_seq_len = 10

    # Inputs
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    past_k = helper.make_tensor_value_info(
        "k", TensorProto.FLOAT, ["batch_size", num_heads, encoder_seq_len, head_size]
    )
    past_v = helper.make_tensor_value_info(
        "v", TensorProto.FLOAT, ["batch_size", num_heads, encoder_seq_len, head_size]
    )
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [hidden_size * 3])
    past_seq_len = helper.make_tensor_value_info("past_seq_len", TensorProto.INT32, [1])
    cache_indir = helper.make_tensor_value_info(
        "cache_indir", TensorProto.INT32, ["batch_size", "num_beams", "max_sequence_length"]
    )
    inputs = [q, past_k, past_v, b, past_seq_len, cache_indir]

    # Outputs
    o = helper.make_tensor_value_info("o", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    qk = helper.make_tensor_value_info(
        "qk", TensorProto.FLOAT, ["batch_size", "num_heads", "sequence_length", "total_sequence_length"]
    )
    outputs = [o, qk]

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MultiHeadAttention",
                    inputs=["q", "k", "v", "b", "", "", "", "", "past_seq_len", "cache_indir"],
                    outputs=["o", "", "", "qk"],
                    name="MultiHeadAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                    unidirectional=0,
                )
            ],
            "dmmha-inside-mha-cross-attn-graph",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    save_model(model, "dmmha_inside_mha_cross_attn.onnx")


# Whisper decoder cross attention with past_kv used directly as K and V, no mask, and bias
# Used in decoder-with-past's cross-attention layers
def dmmha_cross_attn():
    num_heads, head_size = 2, 32
    hidden_size = num_heads * head_size
    encoder_seq_len = 10

    # Inputs
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    past_k = helper.make_tensor_value_info(
        "k", TensorProto.FLOAT, ["batch_size", num_heads, encoder_seq_len, head_size]
    )
    past_v = helper.make_tensor_value_info(
        "v", TensorProto.FLOAT, ["batch_size", num_heads, encoder_seq_len, head_size]
    )
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [hidden_size * 3])
    past_seq_len = helper.make_tensor_value_info("past_seq_len", TensorProto.INT32, [1])
    beam_width = helper.make_tensor_value_info("beam_width", TensorProto.INT32, [1])
    cache_indir = helper.make_tensor_value_info(
        "cache_indir", TensorProto.INT32, ["batch_size", "num_beams", "max_sequence_length"]
    )
    inputs = [q, past_k, past_v, b, past_seq_len, beam_width, cache_indir]

    # Outputs
    o = helper.make_tensor_value_info("o", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    qk = helper.make_tensor_value_info(
        "qk", TensorProto.FLOAT, ["batch_size", "num_heads", "sequence_length", "total_sequence_length"]
    )
    outputs = [o, qk]

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DecoderMaskedMultiHeadAttention",
                    inputs=["q", "k", "v", "", "", "", "", "past_seq_len", "beam_width", "cache_indir", "b"],
                    outputs=["o", "", "", "qk"],
                    name="DecoderMaskedMultiHeadAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                    output_qk=1,
                    past_present_share_buffer=0,
                )
            ],
            "dmmha-cross-attn-graph",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    save_model(model, "dmmha_cross_attn.onnx")


dmmha_inside_mha_self_attn()
dmmha_inside_mha_cross_attn()

dmmha_self_attn()
dmmha_cross_attn()

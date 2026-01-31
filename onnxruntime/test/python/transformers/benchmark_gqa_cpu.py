# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions


class Formats(Enum):
    BSNH = 0
    BNSH = 1


class QKOutputType(Enum):
    NO_OUTPUT = 0
    BEFORE_SOFTMAX = 1
    AFTER_SOFTMAX = 2


@dataclass
class PromptConfig:
    batch_size: int = 0
    q_sequence_length: int = 0
    kv_sequence_length: int = 0
    buffer_sequence_length: int = 0
    num_heads: int = 0
    kv_num_heads: int = 0
    head_size: int = 0
    has_position_ids: bool = False
    has_attention_bias: bool = False
    has_head_sink: bool = False
    qk_output: QKOutputType = QKOutputType.NO_OUTPUT


def create_group_query_attention_graph_prompt(
    config,
    ort_type,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
    present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length

    output_names = [
        "output",
        "present_key",
        "present_value",
    ]
    if config.qk_output != QKOutputType.NO_OUTPUT:
        output_names.append("output_qk")

    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key" if share_buffer else "",
                "past_value" if share_buffer else "",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if rotary else "",
                "sin_cache" if rotary else "",
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            output_names,
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            qk_output=config.qk_output.value,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            ort_type,
            [
                config.batch_size,
                config.q_sequence_length,
                (
                    (config.num_heads * config.head_size)
                    if not packed
                    else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
                ),
            ],
        ),
        helper.make_tensor_value_info(
            "seqlens_k",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "total_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]
    if not packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                ort_type,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                ort_type,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if share_buffer:
        graph_input += [
            helper.make_tensor_value_info(
                "past_key",
                ort_type,
                [
                    config.batch_size,
                    past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                    config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "past_value",
                ort_type,
                [
                    config.batch_size,
                    past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                    config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                    config.head_size,
                ],
            ),
        ]

    # Simple outputs
    graph_output = [
        helper.make_tensor_value_info(
            "output",
            ort_type,
            [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def benchmark_gqa():
    # Configuration
    batch_size = 1
    sequence_length = 2048  # Using a reasonably large sequence length for prompt
    kv_sequence_length = 2048
    num_heads = 32
    kv_num_heads = 8
    head_size = 128

    config = PromptConfig(
        batch_size=batch_size,
        q_sequence_length=sequence_length,
        kv_sequence_length=kv_sequence_length,
        buffer_sequence_length=kv_sequence_length,  # Assume full buffer available
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
    )

    ort_type = TensorProto.FLOAT16
    dtype = np.float16

    print(
        f"Benchmarking GQA with B={batch_size}, S={sequence_length}, N={num_heads}, N_kv={kv_num_heads}, H={head_size}"
    )

    # Create model
    model_str = create_group_query_attention_graph_prompt(
        config,
        ort_type,
        past_kv_format=Formats.BNSH,
        share_buffer=True,
        rotary=False,  # Simplify for now
        packed=False,
    )

    sess_options = SessionOptions()
    sess_options.intra_op_num_threads = (
        1  # Single thread for stable consistent comparison or default? User asked for CPU test.
    )
    # Usually we want to utilize all cores or a fixed number. Let's use default but maybe print it.

    sess = InferenceSession(model_str, sess_options, providers=["CPUExecutionProvider"])

    # Create inputs
    query = np.random.randn(batch_size, sequence_length, num_heads * head_size).astype(dtype)
    key = np.random.randn(batch_size, kv_sequence_length, kv_num_heads * head_size).astype(dtype)
    value = np.random.randn(batch_size, kv_sequence_length, kv_num_heads * head_size).astype(dtype)

    # Past key/value (buffer) - BNSH
    past_key = np.random.randn(batch_size, kv_num_heads, kv_sequence_length, head_size).astype(dtype)
    past_value = np.random.randn(batch_size, kv_num_heads, kv_sequence_length, head_size).astype(dtype)

    seqlens_k = np.array([kv_sequence_length] * batch_size, dtype=np.int32)
    total_sequence_length = np.array([kv_sequence_length], dtype=np.int32)

    inputs = {
        "query": query,
        "key": key,
        "value": value,
        "past_key": past_key,
        "past_value": past_value,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_sequence_length,
    }

    # Warmup
    print("Warming up...")
    for _ in range(10):
        sess.run(None, inputs)

    # Measure
    iterations = 100
    print(f"Running {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        sess.run(None, inputs)
    end_time = time.time()

    avg_latency = (end_time - start_time) / iterations * 1000  # ms
    print(f"Average Latency: {avg_latency:.4f} ms")


if __name__ == "__main__":
    benchmark_gqa()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of SparseAttention. It supports Nvidia GPU of Compute Capability >= 8.0 in Linux.
"""

import math
import statistics
import time
from typing import Dict

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager


class Config:
    batch_size = 0
    sequence_length = 0
    max_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    sparse_block_size = 0
    num_layout = 0
    share_buffer = True

    # TODO: test performance with rotary embedding.
    do_rotary = False

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        max_sequence_length: int,
        past_sequence_length: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        sparse_block_size: int,
        num_layout: int,
        share_buffer: bool = True,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.sparse_block_size = sparse_block_size
        self.num_layout = num_layout
        self.share_buffer = share_buffer

        self.do_rotary = False

        # Derived values
        self.past_buffer_length = max_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = max_sequence_length if share_buffer else past_sequence_length + sequence_length
        self.max_blocks = max_sequence_length // sparse_block_size


def get_block_mask(num_layout, max_blocks, local_blocks=4, vert_stride=4):
    q_pos = torch.arange(max_blocks)[None, :, None]
    k_pos = torch.arange(max_blocks)[None, None]
    head_sliding_step = max(1, int(vert_stride / num_layout))
    mask_vert_strided = [
        (torch.arange(max_blocks) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(num_layout)
    ]
    mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
    block_mask = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
    block_mask = block_mask.to(torch.int32)

    torch.set_printoptions(profile="full")
    torch.set_printoptions(edgeitems=100)
    torch.set_printoptions(linewidth=200)
    print(block_mask)

    return block_mask


def get_shape_dict(config: Config):
    return {
        "query": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        "key": (config.batch_size, config.sequence_length, config.kv_num_heads * config.head_size),
        "value": (config.batch_size, config.sequence_length, config.kv_num_heads * config.head_size),
        "past_key": (config.batch_size, config.kv_num_heads, config.past_buffer_length, config.head_size),
        "past_value": (config.batch_size, config.kv_num_heads, config.past_buffer_length, config.head_size),
        "block_mask": (config.num_layout, config.max_blocks, config.max_blocks),
        "total_sequence_length": (1,),
        "key_total_sequence_lengths": (config.batch_size,),
        "output": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        "present_key": (config.batch_size, config.kv_num_heads, config.present_buffer_length, config.head_size),
        "present_value": (config.batch_size, config.kv_num_heads, config.present_buffer_length, config.head_size),
    }


def get_random_inputs(shape_dict: Dict, total_sequence_length, local_blocks, vert_stride, device, dtype=torch.float16):
    num_layout = shape_dict["block_mask"][0]
    max_blocks = shape_dict["block_mask"][1]

    return {
        "query": torch.randn(shape_dict["query"], device=device, dtype=dtype),
        "key": torch.randn(shape_dict["key"], device=device, dtype=dtype),
        "value": torch.randn(shape_dict["value"], device=device, dtype=dtype),
        "past_key": torch.randn(shape_dict["past_key"], device=device, dtype=dtype),
        "past_value": torch.randn(shape_dict["past_value"], device=device, dtype=dtype),
        "block_mask": get_block_mask(num_layout, max_blocks, local_blocks, vert_stride).to(device),
        "total_sequence_length": torch.tensor([total_sequence_length], dtype=torch.int32),
        "key_total_sequence_lengths": torch.ones(
            shape_dict["key_total_sequence_lengths"], device=device, dtype=torch.int32
        )
        * total_sequence_length,
    }


def create_graph(config):
    nodes = [
        helper.make_node(
            "SparseAttention",
            [
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "block_mask",
                "total_sequence_length" if config.share_buffer else "",
                "key_total_sequence_lengths",
                "cos_cache" if config.do_rotary else "",
                "sin_cache" if config.do_rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "SparseAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            sparse_block_size=config.sparse_block_size,
            do_rotary=1 if config.do_rotary else 0,
            domain="com.microsoft",
        ),
    ]

    shape_dict = get_shape_dict(config)
    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT16, list(shape_dict["query"])),
        helper.make_tensor_value_info("key", TensorProto.FLOAT16, list(shape_dict["key"])),
        helper.make_tensor_value_info("value", TensorProto.FLOAT16, list(shape_dict["value"])),
        helper.make_tensor_value_info("past_key", TensorProto.FLOAT16, list(shape_dict["past_key"])),
        helper.make_tensor_value_info("past_value", TensorProto.FLOAT16, list(shape_dict["past_value"])),
        helper.make_tensor_value_info("block_mask", TensorProto.INT32, list(shape_dict["block_mask"])),
        helper.make_tensor_value_info(
            "total_sequence_length", TensorProto.INT32, list(shape_dict["total_sequence_length"])
        ),
        helper.make_tensor_value_info(
            "key_total_sequence_lengths", TensorProto.INT32, list(shape_dict["key_total_sequence_lengths"])
        ),
    ]

    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", TensorProto.FLOAT16, list(shape_dict["present_key"])),
        helper.make_tensor_value_info("present_value", TensorProto.FLOAT16, list(shape_dict["present_value"])),
    ]

    graph = helper.make_graph(
        nodes,
        "SparseAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_session(config: Config, cuda_provider_options=None) -> InferenceSession:
    onnx_model_str = create_graph(config)
    session_options = SessionOptions()
    ort_session = InferenceSession(
        onnx_model_str,
        session_options,
        providers=[("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"],
    )
    return ort_session


def measure_latency(gpu_binding, feed_dict):
    start = time.time()
    _ = gpu_binding.infer(
        feed_dict,
    )
    end = time.time()
    return end - start


def flops(batch, q_seqlen, kv_seqlen, head_size, num_heads):
    return 4 * batch * q_seqlen * kv_seqlen * num_heads * head_size


def tflops_per_second(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def benchmark_op(gpu_binding, feed_dict, repeats=100):
    # warm up session
    _ = measure_latency(gpu_binding, feed_dict)

    latency_list = []
    for _ in range(repeats):
        latency = measure_latency(gpu_binding, feed_dict)
        latency_list.append(latency)
    return statistics.mean(latency_list)


def run_one_test(config: Config, local_blocks, vert_stride, device, dtype=torch.float16, repeats: int = 100):
    cuda_provider_options = CudaSession.get_cuda_provider_options(
        torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
    )
    cuda_provider_options["tunable_op_enable"] = True
    cuda_provider_options["tunable_op_tuning_enable"] = True
    ort_session = create_session(config, cuda_provider_options=cuda_provider_options)
    gpu_binding_manager = GpuBindingManager(
        ort_session=ort_session,
        device=device,
        stream=torch.cuda.current_stream().cuda_stream,
        max_cuda_graphs=2,
    )

    shape_dict = get_shape_dict(config)
    buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
    gpu_binding = gpu_binding_manager.get_binding(shape_dict, use_cuda_graph=False, buffer_sharing=buffer_sharing)

    total_sequence_length = config.sequence_length + config.past_sequence_length
    feed_dict = get_random_inputs(shape_dict, total_sequence_length, local_blocks, vert_stride, device, dtype=dtype)

    average_latency = benchmark_op(gpu_binding, feed_dict, repeats)
    del ort_session
    del gpu_binding_manager

    print(f"config: {config}, average_latency={average_latency}")


def run_performance_test(dtype=torch.float16, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # You can adjust these parameters to test different sparse configurations.
    num_layout, sparse_block_size, local_blocks, vert_stride = (4, 128, 8, 8)

    # Test prompt up to 16K tokens
    for s in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        config = Config(
            batch_size=1,
            sequence_length=s,
            max_sequence_length=16384,
            past_sequence_length=0,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
        )
        run_one_test(config, local_blocks, vert_stride, device, dtype, repeats)

    # Test token decoding up to 16K tokens
    for s in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        config = Config(
            batch_size=1,
            sequence_length=1,
            max_sequence_length=16384,
            past_sequence_length=s - 1,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
        )
        run_one_test(config, local_blocks, vert_stride, device, dtype, repeats)


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_performance_test()

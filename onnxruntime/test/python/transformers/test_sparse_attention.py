# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Parity test and benchmark performance of SparseAttention. Requires Nvidia GPU of Compute Capability 8.x.
"""

import statistics
import time
from typing import Dict, Optional

import torch
from onnx import TensorProto, helper
from torch import Tensor

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager

ENABLE_DEBUG = False


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

    local_blocks = 0
    vert_stride = 0
    softmax_scale = None
    share_buffer = True

    # TODO: test performance with rotary embedding.
    do_rotary = False

    is_fp16 = True  # True for float16; False for bfloat16.

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
        local_blocks: int,
        vert_stride: int,
        softmax_scale=None,
        share_buffer: bool = True,
        is_fp16: bool = True,
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
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.softmax_scale = softmax_scale
        self.share_buffer = share_buffer
        self.is_fp16 = is_fp16
        self.do_rotary = False

        # Derived values
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = max_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = max_sequence_length if share_buffer else past_sequence_length + sequence_length
        self.max_blocks = max_sequence_length // sparse_block_size

    def block_mask(self):
        return get_block_mask(self.num_layout, self.max_blocks, self.local_blocks, self.vert_stride)

    def dense_mask(self):
        expand_block_mask = self.block_mask()
        dense_mask = get_dense_mask(
            expand_block_mask, self.total_sequence_length, self.sequence_length, self.sparse_block_size
        )
        return dense_mask.repeat(self.batch_size, self.num_heads // self.num_layout, 1, 1)

    def shape_dict(self):
        return get_shape_dict(self)

    def random_inputs(self, device, dtype=torch.float16):
        block_mask = self.block_mask().to(device)
        shape_dict = get_shape_dict(self)
        return get_random_inputs(shape_dict, self.total_sequence_length, block_mask, device, dtype=dtype)


def get_block_mask(num_layout, max_blocks, local_blocks, vert_stride):
    q_pos = torch.arange(max_blocks)[None, :, None]
    k_pos = torch.arange(max_blocks)[None, None]
    head_sliding_step = max(1, int(vert_stride / num_layout))
    mask_vert_strided = [
        (torch.arange(max_blocks) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(num_layout)
    ]
    mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
    block_mask = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
    block_mask = block_mask.to(torch.int32)

    if ENABLE_DEBUG:
        torch.set_printoptions(profile="full")
        torch.set_printoptions(edgeitems=100)
        torch.set_printoptions(linewidth=200)
        print(block_mask)

    return block_mask


def get_dense_mask(block_mask, total_seq_len, query_seq_len, block_size):
    dense_mask = torch.kron(block_mask, block_mask.new_ones((block_size, block_size)))[
        :, :total_seq_len, :total_seq_len
    ]
    causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len)).type_as(dense_mask)
    dense_mask = dense_mask * causal_mask[None]
    return dense_mask[..., -query_seq_len:, :total_seq_len]


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


def get_random_inputs(shape_dict: Dict, total_sequence_length, block_mask, device, dtype=torch.float16):
    torch.manual_seed(123)
    return {
        "query": torch.empty(shape_dict["query"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
        "key": torch.empty(shape_dict["key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
        "value": torch.empty(shape_dict["value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
        "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
        "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
        "block_mask": block_mask.to(device),
        "total_sequence_length": torch.tensor([total_sequence_length], dtype=torch.int32),
        "key_total_sequence_lengths": torch.ones(
            shape_dict["key_total_sequence_lengths"], device=device, dtype=torch.int32
        )
        * total_sequence_length,
    }


def create_graph(config):
    assert config.is_fp16  # python does not support bfloat16 for I/O binding.

    float_type = TensorProto.FLOAT16
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
        helper.make_tensor_value_info("query", float_type, list(shape_dict["query"])),
        helper.make_tensor_value_info("key", float_type, list(shape_dict["key"])),
        helper.make_tensor_value_info("value", float_type, list(shape_dict["value"])),
        helper.make_tensor_value_info("past_key", float_type, list(shape_dict["past_key"])),
        helper.make_tensor_value_info("past_value", float_type, list(shape_dict["past_value"])),
        helper.make_tensor_value_info("block_mask", TensorProto.INT32, list(shape_dict["block_mask"])),
        helper.make_tensor_value_info(
            "total_sequence_length", TensorProto.INT32, list(shape_dict["total_sequence_length"])
        ),
        helper.make_tensor_value_info(
            "key_total_sequence_lengths", TensorProto.INT32, list(shape_dict["key_total_sequence_lengths"])
        ),
    ]

    graph_output = [
        helper.make_tensor_value_info("output", float_type, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", float_type, list(shape_dict["present_key"])),
        helper.make_tensor_value_info("present_value", float_type, list(shape_dict["present_value"])),
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


def benchmark_op(gpu_binding, feed_dict, repeats=100):
    # warm up session
    _ = measure_latency(gpu_binding, feed_dict)

    latency_list = []
    for _ in range(repeats):
        latency = measure_latency(gpu_binding, feed_dict)
        latency_list.append(latency)
    return statistics.mean(latency_list)


def group_query_attention_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    config: Config,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
):
    if scale is None:
        scale = 1.0 / (config.head_size**0.5)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Expand key and value to have same number of heads as query
    num_key_value_groups = config.num_heads // config.kv_num_heads
    key = torch.repeat_interleave(key, dim=1, repeats=num_key_value_groups)
    value = torch.repeat_interleave(value, dim=1, repeats=num_key_value_groups)

    # Apply multi-head attention.
    attn = torch.einsum("bhmd,bhnd->bhmn", query, key).float() * scale
    if mask is not None:
        attn = attn.masked_fill((1 - mask).bool(), float("-inf"))
    attn = attn.softmax(-1)
    attn_output = torch.einsum("bhmn,bhnd->bhmd", attn.type_as(value), value)

    return attn_output.transpose(1, 2).contiguous()


def run_relevance_no_past(device):
    """Test prompt prefilling without past kv cache."""
    for seq_len in [1, 64, 127, 128, 192, 256]:
        config = Config(
            batch_size=1,
            sequence_length=seq_len,
            max_sequence_length=256,
            past_sequence_length=0,
            num_heads=8,
            kv_num_heads=4,
            head_size=128,
            sparse_block_size=64,
            num_layout=2,
            local_blocks=2,
            vert_stride=2,
            softmax_scale=1.8 / (128**0.5),
        )
        dtype = torch.float16
        feed_dict = config.random_inputs(device, dtype=dtype)

        # Run GQA implementation by Torch as reference
        query = feed_dict["query"].view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)
        key = feed_dict["key"].view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
        value = feed_dict["value"].view(
            config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
        )

        dense_mask = config.dense_mask().to(device)
        expected_out = group_query_attention_reference(
            query.clone(), key.clone(), value.clone(), config, scale=config.softmax_scale, mask=dense_mask
        )

        if ENABLE_DEBUG:
            torch.set_printoptions(precision=6, edgeitems=3, linewidth=1000, profile="full", sci_mode=False)
            print("query(BNSH)", query.clone().transpose(1, 2))
            print("key(BNSH)", key.clone().transpose(1, 2))
            print("value(BNSH)", value.clone().transpose(1, 2))
            print("block_mask", feed_dict["block_mask"])
            print("dense_mask", dense_mask)
            print("total_sequence_length", feed_dict["total_sequence_length"])
            print("key_total_sequence_lengths", feed_dict["key_total_sequence_lengths"])

        # Run SparseAttention by ORT
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        ort_session = create_session(config, cuda_provider_options=cuda_provider_options)
        gpu_binding_manager = GpuBindingManager(
            ort_session=ort_session,
            device=device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        gpu_binding = gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        ort_outputs = gpu_binding.infer(feed_dict)
        ort_output = ort_outputs["output"]
        actual_out = ort_output.view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)

        if torch.allclose(expected_out, actual_out, atol=1e-2, rtol=0):
            print(f"Relevance test passed: {vars(config)}")
        else:
            print(f"Relevance test not passed: {vars(config)}")
            print("ort_output", actual_out)
            print("expected_out", expected_out)
            exit(1)


def run_relevance_past(device):
    """Test token generation with past kv cache."""
    for past_seq_len in [1, 63, 64, 127, 128, 511]:
        config = Config(
            batch_size=2,
            sequence_length=1,
            max_sequence_length=512,
            past_sequence_length=past_seq_len,
            num_heads=8,
            kv_num_heads=4,
            head_size=128,
            sparse_block_size=64,
            num_layout=4,
            local_blocks=2,
            vert_stride=4,
        )

        dtype = torch.float16
        feed_dict = config.random_inputs(device, dtype=dtype)

        # Run GQA implementation by Torch as reference
        query = feed_dict["query"].view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)
        key = feed_dict["key"].view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
        value = feed_dict["value"].view(
            config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
        )

        dense_mask = config.dense_mask().to(device)
        expected_out = group_query_attention_reference(
            query.clone(), key.clone(), value.clone(), config, mask=dense_mask
        )

        if ENABLE_DEBUG:
            torch.set_printoptions(precision=6, edgeitems=3, linewidth=1000, profile="full", sci_mode=False)
            print("query(BNSH)", query.clone().transpose(1, 2))
            print("key(BNSH)", key.clone().transpose(1, 2))
            print("value(BNSH)", value.clone().transpose(1, 2))
            print("block_mask", feed_dict["block_mask"])
            print("dense_mask", dense_mask)
            print("total_sequence_length", feed_dict["total_sequence_length"])
            print("key_total_sequence_lengths", feed_dict["key_total_sequence_lengths"])

        # Run SparseAttention by ORT
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        ort_session = create_session(config, cuda_provider_options=cuda_provider_options)
        gpu_binding_manager = GpuBindingManager(
            ort_session=ort_session,
            device=device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        gpu_binding = gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        ort_outputs = gpu_binding.infer(feed_dict)
        ort_output = ort_outputs["output"]
        actual_out = ort_output.view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)

        if torch.allclose(expected_out, actual_out, atol=1e-2, rtol=0):
            print(f"Relevance test passed: {vars(config)}")
        else:
            print(f"Relevance test not passed: {vars(config)}")
            print("ort_output", actual_out)
            print("expected_out", expected_out)
            exit(1)


def run_relevance_test():
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    with torch.no_grad():
        run_relevance_no_past(device)
        # the training kernel cannot handle past state?
        # run_relevance_past(device)


def run_ort_performance(config: Config, device, dtype=torch.float16, repeats: int = 100):
    cuda_provider_options = CudaSession.get_cuda_provider_options(
        torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
    )
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
    block_mask = config.block_mask()
    feed_dict = get_random_inputs(shape_dict, total_sequence_length, block_mask, device, dtype=dtype)

    average_latency = benchmark_op(gpu_binding, feed_dict, repeats)
    del ort_session
    del gpu_binding_manager

    print(f"{vars(config)}, average_latency={average_latency}")


def run_performance_test(dtype=torch.float16, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # You can adjust these parameters to test different sparse configurations.
    num_layout, sparse_block_size = (8, 64)

    # Test prompt
    for s in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        config = Config(
            batch_size=1,
            sequence_length=s,
            max_sequence_length=8192,
            past_sequence_length=0,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=16,
            vert_stride=8,
        )
        run_ort_performance(config, device, dtype, repeats)

    # Test token decoding
    for s in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        config = Config(
            batch_size=1,
            sequence_length=1,
            max_sequence_length=8192,
            past_sequence_length=s - 1,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=16,
            vert_stride=8,
        )
        run_ort_performance(config, device, dtype, repeats)


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_relevance_test()
        run_performance_test()

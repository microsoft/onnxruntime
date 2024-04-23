# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of SparseAttention. It supports Nvidia GPU of Compute Capability 8.x in Linux.
"""

import math
import statistics
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from onnx import TensorProto, helper
from torch import Tensor

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
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = max_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = max_sequence_length if share_buffer else past_sequence_length + sequence_length
        self.max_blocks = max_sequence_length // sparse_block_size


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

    torch.set_printoptions(profile="full")
    torch.set_printoptions(edgeitems=100)
    torch.set_printoptions(linewidth=200)
    print(block_mask)

    return block_mask


def get_dense_mask(block_mask, total_seq_len, query_seq_len, block_size):
    dense_mask = torch.kron(block_mask, block_mask.new_ones((block_size, block_size)))
    causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len)).type_as(dense_mask)[-query_seq_len:]
    return dense_mask[..., -query_seq_len:, :total_seq_len] * causal_mask[None]


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


def benchmark_op(gpu_binding, feed_dict, repeats=100):
    # warm up session
    _ = measure_latency(gpu_binding, feed_dict)

    latency_list = []
    for _ in range(repeats):
        latency = measure_latency(gpu_binding, feed_dict)
        latency_list.append(latency)
    return statistics.mean(latency_list)


def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
):
    """Reference implementation of group query.
        - b: batch size
        - n / s: sequence length
        - h: number of heads
        - g: number of groups
        - d: hidden dimension per head of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, h, n, s), (b, n, s) or (b, s).

    Returns:
        Attention output with shape (b, n, h, d)
    """
    assert (mask is None) or (is_causal is None)
    assert query.ndim == key.ndim == value.ndim == 4
    if scale is None:
        scale = query.size(-1) ** 0.5

    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    assert bq == bk == bv and dq == dk == dv
    assert (hk == hv) or (nk == nv)
    assert hq % hk == 0

    num_head_groups = hq // hk
    if num_head_groups > 1:
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if is_causal:
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)

    out = einsum(attention, value, "b h n s, b h s d -> b h n d")

    out = rearrange(out, "b h n d -> b n h d")

    return out


def run_relevance_no_past(device, dtype=torch.float16):
    local_blocks = 2
    vert_stride = 2
    config = Config(
        batch_size=1,
        sequence_length=256,
        max_sequence_length=256,
        past_sequence_length=0,
        num_heads=2,
        kv_num_heads=2,
        head_size=128,
        sparse_block_size=64,
        num_layout=2,
    )

    # Use a custom scale that is different from the default 1/sqrt(head_size).
    scale = 1.0 / math.sqrt(config.head_size) / 2.0

    block_mask = get_block_mask(config.num_layout, config.max_blocks, local_blocks, vert_stride).to(device)
    shape_dict = get_shape_dict(config)
    feed_dict = get_random_inputs(shape_dict, config.total_sequence_length, block_mask, device, dtype=dtype)

    # Reference implementation using torch SDPA
    dense_mask = get_dense_mask(
        block_mask, config.total_sequence_length, config.sequence_length, config.sparse_block_size
    )

    torch.set_printoptions(precision=6, edgeitems=3, linewidth=1000, profile="full", sci_mode=False)
    print("dense_mask", dense_mask)

    query = feed_dict["query"].view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)
    key = feed_dict["key"].view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
    value = feed_dict["value"].view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
    print("query(BNSH)", query.clone().transpose(1, 2))
    print("key(BNSH)", key.clone().transpose(1, 2))
    print("value(BNSH)", value.clone().transpose(1, 2))
    print("block_mask", feed_dict["block_mask"])
    print("total_sequence_length", feed_dict["total_sequence_length"])
    print("key_total_sequence_lengths", feed_dict["key_total_sequence_lengths"])

    expected_out = scaled_dot_product_gqa(
        query.clone(),
        key.clone(),
        value.clone(),
        scale=1.0 / scale,
        mask=dense_mask.repeat(config.batch_size, 1, 1, 1).bool(),
    )

    # Run ORT
    cuda_provider_options = CudaSession.get_cuda_provider_options(
        torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
    )
    # cuda_provider_options["tunable_op_enable"] = True
    # cuda_provider_options["tunable_op_tuning_enable"] = True
    ort_session = create_session(config, cuda_provider_options=cuda_provider_options)
    gpu_binding_manager = GpuBindingManager(
        ort_session=ort_session,
        device=device,
        stream=torch.cuda.current_stream().cuda_stream,
        max_cuda_graphs=2,
    )
    buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
    gpu_binding = gpu_binding_manager.get_binding(shape_dict, use_cuda_graph=False, buffer_sharing=buffer_sharing)
    ort_outputs = gpu_binding.infer(feed_dict)
    ort_output = ort_outputs["output"]

    actual_out = ort_output.view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)

    print("ort_output", actual_out.shape)
    print(actual_out)

    print("expected_out shape", expected_out.shape)
    print(expected_out)

    if torch.allclose(expected_out, actual_out, atol=0.005, rtol=0.001):
        print("Relevance test passed.")
    else:
        print("Relevance test not passed.")


def run_relevance_test(dtype=torch.float16):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    with torch.no_grad():
        run_relevance_no_past(device, dtype)


def run_ort_performance(config: Config, local_blocks, vert_stride, device, dtype=torch.float16, repeats: int = 100):
    cuda_provider_options = CudaSession.get_cuda_provider_options(
        torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
    )
    # cuda_provider_options["tunable_op_enable"] = True
    # cuda_provider_options["tunable_op_tuning_enable"] = True
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
    block_mask = get_block_mask(config.num_layout, config.max_blocks, local_blocks, vert_stride)
    feed_dict = get_random_inputs(shape_dict, total_sequence_length, block_mask, device, dtype=dtype)

    average_latency = benchmark_op(gpu_binding, feed_dict, repeats)
    del ort_session
    del gpu_binding_manager

    print(f"config: {config}, average_latency={average_latency}")


def run_performance_test(dtype=torch.float16, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # You can adjust these parameters to test different sparse configurations.
    num_layout, sparse_block_size, local_blocks, vert_stride = (8, 64, 16, 8)

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
        )
        run_ort_performance(config, local_blocks, vert_stride, device, dtype, repeats)

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
        )
        run_ort_performance(config, local_blocks, vert_stride, device, dtype, repeats)


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_relevance_test()
        # run_performance_test()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Parity test and benchmark performance of SparseAttention. Requires Nvidia GPU of Compute Capability 8.x.
"""

import math
from typing import Optional

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

    do_rotary = False
    rotary_interleaved = False

    # TODO: test packed qkv
    is_packed_qkv = False

    # TODO: test bfloat16.
    is_fp16 = True  # True for float16; False for bfloat16.

    use_sparse = True  # True for GroupQueryAttention; False for SparseAttention

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
        do_rotary: bool = False,
        rotary_interleaved: bool = False,
        device="cuda",
        operator="SparseAttention",
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
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_size**0.5)

        # Derived values
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = max_sequence_length if self.share_buffer else past_sequence_length
        self.present_buffer_length = (
            max_sequence_length if self.share_buffer else (past_sequence_length + sequence_length)
        )
        self.max_blocks = max_sequence_length // sparse_block_size

        self.do_rotary = do_rotary
        self.rotary_interleaved = rotary_interleaved
        self.device = device
        self.operator = operator

    def block_mask(self):
        return get_block_mask(self.num_layout, self.max_blocks, self.local_blocks, self.vert_stride).to(self.device)

    def dense_mask(self):
        expand_block_mask = self.block_mask()
        dense_mask = get_dense_mask(
            expand_block_mask, self.total_sequence_length, self.sequence_length, self.sparse_block_size
        )
        return dense_mask.repeat(self.batch_size, self.num_heads // self.num_layout, 1, 1).to(self.device)

    def shape_dict(self):
        shape_dict = {
            "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "key": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "value": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "past_key": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "past_value": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "block_mask": (self.num_layout, self.max_blocks, self.max_blocks),
            "total_sequence_length": (1,),
            "key_total_sequence_lengths": (self.batch_size,),
            "seqlens_k": (self.batch_size,),
            "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "present_key": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "present_value": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "cos_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
            "sin_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
        }

        if self.operator == "SparseAttention":
            del shape_dict["seqlens_k"]
        else:
            assert self.operator == "GroupQueryAttention"
            del shape_dict["key_total_sequence_lengths"]
            del shape_dict["block_mask"]
        return shape_dict

    def get_cos_sin_cache(self, dtype=torch.float32):
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * self.head_size) / 16) * 16
        angle = torch.rand(self.max_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        return cos.to(device=self.device), sin.to(device=self.device)

    def random_inputs(self, dtype=torch.float16):
        device = self.device
        shape_dict = self.shape_dict()
        k_seqlens = torch.ones((self.batch_size,), device=device, dtype=torch.int32) * self.total_sequence_length

        torch.manual_seed(123)
        feeds = {
            "query": torch.empty(shape_dict["query"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "key": torch.empty(shape_dict["key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "value": torch.empty(shape_dict["value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "block_mask": self.block_mask(),
            "total_sequence_length": torch.tensor([self.total_sequence_length], dtype=torch.int32),
            "key_total_sequence_lengths": k_seqlens,
            "seqlens_k": k_seqlens - 1,
        }

        if self.do_rotary:
            cos_cache, sin_cache = self.get_cos_sin_cache(dtype)
            feeds["cos_cache"] = cos_cache
            feeds["sin_cache"] = sin_cache

        if "seqlens_k" not in shape_dict:
            del feeds["seqlens_k"]
        else:
            del feeds["key_total_sequence_lengths"]
            del feeds["block_mask"]
        return feeds


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


def create_sparse_attention_onnx_model(config):
    assert config.is_fp16  # python does not support bfloat16 for I/O binding.

    float_type = TensorProto.FLOAT16
    nodes = [
        helper.make_node(
            "SparseAttention",
            [
                "query",
                "key" if not config.is_packed_qkv else "",
                "value" if not config.is_packed_qkv else "",
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
            scale=config.softmax_scale,
            sparse_block_size=config.sparse_block_size,
            do_rotary=1 if config.do_rotary else 0,
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.shape_dict()
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

    if config.do_rotary:
        graph_input += [
            helper.make_tensor_value_info("cos_cache", float_type, list(shape_dict["cos_cache"])),
            helper.make_tensor_value_info("sin_cache", float_type, list(shape_dict["sin_cache"])),
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


def create_group_query_attention_onnx_model(config):
    assert config.is_fp16  # python does not support bfloat16 for I/O binding.

    float_type = TensorProto.FLOAT16
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not config.is_packed_qkv else "",
                "value" if not config.is_packed_qkv else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length" if config.share_buffer else "",
                "cos_cache" if config.do_rotary else "",
                "sin_cache" if config.do_rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=config.local_blocks * config.sparse_block_size if config.use_sparse else -1,
            do_rotary=1 if config.do_rotary else 0,
            rotary_interleaved=config.rotary_interleaved,
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.shape_dict()
    graph_input = [
        helper.make_tensor_value_info("query", float_type, list(shape_dict["query"])),
        helper.make_tensor_value_info("key", float_type, list(shape_dict["key"])),
        helper.make_tensor_value_info("value", float_type, list(shape_dict["value"])),
        helper.make_tensor_value_info("past_key", float_type, list(shape_dict["past_key"])),
        helper.make_tensor_value_info("past_value", float_type, list(shape_dict["past_value"])),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, list(shape_dict["seqlens_k"])),
        helper.make_tensor_value_info(
            "total_sequence_length", TensorProto.INT32, list(shape_dict["total_sequence_length"])
        ),
    ]

    if config.do_rotary:
        graph_input += [
            helper.make_tensor_value_info("cos_cache", float_type, list(shape_dict["cos_cache"])),
            helper.make_tensor_value_info("sin_cache", float_type, list(shape_dict["sin_cache"])),
        ]

    graph_output = [
        helper.make_tensor_value_info("output", float_type, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", float_type, list(shape_dict["present_key"])),
        helper.make_tensor_value_info("present_value", float_type, list(shape_dict["present_value"])),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_session(onnx_model_str, config: Config, cuda_provider_options=None) -> InferenceSession:
    session_options = SessionOptions()
    ort_session = InferenceSession(
        onnx_model_str,
        session_options,
        providers=[("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"],
    )
    return ort_session


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


class TorchGroupQueryAttention:
    """A wrapper of Torch GroupQueryAttention to test relevance and performance."""

    def __init__(self, device, config: Config, feed_dict):
        self.device = device
        self.config = config
        self.query = feed_dict["query"].view(
            config.batch_size, config.sequence_length, config.num_heads, config.head_size
        )
        self.key = feed_dict["key"].view(
            config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
        )
        self.value = feed_dict["value"].view(
            config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
        )
        self.dense_mask = config.dense_mask()
        if ENABLE_DEBUG:
            torch.set_printoptions(precision=6, edgeitems=3, linewidth=1000, profile="full", sci_mode=False)
            print("query(BNSH)", self.query.clone().transpose(1, 2))
            print("key(BNSH)", self.key.clone().transpose(1, 2))
            print("value(BNSH)", self.value.clone().transpose(1, 2))
            print("dense_mask", self.dense_mask)

    def infer(self):
        return group_query_attention_reference(
            self.query, self.key, self.value, self.config, scale=self.config.softmax_scale, mask=self.dense_mask
        )


class OrtGroupQueryAttention:
    """A wrapper of ORT GroupQueryAttention to test relevance and performance."""

    def __init__(self, device, config: Config, feed_dict):
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_group_query_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, config, cuda_provider_options=cuda_provider_options)
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=self.ort_session,
            device=device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        self.gpu_binding = self.gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        self.feed_dict = feed_dict

    def infer(self):
        return self.gpu_binding.infer(self.feed_dict)


class OrtSparseAttention:
    """A wrapper of ORT SparseAttention to test relevance and performance."""

    def __init__(self, device, config: Config, feed_dict):
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_sparse_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, config, cuda_provider_options=cuda_provider_options)
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=self.ort_session,
            device=device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        self.gpu_binding = self.gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        self.feed_dict = feed_dict

    def infer(self):
        return self.gpu_binding.infer(self.feed_dict)


def run_one_relevance_test(device, config: Config):
    dtype = torch.float16

    # Run QGA ort
    config.use_sparse = False
    config.operator = "GroupQueryAttention"
    feed_dict = config.random_inputs(dtype=dtype)
    if config.past_sequence_length == 0:
        obj = TorchGroupQueryAttention(device, config, feed_dict)
        expected_out = obj.infer()
    else:
        obj = OrtGroupQueryAttention(device, config, feed_dict)
        ort_qga_outputs = obj.infer()
        expected_out = ort_qga_outputs["output"].view(
            config.batch_size, config.sequence_length, config.num_heads, config.head_size
        )

    # Run SparseAttention by ORT
    config.use_sparse = True
    config.operator = "SparseAttention"
    if config.past_sequence_length != 0:
        config.local_blocks = config.max_blocks  # Use dense to compare with GQA
    feed_dict = config.random_inputs(dtype=dtype)
    if ENABLE_DEBUG:
        print("block_mask", feed_dict["block_mask"])
        print("total_sequence_length", feed_dict["total_sequence_length"])
        print("key_total_sequence_lengths", feed_dict["key_total_sequence_lengths"])
    obj = OrtSparseAttention(device, config, feed_dict)
    ort_outputs = obj.infer()
    ort_output = ort_outputs["output"]
    actual_out = ort_output.view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)

    if torch.allclose(expected_out, actual_out, atol=1e-2, rtol=0):
        print(f"Relevance test passed: {vars(config)}")
    else:
        print(f"Relevance test not passed: {vars(config)}")
        print("ort_output", actual_out)
        print("expected_out", expected_out)
        print("diff", expected_out - actual_out)
        exit(1)


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
        run_one_relevance_test(device, config)


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
            do_rotary=True,
            rotary_interleaved=(past_seq_len % 2 == 1),
        )
        run_one_relevance_test(device, config)


def run_relevance_test():
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    with torch.no_grad():
        run_relevance_no_past(device)
        run_relevance_past(device)


def plot_prompt_performance(
    batch_size=4,
    num_heads=32,
    max_seq_len=8192,
    head_size=128,
    sparse_block_size=64,
    local_blocks=16,
    vert_stride=8,
    num_layout=8,
    dtype=torch.float16,
):
    import triton

    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(4, 14)],
            line_arg="provider",
            line_vals=["torch_gqa", "ort_gqa", "ort_gqa_local", "ort_sparse_att"],
            line_names=["TORCH-GQA", "ORT-GQA-Dense", "ORT-GQA-Local", "ORT-SparseAtt"],
            styles=[("red", "-"), ("yellow", "-"), ("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"prompt-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config = Config(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_sequence_length=max_seq_len,
            past_sequence_length=0,
            num_heads=num_heads,
            kv_num_heads=8,
            head_size=head_size,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=local_blocks,
            vert_stride=vert_stride,
        )

        config.use_sparse = provider in ["ort_sparse_att", "ort_gqa_local"]
        config.operator = "SparseAttention" if provider in ["ort_sparse_att"] else "GroupQueryAttention"
        feed_dict = config.random_inputs(dtype=dtype)

        if provider in ["ort_gqa", "ort_gqa_local"]:
            obj = OrtGroupQueryAttention(device, config, feed_dict)
        elif provider == "ort_sparse_att":
            obj = OrtSparseAttention(device, config, feed_dict)
        else:
            assert provider == "torch_gqa"
            if sequence_length > 2048:  # out of memory
                return 0
            obj = TorchGroupQueryAttention(device, config, feed_dict)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def plot_token_performance(
    batch_size=4,
    num_heads=32,
    max_seq_len=8192,
    head_size=128,
    sparse_block_size=64,
    local_blocks=16,
    vert_stride=8,
    num_layout=8,
    dtype=torch.float16,
):
    import triton

    configs = [
        triton.testing.Benchmark(
            x_names=["past_sequence_length"],
            x_vals=[2**i for i in range(4, 13)] + [max_seq_len - 1],
            line_arg="provider",
            line_vals=["torch_gqa", "ort_gqa", "ort_gqa_local", "ort_sparse_att"],
            line_names=["TORCH-GQA", "ORT-GQA-Dense", "ORT-GQA-Local", "ORT-SparseAtt"],
            styles=[("red", "-"), ("yellow", "-"), ("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"token-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, past_sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config = Config(
            batch_size=batch_size,
            sequence_length=1,
            max_sequence_length=max_seq_len,
            past_sequence_length=past_sequence_length,
            num_heads=num_heads,
            kv_num_heads=8,
            head_size=head_size,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=local_blocks,
            vert_stride=vert_stride,
        )

        config.use_sparse = provider in ["ort_sparse_att", "ort_gqa_local"]
        config.operator = "SparseAttention" if provider in ["ort_sparse_att"] else "GroupQueryAttention"
        feed_dict = config.random_inputs(dtype=dtype)

        if provider in ["ort_gqa", "ort_gqa_local"]:
            obj = OrtGroupQueryAttention(device, config, feed_dict)
        elif provider == "ort_sparse_att":
            obj = OrtSparseAttention(device, config, feed_dict)
        else:
            assert provider == "torch_gqa"
            if past_sequence_length > 2048:  # out of memory
                return 0
            obj = TorchGroupQueryAttention(device, config, feed_dict)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_relevance_test()
        plot_prompt_performance()
        plot_token_performance()

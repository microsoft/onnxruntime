# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Parity test and benchmark performance of SparseAttention. Requires Nvidia GPU of Compute Capability 8.x.
Install required packages before running this script:
   pip install matplotlib pandas onnx torch onnxruntime-gpu
"""

import math
from typing import Optional

import torch
from onnx import TensorProto, helper
from torch import Tensor

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager

ENABLE_DEBUG = False


class AttentionConfig:
    def __init__(
        self,
        operator: str,
        batch_size: int,
        sequence_length: int,
        max_sequence_length: int,
        past_sequence_length: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        softmax_scale: Optional[float],
        do_rotary: bool,
        rotary_interleaved: bool,
        device="cuda",
        dtype=torch.float16,
        share_buffer: bool = True,
        is_packed_qkv: bool = False,
    ):
        self.operator = operator
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_size**0.5)

        # Derived values
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = max_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = max_sequence_length if share_buffer else (past_sequence_length + sequence_length)

        self.do_rotary = do_rotary
        self.rotary_interleaved = rotary_interleaved
        self.device = device

        self.share_buffer = share_buffer
        self.is_packed_qkv = is_packed_qkv
        self.dtype = dtype

    def shape_dict(self):
        return {
            "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "key": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "value": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "past_key": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "past_value": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "total_sequence_length": (1,),
            "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "present_key": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "present_value": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "cos_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
            "sin_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
        }

    def get_cos_sin_cache(self, dtype):
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * self.head_size) / 16) * 16
        angle = torch.rand(self.max_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        return cos.to(device=self.device), sin.to(device=self.device)

    def random_inputs(self):
        device = self.device
        # Since bfloat16 is not supported in ORT python I/O binding API, we always use float16 as model inputs.
        dtype = torch.float16
        shape_dict = self.shape_dict()

        torch.manual_seed(123)
        feeds = {
            "query": torch.empty(shape_dict["query"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "key": torch.empty(shape_dict["key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "value": torch.empty(shape_dict["value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "total_sequence_length": torch.tensor([self.total_sequence_length], dtype=torch.int32),
        }

        if self.do_rotary:
            cos_cache, sin_cache = self.get_cos_sin_cache(dtype)
            feeds["cos_cache"] = cos_cache
            feeds["sin_cache"] = sin_cache

        return feeds


class GroupQueryAttentionConfig(AttentionConfig):
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        max_sequence_length: int,
        past_sequence_length: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        softmax_scale=None,
        do_rotary: bool = False,
        rotary_interleaved: bool = False,
        device="cuda",
        local_window_size: int = -1,
        attention_mask=None,
    ):
        super().__init__(
            "GroupQueryAttention",
            batch_size,
            sequence_length,
            max_sequence_length,
            past_sequence_length,
            num_heads,
            kv_num_heads,
            head_size,
            softmax_scale,
            do_rotary,
            rotary_interleaved,
            device,
        )
        # local_window_size is for ORT only, not for Torch implementation.
        self.local_window_size = local_window_size

        # attention mask is for Torch implementation only, not for ORT.
        self.attention_mask = attention_mask

    def shape_dict(self):
        shapes = super().shape_dict()
        shapes.update(
            {
                "seqlens_k": (self.batch_size,),
            }
        )
        return shapes

    def random_inputs(self):
        feeds = super().random_inputs()
        k_seqlens = torch.ones((self.batch_size,), device=self.device, dtype=torch.int32) * self.total_sequence_length
        feeds.update(
            {
                "seqlens_k": k_seqlens - 1,
            }
        )
        return feeds


class SparseAttentionConfig(AttentionConfig):
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
        dense_length_threshold = None,
    ):
        super().__init__(
            "SparseAttention",
            batch_size,
            sequence_length,
            max_sequence_length,
            past_sequence_length,
            num_heads,
            kv_num_heads,
            head_size,
            softmax_scale,
            do_rotary,
            rotary_interleaved,
            device,
        )
        self.sparse_block_size = sparse_block_size
        self.num_layout = num_layout
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.max_blocks = max_sequence_length // sparse_block_size
        self.dense_length_threshold = local_blocks * sparse_block_size if dense_length_threshold is None else dense_length_threshold

    def block_mask(self):
        return get_block_mask(self.num_layout, self.max_blocks, self.local_blocks, self.vert_stride).to(self.device)

    def dense_mask(self):
        expand_block_mask = self.block_mask()
        dense_mask = get_dense_mask(
            expand_block_mask, self.total_sequence_length, self.sequence_length, self.sparse_block_size
        )
        return dense_mask.repeat(self.batch_size, self.num_heads // self.num_layout, 1, 1).to(self.device)

    def shape_dict(self):
        shapes = super().shape_dict()
        shapes.update(
            {
                "block_mask": (self.num_layout, self.max_blocks, self.max_blocks),
                "key_total_sequence_lengths": (self.batch_size,),
            }
        )
        return shapes

    def random_inputs(self):
        feeds = super().random_inputs()
        k_seqlens = torch.ones((self.batch_size,), device=self.device, dtype=torch.int32) * self.total_sequence_length
        feeds.update(
            {
                "block_mask": self.block_mask(),
                "total_sequence_length": torch.tensor([self.total_sequence_length], dtype=torch.int32),
                "key_total_sequence_lengths": k_seqlens,
            }
        )
        return feeds

    def get_comparable_gqa_config(self, use_local=False, torch_use_sparse=False) -> GroupQueryAttentionConfig:
        attention_mask = None
        if torch_use_sparse:
            attention_mask = self.dense_mask()[:, :, : self.total_sequence_length, : self.total_sequence_length]
            if self.past_sequence_length > 0:
                attention_mask = attention_mask[:, :, -self.sequence_length :, :]

        return GroupQueryAttentionConfig(
            self.batch_size,
            self.sequence_length,
            self.max_sequence_length,
            self.past_sequence_length,
            self.num_heads,
            self.kv_num_heads,
            self.head_size,
            self.softmax_scale,
            self.do_rotary,
            self.rotary_interleaved,
            self.device,
            local_window_size=self.local_blocks * self.sparse_block_size if use_local else -1,
            attention_mask=attention_mask,
        )


def get_block_mask(num_layout, max_blocks, local_blocks, vert_stride):
    q_pos = torch.arange(max_blocks)[None, :, None]
    k_pos = torch.arange(max_blocks)[None, None]
    head_sliding_step = max(1, int(vert_stride / num_layout))
    mask_vert_strided = [
        (torch.arange(max_blocks) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(num_layout)
    ]
    mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
    local_mask = q_pos - k_pos < local_blocks
    block_mask = (q_pos >= k_pos) & (local_mask | mask_vert_strided)
    block_mask = block_mask.to(torch.int32)

    if ENABLE_DEBUG:
        print(f"{num_layout=} {max_blocks=} {local_blocks=} {vert_stride=}")
        print(f"{block_mask=}")

    return block_mask


def get_dense_mask(block_mask, total_seq_len, query_seq_len, block_size):
    dense_mask = torch.kron(block_mask, block_mask.new_ones((block_size, block_size)))[
        :, :total_seq_len, :total_seq_len
    ]
    causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len)).type_as(dense_mask)
    dense_mask = dense_mask * causal_mask[None]
    return dense_mask[..., -query_seq_len:, :total_seq_len]


def create_sparse_attention_onnx_model(config: SparseAttentionConfig):
    # ORT Python I/O binding API does not support bf16, so always use fp16 as graph inputs/outputs.
    io_float_type = TensorProto.FLOAT16

    suffix = "_bf16" if config.dtype == torch.bfloat16 else ""
    nodes = [
        helper.make_node(
            "SparseAttention",
            [
                "query" + suffix,
                "key" + suffix if not config.is_packed_qkv else "",
                "value" + suffix if not config.is_packed_qkv else "",
                "past_key" + suffix,
                "past_value" + suffix,
                "block_mask",
                "total_sequence_length" if config.share_buffer else "",
                "key_total_sequence_lengths",
                "cos_cache" + suffix if config.do_rotary else "",
                "sin_cache" + suffix if config.do_rotary else "",
            ],
            ["output" + suffix, "present_key" + suffix, "present_value" + suffix],
            "SparseAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            scale=config.softmax_scale,
            sparse_block_size=config.sparse_block_size,
            dense_length_threshold=config.dense_length_threshold,
            do_rotary=1 if config.do_rotary else 0,
            domain="com.microsoft",
        ),
    ]

    # When testing bfloat16, we add cast nodes so that SparseAttention is computed in bfloat16.
    if config.dtype == torch.bfloat16:
        nodes.extend(
            [
                helper.make_node("Cast", [input], [input + suffix], f"Cast_{input}", to=TensorProto.BFLOAT16)
                for input in ["query", "key", "value", "past_key", "past_value"]
            ]
        )
        if config.do_rotary:
            nodes.extend(
                [
                    helper.make_node("Cast", [input], [input + suffix], f"Cast_{input}", to=TensorProto.BFLOAT16)
                    for input in ["cos_cache", "sin_cache"]
                ]
            )
        nodes.extend(
            [
                helper.make_node("Cast", [output + suffix], [output], f"Cast_{output}", to=TensorProto.FLOAT16)
                for output in ["output", "present_key", "present_value"]
            ]
        )

    shape_dict = config.shape_dict()
    graph_input = [
        helper.make_tensor_value_info("query", io_float_type, list(shape_dict["query"])),
        helper.make_tensor_value_info("key", io_float_type, list(shape_dict["key"])),
        helper.make_tensor_value_info("value", io_float_type, list(shape_dict["value"])),
        helper.make_tensor_value_info("past_key", io_float_type, list(shape_dict["past_key"])),
        helper.make_tensor_value_info("past_value", io_float_type, list(shape_dict["past_value"])),
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
            helper.make_tensor_value_info("cos_cache", io_float_type, list(shape_dict["cos_cache"])),
            helper.make_tensor_value_info("sin_cache", io_float_type, list(shape_dict["sin_cache"])),
        ]

    graph_output = [
        helper.make_tensor_value_info("output", io_float_type, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", io_float_type, list(shape_dict["present_key"])),
        helper.make_tensor_value_info("present_value", io_float_type, list(shape_dict["present_value"])),
    ]

    graph = helper.make_graph(
        nodes,
        "SparseAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_group_query_attention_onnx_model(config: GroupQueryAttentionConfig):
    assert config.dtype == torch.float16

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
            scale=config.softmax_scale,
            local_window_size=config.local_window_size,
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


def create_session(onnx_model_str, cuda_provider_options=None) -> InferenceSession:
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
    config: GroupQueryAttentionConfig,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
):
    if scale is None:
        scale = 1.0 / (config.head_size**0.5)

    # Query is in BSNH shape, transpose it here. Note that key/value is BNSH format (transposed).
    query = query.transpose(1, 2)

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

    result = attn_output.transpose(1, 2).contiguous()
    torch.cuda.synchronize()
    return result


class TorchGroupQueryAttention:
    """A wrapper of Torch GroupQueryAttention to test relevance and performance."""

    def __init__(self, config: GroupQueryAttentionConfig):
        self.device = config.device
        self.config = config
        self.feed_dict = config.random_inputs()
        self.dense_mask = config.attention_mask

    @staticmethod
    def concat_cache(past_key_cache, new_key):
        """
        Concatenates a new key to a past key cache.

        Args:
        - past_key (torch.Tensor): Past key cache with shape (batch_size, num_heads, past_sequence_length, head_dim)
        - new_key (torch.Tensor): New key with shape (batch_size, num_heads, sequence_length, head_dim)

        Returns:
        - present_key (torch.Tensor): Concatenated key tensor with shape (batch_size, num_heads, new_length, head_dim)
                                      where new_length = past_sequence_length + sequence_length
        """
        # Check if the past_key_cache and new_key have compatible shapes
        assert past_key_cache.size(0) == new_key.size(0), "Batch sizes do not match"
        assert past_key_cache.size(1) == new_key.size(1), "Number of heads do not match"
        assert past_key_cache.size(3) == new_key.size(3), "Head dimensions do not match"

        # Concatenate the keys along the sequence length dimension
        concatenated_keys = torch.cat((past_key_cache, new_key), dim=2)

        return concatenated_keys

    def infer(self):
        config = self.config
        query = self.feed_dict["query"].view(
            config.batch_size, config.sequence_length, config.num_heads, config.head_size
        )
        key = (
            self.feed_dict["key"]
            .view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
            .transpose(1, 2)
        )
        value = (
            self.feed_dict["value"]
            .view(config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size)
            .transpose(1, 2)
        )

        if config.past_sequence_length > 0:
            past_key = self.feed_dict["past_key"][:, :, : config.past_sequence_length, :]
            past_value = self.feed_dict["past_value"][:, :, : config.past_sequence_length, :]
            present_key = TorchGroupQueryAttention.concat_cache(past_key, key)
            present_value = TorchGroupQueryAttention.concat_cache(past_value, value)
        else:
            present_key = key
            present_value = value

        if ENABLE_DEBUG:
            print("query(BSNH, GQA)", query)
            print("present_key(BNSH, GQA)", present_key)
            print("present_key(BNSH, GQA)", present_value)
            print("dense_mask", self.dense_mask)

        return group_query_attention_reference(
            query, present_key, present_value, config, scale=config.softmax_scale, mask=self.dense_mask
        )


class OrtGroupQueryAttention:
    """A wrapper of ORT GroupQueryAttention to test relevance and performance."""

    def __init__(self, config: GroupQueryAttentionConfig):
        device = config.device
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_group_query_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, cuda_provider_options=cuda_provider_options)
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
        self.feed_dict = config.random_inputs()

        if ENABLE_DEBUG:
            query = self.feed_dict["query"].view(
                config.batch_size, config.sequence_length, config.num_heads, config.head_size
            )
            key = self.feed_dict["key"].view(
                config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
            )
            value = self.feed_dict["value"].view(
                config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
            )
            print(vars(config))
            print("query(BSNH, GQA)", query)
            print("key(BSNH, GQA)", key)
            print("value(BSNH, GQA)", value)
            print("seqlens_k (BSNH, GQA)", self.feed_dict["seqlens_k"])

    def infer(self):
        return self.gpu_binding.infer(self.feed_dict)


class OrtSparseAttention:
    """A wrapper of ORT SparseAttention to test relevance and performance."""

    def __init__(self, config: SparseAttentionConfig):
        device = config.device
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_sparse_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, cuda_provider_options=cuda_provider_options)
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
        self.feed_dict = config.random_inputs()

        if ENABLE_DEBUG:
            query = self.feed_dict["query"].view(
                config.batch_size, config.sequence_length, config.num_heads, config.head_size
            )
            key = self.feed_dict["key"].view(
                config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
            )
            value = self.feed_dict["value"].view(
                config.batch_size, config.sequence_length, config.kv_num_heads, config.head_size
            )
            print(vars(config))
            print("query(BSNH, SA)", query)
            print("key(BSNH, SA)", key)
            print("value(BSNH, SA)", value)
            print("block_mask (SA)", self.feed_dict["block_mask"])
            print("total_sequence_length", self.feed_dict["total_sequence_length"])
            print("key_total_sequence_lengths", self.feed_dict["key_total_sequence_lengths"])

    def infer(self):
        return self.gpu_binding.infer(self.feed_dict)


def run_one_relevance_test(config: SparseAttentionConfig):
    # Run QGA
    if not config.do_rotary:  # config.past_sequence_length == 0:
        gqa_config: GroupQueryAttentionConfig = config.get_comparable_gqa_config(torch_use_sparse=True)
        obj = TorchGroupQueryAttention(gqa_config)
        expected_out = obj.infer()
    else:
        gqa_config: GroupQueryAttentionConfig = config.get_comparable_gqa_config(use_local=False)
        obj = OrtGroupQueryAttention(gqa_config)
        ort_qga_outputs = obj.infer()
        expected_out = ort_qga_outputs["output"].view(
            config.batch_size, config.sequence_length, config.num_heads, config.head_size
        )

    # Run SparseAttention by ORT
    obj = OrtSparseAttention(config)
    ort_outputs = obj.infer()
    ort_output = ort_outputs["output"]
    actual_out = ort_output.view(config.batch_size, config.sequence_length, config.num_heads, config.head_size)

    red_color = "\033[31m"
    green_color = "\033[32m"
    reset_color = "\033[0m"
    if torch.allclose(expected_out, actual_out, atol=1e-2, rtol=0):
        print(f"Relevance test {green_color}passed{reset_color}: {vars(config)}")
    else:
        print(f"Relevance test {red_color}failed{reset_color}: {vars(config)}")
        print("ort_output", actual_out)
        print("expected_out", expected_out)
        print("diff", expected_out - actual_out)
        exit(1)


def run_relevance_no_past(sm: int, device):
    """Test prompt prefilling without past kv cache."""
    for seq_len in [1, 64, 127, 128, 192, 256]:
        config = SparseAttentionConfig(
            batch_size=3,
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
            device=device,
        )
        run_one_relevance_test(config)

        config.dense_length_threshold=0
        run_one_relevance_test(config)

        if sm >= 80:
            config.dtype = torch.bfloat16
            run_one_relevance_test(config)


def run_relevance_past(sm: int, device, do_rotary: bool):
    """Test token generation with past kv cache."""
    for past_seq_len in [1, 63, 64, 127, 128, 511]:
        config = SparseAttentionConfig(
            batch_size=3,
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
            softmax_scale=None,
            do_rotary=do_rotary,
            rotary_interleaved=(past_seq_len % 2 == 1),
            device=device,
        )

        if do_rotary:
            # When there is rotary, we use ORT GQA as reference: ORT GQA does not support mask so here we use dense.
            config.local_blocks = config.max_blocks

        run_one_relevance_test(config)

        if sm >= 80:
            config.dtype = torch.bfloat16
            run_one_relevance_test(config)


def run_relevance_test(sm: int):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    with torch.no_grad():
        run_relevance_no_past(sm, device)
        run_relevance_past(sm, device, do_rotary=False)
        run_relevance_past(sm, device, do_rotary=True)


# ------------------------------------------------------------------
# Below are performance tests


def get_plot_algos(sm: int):
    # GQA with local windows only works in sm=8x
    if sm >= 80:
        return {
            "line_vals": ["torch_gqa", "ort_gqa", "ort_gqa_local", "ort_sparse_att"],
            "line_names": ["TORCH-GQA", "ORT-GQA-Dense", "ORT-GQA-Local", "ORT-SparseAtt"],
            "styles": [("red", "solid"), ("blue", "dashed"), ("yellow", "dashdot"), ("green", "dotted")],
        }
    else:
        return {
            "line_vals": ["torch_gqa", "ort_gqa", "ort_sparse_att"],
            "line_names": ["TORCH-GQA", "ORT-GQA-Dense", "ORT-SparseAtt"],
            "styles": [("red", "solid"), ("blue", "dashed"), ("green", "dashdot")],
        }

def get_plot_prompt_algos(sm: int):
    algos = get_plot_algos(sm)
    algos["line_vals"].append("ort_sparse_att_fallback")
    algos["line_names"].append("ORT-SparseAtt-Fallback")
    algos["styles"].append(("purple", ":"))
    return algos

def plot_prompt_performance(
    sm: int,
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

    algos = get_plot_prompt_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(4, 14)],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"prompt-sm{sm}-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config: SparseAttentionConfig = SparseAttentionConfig(
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
            do_rotary=True,
            dense_length_threshold=None if provider == "ort_sparse_att_fallback" else 0
            )

        if provider in ["ort_gqa", "ort_gqa_local"]:
            gqa_config = config.get_comparable_gqa_config(use_local=(provider == "ort_gqa_local"))
            obj = OrtGroupQueryAttention(gqa_config)
        elif provider in ["ort_sparse_att", "ort_sparse_att_fallback"]:
            obj = OrtSparseAttention(config)
        else:  # Torch GQA
            assert provider == "torch_gqa"
            if sequence_length > 2048:  # out of memory
                return 0
            gqa_config = config.get_comparable_gqa_config(torch_use_sparse=True)
            obj = TorchGroupQueryAttention(gqa_config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)

def plot_token_performance(
    sm: int,
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

    algos = get_plot_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["past_sequence_length"],
            x_vals=[2**i for i in range(4, 13)] + [max_seq_len - 1],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"token-sm{sm}-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, past_sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config: SparseAttentionConfig = SparseAttentionConfig(
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
            do_rotary=True,
        )

        if provider in ["ort_gqa", "ort_gqa_local"]:
            gqa_config = config.get_comparable_gqa_config(use_local=(provider == "ort_gqa_local"))
            obj = OrtGroupQueryAttention(gqa_config)
        elif provider == "ort_sparse_att":
            obj = OrtSparseAttention(config)
        else:
            assert provider == "torch_gqa"
            if past_sequence_length > 2048:  # out of memory
                return 0
            gqa_config = config.get_comparable_gqa_config(torch_use_sparse=True)
            obj = TorchGroupQueryAttention(gqa_config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_performance_test(sm: int):
    """
    Run performance tests for prompt and token generation.

    Example results in Azure Standard_ND96isr_H100_v5 VM with NVIDIA H100-80GB-HBM3 GPU (sm=90):

    prompt-sm90-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0             16.0   0.079877       0.006362       0.006403       0.042758
    1             32.0   0.086920       0.016404       0.016686       0.044183
    2             64.0   0.090727       0.020429       0.020409       0.045343
    3            128.0   0.128148       0.032009       0.031984       0.051516
    4            256.0   0.323933       0.074110       0.073920       0.068308
    5            512.0   1.021856       0.162167       0.161951       0.109226
    6           1024.0   3.596002       0.452629       0.452780       0.231653
    7           2048.0  13.865088       1.499534       1.195749       0.515488
    8           4096.0   0.000000       5.454785       2.669682       1.163233
    9           8192.0   0.000000      22.068159       6.018604       2.772873

    token-sm90-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0                  16.0   0.104460       0.012652       0.012661       0.069549
    1                  32.0   0.113866       0.012776       0.012765       0.069024
    2                  64.0   0.124600       0.016791       0.012672       0.069397
    3                 128.0   0.108658       0.017900       0.018294       0.074844
    4                 256.0   0.115463       0.029409       0.029608       0.078911
    5                 512.0   0.149824       0.033968       0.033701       0.092998
    6                1024.0   0.234050       0.042930       0.042951       0.116920
    7                2048.0   0.390695       0.061462       0.043008       0.121555
    8                4096.0   0.000000       0.097505       0.042948       0.134757
    9                8191.0   0.000000       0.165861       0.043542       0.158796


    Example results in A100-SXM4-80GB (sm=80):

    prompt-sm80-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0             16.0   0.274839       0.008849       0.015198       0.054403
    1             32.0   0.272238       0.022875       0.048804       0.055898
    2             64.0   0.272420       0.027722       0.028318       0.073052
    3            128.0   0.273514       0.085971       0.062785       0.068287
    4            256.0   0.545428       0.108228       0.135093       0.095949
    5            512.0   1.678597       0.278193       0.248580       0.167271
    6           1024.0   6.021056       0.702882       0.701022       0.379936
    7           2048.0  23.512320       2.331175       1.863045       0.895726
    8           4096.0   0.000000       8.789178       4.526275       2.105048
    9           8192.0   0.000000      39.664131      10.046236       5.219436

    token-sm80-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0                  16.0   0.299303       0.020081       0.018587       0.082479
    1                  32.0   0.301700       0.018655       0.041943       0.084583
    2                  64.0   0.305700       0.017825       0.018420       0.085265
    3                 128.0   0.303379       0.023213       0.023152       0.090508
    4                 256.0   0.304119       0.034438       0.035257       0.100197
    5                 512.0   0.306051       0.063312       0.045373       0.114726
    6                1024.0   0.359197       0.092181       0.088628       0.145165
    7                2048.0   0.599463       0.101573       0.062101       0.159452
    8                4096.0   0.000000       0.196258       0.091019       0.180342
    9                8191.0   0.000000       0.334519       0.065158       0.213508


    Example results in Standard_NC4as_T4_v3 Azure VM with T4 GPU (sm=75):

    prompt-sm75-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length   TORCH-GQA  ORT-GQA-Dense  ORT-SparseAtt
    0             16.0    0.165154       3.003173       0.081945
    1             32.0    0.184173       2.994347       0.089064
    2             64.0    0.303300       3.023986       0.107418
    3            128.0    0.887795       3.073728       0.174213
    4            256.0    2.797654       3.246899       0.357869
    5            512.0   10.055048       3.814039       0.893903
    6           1024.0   37.849937       5.818439       2.658720
    7           2048.0  148.641785      13.638480       7.202690
    8           4096.0    0.000000      43.556847      17.680954
    9           8192.0    0.000000     161.628540      44.336670

    token-sm75-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-SparseAtt
    0                  16.0   0.144368       4.179228       0.137407
    1                  32.0   0.110353       2.996305       0.137509
    2                  64.0   0.145088       3.006860       0.165424
    3                 128.0   0.219500       3.036448       0.192001
    4                 256.0   0.347496       3.071341       0.249125
    5                 512.0   0.595842       3.135225       0.398726
    6                1024.0   1.081216       3.261110       0.612744
    7                2048.0   2.060307       3.515578       0.685670
    8                4096.0   0.000000       4.022986       0.819707
    9                8191.0   0.000000       5.024528       1.072912


    """
    with torch.no_grad():
        plot_prompt_performance(sm=sm)
        plot_token_performance(sm=sm)


if __name__ == "__main__":
    torch.set_printoptions(precision=6, edgeitems=3, linewidth=150, profile="default", sci_mode=False)

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_relevance_test(sm)
        run_performance_test(sm)

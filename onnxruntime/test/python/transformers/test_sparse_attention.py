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
import unittest
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
        shapes = {
            "query": (
                self.batch_size,
                self.sequence_length,
                (self.num_heads + 2 * self.kv_num_heads) * self.head_size,
            ),
            "past_key": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "past_value": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "total_sequence_length": (1,),
            "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "present_key": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "present_value": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "cos_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
            "sin_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
        }

        if not self.is_packed_qkv:
            shapes.update(
                {
                    "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                    "key": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
                    "value": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
                }
            )
        return shapes

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

        # Always use non-packed qkv to generate same inputs for Torch and ORT.
        packed = self.is_packed_qkv  # Save the original value.
        self.is_packed_qkv = False
        shape_dict = self.shape_dict()
        self.is_packed_qkv = packed  # Restore the original value.
        torch.manual_seed(123)

        feeds = {
            "query": torch.empty(shape_dict["query"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "key": torch.empty(shape_dict["key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "value": torch.empty(shape_dict["value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "total_sequence_length": torch.tensor([self.total_sequence_length], dtype=torch.int32),
        }

        if packed:
            query = feeds["query"].view(self.batch_size, self.sequence_length, self.num_heads, self.head_size)
            key = feeds["key"].view(self.batch_size, self.sequence_length, self.kv_num_heads, self.head_size)
            value = feeds["value"].view(self.batch_size, self.sequence_length, self.kv_num_heads, self.head_size)
            feeds["query"] = torch.dstack((query, key, value)).reshape(self.batch_size, self.sequence_length, -1)
            del feeds["key"]
            del feeds["value"]

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
        is_packed_qkv=False,
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
            is_packed_qkv=is_packed_qkv,
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
        is_packed_qkv=False,
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
            is_packed_qkv=is_packed_qkv,
        )
        self.sparse_block_size = sparse_block_size
        self.num_layout = num_layout
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.max_blocks = max_sequence_length // sparse_block_size

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

    def get_comparable_ort_gqa_config(self, use_local=False) -> GroupQueryAttentionConfig:
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
            is_packed_qkv=self.is_packed_qkv,
        )

    def get_comparable_torch_gqa_config(self, use_sparse=False) -> GroupQueryAttentionConfig:
        attention_mask = None

        if use_sparse is True:
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
            attention_mask=attention_mask,
            is_packed_qkv=False,  # torch reference implementation does not support packed qkv.
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
            do_rotary=1 if config.do_rotary else 0,
            domain="com.microsoft",
        ),
    ]

    # When testing bfloat16, we add cast nodes so that SparseAttention is computed in bfloat16.
    if config.dtype == torch.bfloat16:
        nodes.extend(
            [
                helper.make_node("Cast", [input], [input + suffix], f"Cast_{input}", to=TensorProto.BFLOAT16)
                for input in (
                    ["query", "key", "value", "past_key", "past_value"]
                    if not config.is_packed_qkv
                    else ["query", "past_key", "past_value"]
                )
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
    ]

    if not config.is_packed_qkv:
        graph_input.extend(
            [
                helper.make_tensor_value_info("key", io_float_type, list(shape_dict["key"])),
                helper.make_tensor_value_info("value", io_float_type, list(shape_dict["value"])),
            ]
        )

    graph_input.extend(
        [
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
    )

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
    ]

    if not config.is_packed_qkv:
        graph_input.extend(
            [
                helper.make_tensor_value_info("key", float_type, list(shape_dict["key"])),
                helper.make_tensor_value_info("value", float_type, list(shape_dict["value"])),
            ]
        )

    graph_input.extend(
        [
            helper.make_tensor_value_info("past_key", float_type, list(shape_dict["past_key"])),
            helper.make_tensor_value_info("past_value", float_type, list(shape_dict["past_value"])),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, list(shape_dict["seqlens_k"])),
            helper.make_tensor_value_info(
                "total_sequence_length", TensorProto.INT32, list(shape_dict["total_sequence_length"])
            ),
        ]
    )

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
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_group_query_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, cuda_provider_options=cuda_provider_options)
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=self.ort_session,
            device=config.device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        self.gpu_binding = self.gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        self.feed_dict = config.random_inputs()

        if ENABLE_DEBUG and not config.is_packed_qkv:
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
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_sparse_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, cuda_provider_options=cuda_provider_options)
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=self.ort_session,
            device=config.device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        self.gpu_binding = self.gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        self.feed_dict = config.random_inputs()

        if ENABLE_DEBUG and not config.is_packed_qkv:
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


class TestSparseAttention(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_sparse_attention(self):
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor

        if sm not in [75, 80, 86, 89, 90]:
            self.skipTest("SparseAttention is not supported on this GPU")

        self.run_relevance_test(sm)

    def run_one_relevance_test(self, config: SparseAttentionConfig):
        if not config.do_rotary:
            # Run QGA by Torch
            gqa_config: GroupQueryAttentionConfig = config.get_comparable_torch_gqa_config(use_sparse=True)
            obj = TorchGroupQueryAttention(gqa_config)
            expected_out = obj.infer()
        else:
            # Run QGA by ORT
            gqa_config: GroupQueryAttentionConfig = config.get_comparable_ort_gqa_config(use_local=False)
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

        passed = torch.allclose(expected_out, actual_out, atol=1e-2, rtol=0)
        if passed:
            print(f"Relevance test {green_color}passed{reset_color}: {vars(config)}")
        else:
            print(f"Relevance test {red_color}failed{reset_color}: {vars(config)}")
            print("ort_output", actual_out)
            print("expected_out", expected_out)
            print("diff", expected_out - actual_out)

        self.assertTrue(passed)

    def run_relevance_no_past(self, sm: int, device):
        """Test prompt prefilling without past kv cache."""
        for seq_len in [1, 64, 127, 128, 192, 256]:
            for packed_qkv in [False, True]:
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
                    is_packed_qkv=packed_qkv,
                )
                self.run_one_relevance_test(config)

                if sm >= 80 and not packed_qkv:
                    config.dtype = torch.bfloat16
                    self.run_one_relevance_test(config)

    def run_relevance_past(self, sm: int, device, do_rotary: bool):
        """Test token generation with past kv cache."""
        for past_seq_len in [1, 63, 64, 127, 128, 511]:
            for packed_qkv in [False, True]:
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
                    is_packed_qkv=packed_qkv,
                )

                if do_rotary:
                    # When there is rotary, we use ORT GQA as reference: ORT GQA does not support mask so here we use dense.
                    config.local_blocks = config.max_blocks

                self.run_one_relevance_test(config)

                if sm >= 80 and not packed_qkv:
                    config.dtype = torch.bfloat16
                    self.run_one_relevance_test(config)

    def run_relevance_test(self, sm: int):
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        with torch.no_grad():
            self.run_relevance_no_past(sm, device)
            self.run_relevance_past(sm, device, do_rotary=False)
            self.run_relevance_past(sm, device, do_rotary=True)


if __name__ == "__main__":
    unittest.main()

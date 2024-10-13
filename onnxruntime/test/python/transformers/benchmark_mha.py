# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of MultiHeadAttention with ORT or PyTorch.

In Linux, run the the following:
   sh benchmark_mha.sh

In Windows, run the the following:
   benchmark_mha.cmd
"""

import argparse
import csv
import math
import os
import platform
import re
import statistics
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.utils.benchmark as benchmark
from onnx import TensorProto, helper
from packaging.version import Version
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from onnxruntime import InferenceSession, SessionOptions, get_available_providers
from onnxruntime.transformers.io_binding_helper import CudaSession


class InputFormats:
    Q_K_V_BSNH_BSNH_BSNH = 0
    QKV_BSN3H = 1
    Q_KV_BSNH_BSN2H = 2
    Q_K_V_BSNH_BNSH_BNSH = 3  # For cross attention

    @staticmethod
    def input_format_str(format: int) -> str:
        names = InputFormats.get_name_list()
        return names[format]

    @staticmethod
    def convert(format_str: str) -> int:
        names = InputFormats.get_name_list()
        return names.index(format_str)

    @staticmethod
    def get_name_list() -> List[str]:
        return ["Q,K,V", "QKV", "Q,KV", "Q,K',V'"]


class SdpaKernel(IntEnum):
    """Bit flags for sdpa_kernel CUDA provider option"""

    DEFAULT = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    TRT_FUSED_ATTENTION = 4
    CUDNN_FLASH_ATTENTION = 8
    MATH = 16
    TRT_FLASH_ATTENTION = 32
    TRT_CROSS_ATTENTION = 64
    TRT_CAUSAL_ATTENTION = 128
    LEAN_ATTENTION = 256


# Since we support attention bias, so we only need support up to 2D mask.
class AttentionMaskFormat(IntEnum):
    Mask_None = 0  # No attention mask.
    Mask_1D_Key_SeqLen = 1  # Shape (batch_size), actual sequence lengths (excluding padding on the right side).
    Mask_2D_Key_PaddingMask = 2  # Shape (batch_size, total_sequence_length), key padding mask mask.


class MultiHeadAttentionConfig:
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        num_heads: int,
        head_size: int,
        causal: bool,
        past_sequence_length: int = 0,
        kv_sequence_length=None,
        max_cache_sequence_length=None,
        scale: float = 0.0,
        provider="CPUExecutionProvider",
        device: Optional[torch.device] = None,
        enable_cuda_graph: bool = False,
        dtype=torch.float,
        use_kv_cache: bool = False,
        has_past_input: bool = False,
        share_past_present_buffer: bool = False,
        input_format: int = InputFormats.Q_K_V_BSNH_BSNH_BSNH,
        verbose: bool = False,
        has_bias: bool = False,  # bias for input projection
        has_attn_bias: bool = False,  # bias added before softmax. For example,relative position bias.
        broadcast_attn_bias_dim_0: bool = False,  # broadcast attention bias dimension 0
        broadcast_attn_bias_dim_1: bool = False,  # broadcast attention bias dimension 1
        mask_format: int = AttentionMaskFormat.Mask_None,
    ):
        self.operator = "MultiHeadAttention"
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.kv_sequence_length = kv_sequence_length or sequence_length
        self.max_cache_sequence_length = max_cache_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.scale = scale or (1.0 / (head_size**0.5))

        # Support the case that there is no past but need present output (for prompt case).
        self.has_past_input = has_past_input
        if has_past_input:
            assert use_kv_cache
        else:  # no past input
            assert past_sequence_length == 0

        self.has_present_output = use_kv_cache

        self.use_kv_cache = use_kv_cache
        if not use_kv_cache:
            assert past_sequence_length == 0
        else:
            assert self.kv_sequence_length == self.sequence_length

        # Only BSNH input format supports past state.
        if input_format != InputFormats.Q_K_V_BSNH_BSNH_BSNH:
            assert not self.has_past_input
            assert not self.has_present_output

        # Derived values
        self.total_sequence_length = self.kv_sequence_length + past_sequence_length
        self.past_buffer_length = self.max_cache_sequence_length if share_past_present_buffer else past_sequence_length
        self.present_buffer_length = (
            self.max_cache_sequence_length if share_past_present_buffer else self.total_sequence_length
        )

        self.provider = provider
        self.device = device
        self.enable_cuda_graph = enable_cuda_graph
        self.dtype = dtype

        self.share_past_present_buffer = share_past_present_buffer
        self.input_format = input_format
        self.is_packed_qkv = input_format == InputFormats.QKV_BSN3H
        self.is_packed_kv = input_format == InputFormats.Q_KV_BSNH_BSN2H
        self.verbose = verbose
        self.has_bias = has_bias
        self.has_attn_bias = has_attn_bias
        self.broadcast_attn_bias_dim_0 = broadcast_attn_bias_dim_0
        self.broadcast_attn_bias_dim_1 = broadcast_attn_bias_dim_1

        assert mask_format in [
            AttentionMaskFormat.Mask_None,
            AttentionMaskFormat.Mask_1D_Key_SeqLen,
            AttentionMaskFormat.Mask_2D_Key_PaddingMask,
        ]
        self.mask_format = mask_format

        # mask_index_q and mask_index_kv will be updated in random_inputs() if mask_format is not Mask_None.
        self.mask_index_kv = torch.ones(self.batch_size, dtype=torch.int32, device=self.device) * self.sequence_length
        self.mask_index_q = (
            torch.ones(self.batch_size, dtype=torch.int32, device=self.device) * self.total_sequence_length
        )

        assert mask_format in [
            AttentionMaskFormat.Mask_None,
            AttentionMaskFormat.Mask_1D_Key_SeqLen,
            AttentionMaskFormat.Mask_2D_Key_PaddingMask,
        ]
        self.mask_format = mask_format

        # mask_index_q and mask_index_kv will be updated in random_inputs() if mask_format is not Mask_None.
        self.mask_index_kv = torch.ones(self.batch_size, dtype=torch.int32, device=self.device) * self.sequence_length
        self.mask_index_q = (
            torch.ones(self.batch_size, dtype=torch.int32, device=self.device) * self.total_sequence_length
        )

    def __repr__(self):
        return (
            f"MultiHeadAttentionConfig(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"num_heads={self.num_heads}, head_size={self.head_size}, "
            f"kv_sequence_length={self.kv_sequence_length}, past_sequence_length={self.past_sequence_length}, "
            f"max_cache_sequence_length={self.max_cache_sequence_length},"
            f"causal={self.causal}), scale={self.scale}, use_kv_cache={self.use_kv_cache}, "
            f"share_past_present_buffer={self.share_past_present_buffer}, "
            f"provider={self.provider}, device={self.device}, enable_cuda_graph={self.enable_cuda_graph}, "
            f"dtype={self.dtype}, input_format={InputFormats.input_format_str(self.input_format)}, "
            f"has_bias={self.has_bias}, mask_format={self.mask_format}, "
            f"has_attn_bias={self.has_attn_bias}, "
            f"broadcast_attn_bias_dim_0={self.broadcast_attn_bias_dim_0}, "
            f"broadcast_attn_bias_dim_1={self.broadcast_attn_bias_dim_1}, "
        )

    def shape_dict(self, input_format=None):
        shapes: Dict[str, Tuple] = {
            "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
        }

        input_format = input_format or self.input_format
        if input_format == InputFormats.QKV_BSN3H:
            shapes = {
                **shapes,
                "query": (self.batch_size, self.sequence_length, self.num_heads, 3, self.head_size),
            }
        elif input_format == InputFormats.Q_KV_BSNH_BSN2H:
            shapes = {
                **shapes,
                "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "key": (self.batch_size, self.sequence_length, self.num_heads, 2, self.head_size),
            }
        elif input_format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
            shapes = {
                **shapes,
                "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "key": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "value": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            }
        else:
            assert input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH
            shapes = {
                **shapes,
                "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "key": (self.batch_size, self.num_heads, self.sequence_length, self.head_size),
                "value": (self.batch_size, self.num_heads, self.sequence_length, self.head_size),
            }

        if self.has_past_input:
            shapes = {
                **shapes,
                "past_key": (self.batch_size, self.num_heads, self.past_buffer_length, self.head_size),
                "past_value": (self.batch_size, self.num_heads, self.past_buffer_length, self.head_size),
            }

        if self.has_present_output:
            shapes = {
                **shapes,
                "present_key": (self.batch_size, self.num_heads, self.present_buffer_length, self.head_size),
                "present_value": (self.batch_size, self.num_heads, self.present_buffer_length, self.head_size),
            }

        if self.has_bias:
            shapes["bias"] = (3 * self.num_heads * self.head_size,)

        if self.mask_format == AttentionMaskFormat.Mask_1D_Key_SeqLen:
            shapes["mask"] = (self.batch_size,)
        elif self.mask_format == AttentionMaskFormat.Mask_2D_Key_PaddingMask:
            shapes["mask"] = (self.batch_size, self.total_sequence_length)
        else:
            assert self.mask_format == AttentionMaskFormat.Mask_None

        if self.has_attn_bias:
            shapes["attn_bias"] = (
                1 if self.broadcast_attn_bias_dim_0 else self.batch_size,
                1 if self.broadcast_attn_bias_dim_1 else self.num_heads,
                self.sequence_length,
                self.total_sequence_length,
            )

        return shapes

    def symbolic_shape_dict(self, input_format=None):
        shapes: Dict[str, Tuple] = {
            "output": ("batch_size", "sequence_length", self.num_heads * self.head_size),
        }

        input_format = input_format or self.input_format
        if input_format == InputFormats.QKV_BSN3H:
            shapes = {
                **shapes,
                "query": ("batch_size", "sequence_length", self.num_heads, 3, self.head_size),
            }
        elif input_format == InputFormats.Q_KV_BSNH_BSN2H:
            shapes = {
                **shapes,
                "query": ("batch_size", "sequence_length", self.num_heads * self.head_size),
                "key": ("batch_size", "sequence_length", self.num_heads, 2, self.head_size),
            }
        elif input_format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
            shapes = {
                **shapes,
                "query": ("batch_size", "sequence_length", self.num_heads * self.head_size),
                "key": ("batch_size", "sequence_length", self.num_heads * self.head_size),
                "value": ("batch_size", "sequence_length", self.num_heads * self.head_size),
            }
        else:
            assert input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH
            shapes = {
                **shapes,
                "query": ("batch_size", "sequence_length", self.num_heads * self.head_size),
                "key": ("batch_size", self.num_heads, "sequence_length", self.head_size),
                "value": ("batch_size", self.num_heads, "sequence_length", self.head_size),
            }

        if self.has_past_input:
            shapes = {
                **shapes,
                "past_key": ("batch_size", self.num_heads, "past_buffer_length", self.head_size),
                "past_value": ("batch_size", self.num_heads, "past_buffer_length", self.head_size),
            }

        if self.has_present_output:
            shapes = {
                **shapes,
                "present_key": ("batch_size", self.num_heads, "present_buffer_length", self.head_size),
                "present_value": ("batch_size", self.num_heads, "present_buffer_length", self.head_size),
            }

        if self.has_bias:
            shapes["bias"] = (3 * self.num_heads * self.head_size,)

        if self.mask_format == AttentionMaskFormat.Mask_1D_Key_SeqLen:
            shapes["mask"] = ("batch_size",)
        elif self.mask_format == AttentionMaskFormat.Mask_2D_Key_PaddingMask:
            shapes["mask"] = ("batch_size", "total_sequence_length")
        else:
            assert self.mask_format == AttentionMaskFormat.Mask_None

        if self.has_attn_bias:
            shapes["attn_bias"] = ("batch_size_or_1", "num_heads_or_1", "sequence_length", "total_sequence_length")

        return shapes

    def right_side_padding_masks(self):
        q_mask = torch.ones(self.batch_size, 1, self.sequence_length, 1, dtype=torch.bool, device=self.device)
        k_mask = torch.ones(self.batch_size, 1, self.total_sequence_length, 1, dtype=torch.bool, device=self.device)
        mask = torch.ones(
            self.batch_size,
            self.num_heads,
            self.sequence_length,
            self.total_sequence_length,
            dtype=torch.bool,
            device=self.device,
        )

        if self.mask_format != AttentionMaskFormat.Mask_None:
            for i, (m, n) in enumerate(zip(self.mask_index_q, self.mask_index_kv)):
                q_mask[i, :, m:, :] = False
                k_mask[i, :, n:, :] = False
                mask[i, :, m:, :] = False
                mask[i, :, :, n:] = False
        return q_mask, k_mask, mask

    def random_inputs(self, seed: int = 123, no_bias_k_v: bool = False):
        device = self.device
        dtype = self.dtype

        shape_dict = self.shape_dict()

        if seed > 0:
            torch.manual_seed(seed)

        shape = (self.batch_size, self.sequence_length, self.num_heads, self.head_size)
        q = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)
        k = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)
        v = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)

        bias_q = torch.empty((self.num_heads * self.head_size,), device=device, dtype=dtype).normal_(mean=0, std=0.1)
        bias_k = torch.empty((self.num_heads * self.head_size,), device=device, dtype=dtype).normal_(mean=0, std=0.1)
        bias_v = torch.empty((self.num_heads * self.head_size,), device=device, dtype=dtype).normal_(mean=0, std=0.1)
        if no_bias_k_v:
            bias_k = torch.zeros_like(bias_k)
            bias_v = torch.zeros_like(bias_v)

        k_bnsh = k.transpose(1, 2)
        v_bnsh = v.transpose(1, 2)

        if self.input_format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
            feeds = {
                "query": q.reshape(shape_dict["query"]),
                "key": k.reshape(shape_dict["key"]),
                "value": v.reshape(shape_dict["value"]),
            }
        elif self.input_format == InputFormats.QKV_BSN3H:
            query = q.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            key = k.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            value = v.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            feeds = {
                "query": torch.dstack((query, key, value)).reshape(shape_dict["query"]).contiguous(),
            }
        elif self.input_format == InputFormats.Q_KV_BSNH_BSN2H:
            key = k.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            value = v.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            feeds = {
                "query": q.reshape(shape_dict["query"]),
                "key": torch.dstack((key, value)).reshape(shape_dict["key"]).contiguous(),
            }
        else:
            assert self.input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH
            feeds = {
                "query": q.reshape(shape_dict["query"]),
                "key": k_bnsh.contiguous(),
                "value": v_bnsh.contiguous(),
            }

        if self.has_past_input:
            feeds = {
                **feeds,
                "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
                "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(
                    mean=0, std=0.1
                ),
            }

        if self.has_bias:
            feeds["bias"] = torch.concat([bias_q, bias_k, bias_v], dim=0).reshape(shape_dict["bias"]).contiguous()

        # Generate padding mask
        if self.mask_format != AttentionMaskFormat.Mask_None:
            self.mask_index_kv = torch.randint(
                1, self.total_sequence_length + 1, (self.batch_size,), dtype=torch.int32, device=self.device
            )
            if self.past_sequence_length > 0:
                self.mask_index_q = (
                    torch.ones(self.batch_size, dtype=torch.int32, device=self.device) * self.sequence_length
                )
            else:  # prompt case
                self.mask_index_q = self.mask_index_kv.clone()

        mask = None
        if self.mask_format == AttentionMaskFormat.Mask_1D_Key_SeqLen:
            mask = self.mask_index_kv.clone()
        elif self.mask_format == AttentionMaskFormat.Mask_2D_Key_PaddingMask:
            k_mask = torch.ones(self.batch_size, 1, self.total_sequence_length, 1, dtype=torch.bool, device=self.device)
            for i, n in enumerate(self.mask_index_kv):
                k_mask[i, :, n:, :] = False
            mask = k_mask.reshape(self.batch_size, self.total_sequence_length)
        else:
            assert self.mask_format == AttentionMaskFormat.Mask_None

        if mask is not None:
            feeds = {**feeds, "mask": mask.to(dtype=torch.int32)}  # mask is int32 (not bool) for MultiHeadAttention op.

        if self.has_attn_bias:
            attn_bias = torch.empty(
                (
                    1 if self.broadcast_attn_bias_dim_0 else self.batch_size,
                    1 if self.broadcast_attn_bias_dim_1 else self.num_heads,
                    self.sequence_length,
                    self.total_sequence_length,
                ),
                device=self.device,
                dtype=dtype,
            ).normal_(mean=0, std=0.1)
            feeds["attn_bias"] = attn_bias

        return feeds

    def get_input_output_names(self):
        if self.input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
            inputs, outputs = ["query", "key", "value"], ["output"]
        elif self.input_format == InputFormats.QKV_BSN3H:
            inputs, outputs = ["query"], ["output"]
        elif self.input_format == InputFormats.Q_KV_BSNH_BSN2H:
            inputs, outputs = ["query", "key"], ["output"]
        else:
            inputs, outputs = ["query", "key", "value"], ["output"]

        if self.has_bias:
            assert self.input_format != InputFormats.Q_KV_BSNH_BSN2H
            inputs = [*inputs, "bias"]

        if self.mask_format != AttentionMaskFormat.Mask_None:
            inputs = [*inputs, "mask"]

        if self.has_attn_bias:
            inputs = [*inputs, "attn_bias"]

        if self.has_past_input:
            inputs = [*inputs, "past_key", "past_value"]

        if self.has_present_output:
            outputs = [*outputs, "present_key", "present_value"]

        return inputs, outputs


def fill_optional_mha_inputs(input_names):
    inputs = ["query", "key", "value", "bias", "mask", "attn_bias", "past_key", "past_value"]

    # Remove optional inputs that are not in input_names with empty string
    inputs_with_optional = [input if input in input_names else "" for input in inputs]

    # Remove empty string at the end of the list.
    while inputs_with_optional[-1] == "":
        inputs_with_optional.pop(-1)

    return inputs_with_optional


def create_multi_head_attention_onnx_model(config: MultiHeadAttentionConfig, use_symbolic_shape=False):
    input_names, output_names = config.get_input_output_names()

    float_type = TensorProto.FLOAT16 if config.dtype == torch.float16 else TensorProto.FLOAT
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            fill_optional_mha_inputs(input_names),
            output_names,
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            unidirectional=int(config.causal),
            scale=config.scale,
            mask_filter_value=float("-inf"),
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.symbolic_shape_dict() if use_symbolic_shape else config.shape_dict()
    inputs = [
        helper.make_tensor_value_info(
            input_name, TensorProto.INT32 if input_name == "mask" else float_type, list(shape_dict[input_name])
        )
        for input_name in input_names
        if input_name
    ]

    outputs = [
        helper.make_tensor_value_info(output_name, float_type, list(shape_dict[output_name]))
        for output_name in output_names
        if output_name
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        inputs,
        outputs,
    )

    model = helper.make_model(graph)

    return model.SerializeToString()


def create_ort_session(
    config: MultiHeadAttentionConfig,
    session_options=None,
    attention_kernel=SdpaKernel.DEFAULT,
    use_symbolic_shape: bool = True,
    use_tf32: bool = True,
) -> CudaSession:
    if config.verbose:
        print(f"create session for {vars(config)}")
    onnx_model_str = create_multi_head_attention_onnx_model(config, use_symbolic_shape=use_symbolic_shape)

    if config.provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device() if isinstance(config.device, str) else config.device.index
        provider_options = CudaSession.get_cuda_provider_options(device_id, config.enable_cuda_graph)
        provider_options["sdpa_kernel"] = int(attention_kernel)
        provider_options["use_tf32"] = int(use_tf32)
        providers = [(config.provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    ort_session = InferenceSession(onnx_model_str, session_options, providers=providers)
    return ort_session


def create_session(
    config: MultiHeadAttentionConfig, session_options=None, attention_kernel=SdpaKernel.DEFAULT, use_tf32: bool = True
) -> CudaSession:
    ort_session = create_ort_session(
        config, session_options, attention_kernel, use_symbolic_shape=False, use_tf32=use_tf32
    )
    cuda_session = CudaSession(ort_session, config.device, config.enable_cuda_graph)
    shape_dict = config.shape_dict()
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


class OrtMultiHeadAttention:
    """A wrapper of ORT MultiHeadAttention to test relevance and performance."""

    def __init__(self, config: MultiHeadAttentionConfig, session_options=None, use_tf32: bool = True):
        self.ort_session = create_session(config, session_options, use_tf32=use_tf32)
        self.feed_dict = config.random_inputs()

    def infer(self, run_options=None, synchronize=True):
        return self.ort_session.infer(self.feed_dict, run_options=run_options, synchronize=synchronize)


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


def flops(batch, sequence_length_q, sequence_length_kv, head_size, num_heads, causal):
    return 4 * batch * sequence_length_q * sequence_length_kv * num_heads * head_size // (2 if causal else 1)


def tflops_per_second(flop, time):
    try:
        return (flop / time / 10**12) if not math.isnan(time) else 0.0
    except ZeroDivisionError:
        return None


def get_gpu_kernel_name(attention_kernel: SdpaKernel) -> str:
    kernel_names = {
        SdpaKernel.DEFAULT: "ort:default",
        SdpaKernel.FLASH_ATTENTION: "ort:flash",
        SdpaKernel.LEAN_ATTENTION: "ort:lean",
        SdpaKernel.EFFICIENT_ATTENTION: "ort:efficient",
        SdpaKernel.CUDNN_FLASH_ATTENTION: "ort:cudnn",
        SdpaKernel.MATH: "ort:math",
    }
    assert attention_kernel in kernel_names
    return kernel_names[attention_kernel]


def get_cpu_kernel_name(config: MultiHeadAttentionConfig) -> str:
    # CPU Flash Attention does not support causal and kv cache etc.
    if not (config.causal or config.use_kv_cache or config.past_sequence_length > 0):
        if os.getenv("ORT_DISABLE_FLASH_ATTENTION") != "1":
            return "ort:flash"

    return "ort:math"


# ------------------------------------------------------------------
# Functions for benchmarking PyTorch SDPA
# ------------------------------------------------------------------
def benchmark_torch_function(repeats: int, func: Callable, *args, **kwargs) -> float:
    warmup = 5
    for _ in range(warmup):
        func(*args, **kwargs)

    timer = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )

    return timer.timeit(number=repeats).median


def run_torch_sdpa(
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    head_size: int,
    causal: bool,
    device,
    dtype,
    has_mask: bool = False,
    mask_dim: int = 2,
    mask_dtype=torch.bool,
    backend: Optional[int] = None,
    repeats: int = 100,
):
    q_shape = (batch_size, num_heads, q_seq_len, head_size)
    kv_shape = (batch_size, num_heads, kv_seq_len, head_size)
    q = torch.randn(q_shape, device=device, dtype=dtype)
    k = torch.randn(kv_shape, device=device, dtype=dtype)
    v = torch.randn(kv_shape, device=device, dtype=dtype)

    attn_mask = None
    if has_mask:
        mask_shape = (batch_size, num_heads, q_seq_len, kv_seq_len) if mask_dim == 4 else (q_seq_len, kv_seq_len)
        attn_mask = torch.ones(mask_shape, dtype=mask_dtype, device=device)

    context = sdpa_kernel(backend) if backend is not None else nullcontext()

    with context:
        average_latency = benchmark_torch_function(
            repeats,
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=causal,
            attn_mask=attn_mask,
        )
    return average_latency


def get_test_configs(args: argparse.Namespace):
    use_gpu: bool = args.use_gpu

    if args.batch_size > 0:
        run_unfused = args.sequence_length + args.past_sequence_length <= (2048 if use_gpu else 1024)
        return [
            (
                args.batch_size,
                args.sequence_length,
                args.past_sequence_length,
                args.num_heads,
                args.head_size,
                run_unfused,
            ),
        ]

    if use_gpu:
        # (batch_size, sequence_length, past_sequence_length, num_heads, head_size, run_unfused)
        configs = [
            (32, 512, 0, 64, 32, True),
            (32, 512, 0, 128, 16, True),
            (16, 1024, 0, 64, 32, True),
            (16, 1024, 0, 128, 16, True),
            (8, 2048, 0, 64, 32, True),
            (8, 2048, 0, 128, 16, False),
            (4, 4096, 0, 64, 32, False),
            (4, 4096, 0, 128, 16, False),
            (2, 8192, 0, 64, 32, False),
            (2, 8192, 0, 128, 16, False),
            (1, 16384, 0, 64, 32, False),
            (1, 16384, 0, 128, 16, False),
            # stable diffusion
            (1, 4096, 0, 8, 40, False),
            (1, 4096, 0, 8, 80, False),
            (1, 4096, 0, 8, 160, False),
            (4, 4096, 0, 8, 40, False),
            (4, 4096, 0, 8, 80, False),
            (4, 4096, 0, 8, 160, False),
            (1, 16384, 0, 8, 40, False),
            (1, 16384, 0, 8, 80, False),
            (1, 16384, 0, 8, 160, False),
            # bert-base
            (128, 128, 0, 12, 64, True),
            (64, 128, 0, 12, 64, True),
            (128, 384, 0, 12, 64, True),
            (64, 384, 0, 12, 64, True),
            (128, 512, 0, 12, 64, True),
            (64, 512, 0, 12, 64, True),
            # TNLGv4
            (4, 2048, 0, 32, 128, True),
            (4, 4096, 0, 32, 128, False),
            (8, 2048, 0, 32, 128, False),
            (8, 4096, 0, 32, 128, False),
        ]
    else:
        configs = [
            # TNLGv4
            (1, 128, 0, 32, 128, True),
            (1, 256, 0, 32, 128, True),
            (1, 512, 0, 32, 128, True),
            (1, 1024, 0, 32, 128, True),
            # (1, 2048, 0, 32, 128, True),
            # bert-base
            (1, 128, 0, 12, 64, True),
            (1, 384, 0, 12, 64, True),
            (1, 512, 0, 12, 64, True),
            (4, 128, 0, 12, 64, True),
            (4, 384, 0, 12, 64, True),
            (4, 512, 0, 12, 64, True),
            # bert-large
            (1, 128, 0, 16, 64, True),
            (1, 384, 0, 16, 64, True),
            (1, 512, 0, 16, 64, True),
            (4, 128, 0, 16, 64, True),
            (4, 384, 0, 16, 64, True),
            (4, 512, 0, 16, 64, True),
        ]
    return configs


def get_compute_capability():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return sm


class CaptureStdout:
    def __init__(self):
        self.fd = sys.stdout.fileno()
        self.chunk_size = 1024
        self.output = b""

    def _capture(self):
        chunks = []
        while chunk := os.read(self._pipe_reader, self.chunk_size):
            chunks.append(chunk)
        self.output = b"".join(chunks)

    def __enter__(self):
        self._duped_fd = os.dup(self.fd)
        self._pipe_reader, pipe_writer = os.pipe()
        os.dup2(pipe_writer, self.fd)
        os.close(pipe_writer)
        self._capture_thread = threading.Thread(target=self._capture)
        self._capture_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(self.fd)
        self._capture_thread.join()
        os.close(self._pipe_reader)
        os.dup2(self._duped_fd, self.fd)
        os.close(self._duped_fd)


def sdpa_kernel_from_debug_info(
    config: MultiHeadAttentionConfig, attention_kernel: SdpaKernel, sess_options: SessionOptions
):
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    captured_text = None

    try:
        with CaptureStdout() as captured:
            session = create_session(config, sess_options, attention_kernel=attention_kernel)
            input_dict = config.random_inputs()
            session.infer(input_dict)
        captured_text = captured.output.decode()
    except Exception as e:
        print(f"Failed to run {attention_kernel=} for {config=}. Exception: {e}")

    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "0"

    if captured_text is not None:
        m = re.search("SdpaKernel=(?P<kernel>[A-Z_]+)", captured_text)
        if m is not None:
            name = m.group("kernel")
            kernel_names = {
                "FLASH_ATTENTION": "ort:flash",
                "LEAN_ATTENTION": "ort:lean",
                "EFFICIENT_ATTENTION": "ort:efficient",
                "CUDNN_FLASH_ATTENTION": "ort:cudnn",
                "MATH": "ort:math",
                "TRT_FUSED_ATTENTION": "ort:trt_fmha",
                "TRT_FLASH_ATTENTION": "ort:trt_flash",
                "TRT_CROSS_ATTENTION": "ort:trt_cross",
                "TRT_CAUSAL_ATTENTION": "ort:trt_causal",
            }
            return kernel_names[name]
        else:
            print("Failed to get sdpa kernel from debug info:", captured_text)

    return None


def run_tflops_test(
    csv_writer: csv.DictWriter,
    args: argparse.Namespace,
):
    use_gpu: bool = args.use_gpu
    enable_cuda_graph: bool = args.use_cuda_graph
    causal: bool = args.causal
    intra_op_num_threads: int = args.intra_op_num_threads
    repeats: int = args.repeats

    print(f"run_tflops_test: causal={causal}")

    if use_gpu:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH, InputFormats.Q_KV_BSNH_BSN2H, InputFormats.QKV_BSN3H]
        provider = "CUDAExecutionProvider"
        # flash attention is available for sm >= 80
        sm = get_compute_capability()
        if sm >= 80:
            backends = [
                SdpaKernel.DEFAULT,
                SdpaKernel.FLASH_ATTENTION,
                SdpaKernel.EFFICIENT_ATTENTION,
                SdpaKernel.CUDNN_FLASH_ATTENTION,
                SdpaKernel.MATH,
            ]

            if args.past_sequence_length > 0:
                backends.append(SdpaKernel.LEAN_ATTENTION)

            if args.past_sequence_length > 0 and causal:
                backends.remove(SdpaKernel.CUDNN_FLASH_ATTENTION)

            if args.past_sequence_length > 4096:
                backends.remove(SdpaKernel.MATH)
        else:
            backends = [SdpaKernel.DEFAULT, SdpaKernel.EFFICIENT_ATTENTION, SdpaKernel.MATH]
    else:
        device_id = 0
        device = torch.device("cpu")
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH]
        enable_cuda_graph = False
        provider = "CPUExecutionProvider"
        backends = [SdpaKernel.DEFAULT]

    configs = get_test_configs(args)
    print(
        "\nformat\tcausal\tattBias\tbatch\tseqlen\tpast\theads\th_dim\tthreads\tms\tTFLOPS\tsdpa_kernel\trequest_kernel"
    )

    for input_format in formats:
        for batch_size, sequence_length, past_sequence_length, num_heads, head_size, enable_unfused in configs:
            if past_sequence_length > 0 and input_format not in [InputFormats.Q_K_V_BSNH_BSNH_BSNH]:
                continue
            config = MultiHeadAttentionConfig(
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_heads=num_heads,
                head_size=head_size,
                causal=causal,
                use_kv_cache=past_sequence_length > 0,
                past_sequence_length=past_sequence_length,
                max_cache_sequence_length=None,
                kv_sequence_length=None,
                provider=provider,
                enable_cuda_graph=enable_cuda_graph,
                device=device,
                dtype=torch.float16 if use_gpu else torch.float,
                share_past_present_buffer=False,
                input_format=input_format,
                has_past_input=past_sequence_length > 0,
                has_attn_bias=args.has_attn_bias,
                broadcast_attn_bias_dim_0=args.broadcast_attn_bias_dim_0,
                broadcast_attn_bias_dim_1=args.broadcast_attn_bias_dim_1,
            )
            for attention_kernel in backends:
                sess_options = SessionOptions()
                sess_options.intra_op_num_threads = intra_op_num_threads

                if use_gpu:
                    request_kernel = get_gpu_kernel_name(attention_kernel)
                else:
                    request_kernel = get_cpu_kernel_name(config)

                if "math" in request_kernel:
                    # Skip large sequence length for Unfused kernel to avoid OOM.
                    if not enable_unfused:
                        if config.verbose:
                            print(f"skip unfused kernel for {vars(config)}")
                        continue

                    # Unfused kernel does not support packed QKV or packed KV formats.
                    if input_format not in [InputFormats.Q_K_V_BSNH_BSNH_BSNH]:
                        if config.verbose:
                            print(f"skip input_format for {vars(config)}")
                        continue

                    if use_gpu and config.total_sequence_length > 8192:
                        if config.verbose:
                            print(f"skip large sequence length for {vars(config)}")
                        continue

                if use_gpu:
                    actual_kernel = sdpa_kernel_from_debug_info(config, attention_kernel, sess_options)
                    if actual_kernel is None:
                        print(f"Warning: skip {config} since kernel from debug info is None")
                        continue
                    if actual_kernel != request_kernel and request_kernel != "ort:default":
                        print(f"Skip since {actual_kernel=} != {request_kernel=}")
                        continue
                else:
                    # CPU has no debug info for now.
                    actual_kernel = request_kernel

                session = create_session(config, sess_options, attention_kernel=attention_kernel)
                input_dict = config.random_inputs()

                # warm up session
                try:
                    _ = measure_latency(session, input_dict)
                except Exception as e:
                    print(f"Failed to run {request_kernel=} for {config=}. Exception: {e}")
                    continue

                latency_list = []
                for _ in range(repeats):
                    latency = measure_latency(session, input_dict)
                    latency_list.append(latency)
                average_latency = statistics.mean(latency_list)

                del session

                format_str = InputFormats.input_format_str(input_format)

                # compute TFLOPS per second
                speed = tflops_per_second(
                    flops(
                        batch_size,
                        sequence_length,
                        sequence_length + past_sequence_length,
                        head_size,
                        num_heads,
                        causal,
                    ),
                    average_latency,
                )

                row = {
                    "use_gpu": use_gpu,
                    "enable_cuda_graph": enable_cuda_graph,
                    "format": format_str,
                    "causal": causal,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "past_sequence_length": past_sequence_length,
                    "num_heads": num_heads,
                    "head_size": head_size,
                    "has_attn_bias": args.has_attn_bias,
                    "broadcast_attn_bias_dim_0": args.broadcast_attn_bias_dim_0,
                    "broadcast_attn_bias_dim_1": args.broadcast_attn_bias_dim_1,
                    "intra_op_num_threads": intra_op_num_threads,
                    "average_latency": average_latency,
                    "tflops": speed,
                    "request_kernel": request_kernel,
                    "kernel": actual_kernel,
                }
                csv_writer.writerow(row)

                speed = f"{speed:.3f}" if speed is not None else "NA"
                print(
                    f"{format_str}\t{causal}\t{args.has_attn_bias}\t{batch_size}\t"
                    f"{sequence_length}\t{past_sequence_length}\t{num_heads}\t{head_size}\t"
                    f"{intra_op_num_threads}\t{average_latency * 1000:.3f}\t{speed}\t{actual_kernel}\t{request_kernel}"
                )


def run_torch_test(
    csv_writer: csv.DictWriter,
    args: argparse.Namespace,
):
    use_gpu: bool = args.use_gpu
    causal: bool = args.causal

    configs = get_test_configs(args)

    if use_gpu:
        if not torch.cuda.is_available():
            return
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        dtype = torch.float16
        backends = [
            None,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.MATH,
        ]
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        backends = [None]

    backend_names = {
        SDPBackend.FLASH_ATTENTION: "torch:flash",
        SDPBackend.EFFICIENT_ATTENTION: "torch:efficient",
        SDPBackend.CUDNN_ATTENTION: "torch:cudnn",
        SDPBackend.MATH: "torch:math",
        None: "torch:default",
    }

    # Test PyTorch latency
    for batch_size, sequence_length, past_sequence_length, num_heads, head_size, enable_unfused in configs:
        for backend in backends:
            if backend == SDPBackend.MATH and not enable_unfused:
                continue
            if backend == SDPBackend.FLASH_ATTENTION and platform.system() != "Linux":
                continue

            backend_name = backend_names[backend]
            try:
                with torch.no_grad():
                    torch_latency = run_torch_sdpa(
                        batch_size,
                        sequence_length,
                        sequence_length,
                        num_heads,
                        head_size,
                        causal,
                        has_mask=False,
                        mask_dim=2,
                        mask_dtype=torch.bool,
                        device=device,
                        dtype=dtype,
                        backend=backend,
                        repeats=args.repeats,
                    )
            except RuntimeError:
                continue

            speed = tflops_per_second(
                flops(
                    batch_size,
                    sequence_length,
                    sequence_length + past_sequence_length,
                    head_size,
                    num_heads,
                    causal,
                ),
                torch_latency,
            )
            input_format = "Q,K,V"
            print(
                f"{input_format}\t{causal}\t{False}\t{batch_size}\t"
                f"{sequence_length}\t{past_sequence_length}\t{num_heads}\t{head_size}\t"
                f"{torch.get_num_threads()}\t{torch_latency * 1000:.2f}\t{speed}\t{backend_name}\t{backend_name}"
            )
            row = {
                "use_gpu": use_gpu,
                "enable_cuda_graph": False,
                "format": input_format,
                "causal": causal,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "past_sequence_length": past_sequence_length,
                "num_heads": num_heads,
                "head_size": head_size,
                "has_attn_bias": False,
                "broadcast_attn_bias_dim_0": False,
                "broadcast_attn_bias_dim_1": False,
                "intra_op_num_threads": torch.get_num_threads(),
                "average_latency": torch_latency,
                "tflops": speed,
                "request_kernel": backend_name,
                "kernel": backend_name,
            }
            csv_writer.writerow(row)


def run_tflops_tests(args):
    features = "gpu" if args.use_gpu else "cpu"
    if args.causal:
        features += "_causal"
    if args.past_sequence_length > 0:
        features += "_past"
    csv_filename = "{}_{}_{}_{}.csv".format(
        args.csv_filename_prefix,
        features,
        "torch" if args.torch else "ort",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    with open(csv_filename, mode="a", newline="") as csv_file:
        column_names = [
            "use_gpu",
            "enable_cuda_graph",
            "format",
            "causal",
            "batch_size",
            "sequence_length",
            "past_sequence_length",
            "num_heads",
            "head_size",
            "has_attn_bias",
            "broadcast_attn_bias_dim_0",
            "broadcast_attn_bias_dim_1",
            "intra_op_num_threads",
            "average_latency",
            "tflops",
            "request_kernel",
            "kernel",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        if args.torch:
            run_torch_test(csv_writer, args)
        else:
            run_tflops_test(csv_writer, args)


def plot_prompt_performance(
    model_name: str,
    batch_size: int,
    num_heads: int,
    head_size: int,
    max_seq_len: int,
):
    import triton

    formats = InputFormats.get_name_list()

    # Exclude cross attention since kernel crashes for some configuration.
    formats = formats[:-1]

    settings = {
        "line_vals": formats,
        "line_names": ["ORT-MHA:" + name for name in formats],
        "styles": [("red", "solid"), ("yellow", "dashdot"), ("blue", "dashed"), ("green", "dotted")][0 : len(formats)],
    }

    sm = get_compute_capability()
    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(6, 17) if 2**i <= max_seq_len],
            line_arg="input_format",
            ylabel="ms",
            **settings,
            plot_name=f"prompt-sm{sm}-{model_name}-b{batch_size}-h{num_heads}_{head_size}-fp16",
            args={
                "batch_size": batch_size,
                "num_heads": num_heads,
                "head_size": head_size,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(
        input_format: str,
        sequence_length: int,
        batch_size: int,
        num_heads: int,
        head_size: int,
        device="cuda",
    ):
        warmup = 15
        repeat = 100

        config: MultiHeadAttentionConfig = MultiHeadAttentionConfig(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            past_sequence_length=0,
            kv_sequence_length=sequence_length if input_format == "Q,K',V'" else None,
            max_cache_sequence_length=max_seq_len,
            provider="CUDAExecutionProvider",
            enable_cuda_graph=False,
            device=device,
            dtype=torch.float16,
            use_kv_cache=False,
            input_format=InputFormats.convert(input_format),
        )

        obj = OrtMultiHeadAttention(config)
        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_bert_performance_test():
    """
    Run performance tests for prompt and token generation.

    """
    configures = [
        # (1, 32, 128, 8192, "TNLGv4"),
        # (4, 32, 128, 8192, "TNLGv4"),
        (1, 12, 64, 1024, "BertBase"),
        (16, 12, 64, 1024, "BertBase"),
        (1, 16, 64, 1024, "BertLarge"),
        (8, 16, 64, 1024, "BertLarge"),
    ]

    for batch_size, num_heads, head_size, max_seq_len, model_name in configures:
        plot_prompt_performance(
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_len=max_seq_len,
            model_name=model_name,
        )


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark MultiHeadAttention for ONNX Runtime and PyTorch.")

    parser.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="Use GPU for inference.",
    )
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "--use_cuda_graph",
        required=False,
        action="store_true",
        help="Use cuda graph in onnxruntime.",
    )
    parser.set_defaults(use_cuda_graph=False)

    parser.add_argument(
        "--intra_op_num_threads",
        required=False,
        type=int,
        choices=[0, 1, 2, 4, 8, 16],
        default=0,
        help="intra_op_num_threads for onnxruntime. ",
    )

    parser.add_argument(
        "--causal",
        required=False,
        action="store_true",
        help="test unidirectional",
    )
    parser.set_defaults(causal=False)

    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=0,
        help="batch size",
    )

    parser.add_argument(
        "-s",
        "--sequence_length",
        required=False,
        type=int,
        default=512,
        help="sequence length",
    )

    parser.add_argument(
        "-p",
        "--past_sequence_length",
        required=False,
        type=int,
        default=0,
        help="past sequence length",
    )

    parser.add_argument(
        "-n",
        "--num_heads",
        required=False,
        type=int,
        default=16,
        help="number of attention heads",
    )

    parser.add_argument(
        "-d",
        "--head_size",
        required=False,
        type=int,
        default=64,
        help="hidden dimension per head",
    )

    parser.add_argument(
        "-r",
        "--repeats",
        required=False,
        type=int,
        default=0,
        help="number of repeats for performance test",
    )

    parser.add_argument(
        "--torch",
        required=False,
        action="store_true",
        help="test pytorch instead of onnxruntime",
    )
    parser.set_defaults(torch=False)

    parser.add_argument(
        "--has_attn_bias",
        required=False,
        action="store_true",
        help="has attention bias",
    )
    parser.set_defaults(has_attn_bias=False)

    parser.add_argument(
        "--broadcast_attn_bias_dim_0",
        required=False,
        action="store_true",
        help="broadcast attention bias dimension 0",
    )
    parser.set_defaults(broadcast_attn_bias_dim_0=False)

    parser.add_argument(
        "--broadcast_attn_bias_dim_1",
        required=False,
        action="store_true",
        help="broadcast attention bias dimension 1",
    )
    parser.set_defaults(broadcast_attn_bias_dim_1=False)

    parser.add_argument(
        "--csv_filename_prefix",
        required=False,
        type=str,
        default="benchmark_mha",
        help="Prefix of csv filename",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_arguments()
    print(f"arguments:{args}")

    if args.repeats == 0:
        args.repeats = 10000 if args.use_gpu else 100

    if args.use_gpu:
        assert torch.cuda.is_available()
        if not args.torch:
            assert "CUDAExecutionProvider" in get_available_providers()

    if args.torch:
        assert Version(torch.__version__) >= Version("2.3.0")
        assert args.past_sequence_length == 0

    if args.use_gpu and args.batch_size == 0 and not args.torch:
        if platform.system() == "Linux":
            s = torch.cuda.Stream()
            with torch.cuda.stream(s), torch.no_grad():
                run_bert_performance_test()

    run_tflops_tests(args)

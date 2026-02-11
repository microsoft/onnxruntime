# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math

import numpy
import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession

# --- Quantization Helpers (from test_gqa.py) ---

ONNX_TENSOR_TYPE_MAP = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
    "int32": TensorProto.INT32,
    "int8": TensorProto.INT8,
    "int4": TensorProto.UINT8,
}

TORCH_DTYPE_TO_ONNX_MAP = {
    torch.float32: TensorProto.FLOAT,
    torch.float16: TensorProto.FLOAT16,
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.int32: TensorProto.INT32,
    torch.int8: TensorProto.INT8,
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.uint8,
}

NUMPY_DTYPE_MAP = {
    "float32": numpy.float32,
    "float16": numpy.float16,
    "bfloat16": numpy.uint16,
    "int8": numpy.int8,
    "int4": numpy.uint8,
}


def get_q_range(q_type_str):
    q_type_str = str(q_type_str)
    if q_type_str.endswith("int8"):
        return -128, 127
    if q_type_str.endswith("int4"):
        return -8, 7
    raise ValueError(f"Unsupported quantization type for range: {q_type_str}")


def pack_int4(tensor_int8):
    assert tensor_int8.shape[-1] % 2 == 0
    t_low = tensor_int8[..., 0::2] + 8
    t_high = tensor_int8[..., 1::2] + 8
    packed = (t_low & 0x0F) | (t_high << 4)
    return packed.to(torch.uint8)


def unpack_int4(packed_tensor_uint8):
    t_low = (packed_tensor_uint8 & 0x0F) - 8
    t_high = (packed_tensor_uint8 >> 4) - 8
    unpacked = torch.empty(
        (*packed_tensor_uint8.shape[:-1], packed_tensor_uint8.shape[-1] * 2),
        dtype=torch.int8,
        device=packed_tensor_uint8.device,
    )
    unpacked[..., 0::2] = t_low
    unpacked[..., 1::2] = t_high
    return unpacked


def compute_scale(tensor_float, quant_type, q_type_str):
    if quant_type == "NONE":
        return None

    qmin, qmax = get_q_range(q_type_str)

    if quant_type == "PER_TENSOR":
        t_max = torch.max(torch.abs(tensor_float))
        scale = t_max / qmax if t_max > 1e-6 else torch.tensor(1.0, device=tensor_float.device, dtype=torch.float32)
        return scale.unsqueeze(0).to(torch.float32)

    if quant_type == "PER_CHANNEL":
        # Per-channel scale is computed independently for each channel across the batch and sequence length dimensions.
        t_max = torch.max(torch.abs(tensor_float), dim=2, keepdim=True)[0]
        t_max = torch.max(t_max, dim=0, keepdim=True)[0]
        scale = t_max / qmax
        scale[scale < 1e-6] = 1.0
        return scale.to(torch.float32)

    raise ValueError(f"Unsupported quant_type: {quant_type}")


def dequantize_tensor(quantized_tensor, scale, quant_type, q_type_str):
    if quant_type == "NONE":
        return quantized_tensor

    # Ensure scale is on the same device as quantized_tensor
    if isinstance(scale, torch.Tensor):
        scale = scale.to(quantized_tensor.device)

    unpacked_tensor = quantized_tensor
    q_type_str_s = str(q_type_str)
    if q_type_str_s.endswith("int4"):
        unpacked_tensor = unpack_int4(quantized_tensor)

    return unpacked_tensor.to(torch.float32) * scale


def quantize_tensor_with_scale(tensor_float, scale, quant_type, q_type_str):
    """Quantizes a tensor using a provided scale."""
    if quant_type == "NONE":
        return tensor_float

    qmin, qmax = get_q_range(q_type_str)
    quantized = torch.clamp(torch.round(tensor_float / scale), qmin, qmax)

    q_type_str_s = str(q_type_str)
    if q_type_str_s.endswith("int4"):
        quantized = pack_int4(quantized.to(torch.int8))
    else:
        target_dtype = TORCH_DTYPE_MAP[q_type_str]
        quantized = quantized.to(target_dtype)
    return quantized


# --- Classes moved from test_sparse_attention.py ---


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
        softmax_scale: float | None,
        do_rotary: bool,
        rotary_interleaved: bool,
        provider: str = "CUDAExecutionProvider",
        device="cuda",
        dtype=torch.float16,
        share_buffer: bool = True,
        is_packed_qkv: bool = False,
        max_cache_sequence_length=None,
        max_rotary_sequence_length=None,
        use_smooth_softmax: bool = False,
    ):
        self.operator = operator
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.max_cache_sequence_length = max_cache_sequence_length or max_sequence_length
        self.max_rotary_sequence_length = max_rotary_sequence_length or max_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale or (1.0 / (head_size**0.5))

        # Derived values
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = self.max_cache_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = (
            self.max_cache_sequence_length if share_buffer else (past_sequence_length + sequence_length)
        )

        self.do_rotary = do_rotary
        self.rotary_interleaved = rotary_interleaved

        self.provider = provider
        self.device = device
        self.dtype = dtype

        self.share_buffer = share_buffer
        self.is_packed_qkv = is_packed_qkv

        self.use_smooth_softmax = use_smooth_softmax

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
            "cos_cache": (self.max_rotary_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
            "sin_cache": (self.max_rotary_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
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
        angle = torch.rand(self.max_rotary_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        return cos.to(device=self.device), sin.to(device=self.device)

    def random_inputs(self):
        device = self.device
        # ORT python I/O binding API supports bf16 via torch tensor.
        dtype = self.dtype

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
        provider: str = "CUDAExecutionProvider",
        device="cuda",
        dtype=torch.float16,
        local_window_size: int = -1,
        attention_mask=None,
        is_packed_qkv=False,
        max_cache_sequence_length=None,
        max_rotary_sequence_length=None,
        use_smooth_softmax: bool = False,
        k_quant_type: str = "NONE",
        v_quant_type: str = "NONE",
        kv_cache_type: str = "float16",
        share_kv_scale: bool = False,
    ):
        super().__init__(
            "GroupQueryAttention",
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            past_sequence_length=past_sequence_length,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            do_rotary=do_rotary,
            rotary_interleaved=rotary_interleaved,
            provider=provider,
            device=device,
            dtype=dtype,
            is_packed_qkv=is_packed_qkv,
            max_cache_sequence_length=max_cache_sequence_length,
            max_rotary_sequence_length=max_rotary_sequence_length,
            use_smooth_softmax=use_smooth_softmax,
        )
        # local_window_size is for ORT only, not for Torch implementation.
        self.local_window_size = local_window_size

        # attention mask is for Torch implementation only, not for ORT.
        self.attention_mask = attention_mask

        # Quantization parameters
        self.k_quant_type = k_quant_type
        self.v_quant_type = v_quant_type
        self.kv_cache_type = kv_cache_type
        # Determine bit width from cache type if applicable
        self.kv_cache_bit_width = 4 if kv_cache_type == "int4" else (8 if kv_cache_type == "int8" else 0)
        self.share_kv_scale = share_kv_scale

    def shape_dict(self):
        shapes = super().shape_dict()
        shapes.update(
            {
                "seqlens_k": (self.batch_size,),
            }
        )
        # Note: We don't adjust shapes for int4 here because the parent's random_inputs
        # creates float tensors first, then quantization will pack them
        return shapes

    def random_inputs(self):
        feeds = super().random_inputs()
        k_seqlens = torch.ones((self.batch_size,), device=self.device, dtype=torch.int32) * self.total_sequence_length
        feeds.update(
            {
                "seqlens_k": k_seqlens - 1,
            }
        )

        # Generate quantized cache and scales if quantization is enabled
        if self.k_quant_type != "NONE":
            # Compute scales from the generated float cache
            k_scale = compute_scale(feeds["past_key"], self.k_quant_type, self.kv_cache_type)
            if self.share_kv_scale:
                v_scale = k_scale
            else:
                v_scale = compute_scale(feeds["past_value"], self.v_quant_type, self.kv_cache_type)

            # Scale tensors must be float32 (required by GQA operator)
            if k_scale is not None:
                k_scale = k_scale.to(torch.float32)
                feeds["k_scale"] = k_scale
            if v_scale is not None:
                v_scale = v_scale.to(torch.float32)
                feeds["v_scale"] = v_scale

            # Quantize the cache tensors
            feeds["past_key"] = quantize_tensor_with_scale(
                feeds["past_key"], k_scale, self.k_quant_type, self.kv_cache_type
            )
            feeds["past_value"] = quantize_tensor_with_scale(
                feeds["past_value"], v_scale, self.v_quant_type, self.kv_cache_type
            )

        return feeds


def create_group_query_attention_onnx_model(config: GroupQueryAttentionConfig):
    assert config.dtype in [torch.float16, torch.float32, torch.bfloat16]

    if config.dtype == torch.float16:
        float_type = TensorProto.FLOAT16
    elif config.dtype == torch.bfloat16:
        float_type = TensorProto.BFLOAT16
    else:
        float_type = TensorProto.FLOAT

    # Build input list for the GQA node
    node_inputs = [
        "query",
        "key" if not config.is_packed_qkv else "",
        "value" if not config.is_packed_qkv else "",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length" if config.share_buffer else "",
        "cos_cache" if config.do_rotary else "",
        "sin_cache" if config.do_rotary else "",
        "",  # position_ids (optional, not used in benchmark)
        "",  # attention_bias (optional, not used in benchmark)
        "",  # head_sink (optional, not used in benchmark)
        "k_scale" if config.k_quant_type != "NONE" else "",
        "v_scale" if config.v_quant_type != "NONE" else "",
    ]
    # Remove trailing empty strings
    while node_inputs and node_inputs[-1] == "":
        node_inputs.pop()

    # Build attributes dictionary
    node_attrs = {
        "num_heads": config.num_heads,
        "kv_num_heads": config.kv_num_heads,
        "scale": config.softmax_scale,
        "local_window_size": config.local_window_size,
        "do_rotary": 1 if config.do_rotary else 0,
        "rotary_interleaved": config.rotary_interleaved,
        "smooth_softmax": 1 if config.use_smooth_softmax else 0,
        "domain": "com.microsoft",
    }

    # Add quantization attributes if enabled
    if config.k_quant_type != "NONE":
        node_attrs["k_quant_type"] = config.k_quant_type
        node_attrs["v_quant_type"] = config.v_quant_type
        node_attrs["kv_cache_bit_width"] = config.kv_cache_bit_width

    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            node_inputs,
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            **node_attrs,
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

    # Determine cache tensor type based on quantization
    # Note: INT8 uses INT8 type, INT4 uses UINT8 (for packing 2x4-bit values per byte)
    cache_type = float_type
    if config.kv_cache_type == "int4":
        cache_type = TensorProto.UINT8
    elif config.kv_cache_type == "int8":
        cache_type = TensorProto.INT8

    # Compute actual cache shapes (packed for INT4)
    past_key_shape = list(shape_dict["past_key"])
    past_value_shape = list(shape_dict["past_value"])
    present_key_shape = list(shape_dict["present_key"])
    present_value_shape = list(shape_dict["present_value"])

    # For INT4, the last dimension is packed (2 values per byte)
    if config.kv_cache_type == "int4":
        past_key_shape[-1] = past_key_shape[-1] // 2
        past_value_shape[-1] = past_value_shape[-1] // 2
        present_key_shape[-1] = present_key_shape[-1] // 2
        present_value_shape[-1] = present_value_shape[-1] // 2

    graph_input.extend(
        [
            helper.make_tensor_value_info("past_key", cache_type, past_key_shape),
            helper.make_tensor_value_info("past_value", cache_type, past_value_shape),
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

    # Add scale inputs for quantization
    # Shape depends on quantization type:
    # - PER_TENSOR: [1]
    # - PER_CHANNEL: [1, kv_num_heads, 1, head_size]
    # Note: k_scale and v_scale are always float32 regardless of the model's dtype
    if config.k_quant_type != "NONE":
        if config.k_quant_type == "PER_TENSOR":
            k_scale_shape = [1]
        else:  # PER_CHANNEL
            k_scale_shape = [1, config.kv_num_heads, 1, config.head_size]
        graph_input.append(helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, k_scale_shape))

    if config.v_quant_type != "NONE":
        if config.v_quant_type == "PER_TENSOR":
            v_scale_shape = [1]
        else:  # PER_CHANNEL
            v_scale_shape = [1, config.kv_num_heads, 1, config.head_size]
        graph_input.append(helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, v_scale_shape))

    graph_output = [
        helper.make_tensor_value_info("output", float_type, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", cache_type, present_key_shape),
        helper.make_tensor_value_info("present_value", cache_type, present_value_shape),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_gqa_ort_session(
    config: GroupQueryAttentionConfig, session_options=None, enable_cuda_graph=False
) -> CudaSession:
    onnx_model_str = create_group_query_attention_onnx_model(config)

    if config.provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device() if isinstance(config.device, str) else config.device.index
        provider_options = CudaSession.get_cuda_provider_options(
            device_id, enable_cuda_graph=enable_cuda_graph, stream=torch.cuda.current_stream().cuda_stream
        )
        providers = [(config.provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    ort_session = InferenceSession(onnx_model_str, session_options, providers=providers)
    # Note that CudaSession could work with both CUDA and CPU providers.
    cuda_session = CudaSession(ort_session, config.device, enable_cuda_graph=enable_cuda_graph)
    shape_dict = config.shape_dict()
    cuda_session.allocate_buffers(shape_dict)

    buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
    for input_name, output_name in buffer_sharing.items():
        cuda_session.set_buffer_sharing(input_name, output_name)

    return cuda_session


class OrtGroupQueryAttention:
    """A wrapper of ORT GroupQueryAttention to test relevance and performance."""

    def __init__(self, config: GroupQueryAttentionConfig):
        self.session = create_gqa_ort_session(config)

        self.feed_dict = config.random_inputs()

        # ENABLE_DEBUG is not defined in this module, so we assume False or pass it as arg if needed.
        # But looking at original code, it was a global. Since this is a helper, we might skip the debug print or make it optional.
        # For strict refactoring, I'll remove the debug print block or comment it out unless I import ENABLE_DEBUG.
        # I'll check if ENABLE_DEBUG was used in the class. It was.
        # I'll skip it for now to avoid dependency on global var.

    def infer(self):
        return self.session.infer(self.feed_dict)

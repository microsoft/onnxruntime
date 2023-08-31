# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of MultiHeadAttention with Nvidia GPU of Compute Capability 8.0, 8.6 or 8.9 in Linux:
sh benchmark_mha.sh
"""

import math
import os
import statistics
import time

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession


class InputFormats:
    Q_K_V_BSNH = 0
    QKV_BSN3H = 1
    Q_KV_BSNH_BSN2H = 2

    @staticmethod
    def input_format_str(format: int) -> str:
        return "QKV" if format == 1 else "Q,KV" if format == 2 else "Q,K,V"


class Config:
    batch_size: int = 0
    sequence_length: int = 0
    kv_sequence_length: int = 0
    num_heads: int = 0
    head_size: int = 0
    causal: bool = False
    input_format: int = InputFormats.Q_K_V_BSNH

    def __init__(self, b, s, s2, n, h, causal, input_format):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.num_heads = n
        self.head_size = h
        self.causal = causal
        self.input_format = input_format


def create_multihead_attention_graph(config: Config):
    query = helper.make_tensor_value_info(
        "query",
        TensorProto.FLOAT16,
        [
            config.batch_size,
            config.sequence_length,
            config.num_heads * config.head_size,
        ],
    )

    key = helper.make_tensor_value_info(
        "key",
        TensorProto.FLOAT16,
        [
            config.batch_size,
            config.kv_sequence_length,
            config.num_heads * config.head_size,
        ],
    )

    value = helper.make_tensor_value_info(
        "value",
        TensorProto.FLOAT16,
        [
            config.batch_size,
            config.kv_sequence_length,
            config.num_heads * config.head_size,
        ],
    )

    packed_qkv = helper.make_tensor_value_info(
        "query",
        TensorProto.FLOAT16,
        [
            config.batch_size,
            config.sequence_length,
            config.num_heads,
            3,
            config.head_size,
        ],
    )

    packed_kv = helper.make_tensor_value_info(
        "key",
        TensorProto.FLOAT16,
        [
            config.batch_size,
            config.kv_sequence_length,
            config.num_heads,
            2,
            config.head_size,
        ],
    )

    if config.input_format == InputFormats.QKV_BSN3H:
        input_names = ["query"]
        inputs = [packed_qkv]
    elif config.input_format == InputFormats.Q_KV_BSNH_BSN2H:
        input_names = ["query", "key"]
        inputs = [query, packed_kv]
    else:  # input_format==InputFormats.Q_K_V_BSNH
        input_names = ["query", "key", "value"]
        inputs = [query, key, value]

    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            input_names,
            ["output"],
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        inputs,
        outputs,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def input_output_shapes(config: Config):
    if config.input_format == InputFormats.QKV_BSN3H:
        return {
            "query": (config.batch_size, config.sequence_length, config.num_heads, 3, config.head_size),
            "output": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        }

    if config.input_format == InputFormats.Q_KV_BSNH_BSN2H:
        return {
            "query": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
            "key": (config.batch_size, config.kv_sequence_length, config.num_heads, 2, config.head_size),
            "output": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        }

    return {
        "query": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        "key": (config.batch_size, config.kv_sequence_length, config.num_heads * config.head_size),
        "value": (config.batch_size, config.kv_sequence_length, config.num_heads * config.head_size),
        "output": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
    }


def create_session(
    device_id: int, config: Config, provider: str = "CUDAExecutionProvider", enable_cuda_graph: bool = False
) -> CudaSession:
    onnx_model_str = create_multihead_attention_graph(config)
    provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
    ort_session = InferenceSession(onnx_model_str, providers=[(provider, provider_options), "CPUExecutionProvider"])
    device = torch.device("cuda", device_id)
    cuda_session = CudaSession(ort_session, device, enable_cuda_graph)
    shape_dict = input_output_shapes(config)
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


def flops(batch, sequence_length, head_size, num_heads, causal):
    return 4 * batch * sequence_length**2 * num_heads * head_size // (2 if causal else 1)


def tflops_per_second(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def get_sm8x_kernel_name(config: Config) -> str:
    # This classification is for Nvidia GPU of Compute Capability 8.* like A100.
    # Note that some kernel might not exist in older or newer GPUs.
    if os.getenv("ORT_DISABLE_FLASH_ATTENTION") != "1":
        if config.input_format == InputFormats.QKV_BSN3H:
            min_seq_len = os.getenv("ORT_MIN_SEQ_LEN_FLASH_ATTENTION_PACKED_QKV")
            min_length = int(min_seq_len) if min_seq_len is not None else 513
            if config.sequence_length >= min_length:
                return "Flash"
        else:
            return "Flash"

    if (os.getenv("ORT_DISABLE_FUSED_CROSS_ATTENTION") != "1" and config.kv_sequence_length <= 128) or (
        os.getenv("ORT_DISABLE_FUSED_ATTENTION") != "1"
        and (config.sequence_length <= 384 or os.getenv("ORT_DISABLE_TRT_FLASH_ATTENTION") != "1")
    ):
        return "TRT"

    if os.getenv("ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION") != "1":
        return "MemEff"

    return "Unfused"


def run_tflops_test(dtype=torch.float16, enable_cuda_graph: bool = False, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # (batch_size, sequence_length, num_heads, head_size)
    configs = [
        (32, 512, 64, 32),
        (32, 512, 128, 16),
        (16, 1024, 64, 32),
        (16, 1024, 128, 16),
        (8, 2048, 64, 32),
        (8, 2048, 128, 16),
        (4, 4096, 64, 32),
        (4, 4096, 128, 16),
        (2, 8192, 64, 32),
        (2, 8192, 128, 16),
        (1, 16384, 64, 32),
        (1, 16384, 128, 16),
        # stable diffusion
        (1, 4096, 8, 40),
        (1, 4096, 8, 80),
        (1, 4096, 8, 160),
        (4, 4096, 8, 40),
        (4, 4096, 8, 80),
        (4, 4096, 8, 160),
        (1, 16384, 8, 40),
        (1, 16384, 8, 80),
        (1, 16384, 8, 160),
        # bert-base
        (128, 128, 12, 64),
        (64, 128, 12, 64),
        (128, 384, 12, 64),
        (64, 384, 12, 64),
        (128, 512, 12, 64),
        (64, 512, 12, 64),
        # TNLGv4
        (4, 2048, 32, 128),
        (4, 4096, 32, 128),
        (8, 2048, 32, 128),
        (8, 4096, 32, 128),
    ]

    print(f"enable_cuda_graph={enable_cuda_graph}")

    # List of environment variables to enable/disable attention kernels
    print("Environment Variables:")
    env_names = [
        "ORT_DISABLE_FLASH_ATTENTION",
        "ORT_MIN_SEQ_LEN_FLASH_ATTENTION_PACKED_QKV",
        "ORT_DISABLE_FUSED_ATTENTION",
        "ORT_DISABLE_TRT_FLASH_ATTENTION",
        "ORT_ENABLE_FUSED_CAUSAL_ATTENTION",
        "ORT_DISABLE_FUSED_CROSS_ATTENTION",
        "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION",
    ]
    for name in env_names:
        value = os.getenv(name)
        if value is not None:
            print(f"{name}={value}")
    print()

    print("format\tcausal\tbatch\tseqlen\theads\th_dim\tms\tTFLOPS\tkernel")
    causal = False
    for input_format in [InputFormats.Q_K_V_BSNH, InputFormats.Q_KV_BSNH_BSN2H, InputFormats.QKV_BSN3H]:
        for batch_size, sequence_length, num_heads, head_size in configs:
            config = Config(batch_size, sequence_length, sequence_length, num_heads, head_size, causal, input_format)

            session = create_session(device_id, config, enable_cuda_graph=enable_cuda_graph)

            qkv = torch.randn(batch_size, sequence_length, 3, num_heads, head_size, device=device, dtype=dtype)
            q, k, v = qkv.unbind(dim=2)

            if input_format == InputFormats.QKV_BSN3H:
                if config.sequence_length != config.kv_sequence_length:
                    continue
                q = torch.reshape(q, (-1, config.num_heads, config.head_size))
                k = torch.reshape(k, (-1, config.num_heads, config.head_size))
                v = torch.reshape(v, (-1, config.num_heads, config.head_size))
                packed_qkv = torch.dstack((q, k, v)).reshape(
                    config.batch_size, config.sequence_length, config.num_heads, 3, config.head_size
                )
                input_dict = {"query": packed_qkv.contiguous()}
            elif input_format == InputFormats.Q_KV_BSNH_BSN2H:
                q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
                k = torch.reshape(k, (-1, config.num_heads, config.head_size))
                v = torch.reshape(v, (-1, config.num_heads, config.head_size))
                packed_kv = torch.dstack((k, v)).reshape(
                    config.batch_size, config.sequence_length, config.num_heads, 2, config.head_size
                )
                input_dict = {"query": q.contiguous(), "key": packed_kv.contiguous()}
            else:  # input_format == InputFormats.Q_K_V_BSNH
                q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
                k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
                v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))
                input_dict = {
                    "query": q.contiguous(),
                    "key": k.contiguous(),
                    "value": v.contiguous(),
                }

            # warm up session
            _ = measure_latency(session, input_dict)

            latency_list = []
            for _ in range(repeats):
                latency = measure_latency(session, input_dict)
                latency_list.append(latency)
            average_latency = statistics.mean(latency_list)

            del session

            # compute TFLOPS per second
            speed = tflops_per_second(flops(batch_size, sequence_length, head_size, num_heads, causal), average_latency)

            kernel = get_sm8x_kernel_name(config)
            format = InputFormats.input_format_str(input_format)
            print(
                f"{format}\t{causal}\t{batch_size}\t{sequence_length}\t{num_heads}\t{head_size}\t{average_latency * 1000:.2f}\t{speed:.2f}\t{kernel}"
            )


if __name__ == "__main__":
    run_tflops_test(enable_cuda_graph=False)

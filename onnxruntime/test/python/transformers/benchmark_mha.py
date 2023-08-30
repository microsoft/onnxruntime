# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark the performance of MultiHeadAttention. Examples in Linux:

flash attention v2:
ORT_DISABLE_FLASH_ATTENTION=0  python benchmark_mha.py >> result.txt

TensorRT flash attention:
ORT_DISABLE_FLASH_ATTENTION=1  python benchmark_mha.py >> result.txt

Memory Efficient attention:
ORT_DISABLE_FLASH_ATTENTION=1 ORT_DISABLE_TRT_FLASH_ATTENTION=1 python benchmark_mha.py >> result.txt

Unfused attention (might fail):
ORT_DISABLE_FLASH_ATTENTION=1 ORT_DISABLE_TRT_FLASH_ATTENTION=1 ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION=1 python benchmark_mha.py >> result.txt
"""
import math
import os
import statistics
import time
from typing import Dict

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import OrtCudaSession as CudaSession


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    num_heads = 0
    head_size = 0
    causal = False

    def __init__(self, b, s, s2, n, h, causal):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.num_heads = n
        self.head_size = h
        self.causal = causal


def create_multihead_attention_graph(config):
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "key",
                "value",
            ],
            ["output"],
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_session(
    device_id: int, config: Config, provider: str = "CUDAExecutionProvider", enable_cuda_graph: bool = False
) -> CudaSession:
    onnx_model_str = create_multihead_attention_graph(config)
    provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
    ort_session = InferenceSession(onnx_model_str, providers=[(provider, provider_options), "CPUExecutionProvider"])
    device = torch.device("cuda", device_id)
    cuda_session = CudaSession(ort_session, device, enable_cuda_graph)
    shape_dict: Dict[str, tuple] = {
        "query": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
        "key": (config.batch_size, config.kv_sequence_length, config.num_heads * config.head_size),
        "value": (config.batch_size, config.kv_sequence_length, config.num_heads * config.head_size),
        "output": (config.batch_size, config.sequence_length, config.num_heads * config.head_size),
    }
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


def flops(batch, sequence_length, head_size, num_heads, causal):
    return 4 * batch * sequence_length**2 * num_heads * head_size // (2 if causal else 1)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def run_tflops_test(dtype=torch.float16, enable_cuda_graph: bool = False, repeats: int = 30):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

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
        (1, 16384, 8, 40),  # stable diffusion self attention
        (1, 16384, 8, 80),
        (1, 16384, 8, 160),
    ]
    causal = False

    env_names = [
        "ORT_DISABLE_FLASH_ATTENTION",
        "ORT_DISABLE_FUSED_ATTENTION",
        "ORT_DISABLE_TRT_FLASH_ATTENTION",
        "ORT_ENABLE_FUSED_CAUSAL_ATTENTION",
        "ORT_DISABLE_FUSED_CROSS_ATTENTION",
        "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION",
    ]
    for name in env_names:
        print(f"{name}={os.getenv(name)}")

    if os.getenv("ORT_DISABLE_FLASH_ATTENTION") != "1":
        kernel = "Flash_v2"
    elif os.getenv("ORT_DISABLE_TRT_FLASH_ATTENTION") != "1":
        kernel = "TRT_Flash"
    elif os.getenv("ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION") != "1":
        kernel = "MemoryEfficientAttention"
    else:
        kernel = "Unfused"

    print(f"enable_cuda_graph={enable_cuda_graph}")
    print("causal\tbatch\tseqlen\theads\th_dim\tTFLOPS\tkernel")
    for batch_size, sequence_length, num_heads, head_size in configs:
        config = Config(batch_size, sequence_length, sequence_length, num_heads, head_size, causal)

        session = create_session(device_id, config, enable_cuda_graph=enable_cuda_graph)

        qkv = torch.randn(batch_size, sequence_length, 3, num_heads, head_size, device=device, dtype=dtype)
        q, k, v = qkv.unbind(dim=2)

        # configuous is required for IO Binding
        q = torch.reshape(q, (config.batch_size, config.sequence_length, -1)).contiguous()
        k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1)).contiguous()
        v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1)).contiguous()
        input_dict = {
            "query": q,
            "key": k,
            "value": v,
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
        speed = efficiency(flops(batch_size, sequence_length, head_size, num_heads, causal), average_latency)

        print(f"{causal}\t{batch_size}\t{sequence_length}\t{num_heads}\t{head_size}\t{speed:.2f}\t{kernel}")


if __name__ == "__main__":
    run_tflops_test(enable_cuda_graph=False)

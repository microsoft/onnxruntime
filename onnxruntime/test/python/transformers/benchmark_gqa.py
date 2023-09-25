# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of MultiHeadAttention with Nvidia GPU of Compute Capability 8.0, 8.6 or 8.9 in Linux:
sh benchmark_mha.sh
"""

import math
import random
import statistics
import time

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession


class InputFormats:
    QKV_BSNH = 0
    QKV_BNSH = 1


class Config:
    batch_size = 0
    kv_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    head_size = 0

    def __init__(self, b, s2, sp, n, h):
        self.batch_size = b
        self.kv_sequence_length = s2
        self.past_sequence_length = sp
        self.num_heads = n
        self.head_size = h


def create_group_query_attention_graph(config):
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "past_sequence_length",
            ],
            ["output"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.num_heads,
            unidirectional=0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length,
                config.num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length,
                config.num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, 1, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_decoder_masked_multihead_attention_graph(config):
    nodes = [
        helper.make_node(
            "DecoderMaskedMultiHeadAttention",
            [
                "query",
                "key",
                "value",
                "",
                "",
                "past_key",
                "past_value",
                "past_sequence_length",
            ],
            ["output", "present_key", "present_value"],
            "DecoderMaskedMultiHeadAttention_0",
            num_heads=config.num_heads,
            past_present_share_buffer=1,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                1,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.num_heads,
                config.kv_sequence_length,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.num_heads,
                config.kv_sequence_length,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, 1, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [config.batch_size, config.num_heads, config.kv_sequence_length, config.head_size],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [config.batch_size, config.num_heads, config.kv_sequence_length, config.head_size],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "DecoderMaskedMultiHeadAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def input_output_shapes(config: Config, input_format):
    if input_format == InputFormats.QKV_BSNH:
        return {
            "query": (config.batch_size, 1, config.num_heads * config.head_size),
            "key": (config.batch_size, 1, config.num_heads * config.head_size),
            "value": (config.batch_size, 1, config.num_heads * config.head_size),
            "output": (config.batch_size, 1, config.num_heads * config.head_size),
            "past_key": (config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size),
            "past_value": (config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size),
        }

    # DecoderMaskedMultiHeadAttention uses BNSH for past
    if input_format == InputFormats.QKV_BNSH:
        return {
            "query": (config.batch_size, 1, config.num_heads * config.head_size),
            "key": (config.batch_size, 1, config.num_heads * config.head_size),
            "value": (config.batch_size, 1, config.num_heads * config.head_size),
            "output": (config.batch_size, 1, config.num_heads * config.head_size),
            "past_key": (config.batch_size, config.num_heads, config.kv_sequence_length, config.head_size),
            "past_value": (config.batch_size, config.num_heads, config.kv_sequence_length, config.head_size),
        }


def create_gqa_session(
    device_id: int, config: Config, provider: str = "CUDAExecutionProvider", enable_cuda_graph: bool = False
) -> CudaSession:
    onnx_model_str = create_group_query_attention_graph(config)
    provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
    ort_session = InferenceSession(onnx_model_str, providers=[(provider, provider_options), "CPUExecutionProvider"])
    device = torch.device("cuda", device_id)
    cuda_session = CudaSession(ort_session, device, enable_cuda_graph)
    shape_dict = input_output_shapes(config, InputFormats.QKV_BSNH)
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


def create_dmmha_session(
    device_id: int, config: Config, provider: str = "CUDAExecutionProvider", enable_cuda_graph: bool = False
) -> CudaSession:
    onnx_model_str = create_decoder_masked_multihead_attention_graph(config)
    provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
    ort_session = InferenceSession(onnx_model_str, providers=[(provider, provider_options), "CPUExecutionProvider"])
    device = torch.device("cuda", device_id)
    cuda_session = CudaSession(ort_session, device, enable_cuda_graph)
    shape_dict = input_output_shapes(config, InputFormats.QKV_BNSH)
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


# TODO(aciddelgado): what is the correct number of flops in this case?
def flops(batch, kv_sequence_length, head_size, num_heads):
    return 4 * batch * kv_sequence_length * num_heads * head_size


def tflops_per_second(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def benchmark_op(session, input_dict, repeats=100):
    # warm up session
    _ = measure_latency(session, input_dict)

    latency_list = []
    for _ in range(repeats):
        latency = measure_latency(session, input_dict)
        latency_list.append(latency)
    return statistics.mean(latency_list)


def run_tflops_test(dtype=torch.float16, enable_cuda_graph: bool = False, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    print(f"enable_cuda_graph={enable_cuda_graph}")

    print("op\tbatch\ts_kv\theads\th_dim\tms\tTFLOPS")
    mean_gqa_lat = 0
    mean_dmmha_lat = 0
    num_trials = 0
    for b in [1, 3, 8, 16]:
        for s_kv in [128, 256, 512, 1024, 2048]:
            for n in [8, 16, 32, 64, 128]:
                for h in [32, 64, 128]:
                    random.seed(69)
                    sp = random.randint(1, s_kv - 1) if s_kv - 1 > 0 else 0
                    config = Config(b, s_kv, sp, n, h)

                    gqa_session = create_gqa_session(device_id, config, enable_cuda_graph=enable_cuda_graph)
                    dmmha_session = create_dmmha_session(device_id, config, enable_cuda_graph=enable_cuda_graph)

                    qkv = torch.randn(b, 1, 3, n * h, device=device, dtype=dtype)
                    q, k, v = qkv.unbind(dim=2)

                    past_kv = torch.rand(b, s_kv, 2, n, h, device=device, dtype=dtype)
                    past_k, past_v = past_kv.unbind(dim=2)

                    input_dict_gqa = {
                        "query": q.contiguous(),
                        "key": k.contiguous(),
                        "value": v.contiguous(),
                        "past_key": past_k.contiguous(),
                        "past_value": past_v.contiguous(),
                        "past_sequence_length": torch.tensor([sp], device=device, dtype=torch.int32),
                    }

                    input_dict_dmmha = {
                        "query": q.contiguous(),
                        "key": k.contiguous(),
                        "value": v.contiguous(),
                        "past_key": past_k.transpose(1, 2).contiguous(),
                        "past_value": past_v.transpose(1, 2).contiguous(),
                        "past_sequence_length": torch.tensor([sp], device=device, dtype=torch.int32),
                    }

                    average_gqa_latency = benchmark_op(gqa_session, input_dict_gqa, repeats)
                    average_dmmha_latency = benchmark_op(dmmha_session, input_dict_dmmha, repeats)

                    del gqa_session
                    del dmmha_session

                    # compute TFLOPS per second
                    gqa_speed = tflops_per_second(flops(b, s_kv, h, n), average_gqa_latency)
                    print(f"gqa\t{b}\t{s_kv}\t{n}\t{h}\t{average_gqa_latency * 1000:.2f}\t{gqa_speed:.2f}")
                    dmmha_speed = tflops_per_second(flops(b, s_kv, h, n), average_dmmha_latency)
                    print(f"dmmha\t{b}\t{s_kv}\t{n}\t{h}\t{average_dmmha_latency * 1000:.2f}\t{dmmha_speed:.2f}")
                    print("---------")
                    if average_dmmha_latency > 10 * average_gqa_latency:
                        continue
                    num_trials += 1
                    mean_gqa_lat += average_gqa_latency
                    mean_dmmha_lat += average_dmmha_latency
    mean_gqa_lat /= num_trials
    mean_dmmha_lat /= num_trials
    print(f"average gqa latency:\t{mean_gqa_lat}")
    print(f"average dmmha latency:\t{mean_dmmha_lat}")


if __name__ == "__main__":
    run_tflops_test(enable_cuda_graph=False)

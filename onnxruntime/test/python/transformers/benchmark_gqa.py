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

from onnxruntime import InferenceSession, OrtValue, SessionOptions


class InputFormats:
    QKV_BSNH = 0
    QKV_BNSH = 1


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0

    def __init__(self, b, s, s2, sp, n, n2, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.past_sequence_length = sp
        self.num_heads = n
        self.kv_num_heads = n2
        self.head_size = h


def create_group_query_attention_graph_past(
    config, causal=False, past_kv_format=InputFormats.QKV_BSNH, share_buffer=True
):
    past_kv_seqlen = config.kv_sequence_length if share_buffer else config.past_sequence_length
    present_kv_seqlen = (
        config.kv_sequence_length if share_buffer else config.past_sequence_length + config.sequence_length
    )
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "past_sequence_length" if share_buffer else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            unidirectional=1 if causal else 0,
            is_past_bsnh=1 if past_kv_format == InputFormats.QKV_BSNH else 0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
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
                config.sequence_length,
                config.kv_num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.sequence_length,
                config.kv_num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == InputFormats.QKV_BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == InputFormats.QKV_BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == InputFormats.QKV_BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == InputFormats.QKV_BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
    ]
    if share_buffer:
        graph_input += [
            helper.make_tensor_value_info(
                "past_sequence_length",
                TensorProto.INT32,
                [1],
            )
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == InputFormats.QKV_BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == InputFormats.QKV_BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == InputFormats.QKV_BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == InputFormats.QKV_BSNH else present_kv_seqlen,
                config.head_size,
            ],
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


def create_gqa_session(
    config: Config,
    causal: bool = False,
    past_format=InputFormats.QKV_BSNH,
    share_buffer: bool = True,
) -> InferenceSession:
    onnx_model_str = create_group_query_attention_graph_past(config, causal, past_format, share_buffer)
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
    return ort_session


def bind_io(io_binding, input_dict, device, share_buffer=True):
    io_binding.bind_cpu_input("query", input_dict["query"])
    io_binding.bind_cpu_input("key", input_dict["key"])
    io_binding.bind_cpu_input("value", input_dict["value"])
    io_binding.bind_input(
        "past_key", "cuda", 0, "float16", input_dict["past_key"].shape(), input_dict["past_key"].data_ptr()
    )
    io_binding.bind_input(
        "past_value",
        "cuda",
        0,
        "float16",
        input_dict["past_value"].shape(),
        input_dict["past_value"].data_ptr(),
    )
    io_binding.bind_output("output")
    if share_buffer:
        io_binding.bind_cpu_input("past_sequence_length", input_dict["past_sequence_length"])
        io_binding.bind_output(
            "present_key",
            device_type="cuda",
            device_id=device,
            element_type="float16",
            shape=input_dict["past_key"].shape(),
            buffer_ptr=input_dict["past_key"].data_ptr(),
        )
        io_binding.bind_output(
            "present_value",
            device_type="cuda",
            device_id=device,
            element_type="float16",
            shape=input_dict["past_value"].shape(),
            buffer_ptr=input_dict["past_value"].data_ptr(),
        )
    else:
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")


def measure_latency(ort_session, io_binding):
    start = time.time()
    _ = ort_session.run_with_iobinding(io_binding)
    end = time.time()
    return end - start


def flops(batch, q_seqlen, kv_seqlen, head_size, num_heads):
    return 4 * batch * q_seqlen * kv_seqlen * num_heads * head_size


def tflops_per_second(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def benchmark_op(session, io_binding, repeats=100):
    # warm up session
    _ = measure_latency(session, io_binding)

    latency_list = []
    for _ in range(repeats):
        latency = measure_latency(session, io_binding)
        latency_list.append(latency)
    return statistics.mean(latency_list)


def run_tflops_test(dtype=torch.float16, repeats: int = 100):
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    print("---- GQA BSNH vs GQA BNSH ----")
    print("op\tbatch\ts_kv\theads\th_dim\tms\tTFLOPS")
    mean_bsnh_lat = 0
    mean_bnsh_lat = 0
    num_trials = 0
    share_buffer = True
    random.seed(69)
    for b in [1, 3, 8, 16]:
        for s_q, s_kv in [(1, 128), (128, 256), (512, 512), (128, 1024), (1, 2048)]:
            for n_q, n_kv in [(8, 8), (16, 8), (32, 32), (12, 3), (128, 64)]:
                for h in [32, 64, 128]:
                    sp = random.randint(1, s_kv - 1) if s_kv - 1 > 0 else 0
                    config = Config(b, s_q, s_kv, sp, n_q, n_kv, h)

                    bsnh_session = create_gqa_session(
                        config,
                        causal=False,
                        past_format=InputFormats.QKV_BSNH,
                        share_buffer=share_buffer,
                    )
                    bnsh_session = create_gqa_session(
                        config,
                        causal=False,
                        past_format=InputFormats.QKV_BNSH,
                        share_buffer=share_buffer,
                    )

                    q = torch.randn(b, s_q, n_q * h, device=device, dtype=dtype)
                    kv = torch.randn(b, s_q, 2, n_kv * h, device=device, dtype=dtype)
                    k, v = kv.unbind(dim=2)

                    past_kv = torch.rand(b, s_kv if share_buffer else sp, 2, n_kv, h, device=device, dtype=dtype)
                    past_k, past_v = past_kv.unbind(dim=2)

                    input_dict_bsnh = {
                        "query": q.detach().cpu().numpy(),
                        "key": k.detach().cpu().numpy(),
                        "value": v.detach().cpu().numpy(),
                        "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cuda", device_id),
                        "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cuda", device_id),
                    }
                    input_dict_bnsh = {
                        "query": q.detach().cpu().numpy(),
                        "key": k.detach().cpu().numpy(),
                        "value": v.detach().cpu().numpy(),
                        "past_key": OrtValue.ortvalue_from_numpy(
                            past_k.transpose(1, 2).detach().cpu().numpy(), "cuda", 0
                        ),
                        "past_value": OrtValue.ortvalue_from_numpy(
                            past_v.transpose(1, 2).detach().cpu().numpy(), "cuda", 0
                        ),
                    }
                    if share_buffer:
                        input_dict_bsnh["past_sequence_length"] = (
                            torch.tensor([sp], device="cuda", dtype=torch.int32).detach().cpu().numpy()
                        )
                        input_dict_bnsh["past_sequence_length"] = (
                            torch.tensor([sp], device="cuda", dtype=torch.int32).detach().cpu().numpy()
                        )

                    io_binding_bsnh = bsnh_session.io_binding()
                    io_binding_bnsh = bnsh_session.io_binding()
                    bind_io(io_binding_bsnh, input_dict_bsnh, device_id, share_buffer)
                    bind_io(io_binding_bnsh, input_dict_bnsh, device_id, share_buffer)
                    average_gqa_bsnh_latency = benchmark_op(bsnh_session, io_binding_bsnh, repeats)
                    average_gqa_bnsh_latency = benchmark_op(bnsh_session, io_binding_bnsh, repeats)

                    del bsnh_session
                    del bnsh_session

                    # compute TFLOPS per second
                    bsnh_speed = tflops_per_second(flops(b, s_q, s_kv, h, n_q), average_gqa_bsnh_latency)
                    print(f"bsnh\t{b}\t{s_kv}\t{n_q}\t{h}\t{average_gqa_bsnh_latency * 1000:.2f}\t{bsnh_speed:.2f}")
                    bnsh_speed = tflops_per_second(flops(b, s_q, s_kv, h, n_q), average_gqa_bnsh_latency)
                    print(f"bnsh\t{b}\t{s_kv}\t{n_q}\t{h}\t{average_gqa_bnsh_latency * 1000:.2f}\t{bnsh_speed:.2f}")
                    print("---------")
                    if average_gqa_bsnh_latency > 10 * average_gqa_bnsh_latency:
                        continue
                    num_trials += 1
                    mean_bsnh_lat += average_gqa_bsnh_latency
                    mean_bnsh_lat += average_gqa_bnsh_latency
    mean_bsnh_lat /= num_trials
    mean_bnsh_lat /= num_trials
    print(f"average bsnh latency:\t{mean_bsnh_lat}")
    print(f"average bnsh latency:\t{mean_bnsh_lat}")


if __name__ == "__main__":
    run_tflops_test()

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
import platform
import statistics
import time
from typing import List, Optional

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, get_available_providers
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
        softmax_scale: float = 0.0,
        provider="CPUExecutionProvider",
        device: Optional[torch.device] = None,
        enable_cuda_graph: bool = False,
        dtype=torch.float,
        use_kv_cache: bool = False,
        share_past_present_buffer: bool = False,
        input_format: int = InputFormats.Q_K_V_BSNH_BSNH_BSNH,
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
        self.softmax_scale = softmax_scale or (1.0 / (head_size**0.5))

        self.use_kv_cache = use_kv_cache
        if not use_kv_cache:
            assert past_sequence_length == 0
        else:
            assert self.kv_sequence_length == self.sequence_length

        if input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
            # cross attention does not have past state
            assert not use_kv_cache

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

    def __repr__(self):
        return (
            f"MultiHeadAttentionConfig(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"num_heads={self.num_heads}, head_size={self.head_size}, "
            f"kv_sequence_length={self.kv_sequence_length}, past_sequence_length={self.past_sequence_length}, "
            f"max_cache_sequence_length={self.max_cache_sequence_length},"
            f"causal={self.causal}), softmax_scale={self.softmax_scale}, use_kv_cache={self.use_kv_cache}, "
            f"share_past_present_buffer={self.share_past_present_buffer}, "
            f"provider={self.provider}, device={self.device}, enable_cuda_graph={self.enable_cuda_graph}, "
            f"dtype={self.dtype}, input_format={InputFormats.input_format_str(self.input_format)}"
        )

    def shape_dict(self, input_format=None):
        input_format = input_format or self.input_format
        if input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
            # cross attention does not have past state
            return {
                "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "key": (self.batch_size, self.num_heads, self.sequence_length, self.head_size),
                "value": (self.batch_size, self.num_heads, self.sequence_length, self.head_size),
                "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            }

        if self.use_kv_cache:
            shapes = {
                "past_key": (self.batch_size, self.num_heads, self.past_buffer_length, self.head_size),
                "past_value": (self.batch_size, self.num_heads, self.past_buffer_length, self.head_size),
                "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                "present_key": (self.batch_size, self.num_heads, self.present_buffer_length, self.head_size),
                "present_value": (self.batch_size, self.num_heads, self.present_buffer_length, self.head_size),
            }
        else:
            shapes = {
                "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            }

        if input_format == InputFormats.QKV_BSN3H:
            shapes.update({"query": (self.batch_size, self.sequence_length, self.num_heads, 3, self.head_size)})
        elif input_format == InputFormats.Q_KV_BSNH_BSN2H:
            shapes.update(
                {
                    "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                    "key": (self.batch_size, self.sequence_length, self.num_heads, 2, self.head_size),
                }
            )
        else:  # input_format == InputFormats.Q_K_V_BSNH_BSNH_BSNH
            shapes.update(
                {
                    "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                    "key": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                    "value": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
                }
            )
        return shapes

    def random_inputs(self, seed: int = 123):
        device = self.device
        dtype = self.dtype

        shape_dict = self.shape_dict()

        if seed > 0:
            torch.manual_seed(seed)

        shape = (self.batch_size, self.sequence_length, self.num_heads, self.head_size)
        q = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)
        k = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)
        v = torch.empty(shape, device=device, dtype=dtype).normal_(mean=0, std=0.1)
        k_bnsh = k.transpose(1, 2)
        v_bnsh = v.transpose(1, 2)

        if self.input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
            return {
                "query": q.reshape(shape_dict["query"]),
                "key": k_bnsh.contiguous(),
                "value": v_bnsh.contiguous(),
            }

        feeds = {}
        if self.use_kv_cache:
            feeds.update(
                {
                    "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(
                        mean=0, std=0.1
                    ),
                    "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(
                        mean=0, std=0.1
                    ),
                }
            )

        if self.input_format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
            feeds.update(
                {
                    "query": q.reshape(shape_dict["query"]),
                    "key": k.reshape(shape_dict["key"]),
                    "value": v.reshape(shape_dict["value"]),
                }
            )
        elif self.input_format == InputFormats.QKV_BSN3H:
            query = q.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            key = k.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            value = v.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            feeds["query"] = torch.dstack((query, key, value)).reshape(shape_dict["query"]).contiguous()
        elif self.input_format == InputFormats.Q_KV_BSNH_BSN2H:
            key = k.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            value = v.view(self.batch_size * self.sequence_length, self.num_heads, self.head_size)
            feeds["query"] = q.reshape(shape_dict["query"])
            feeds["key"] = torch.dstack((key, value)).reshape(shape_dict["key"]).contiguous()

        return feeds

    def get_input_output_names(self):
        if self.input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
            return ["query", "key"], ["output"]

        if self.input_format == InputFormats.QKV_BSN3H:
            inputs, outputs = ["query"], ["output"]
        elif self.input_format == InputFormats.Q_KV_BSNH_BSN2H:
            inputs, outputs = ["query", "key"], ["output"]
        else:
            inputs, outputs = ["query", "key", "value"], ["output"]

        if self.use_kv_cache:
            return [*inputs, "past_key", "past_value"], [*outputs, "present_key", "present_value"]
        else:
            return inputs, outputs


def fill_optional_mha_inputs(input_names):
    inputs = ["query", "key", "value", "bias", "key_padding_mask", "relative_position_bias", "past_key", "past_value"]
    return input_names[:-2] + [""] * (len(inputs) - len(input_names)) + input_names[-2:]


def create_multi_head_attention_onnx_model(config: MultiHeadAttentionConfig):
    input_names, output_names = config.get_input_output_names()

    float_type = TensorProto.FLOAT16 if config.dtype == torch.float16 else TensorProto.FLOAT
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            fill_optional_mha_inputs(input_names) if config.use_kv_cache else input_names,
            output_names,
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            unidirectional=int(config.causal),
            scale=config.softmax_scale,
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.shape_dict()
    inputs = [
        helper.make_tensor_value_info(input_name, float_type, list(shape_dict[input_name]))
        for input_name in input_names
    ]

    outputs = [
        helper.make_tensor_value_info(output_name, float_type, list(shape_dict[output_name]))
        for output_name in output_names
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        inputs,
        outputs,
    )

    model = helper.make_model(graph)

    return model.SerializeToString()


def create_session(
    config: MultiHeadAttentionConfig,
) -> CudaSession:
    onnx_model_str = create_multi_head_attention_onnx_model(config)

    if config.provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device() if isinstance(config.device, str) else config.device.index
        provider_options = CudaSession.get_cuda_provider_options(device_id, config.enable_cuda_graph)
        providers = [(config.provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    ort_session = InferenceSession(onnx_model_str, providers=providers)
    cuda_session = CudaSession(ort_session, config.device, config.enable_cuda_graph)
    shape_dict = config.shape_dict()
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


class OrtMultiHeadAttention:
    """A wrapper of ORT MultiHeadAttention to test relevance and performance."""

    def __init__(
        self,
        config: MultiHeadAttentionConfig,
    ):
        self.ort_session = create_session(config)
        self.feed_dict = config.random_inputs()

    def infer(self):
        return self.ort_session.infer(self.feed_dict)


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


def flops(batch, sequence_length, head_size, num_heads, causal):
    return 4 * batch * sequence_length**2 * num_heads * head_size // (2 if causal else 1)


def tflops_per_second(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def get_gpu_kernel_name(config: MultiHeadAttentionConfig) -> str:
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


def get_cpu_kernel_name() -> str:
    if os.getenv("ORT_DISABLE_FLASH_ATTENTION") != "1":
        return "CPU:Flash"
    return "CPU:Unfused"


def run_tflops_test(use_gpu: bool = True, enable_cuda_graph: bool = False, repeats: int = 100):
    if use_gpu:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH, InputFormats.Q_KV_BSNH_BSN2H, InputFormats.QKV_BSN3H]
        provider = "CUDAExecutionProvider"
        print(f"enable_cuda_graph={enable_cuda_graph}")
    else:
        device_id = 0
        device = torch.device("cpu")
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH]
        enable_cuda_graph = False
        provider = "CPUExecutionProvider"

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
            (1, 128, 0, 32, 128, True),
            (1, 256, 0, 32, 128, True),
            (1, 512, 0, 32, 128, True),
            (1, 1024, 0, 32, 128, True),
            (1, 2048, 0, 32, 128, True),
        ]

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

    print("\nformat\tcausal\tbatch\tseqlen\theads\th_dim\tms\tTFLOPS\tkernel")
    causal = False

    for input_format in formats:
        for batch_size, sequence_length, past_sequence_length, num_heads, head_size, enable_unfused in configs:
            for use_kv_cache in [False]:
                config = MultiHeadAttentionConfig(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    num_heads=num_heads,
                    head_size=head_size,
                    causal=True,
                    use_kv_cache=use_kv_cache,
                    past_sequence_length=past_sequence_length,
                    max_cache_sequence_length=None,
                    kv_sequence_length=None,
                    provider=provider,
                    enable_cuda_graph=enable_cuda_graph,
                    device=device,
                    dtype=torch.float16 if use_gpu else torch.float,
                    share_past_present_buffer=False,
                    input_format=input_format,
                )

                session = create_session(config)

                if use_gpu:
                    kernel = get_gpu_kernel_name(config)
                else:
                    kernel = get_cpu_kernel_name()

                if kernel == "Unfused":
                    # Skip large sequence length for Unfused kernel to avoid OOM.
                    if not enable_unfused:
                        continue

                    # Unfused kernel does not support packed QKV or packed KV formats.
                    if input_format not in [InputFormats.Q_K_V_BSNH_BSNH_BSNH]:
                        continue

                input_dict = config.random_inputs()

                # warm up session
                _ = measure_latency(session, input_dict)

                latency_list = []
                for _ in range(repeats):
                    latency = measure_latency(session, input_dict)
                    latency_list.append(latency)
                average_latency = statistics.mean(latency_list)

                del session

                # compute TFLOPS per second
                speed = tflops_per_second(
                    flops(batch_size, sequence_length, head_size, num_heads, causal), average_latency
                )

                format = InputFormats.input_format_str(input_format)
                print(
                    f"{format}\t{causal}\t{batch_size}\t{sequence_length}\t{num_heads}\t{head_size}\t{average_latency * 1000:.2f}\t{speed:.2f}\t{kernel}"
                )


def plot_prompt_performance(
    sm: int,
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
            causal=True,
            past_sequence_length=0,
            kv_sequence_length=sequence_length if input_format == InputFormats.get_name_list()[-1] else None,
            max_cache_sequence_length=max_seq_len,
            provider="CUDAExecutionProvider",
            enable_cuda_graph=False,
            device=device,
            use_kv_cache=False,
            input_format=InputFormats.convert(input_format),
        )

        obj = OrtMultiHeadAttention(config)
        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_performance_test(sm: int):
    """
    Run performance tests for prompt and token generation.

    """
    configures = [
        (1, 32, 128, 8192, "TNLGv4"),
        (4, 32, 128, 8192, "TNLGv4"),
        (1, 12, 64, 1024, "BertBase"),
        (16, 12, 64, 1024, "BertBase"),
        (1, 16, 64, 1024, "BertLarge"),
        (8, 16, 64, 1024, "BertLarge"),
    ]

    for batch_size, num_heads, head_size, max_seq_len, model_name in configures:
        plot_prompt_performance(
            sm=sm,
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_len=max_seq_len,
            model_name=model_name,
        )


if __name__ == "__main__":
    if torch.cuda.is_available() and "CUDAExecutionProvider" in get_available_providers():
        # Test CUDA provider
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor

        if platform.system() == "Linux":
            s = torch.cuda.Stream()
            with torch.cuda.stream(s), torch.no_grad():
                run_performance_test(sm)

        run_tflops_test(use_gpu=True, enable_cuda_graph=True)

    # Test CPU provider
    run_tflops_test(use_gpu=False, enable_cuda_graph=False)

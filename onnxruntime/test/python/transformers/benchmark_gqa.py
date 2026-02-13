# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of GroupQueryAttention.
"""

from dataclasses import dataclass

import torch
from gqa_test_helper import GroupQueryAttentionConfig, OrtGroupQueryAttention

try:
    import triton
except ImportError:
    triton = None


@dataclass
class TestConfig:
    test_int4: bool = False
    test_int8: bool = False


def get_plot_algos(sm: int, local_window_size: int | None, config: TestConfig | None):
    # GQA with local windows only works in sm=8x
    if sm >= 80 and local_window_size:
        line_vals = ["ort_gqa", "ort_gqa_local", "ort_gqa_packed", "ort_gqa_local_packed"]
        line_names = ["ORT-GQA-Dense", "ORT-GQA-Local", "ORT-GQA-Dense-PackedQKV", "ORT-GQA-Local-PackedQKV"]
        styles = [("red", "solid"), ("yellow", "dashdot"), ("blue", "dashed"), ("green", "dotted")]
    else:
        line_vals = ["ort_gqa", "ort_gqa_packed"]
        line_names = ["ORT-GQA-Dense", "ORT-GQA-Dense-PackedQKV"]
        styles = [("red", "solid"), ("blue", "dashed")]

    # Add quantized variants if requested
    if sm >= 80 and config:
        quant_vals = ["ort_gqa_int4", "ort_gqa_int8"]
        quant_names = ["ORT-GQA-INT4", "ORT-GQA-INT8"]
        quant_styles = [("purple", "dotted"), ("orange", "dashdot")]
        if config.test_int4:
            line_vals.extend(quant_vals[:1])
            line_names.extend(quant_names[:1])
            styles.extend(quant_styles[:1])
        if config.test_int8:
            line_vals.extend(quant_vals[1:])
            line_names.extend(quant_names[1:])
            styles.extend(quant_styles[1:])

    return {
        "line_vals": line_vals,
        "line_names": line_names,
        "styles": styles,
    }


def plot_prompt_performance(
    sm: int,
    model_name: str,
    batch_size: int,
    num_heads: int,
    kv_num_heads: int,
    head_size: int,
    max_seq_len: int,
    local_window_size: int | None = None,
    use_smooth_softmax: bool = False,
    config: TestConfig | None = None,
    dtype: str = "float16",
):
    algos = get_plot_algos(sm, local_window_size, config)
    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(6, 17) if 2**i <= max_seq_len],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"prompt-sm{sm}-{model_name}-b{batch_size}-h{num_heads}_{kv_num_heads}x{head_size}-{dtype}",
            args={
                "batch_size": batch_size,
                "num_heads": num_heads,
                "kv_num_heads": kv_num_heads,
                "head_size": head_size,
                "local_window_size": local_window_size,
                "use_smooth_softmax": use_smooth_softmax,
                "dtype": dtype,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(
        provider: str,
        sequence_length: int,
        batch_size: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        local_window_size: int | None = None,
        use_smooth_softmax: bool = False,
        dtype: str = "float16",
        device="cuda",
    ):
        warmup = 15
        repeat = 100

        # Determine quantization settings based on provider
        k_quant_type = "NONE"
        v_quant_type = "NONE"
        kv_cache_type = "float16" if dtype == "float16" else "bfloat16"
        if "_int4" in provider:
            k_quant_type = v_quant_type = "PER_CHANNEL"
            kv_cache_type = "int4"
        elif "_int8" in provider:
            k_quant_type = v_quant_type = "PER_TENSOR"
            kv_cache_type = "int8"

        config: GroupQueryAttentionConfig = GroupQueryAttentionConfig(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_sequence_length=max_seq_len,
            past_sequence_length=0,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            local_window_size=local_window_size if provider in ["ort_gqa_local", "ort_gqa_local_packed"] else -1,
            use_smooth_softmax=use_smooth_softmax,
            device=device,
            dtype=torch.float16 if dtype == "float16" else torch.bfloat16,
            is_packed_qkv=provider in ["ort_gqa_packed", "ort_gqa_local_packed"],
            k_quant_type=k_quant_type,
            v_quant_type=v_quant_type,
            kv_cache_type=kv_cache_type,
        )

        obj = OrtGroupQueryAttention(config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def plot_token_performance(
    sm: int,
    model_name: str,
    batch_size: int,
    num_heads: int,
    kv_num_heads: int,
    head_size: int,
    max_seq_len: int,
    local_window_size: int | None = None,
    use_smooth_softmax: bool = False,
    config: TestConfig | None = None,
    dtype: str = "float16",
):
    algos = get_plot_algos(sm, local_window_size, config)
    configs = [
        triton.testing.Benchmark(
            x_names=["past_sequence_length"],
            x_vals=[2**i for i in range(6, 17) if 2**i < max_seq_len] + [max_seq_len - 1],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"token-sm{sm}-{model_name}-b{batch_size}-h{num_heads}_{kv_num_heads}_d{head_size}-{dtype}",
            args={
                "batch_size": batch_size,
                "num_heads": num_heads,
                "kv_num_heads": kv_num_heads,
                "head_size": head_size,
                "local_window_size": local_window_size,
                "use_smooth_softmax": use_smooth_softmax,
                "dtype": dtype,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(
        provider: str,
        past_sequence_length: int,
        batch_size: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        local_window_size: int | None = None,
        use_smooth_softmax: bool = False,
        dtype: str = "float16",
        device="cuda",
    ):
        warmup = 15
        repeat = 100

        # Determine quantization settings based on provider
        k_quant_type = "NONE"
        v_quant_type = "NONE"
        kv_cache_type = "float16" if dtype == "float16" else "bfloat16"
        share_kv_scale = False
        if "_int4" in provider:
            k_quant_type = v_quant_type = "PER_CHANNEL"
            kv_cache_type = "int4"
        elif "_int8" in provider:
            k_quant_type = v_quant_type = "PER_TENSOR"
            kv_cache_type = "int8"
            share_kv_scale = True  # XQA requires shared scale

        config: GroupQueryAttentionConfig = GroupQueryAttentionConfig(
            batch_size=batch_size,
            sequence_length=1,
            max_sequence_length=max_seq_len,
            past_sequence_length=past_sequence_length,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            local_window_size=local_window_size if provider in ["ort_gqa_local", "ort_gqa_local_packed"] else -1,
            do_rotary=True,  # Most models use rotary positional embeddings
            is_packed_qkv=provider in ["ort_gqa_packed", "ort_gqa_local_packed"],
            use_smooth_softmax=use_smooth_softmax,
            device=device,
            dtype=torch.float16 if dtype == "float16" else torch.bfloat16,
            k_quant_type=k_quant_type,
            v_quant_type=v_quant_type,
            kv_cache_type=kv_cache_type,
            share_kv_scale=share_kv_scale,
        )

        obj = OrtGroupQueryAttention(config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_performance_test(
    sm: int, fast: bool = False, config: TestConfig | None = None, dtype: str = "float16", is_prompt: bool = True
):
    """
    Run performance tests for prompt and token generation.

    """
    device_id = torch.cuda.current_device()
    memory_in_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024 * 1024)

    # Note: some models use bf16.
    # We use fp16/bf16 for all models in this test.
    configures = [
        (32, 128, 8, 8192, None, "Llama3-8B"),
        (64, 128, 8, 8192, None, "Llama3-70B"),
        (32, 128, 8, 32768, 4096, "Mistral-7B-v0.1"),
        (48, 128, 8, 65536, None, "Mixtral-8x22B-v0.1"),
        (32, 96, 32, 131072, None, "Phi-3-mini-128k"),
        (32, 128, 8, 131072, None, "Phi-3-small-128k"),  # Sparsity is not used in this test
        (40, 128, 10, 131072, None, "Phi-3-medium-128K"),
    ]

    if fast:
        configures = configures[:1]
    batch_sizes = [1] if fast else [1, 4]
    smooth_softmax_options = [False] if fast else [False, True]

    # Reduce max sequence length when GPU memory is not enough.
    threshold = 131072 if memory_in_gb > 24 else 65536 if memory_in_gb > 12 else 32768

    for num_heads, head_size, kv_num_heads, max_seq_len, local_window_size, model_name in configures:
        for batch_size in batch_sizes:
            for use_smooth_softmax in smooth_softmax_options:
                if is_prompt:
                    plot_prompt_performance(
                        sm=sm,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        head_size=head_size,
                        max_seq_len=min(threshold, max_seq_len),
                        local_window_size=local_window_size,
                        use_smooth_softmax=use_smooth_softmax,
                        model_name=model_name,
                        config=config,
                        dtype=dtype,
                    )
                else:
                    plot_token_performance(
                        sm=sm,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        head_size=head_size,
                        max_seq_len=min(threshold, max_seq_len),
                        local_window_size=local_window_size,
                        use_smooth_softmax=use_smooth_softmax,
                        model_name=model_name,
                        config=config,
                        dtype=dtype,
                    )


if __name__ == "__main__":
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.no_grad():
        config = TestConfig(test_int4=False, test_int8=True)
        run_performance_test(sm, fast=True, config=config, dtype="float16", is_prompt=True)
        run_performance_test(sm, fast=True, config=config, dtype="float16", is_prompt=False)
        # run_performance_test(sm, fast=True, config=config, dtype="bfloat16", is_prompt=True)
        # run_performance_test(sm, fast=True, config=config, dtype="bfloat16", is_prompt=False)

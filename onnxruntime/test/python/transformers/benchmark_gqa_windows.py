import argparse
import os
import time
from typing import Optional

import torch
from test_sparse_attention import GroupQueryAttentionConfig, OrtGroupQueryAttention


def save_results(results, filename):
    import pandas as pd

    df = pd.DataFrame(
        results,
        columns=[
            "Inference Interval (ms)",
            "Throughput (samples/second)",
            "Batch Size",
            "Max Sequence Length",
            "Sequence Length",
            "Past Sequence Length",
            "Smooth Softmax",
            "Model Name",
        ],
    )
    # df = df.transpose()  # This line swaps the rows and columns
    df.to_csv(filename, header=True, index=False)
    print(f"Results saved in {filename}!")


def benchmark(
    batch_size: int,
    num_heads: int,
    kv_num_heads: int,
    head_size: int,
    max_seq_len: int,
    sequence_length: int = 1,
    past_sequence_length: int = 0,
    local_window_size: Optional[int] = None,
    use_smooth_softmax: bool = False,
    model_name: str = "Llama3-8B",
):
    warmup = 15
    repeat = 100

    config: GroupQueryAttentionConfig = GroupQueryAttentionConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_sequence_length=max_seq_len,
        past_sequence_length=past_sequence_length,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        local_window_size=local_window_size if local_window_size else -1,
        use_smooth_softmax=use_smooth_softmax,
        do_rotary=True,  # Most models use rotary positional embeddings
        is_packed_qkv=model_name in ["Phi-3-mini-128k", "Phi-3-small-128k"],
        device="cuda",
    )

    obj = OrtGroupQueryAttention(config)

    for _ in range(warmup):
        obj.infer()

    intervals = []
    for _ in range(repeat):
        infer_start = time.perf_counter()
        obj.infer()
        infer_interval = time.perf_counter() - infer_start
        intervals.append(infer_interval)
    avg_infer_interval = sum(intervals) / len(intervals)
    avg_infer_interval_ms = avg_infer_interval * 1000
    print(f"Average inference interval: {avg_infer_interval_ms:.6f} milliseconds")
    avg_throughput = batch_size / avg_infer_interval
    print(f"Average throughput: {avg_throughput:.6f} samples/second")

    return [avg_infer_interval_ms, avg_throughput]


def run_performance_tests(args):
    device_id = torch.cuda.current_device()
    memory_in_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024 * 1024)

    configures = [
        (32, 128, 8, 8192, None, "Llama3-8B"),
        (64, 128, 8, 8192, None, "Llama3-70B"),
        (48, 128, 8, 32768, None, "Mixtral-8x22B-v0.1"),
        (32, 96, 32, 131072, None, "Phi-3-mini-128k"),
        (32, 128, 8, 65536, None, "Phi-3-small-128k"),  # Sparsity is not used in this test
        (40, 128, 10, 32768, None, "Phi-3-medium-128K"),
    ]
    if args.kernel == "flash_attention":
        configures.append((32, 128, 8, 32768, 4096, "Mistral-7B-v0.1"))

    # Reduce max sequence length when GPU memory is not enough.
    threshold = 131072 if memory_in_gb > 24 else 65536 if memory_in_gb > 12 else 32768

    smooth_softmax = args.use_smooth_softmax

    all_metrics = []
    for num_heads, head_size, kv_num_heads, max_seq_len, local_window_size, model_name in configures:
        prompt_metrics_model = []
        token_metrics_model = []
        for batch_size in [1, 4]:
            # Benchmark prompt
            for sequence_length in [
                1,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
            ]:
                if sequence_length >= min(threshold, max_seq_len):
                    continue
                print(
                    f"Prompt: batch_size={batch_size}, num_heads={num_heads}, kv_num_heads={kv_num_heads}, head_size={head_size}, sequence_length={sequence_length}, max_seq_len={max_seq_len}, local_window_size={local_window_size}, model_name={model_name}"
                )
                metrics = benchmark(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    head_size=head_size,
                    sequence_length=sequence_length,
                    max_seq_len=min(threshold, max_seq_len),
                    local_window_size=local_window_size,
                    use_smooth_softmax=smooth_softmax,
                    model_name=model_name,
                )
                metrics = [*metrics, batch_size, max_seq_len, sequence_length, 0, model_name]
                prompt_metrics_model.append(metrics)
                all_metrics.append(metrics)
            # Benchmark token
            for past_sequence_length in [
                0,
                3,
                7,
                15,
                31,
                63,
                127,
                255,
                511,
                1023,
                2047,
                4095,
                8191,
                16383,
                32767,
                65535,
                131071,
            ]:
                if past_sequence_length >= min(threshold, max_seq_len):
                    continue
                print(
                    f"Token: batch_size={batch_size}, num_heads={num_heads}, kv_num_heads={kv_num_heads}, head_size={head_size}, past_sequence_length={past_sequence_length}, max_seq_len={max_seq_len}, local_window_size={local_window_size}, model_name={model_name}"
                )
                metrics = benchmark(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    head_size=head_size,
                    past_sequence_length=past_sequence_length,
                    max_seq_len=min(threshold, max_seq_len),
                    local_window_size=local_window_size,
                    use_smooth_softmax=smooth_softmax,
                    model_name=model_name,
                )
                metrics = [*metrics, batch_size, max_seq_len, 1, past_sequence_length, smooth_softmax, model_name]
                token_metrics_model.append(metrics)
                all_metrics.append(metrics)
        # Calculate average inference interval and throughput for each model
        avg_prompt_infer_interval = sum([metrics[0] for metrics in prompt_metrics_model]) / len(prompt_metrics_model)
        avg_prompt_throughput = sum([metrics[1] for metrics in prompt_metrics_model]) / len(prompt_metrics_model)
        avg_token_infer_interval = sum([metrics[0] for metrics in token_metrics_model]) / len(token_metrics_model)
        avg_token_throughput = sum([metrics[1] for metrics in token_metrics_model]) / len(token_metrics_model)
        print(f"Average {model_name} prompt inference interval: {avg_prompt_infer_interval:.6f} milliseconds")
        print(f"Average {model_name} prompt throughput: {avg_prompt_throughput:.6f} samples/second")
        print(f"Average {model_name} token inference interval: {avg_token_infer_interval:.6f} milliseconds")
        print(f"Average {model_name} token throughput: {avg_token_throughput:.6f} samples/second")
        all_metrics.append(
            [avg_prompt_infer_interval, avg_prompt_throughput, 0, max_seq_len, 0, 0, model_name + " (Average Prompt)"]
        )
        all_metrics.append(
            [avg_token_infer_interval, avg_token_throughput, 0, max_seq_len, 0, 0, model_name + " (Average Token)"]
        )

    save_results(all_metrics, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file name or path (with .csv extension)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        default="flash_attention",
        help="GQA Kernel to use for benchmarking. Options: flash_attention, memory_efficient",
    )

    parser.add_argument(
        "--use_smooth_softmax",
        required=False,
        action="store_true",
        help="test smooth softmax",
    )
    parser.set_defaults(use_smooth_softmax=False)

    args = parser.parse_args()

    if args.kernel == "memory_efficient":
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
    else:
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.no_grad():
        run_performance_tests(args)

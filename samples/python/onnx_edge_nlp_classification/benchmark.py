#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Performance benchmarking script for edge AI sequence classification.
Measures model file size, session initialization time, and inference execution latency distributions.
"""

import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def run_benchmark(model_path: str, tokenizer_id: str, runs: int) -> None:
    """
    Benchmarks model loading, initialization, and execution performance.
    
    Args:
        model_path: Path to the exported ONNX model.
        tokenizer_id: Hugging Face model ID for tokenizer.
        runs: Number of inference iterations to run for stats.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please run export_model.py first.", file=sys.stderr)
        sys.exit(1)

    # 1. Model size footprint
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model File Size: {file_size_mb:.2f} MB")

    # 2. Tokenization setup
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    sample_text = "I feel stressed and ignored today. Hopefully tomorrow will be better."
    inputs = tokenizer(sample_text, truncation=True, max_length=128)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

    # 3. Benchmark Session Initialization (Cold Startup Latency)
    print("Measuring ONNX Runtime Session initialization latency...")
    start_init = time.perf_counter()
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Failed to initialize ORT Session: {e}", file=sys.stderr)
        sys.exit(1)
    init_time_ms = (time.perf_counter() - start_init) * 1000
    print(f"Session Initialization Latency: {init_time_ms:.2f} ms")

    # 4. Warm-up runs (to prime session execution buffers)
    print("\nPerforming warm-up runs...")
    for _ in range(5):
        session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

    # 5. Iterated execution benchmarks
    print(f"Executing {runs} inference iterations...")
    latencies = []
    
    for i in range(runs):
        start_run = time.perf_counter()
        session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        run_time_ms = (time.perf_counter() - start_run) * 1000
        latencies.append(run_time_ms)

    latencies = np.array(latencies)
    
    # 6. Reporting statistical distribution
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)

    print("\n--- Latency Performance Report (CPU) ---")
    print(f"Average Inference Latency: {mean_latency:.2f} ms")
    print(f"Median Inference Latency:  {median_latency:.2f} ms")
    print(f"50th Percentile (p50):    {p50:.2f} ms")
    print(f"90th Percentile (p90):    {p90:.2f} ms")
    print(f"99th Percentile (p99):    {p99:.2f} ms")
    print("----------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime sequence classification performance.")
    parser.add_argument(
        "--model",
        type=str,
        default="model.onnx",
        help="Path to the ONNX model file (default: model.onnx)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bhadresh-savani/distilbert-base-uncased-emotion",
        help="Tokenizer ID from Hugging Face (default: bhadresh-savani/distilbert-base-uncased-emotion)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of inference loops for statistical benchmarking (default: 100)"
    )
    args = parser.parse_args()

    run_benchmark(args.model, args.tokenizer, args.runs)


if __name__ == "__main__":
    main()

"""
ORT QDQ vs MatMulNBits Performance Investigation - Benchmark Script

This script benchmarks onnxruntime-genai models comparing:
1. MatMulNBits (QOperator format) - Scenario A
2. QDQ with fusion enabled (auto-fused to MatMulNBits) - Scenario B
3. QDQ with fusion disabled (pure DequantizeLinear→MatMul) - Scenario C

Usage:
    python benchmark_runner.py --config config.json
    python benchmark_runner.py --model-dir C:/models --output results.csv
"""

import onnxruntime_genai as og
import onnxruntime as ort
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    path: Path
    format_type: str  # "matmulnbits" or "qdq"
    bits: int
    block_size: int
    symmetric: bool
    base_model: str


@dataclass
class ExperimentConfig:
    """Configuration for the benchmark experiment."""

    # Prompt settings
    prompt_length: int = 128
    generation_length: int = 256
    max_length: int = 512

    # Thread configuration
    num_threads: int = 0  # 0 = auto

    # Benchmark iterations
    num_warmup: int = 3
    num_iterations: int = 10

    # Test scenarios
    test_scenarios: List[str] = field(default_factory=lambda: [
        "native",
        "qdq_fused",
        "qdq_unfused",
    ])

    # Generation settings (greedy for reproducibility)
    do_sample: bool = False

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./results"))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_name: str
    scenario: str
    format_type: str
    bits: int
    block_size: int
    symmetric: bool
    base_model: str

    # TTFT metrics (seconds)
    ttft_mean: float
    ttft_std: float
    ttft_min: float
    ttft_max: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float

    # TPS metrics
    tps_mean: float
    tps_std: float
    tps_min: float
    tps_max: float
    tps_p50: float
    tps_p95: float
    tps_p99: float

    # Additional
    prompt_length: int
    tokens_generated: int
    total_time_mean: float
    num_iterations: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "scenario": self.scenario,
            "format_type": self.format_type,
            "bits": self.bits,
            "block_size": self.block_size,
            "symmetric": self.symmetric,
            "base_model": self.base_model,
            "ttft_mean_ms": self.ttft_mean * 1000,
            "ttft_std_ms": self.ttft_std * 1000,
            "ttft_min_ms": self.ttft_min * 1000,
            "ttft_max_ms": self.ttft_max * 1000,
            "ttft_p50_ms": self.ttft_p50 * 1000,
            "ttft_p95_ms": self.ttft_p95 * 1000,
            "ttft_p99_ms": self.ttft_p99 * 1000,
            "tps_mean": self.tps_mean,
            "tps_std": self.tps_std,
            "tps_min": self.tps_min,
            "tps_max": self.tps_max,
            "tps_p50": self.tps_p50,
            "tps_p95": self.tps_p95,
            "tps_p99": self.tps_p99,
            "prompt_length": self.prompt_length,
            "tokens_generated": self.tokens_generated,
            "total_time_mean_s": self.total_time_mean,
            "num_iterations": self.num_iterations,
        }


# =============================================================================
# Test Prompts
# =============================================================================

TEST_PROMPTS = [
    """Explain the concept of machine learning in simple terms.
    Machine learning is a subset of artificial intelligence that enables computers to learn from data
    and make predictions or decisions without being explicitly programmed.""",

    """Write a Python function to calculate the Fibonacci sequence.
    The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones.""",

    """What are the key differences between TCP and UDP protocols?
    TCP and UDP are both transport layer protocols used for sending data over networks.""",
]


# =============================================================================
# Utility Functions
# =============================================================================

def set_thread_config(num_threads: int = 0):
    """Configure thread settings via environment variables."""
    if num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(num_threads)
        print(f"Thread configuration set to {num_threads} threads")
    else:
        for var in ["OMP_NUM_THREADS", "ORT_INTRA_OP_NUM_THREADS"]:
            if var in os.environ:
                del os.environ[var]
        import multiprocessing
        print(f"Using default thread configuration (cores: {multiprocessing.cpu_count()})")


def generate_prompt_tokens(tokenizer, target_length: int, base_prompt: str) -> List[int]:
    """Generate a prompt of approximately target_length tokens."""
    tokens = tokenizer.encode(base_prompt)

    while len(tokens) < target_length:
        base_prompt = base_prompt + " " + base_prompt
        tokens = tokenizer.encode(base_prompt)

    if len(tokens) > target_length:
        tokens = tokens[:target_length]

    return list(tokens)


def compute_statistics(values: List[float]) -> dict:
    """Compute comprehensive statistics."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# =============================================================================
# Core Benchmark Functions
# =============================================================================

def run_single_iteration(
    model: og.Model,
    prompt_tokens: List[int],
    generation_length: int,
    max_length: int,
    do_sample: bool = False,
) -> Tuple[float, float, int]:
    """Run a single benchmark iteration."""
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=do_sample,
        max_length=max_length,
        min_length=0,
    )
    generator = og.Generator(model, params)

    # Measure TTFT
    prompt_start = time.perf_counter()
    generator.append_tokens(prompt_tokens)
    generator.generate_next_token()
    first_token_time = time.perf_counter()

    ttft = first_token_time - prompt_start

    # Measure decode phase
    decode_start = time.perf_counter()
    tokens_decoded = 1

    while not generator.is_done() and tokens_decoded < generation_length:
        generator.generate_next_token()
        tokens_decoded += 1

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start

    tps = (tokens_decoded - 1) / decode_time if decode_time > 0 and tokens_decoded > 1 else 0.0

    del generator
    return ttft, tps, tokens_decoded


def run_benchmark_iterations(
    model_path: str,
    prompt_tokens: List[int],
    generation_length: int,
    max_length: int,
    num_warmup: int = 3,
    num_iterations: int = 10,
    do_sample: bool = False,
) -> Tuple[List[float], List[float], List[int]]:
    """Run benchmark with warmup and multiple iterations."""

    print(f"  Loading model from: {model_path}")
    model = og.Model(model_path)

    ttft_times = []
    tps_values = []
    tokens_generated_list = []

    # Warmup
    print(f"  Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = run_single_iteration(
            model, prompt_tokens, generation_length, max_length, do_sample
        )

    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    for i in tqdm(range(num_iterations), desc="  Iterations", leave=False):
        ttft, tps, tokens = run_single_iteration(
            model, prompt_tokens, generation_length, max_length, do_sample
        )
        ttft_times.append(ttft)
        tps_values.append(tps)
        tokens_generated_list.append(tokens)

    del model
    return ttft_times, tps_values, tokens_generated_list


def benchmark_model(
    model_config: ModelConfig,
    scenario: str,
    experiment_config: ExperimentConfig,
) -> Optional[BenchmarkResult]:
    """Benchmark a single model configuration."""

    # Check scenario applicability
    if scenario == "native" and model_config.format_type != "matmulnbits":
        return None
    if scenario in ["qdq_fused", "qdq_unfused"] and model_config.format_type != "qdq":
        return None

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_config.name}")
    print(f"Scenario: {scenario}")
    print(f"Path: {model_config.path}")
    print(f"{'='*60}")

    if not model_config.path.exists():
        print(f"  WARNING: Model path does not exist, skipping")
        return None

    if scenario == "qdq_unfused":
        print(f"  NOTE: QDQ unfused requires pre-processed model with fusion disabled")

    try:
        # Get tokenizer
        temp_model = og.Model(str(model_config.path))
        tokenizer = og.Tokenizer(temp_model)

        prompt_tokens = generate_prompt_tokens(
            tokenizer,
            experiment_config.prompt_length,
            TEST_PROMPTS[0]
        )
        print(f"  Prompt tokens: {len(prompt_tokens)}")

        del temp_model, tokenizer

        # Run benchmark
        ttft_times, tps_values, tokens_generated = run_benchmark_iterations(
            model_path=str(model_config.path),
            prompt_tokens=prompt_tokens,
            generation_length=experiment_config.generation_length,
            max_length=experiment_config.max_length,
            num_warmup=experiment_config.num_warmup,
            num_iterations=experiment_config.num_iterations,
            do_sample=experiment_config.do_sample,
        )

        # Compute statistics
        ttft_stats = compute_statistics(ttft_times)
        tps_stats = compute_statistics(tps_values)

        result = BenchmarkResult(
            model_name=model_config.name,
            scenario=scenario,
            format_type=model_config.format_type,
            bits=model_config.bits,
            block_size=model_config.block_size,
            symmetric=model_config.symmetric,
            base_model=model_config.base_model,
            ttft_mean=ttft_stats["mean"],
            ttft_std=ttft_stats["std"],
            ttft_min=ttft_stats["min"],
            ttft_max=ttft_stats["max"],
            ttft_p50=ttft_stats["p50"],
            ttft_p95=ttft_stats["p95"],
            ttft_p99=ttft_stats["p99"],
            tps_mean=tps_stats["mean"],
            tps_std=tps_stats["std"],
            tps_min=tps_stats["min"],
            tps_max=tps_stats["max"],
            tps_p50=tps_stats["p50"],
            tps_p95=tps_stats["p95"],
            tps_p99=tps_stats["p99"],
            prompt_length=len(prompt_tokens),
            tokens_generated=int(np.mean(tokens_generated)),
            total_time_mean=ttft_stats["mean"] + np.mean(tokens_generated) / tps_stats["mean"] if tps_stats["mean"] > 0 else 0,
            num_iterations=experiment_config.num_iterations,
        )

        print(f"\n  TTFT: {result.ttft_mean*1000:.2f} ± {result.ttft_std*1000:.2f} ms")
        print(f"  TPS:  {result.tps_mean:.2f} ± {result.tps_std:.2f} tokens/sec")

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_full_benchmark(
    model_configs: List[ModelConfig],
    experiment_config: ExperimentConfig,
) -> pd.DataFrame:
    """Run the complete benchmark."""

    set_thread_config(experiment_config.num_threads)
    experiment_config.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_configs = len(model_configs)

    print(f"\n{'#'*60}")
    print(f"# Starting Full Benchmark")
    print(f"# Models: {total_configs}")
    print(f"# Scenarios: {experiment_config.test_scenarios}")
    print(f"{'#'*60}\n")

    for idx, model_config in enumerate(model_configs, 1):
        print(f"\n[{idx}/{total_configs}] Processing: {model_config.name}")

        if model_config.format_type == "matmulnbits":
            applicable_scenarios = ["native"]
        else:
            applicable_scenarios = ["qdq_fused", "qdq_unfused"]

        applicable_scenarios = [
            s for s in applicable_scenarios
            if s in experiment_config.test_scenarios
        ]

        for scenario in applicable_scenarios:
            result = benchmark_model(model_config, scenario, experiment_config)

            if result is not None:
                all_results.append(result.to_dict())

                # Save intermediate
                intermediate_df = pd.DataFrame(all_results)
                intermediate_df.to_csv(
                    experiment_config.output_dir / "benchmark_results_intermediate.csv",
                    index=False
                )

    df = pd.DataFrame(all_results)

    if not df.empty:
        output_path = experiment_config.output_dir / "benchmark_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"Total results: {len(df)}")
        print(f"{'='*60}")

    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results(df: pd.DataFrame):
    """Analyze and summarize results."""
    if df.empty:
        print("No results to analyze")
        return

    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)

    # By scenario
    print("\n1. Summary by Scenario")
    print("-"*50)
    summary = df.groupby('scenario').agg({
        'ttft_mean_ms': ['mean', 'std'],
        'tps_mean': ['mean', 'std'],
    }).round(2)
    print(summary)

    # By base model
    print("\n2. Summary by Base Model")
    print("-"*50)
    model_summary = df.groupby(['base_model', 'scenario']).agg({
        'ttft_mean_ms': 'mean',
        'tps_mean': 'mean',
    }).round(2)
    print(model_summary)

    return summary


def compare_matmulnbits_vs_qdq(df: pd.DataFrame):
    """Compare MatMulNBits vs QDQ performance."""
    if df.empty:
        return

    print("\n" + "="*70)
    print("KEY COMPARISON: MatMulNBits (native) vs QDQ (unfused)")
    print("="*70)

    for base_model in df['base_model'].unique():
        model_df = df[df['base_model'] == base_model]

        native = model_df[model_df['scenario'] == 'native']
        qdq_unfused = model_df[model_df['scenario'] == 'qdq_unfused']

        print(f"\n{base_model}:")
        print("-" * 50)

        if native.empty or qdq_unfused.empty:
            print("  Insufficient data for comparison")
            continue

        native_ttft = native['ttft_mean_ms'].mean()
        qdq_ttft = qdq_unfused['ttft_mean_ms'].mean()
        ttft_diff = qdq_ttft - native_ttft
        ttft_pct = (ttft_diff / native_ttft) * 100 if native_ttft > 0 else 0

        native_tps = native['tps_mean'].mean()
        qdq_tps = qdq_unfused['tps_mean'].mean()
        tps_diff = native_tps - qdq_tps
        tps_pct = (tps_diff / native_tps) * 100 if native_tps > 0 else 0

        print(f"  TTFT: Native={native_ttft:.2f}ms, QDQ={qdq_ttft:.2f}ms, Diff={ttft_diff:+.2f}ms ({ttft_pct:+.1f}%)")
        print(f"  TPS:  Native={native_tps:.2f}, QDQ={qdq_tps:.2f}, Diff={tps_diff:+.2f} ({tps_pct:+.1f}%)")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ORT QDQ vs MatMulNBits Benchmark")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model-dir", type=str, help="Base directory for models")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--prompt-length", type=int, default=128, help="Prompt length in tokens")
    parser.add_argument("--generation-length", type=int, default=256, help="Tokens to generate")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads (0=auto)")
    parser.add_argument("--analyze-only", type=str, help="Path to existing results CSV to analyze")
    return parser.parse_args()


def main():
    args = parse_args()

    # Analyze existing results
    if args.analyze_only:
        df = pd.read_csv(args.analyze_only)
        analyze_results(df)
        compare_matmulnbits_vs_qdq(df)
        return

    # Create experiment config
    experiment_config = ExperimentConfig(
        prompt_length=args.prompt_length,
        generation_length=args.generation_length,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        num_threads=args.threads,
        output_dir=Path(args.output),
    )

    # Load model configs
    model_configs = []

    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
        for name, cfg in config_data.get("models", {}).items():
            model_configs.append(ModelConfig(
                name=name,
                path=Path(cfg["path"]),
                format_type=cfg["format"],
                bits=cfg["bits"],
                block_size=cfg["block_size"],
                symmetric=cfg["symmetric"],
                base_model=cfg["base_model"],
            ))
    elif args.model_dir:
        # Auto-discover models (simplified)
        model_dir = Path(args.model_dir)
        print(f"Auto-discovering models in: {model_dir}")
        # Add discovery logic here
    else:
        print("Please provide --config or --model-dir")
        return

    if not model_configs:
        print("No model configurations found")
        return

    # Run benchmark
    df = run_full_benchmark(model_configs, experiment_config)

    # Analyze
    if not df.empty:
        analyze_results(df)
        compare_matmulnbits_vs_qdq(df)


if __name__ == "__main__":
    main()

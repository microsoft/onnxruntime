"""
ONNX Model Latency Benchmark for QDQ Investigation

Measures raw inference latency by calling the ONNX model directly with dummy
inputs.

Approach:
    - Creates "prefill-like" inputs: empty past_key_values, multiple input_ids
    - Quantized MatMuls don't differentiate prefill vs decode - they only run on
      new tokens, so we measure with zero past and varying input lengths
    - No need to measure prefill/decode separately for this investigation

Usage:
    # Basic: benchmark with default sequence lengths (128, 256, 512, 1024)
    python benchmark.py -m model.onnx

    # Compare QDQ (unfused) vs MatMulNBits
    python benchmark.py -m model_qdq.onnx --disable-qdq-fusion
    python benchmark.py -m model_matmulnbits.onnx

    # For 2-bit quantized models, enable LUT GEMM
    python benchmark.py -m model_2bit.onnx --enable-lut-gemm

    # Custom sequence lengths and iterations
    python benchmark.py -m model.onnx -s 128 512 -i 50 -w 10

    # Save results to JSON
    python benchmark.py -m model.onnx -o results.json

    # Validate with onnxruntime_perf_test (optional cross-check)
    python benchmark.py -m model.onnx --perf-test

    # With profiling (generates Chrome tracing JSON)
    python benchmark.py -m model.onnx --perf-test --profile trace.json

References:
    - DQMatMulToMatMulNBits fusion: onnxruntime/core/optimizer/qdq_transformer/
      selectors_actions/qdq_selector_action_transformer.cc
    - LUT GEMM option: onnxruntime/core/session/onnxruntime_session_options_config_keys.h
      (kOrtSessionOptionsMlasLutGemm = "mlas.use_lut_gemm")
    - ORT default optimization level is ORT_ENABLE_ALL (99)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SEQ_LENGTHS = [128, 256, 512, 1024]
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 1


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    model_path: str
    seq_lengths: List[int] = field(default_factory=lambda: DEFAULT_SEQ_LENGTHS.copy())
    batch_size: int = DEFAULT_BATCH_SIZE
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERATIONS
    disable_qdq_fusion: bool = False
    enable_lut_gemm: bool = False  # For 2-bit quantized models
    output_file: Optional[str] = None
    perf_test_path: Optional[str] = None  # Optional: run onnxruntime_perf_test for validation
    profile_output: Optional[str] = None  # Profile output path for perf_test -p
    verbose: bool = False


@dataclass
class LatencyResult:
    """Results from a single sequence length benchmark."""
    seq_length: int
    batch_size: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_length": self.seq_length,
            "batch_size": self.batch_size,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "iterations": self.iterations,
        }


# =============================================================================
# Session Creation
# =============================================================================

def create_session(config: BenchmarkConfig) -> ort.InferenceSession:
    """
    Create an ONNX Runtime inference session.

    Notes:
        - ORT defaults to ORT_ENABLE_ALL optimization level
        - ORT auto-detects thread count when set to 0 (default)
        - disabled_optimizers prevents DequantizeLinear+MatMul -> MatMulNBits fusion
    """
    sess_options = ort.SessionOptions()

    # For 2-bit quantized models, enable LUT-based GEMM
    # See: onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    if config.enable_lut_gemm:
        sess_options.add_session_config_entry("mlas.use_lut_gemm", "1")
        print("  Enabled LUT GEMM for 2-bit quantization")

    # Disable DQ+MatMul -> MatMulNBits fusion for unfused QDQ comparison
    # Optimizer name from: onnxruntime/core/optimizer/qdq_transformer/
    #   selectors_actions/qdq_selector_action_transformer.cc
    disabled_optimizers = None
    if config.disable_qdq_fusion:
        disabled_optimizers = ["DQMatMulToMatMulNBits"]
        print("  Disabled DQMatMulToMatMulNBits fusion")

    session = ort.InferenceSession(
        config.model_path,
        sess_options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=disabled_optimizers,
    )

    return session


# =============================================================================
# Input Generation
# =============================================================================

def inspect_model_inputs(session: ort.InferenceSession) -> List[Dict[str, Any]]:
    """Get information about model inputs."""
    inputs = []
    for inp in session.get_inputs():
        inputs.append({
            "name": inp.name,
            "shape": inp.shape,
            "type": inp.type,
        })
    return inputs


def numpy_dtype_from_onnx_type(onnx_type: str) -> np.dtype:
    """Convert ONNX type string to numpy dtype."""
    type_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return type_map.get(onnx_type, np.float32)


def generate_dummy_input(
    input_info: Dict[str, Any],
    seq_length: int,
    batch_size: int,
    num_layers: int = 32,  # Default for most LLMs
    num_heads: int = 32,
    head_dim: int = 128,
) -> np.ndarray:
    """
    Generate dummy input data based on input name and shape.

    Handles common LLM input patterns:
    - input_ids: random token IDs
    - attention_mask: all ones
    - position_ids: sequential positions
    - past_key_values.*: zeros (empty cache for prefill-like mode)
    """
    name = input_info["name"]
    shape = input_info["shape"]
    dtype = numpy_dtype_from_onnx_type(input_info["type"])

    # Resolve dynamic dimensions
    resolved_shape = []
    for dim in shape:
        if isinstance(dim, str) or dim is None:
            # Dynamic dimension - need to infer based on name
            dim_lower = str(dim).lower() if dim else ""
            if "batch" in dim_lower:
                resolved_shape.append(batch_size)
            elif "seq" in dim_lower or "length" in dim_lower or "past" in dim_lower:
                # For past_key_values, use 0 for prefill mode
                if "past" in name.lower():
                    resolved_shape.append(0)  # Empty past for prefill
                else:
                    resolved_shape.append(seq_length)
            elif "head" in dim_lower:
                resolved_shape.append(num_heads)
            else:
                # Default to seq_length for unknown dynamic dims
                resolved_shape.append(seq_length)
        else:
            resolved_shape.append(dim)

    # Generate appropriate data based on input name
    name_lower = name.lower()

    if "input_ids" in name_lower:
        # Random token IDs (typical vocab size range)
        return np.random.randint(0, 32000, size=resolved_shape, dtype=dtype)

    elif "attention_mask" in name_lower:
        # All ones (attend to everything)
        return np.ones(resolved_shape, dtype=dtype)

    elif "position_ids" in name_lower:
        # Sequential positions
        positions = np.arange(seq_length, dtype=dtype)
        return np.broadcast_to(positions, resolved_shape).copy()

    elif "past" in name_lower or "cache" in name_lower:
        # Empty past key/values for prefill mode
        # Shape is typically [batch, num_heads, 0, head_dim] for empty cache
        return np.zeros(resolved_shape, dtype=dtype)

    else:
        # Default: random data
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 100, size=resolved_shape, dtype=dtype)
        else:
            return np.random.randn(*resolved_shape).astype(dtype)


def generate_all_dummy_inputs(
    session: ort.InferenceSession,
    seq_length: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Generate dummy inputs for all model inputs."""
    inputs = {}
    input_infos = inspect_model_inputs(session)

    for info in input_infos:
        inputs[info["name"]] = generate_dummy_input(info, seq_length, batch_size)

    return inputs


# =============================================================================
# Benchmarking Functions
# =============================================================================

def run_benchmark(
    session: ort.InferenceSession,
    inputs: Dict[str, np.ndarray],
    warmup: int,
    iterations: int,
    verbose: bool = False,
) -> List[float]:
    """Run benchmark iterations and return latencies in milliseconds."""
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    if verbose:
        print(f"    Running {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = session.run(output_names, inputs)

    # Benchmark
    latencies = []
    if verbose:
        print(f"    Running {iterations} benchmark iterations...")

    for i in range(iterations):
        start = time.perf_counter()
        _ = session.run(output_names, inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return latencies


def compute_statistics(latencies: List[float], seq_length: int, batch_size: int,
                       iterations: int) -> LatencyResult:
    """Compute statistics from latency measurements."""
    arr = np.array(latencies)

    return LatencyResult(
        seq_length=seq_length,
        batch_size=batch_size,
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        iterations=iterations,
    )


def benchmark_sequence_lengths(config: BenchmarkConfig) -> List[LatencyResult]:
    """Run benchmarks for all specified sequence lengths."""
    print(f"\n{'='*60}")
    print(f"Raw Latency Benchmark")
    print(f"{'='*60}")
    print(f"Model: {config.model_path}")
    print(f"Sequence lengths: {config.seq_lengths}")
    print(f"Batch size: {config.batch_size}")
    print(f"Warmup: {config.warmup}, Iterations: {config.iterations}")
    print(f"QDQ fusion disabled: {config.disable_qdq_fusion}")
    print(f"{'='*60}\n")

    # Create session once
    print("Loading model...")
    session = create_session(config)

    # Print model input info
    input_infos = inspect_model_inputs(session)
    print(f"\nModel inputs ({len(input_infos)}):")
    for info in input_infos:
        print(f"  - {info['name']}: shape={info['shape']}, type={info['type']}")
    print()

    results = []

    for seq_len in config.seq_lengths:
        print(f"\nBenchmarking seq_length={seq_len}...")

        # Generate dummy inputs
        inputs = generate_all_dummy_inputs(session, seq_len, config.batch_size)

        if config.verbose:
            print("  Generated inputs:")
            for name, arr in inputs.items():
                print(f"    - {name}: shape={arr.shape}, dtype={arr.dtype}")

        # Run benchmark
        latencies = run_benchmark(
            session, inputs, config.warmup, config.iterations, config.verbose
        )

        # Compute statistics
        result = compute_statistics(
            latencies, seq_len, config.batch_size, config.iterations
        )
        results.append(result)

        # Print summary
        print(f"  Mean: {result.mean_ms:.2f} ms ± {result.std_ms:.2f} ms")
        print(f"  P50: {result.p50_ms:.2f} ms, P95: {result.p95_ms:.2f} ms, P99: {result.p99_ms:.2f} ms")

    return results


# =============================================================================
# onnxruntime_perf_test Validation (Optional)
# =============================================================================

def find_perf_test_executable() -> Optional[str]:
    """Try to find onnxruntime_perf_test executable."""
    candidates = [
        "onnxruntime_perf_test",
        "onnxruntime_perf_test.exe",
        # Common build directories
        "build/Windows/Release/onnxruntime_perf_test.exe",
        "build/Release/onnxruntime_perf_test.exe",
        "../build/Windows/Release/onnxruntime_perf_test.exe",
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found

    return None


def run_perf_test_validation(
    config: BenchmarkConfig,
    seq_length: int = 128,
) -> None:
    """
    Run onnxruntime_perf_test for validation/cross-checking.

    Note: perf_test uses -g flag for random data generation, treating dynamic dims as 1.
    This may not match our LLM-aware dummy data, but useful for sanity checking.
    """
    perf_test = config.perf_test_path or find_perf_test_executable()

    if not perf_test:
        print("\nWarning: onnxruntime_perf_test not found. Skipping validation.")
        print("  Hint: Pass --perf-test /path/to/onnxruntime_perf_test")
        return

    print(f"\n{'='*60}")
    print("Running onnxruntime_perf_test for validation")
    print(f"{'='*60}")
    print(f"Executable: {perf_test}")

    # Build command
    # -m times: run for specified number of iterations
    # -r: number of iterations
    # -e cpu: use CPU EP
    # -g: generate random input data
    # -z: show statistics (P50/P90/P95/P99)
    # -C: session config entries
    cmd = [
        perf_test,
        "-m", "times",
        "-r", str(config.iterations),
        "-e", "cpu",
        "-g",  # Generate random data (dynamic dims treated as 1)
        "-z",  # Show statistics
    ]

    # Pass same session config as Python benchmark
    if config.enable_lut_gemm:
        cmd.extend(["-C", "mlas.use_lut_gemm|1"])
        print("  Session config: mlas.use_lut_gemm=1")

    # Add profiling if requested
    profile_file = None
    if config.profile_output:
        profile_file = config.profile_output
        if not profile_file.endswith(".json"):
            profile_file = profile_file + ".json"
        cmd.extend(["-p", profile_file])
        print(f"Profiling enabled: {profile_file}")

    cmd.append(config.model_path)

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        if result.returncode != 0:
            print(f"Warning: perf_test exited with code {result.returncode}")

        if profile_file and os.path.exists(profile_file):
            print(f"\nProfile saved to: {profile_file}")
            print("  View in Chrome: chrome://tracing")
            print("  Or use: https://ui.perfetto.dev/")
    except subprocess.TimeoutExpired:
        print("Error: onnxruntime_perf_test timed out")
    except Exception as e:
        print(f"Error running onnxruntime_perf_test: {e}")


# =============================================================================
# Output Functions
# =============================================================================

def save_results(results: List[LatencyResult], output_file: str, config: BenchmarkConfig):
    """Save benchmark results to JSON file."""
    metadata = {
        "model_path": config.model_path,
        "batch_size": config.batch_size,
        "warmup": config.warmup,
        "iterations": config.iterations,
        "disable_qdq_fusion": config.disable_qdq_fusion,
        "enable_lut_gemm": config.enable_lut_gemm,
    }

    results_data = [r.to_dict() for r in results]

    # Ensure .json extension
    if not output_file.endswith(".json"):
        output_file = output_file + ".json"

    with open(output_file, "w") as f:
        json.dump({"metadata": metadata, "results": results_data}, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def print_summary_table(results: List[LatencyResult]):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'Seq Len':>10} {'Mean (ms)':>12} {'Std (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r.seq_length:>10} {r.mean_ms:>12.2f} {r.std_ms:>10.2f} {r.p50_ms:>10.2f} {r.p95_ms:>10.2f} {r.p99_ms:>10.2f}")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Raw ONNX Model Latency Benchmark for QDQ Investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py -m model.onnx
  python benchmark.py -m model.onnx -s 128 512 1024
  python benchmark.py -m model_qdq.onnx --disable-qdq-fusion
  python benchmark.py -m model_2bit.onnx --enable-lut-gemm
  python benchmark.py -m model.onnx -o results.json
        """,
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--seq-lengths", "-s",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
        help=f"Sequence lengths to benchmark (default: {DEFAULT_SEQ_LENGTHS})",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of benchmark iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--disable-qdq-fusion",
        action="store_true",
        help="Disable DQMatMulToMatMulNBits fusion (for unfused QDQ comparison)",
    )
    parser.add_argument(
        "--enable-lut-gemm",
        action="store_true",
        help="Enable LUT-based GEMM for 2-bit quantized models",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (e.g., results.json)",
    )
    parser.add_argument(
        "--perf-test",
        metavar="PATH",
        nargs="?",
        const="auto",
        help="Run onnxruntime_perf_test for validation (optional path to executable)",
    )
    parser.add_argument(
        "--profile",
        metavar="FILE",
        nargs="?",
        const="profile.json",
        help="Enable profiling with perf_test, output trace file (default: profile.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        model_path=args.model,
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        disable_qdq_fusion=args.disable_qdq_fusion,
        enable_lut_gemm=args.enable_lut_gemm,
        output_file=args.output,
        perf_test_path=args.perf_test if args.perf_test != "auto" else None,
        profile_output=args.profile,
        verbose=args.verbose,
    )


def main():
    # Parse args once, check for perf-test flag
    run_perf_test = '--perf-test' in sys.argv
    config = parse_args()

    if not os.path.isfile(config.model_path):
        print(f"Error: Model file not found: {config.model_path}")
        sys.exit(1)

    results = benchmark_sequence_lengths(config)
    print_summary_table(results)

    if config.output_file:
        save_results(results, config.output_file, config)

    # Run perf_test validation if requested
    if run_perf_test:
        run_perf_test_validation(config)

    print("\nDone!")


if __name__ == "__main__":
    main()

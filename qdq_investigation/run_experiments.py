"""
Batch Experiment Runner for ONNX Runtime CPU EP Benchmarking

Orchestrates running benchmark.py across multiple models with different
configurations. Supports presets, filtering, and result aggregation.

Usage:
    # Dry run to see what would be executed
    python run_experiments.py --preset validate --dry-run

    # Quick validation (0.5B 4-bit models, 1 seq length, fast)
    python run_experiments.py --preset validate

    # Quick experiment (0.5B + 1.5B, 2 seq lengths)
    python run_experiments.py --preset quick

    # Full experiment (all 96 models, 5 seq lengths)
    python run_experiments.py --preset full

    # Custom filtering
    python run_experiments.py --preset full --model-sizes 0.5b 1.5b --bits 4

    # Aggregate existing results only
    python run_experiments.py --aggregate-only
"""

import argparse
import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import onnxruntime as ort
    ORT_VERSION = ort.__version__
except ImportError:
    ORT_VERSION = "unknown"


# =============================================================================
# Model Metadata
# =============================================================================

@dataclass
class ModelMetadata:
    """Parsed metadata from model directory name."""
    format_type: str       # "mnb" or "qdq"
    model_family: str      # "qwen"
    model_size: str        # "0.5b", "1.5b", "3b", "7b"
    bits: int              # 2, 4, 8
    symmetry: str          # "sym" or "asym"
    granularity: Optional[str] = None   # "block" or "channel" (QDQ only)
    signedness: Optional[str] = None    # "signed" or "unsigned" (QDQ only)
    dir_name: str = ""
    model_path: str = ""   # full path to model.onnx


def parse_model_dirname(dirname: str) -> Optional[ModelMetadata]:
    """Parse a model directory name into ModelMetadata.

    Expected formats:
        mnb-qwen-0.5b-4-sym
        mnb-qwen-0.5b-4-sym-transpose
        qdq-qwen-0.5b-4-block-sym-signed
        qdq-qwen-0.5b-4-block-sym-signed-transpose
    """
    # Strip optional "-transpose" suffix before parsing
    stripped = dirname
    if dirname.endswith("-transpose"):
        stripped = dirname[: -len("-transpose")]

    parts = stripped.split("-")

    try:
        if parts[0] == "mnb" and len(parts) == 5:
            return ModelMetadata(
                format_type="mnb",
                model_family=parts[1],
                model_size=parts[2],
                bits=int(parts[3]),
                symmetry=parts[4],
                dir_name=dirname,
            )
        elif parts[0] == "qdq" and len(parts) == 7:
            return ModelMetadata(
                format_type="qdq",
                model_family=parts[1],
                model_size=parts[2],
                bits=int(parts[3]),
                granularity=parts[4],
                symmetry=parts[5],
                signedness=parts[6],
                dir_name=dirname,
            )
    except (IndexError, ValueError):
        pass

    return None


# =============================================================================
# Model Discovery
# =============================================================================

def discover_models(base_dir: str) -> List[ModelMetadata]:
    """Scan base directory for model directories and parse their metadata."""
    models = []

    if not os.path.isdir(base_dir):
        print(f"Error: Model directory not found: {base_dir}")
        return models

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        if not (entry.startswith("mnb-") or entry.startswith("qdq-")):
            continue

        model_onnx = os.path.join(entry_path, "model.onnx")
        if not os.path.isfile(model_onnx):
            print(f"  Warning: No model.onnx in {entry}, skipping")
            continue

        meta = parse_model_dirname(entry)
        if meta is None:
            print(f"  Warning: Could not parse directory name: {entry}, skipping")
            continue

        meta.model_path = model_onnx
        models.append(meta)

    return models


def filter_models(
    models: List[ModelMetadata],
    model_sizes: Optional[List[str]] = None,
    format_types: Optional[List[str]] = None,
    bit_widths: Optional[List[int]] = None,
) -> List[ModelMetadata]:
    """Filter models based on criteria."""
    filtered = models

    if model_sizes:
        filtered = [m for m in filtered if m.model_size in model_sizes]
    if format_types:
        filtered = [m for m in filtered if m.format_type in format_types]
    if bit_widths:
        filtered = [m for m in filtered if m.bits in bit_widths]

    return filtered


# =============================================================================
# Run Matrix
# =============================================================================

@dataclass
class RunConfig:
    """A single benchmark run configuration."""
    model: ModelMetadata
    scenario: str          # "native", "qdq_fused", "qdq_unfused"
    seq_lengths: List[int]
    warmup: int
    iterations: int
    batch_size: int
    seed: Optional[int]
    enable_lut_gemm: bool
    output_file: str


def build_run_matrix(
    models: List[ModelMetadata],
    seq_lengths: List[int],
    warmup: int,
    iterations: int,
    batch_size: int,
    seed: Optional[int],
    results_dir: str,
    run_unfused: bool = True,
    auto_lut_gemm: bool = True,
) -> List[RunConfig]:
    """Build the full matrix of (model, scenario) runs."""
    runs = []
    per_model_dir = os.path.join(results_dir, "per_model")

    for model in models:
        enable_lut = auto_lut_gemm and model.bits == 2

        if model.format_type == "mnb":
            scenarios = ["native"]
        else:
            scenarios = ["qdq_fused"]
            # Unfused only meaningful for 4-bit block-quantized models:
            # DQ+MatMul->MatMulNBits fusion only matches int4/uint4 weights
            # (qdq_selectors.cc:Is4BitIntType) with block quantization structure.
            # Channel-quantized models don't match the fusion pattern.
            if run_unfused and model.bits == 4 and model.granularity == "block":
                scenarios.append("qdq_unfused")

        for scenario in scenarios:
            output_file = os.path.join(per_model_dir, f"{model.dir_name}_{scenario}.json")
            runs.append(RunConfig(
                model=model,
                scenario=scenario,
                seq_lengths=seq_lengths,
                warmup=warmup,
                iterations=iterations,
                batch_size=batch_size,
                seed=seed,
                enable_lut_gemm=enable_lut,
                output_file=output_file,
            ))

    return runs


# =============================================================================
# Subprocess Execution
# =============================================================================

def build_command(run: RunConfig, benchmark_script: str) -> List[str]:
    """Build the benchmark.py subprocess command."""
    cmd = [
        sys.executable, benchmark_script,
        "-m", run.model.model_path,
        "-s", *[str(s) for s in run.seq_lengths],
        "-w", str(run.warmup),
        "-i", str(run.iterations),
        "-b", str(run.batch_size),
        "-o", run.output_file,
    ]

    if run.seed is not None:
        cmd.extend(["--seed", str(run.seed)])

    if run.scenario == "qdq_unfused":
        cmd.append("--disable-qdq-fusion")

    if run.enable_lut_gemm:
        cmd.append("--enable-lut-gemm")

    return cmd


def run_single_benchmark(
    run: RunConfig,
    benchmark_script: str,
    timeout: int = 600,
) -> dict:
    """Execute a single benchmark run.

    Returns dict with keys: success, elapsed_s, error
    """
    cmd = build_command(run, benchmark_script)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout or None
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            print(f"  FAILED (exit code {result.returncode}): {error_msg[:200]}")
            return {"success": False, "elapsed_s": elapsed, "error": error_msg[:500]}

        return {"success": True, "elapsed_s": elapsed, "error": None}

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  TIMEOUT after {timeout}s")
        return {"success": False, "elapsed_s": elapsed, "error": "timeout"}
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  EXCEPTION: {e}")
        return {"success": False, "elapsed_s": elapsed, "error": str(e)}


# =============================================================================
# Result Aggregation
# =============================================================================

def aggregate_results(results_dir: str) -> List[Dict[str, Any]]:
    """Merge all per-model JSON files into a single list of result rows."""
    per_model_dir = os.path.join(results_dir, "per_model")
    all_rows = []

    json_files = sorted(glob.glob(os.path.join(per_model_dir, "*.json")))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        # Parse: {dir_name}_{scenario}.json
        name_no_ext = filename.rsplit(".", 1)[0]
        # Find the last underscore that separates dir_name from scenario
        # Scenarios are: native, qdq_fused, qdq_unfused
        for suffix in ["_qdq_unfused", "_qdq_fused", "_native"]:
            if name_no_ext.endswith(suffix):
                dir_name = name_no_ext[:-len(suffix)]
                scenario = suffix[1:]  # strip leading underscore
                break
        else:
            print(f"  Warning: Could not parse filename: {filename}, skipping")
            continue

        meta = parse_model_dirname(dir_name)
        if meta is None:
            print(f"  Warning: Could not parse model name from: {dir_name}, skipping")
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {filename}: {e}, skipping")
            continue

        file_metadata = data.get("metadata", {})

        for result in data.get("results", []):
            row = {
                "format_type": meta.format_type,
                "model_family": meta.model_family,
                "model_size": meta.model_size,
                "bits": meta.bits,
                "symmetry": meta.symmetry,
                "granularity": meta.granularity or "",
                "signedness": meta.signedness or "",
                "dir_name": dir_name,
                "scenario": scenario,
                "model_load_time_s": file_metadata.get("model_load_time_s", ""),
                "ort_version": file_metadata.get("ort_version", ""),
                "platform": file_metadata.get("platform", ""),
                **result,
            }
            all_rows.append(row)

    return all_rows


def save_aggregate_csv(rows: List[Dict[str, Any]], output_path: str):
    """Save aggregated results to CSV."""
    if not rows:
        print("No results to aggregate.")
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated CSV saved to: {output_path} ({len(rows)} rows)")


def save_summary_csv(rows: List[Dict[str, Any]], output_path: str):
    """Save a lightweight summary CSV with just model info and mean latency."""
    if not rows:
        return

    fieldnames = [
        "format_type", "model_size", "bits", "symmetry",
        "granularity", "signedness", "scenario", "seq_length",
        "mean_ms", "model_load_time_s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary CSV saved to: {output_path} ({len(rows)} rows)")


def get_system_info() -> Dict[str, str]:
    """Collect system information."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "ort_version": ORT_VERSION,
        "machine": platform.machine(),
    }


# =============================================================================
# Console Summary
# =============================================================================

def print_summary(rows: List[Dict[str, Any]]):
    """Print a summary table of aggregated results."""
    if not rows:
        print("No results to summarize.")
        return

    header = (f"{'Format':>6} {'Size':>5} {'Bits':>4} {'Sym':>5} {'Gran':>8} "
              f"{'Sign':>10} {'Scenario':>13} {'SeqLen':>7} {'Mean(ms)':>10}")
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['format_type']:>6} "
            f"{r['model_size']:>5} "
            f"{r['bits']:>4} "
            f"{r['symmetry']:>5} "
            f"{r.get('granularity', ''):>8} "
            f"{r.get('signedness', ''):>10} "
            f"{r['scenario']:>13} "
            f"{r['seq_length']:>7} "
            f"{r['mean_ms']:>10.2f}"
        )


# =============================================================================
# Experiment Presets
# =============================================================================

PRESETS = {
    "validate": {
        "seq_lengths": [128],
        "warmup": 1,
        "iterations": 3,
        "model_sizes": ["0.5b"],
        "bit_widths": [4],
        "format_types": None,  # both mnb and qdq
        "run_unfused": False,
    },
    "quick": {
        "seq_lengths": [128, 512],
        "warmup": 3,
        "iterations": 10,
        "model_sizes": ["0.5b", "1.5b"],
        "bit_widths": None,
        "format_types": None,
        "run_unfused": True,
    },
    "full": {
        "seq_lengths": [1, 128, 256, 512, 1024],
        "warmup": 3,
        "iterations": 10,
        "model_sizes": None,
        "bit_widths": None,
        "format_types": None,
        "run_unfused": True,
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for ONNX Runtime CPU EP benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  validate  0.5B 4-bit models, 1 seq length, 3 iterations (~2 min)
  quick     0.5B + 1.5B models, 2 seq lengths, 10 iterations
  full      All models, 5 seq lengths, 10 iterations

Examples:
  python run_experiments.py --preset validate --dry-run
  python run_experiments.py --preset quick --model-sizes 0.5b
  python run_experiments.py --preset full --bits 4 8
  python run_experiments.py --aggregate-only
        """,
    )

    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="validate",
                        help="Experiment preset (default: validate)")
    parser.add_argument("--model-dir", default="C:/dev/llm",
                        help="Base directory containing model folders")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory for results (default: ./results)")
    parser.add_argument("--benchmark-script", default=None,
                        help="Path to benchmark.py (default: auto-detect)")

    # Filtering overrides
    parser.add_argument("--model-sizes", nargs="+",
                        help="Filter by model size (e.g., 0.5b 1.5b)")
    parser.add_argument("--format-types", nargs="+", choices=["mnb", "qdq"],
                        help="Filter by format type")
    parser.add_argument("--bits", type=int, nargs="+",
                        help="Filter by bit width (e.g., 2 4 8)")

    # Benchmark parameter overrides
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        help="Override sequence lengths")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations")
    parser.add_argument("--iterations", type=int, help="Override benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Behavior flags
    parser.add_argument("--no-unfused", action="store_true",
                        help="Skip unfused QDQ runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-model timeout in seconds (default: 600)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, don't run benchmarks")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only re-run models from failed_models.json (use with --timeout 0 for timed-out models)")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(script_dir, "results")
    benchmark_script = args.benchmark_script or os.path.join(script_dir, "benchmark.py")

    if not os.path.isfile(benchmark_script):
        print(f"Error: benchmark.py not found at: {benchmark_script}")
        sys.exit(1)

    # Aggregate-only mode
    if args.aggregate_only:
        print(f"Aggregating results from: {results_dir}")
        rows = aggregate_results(results_dir)
        if rows:
            save_aggregate_csv(rows, os.path.join(results_dir, "benchmark_results.csv"))
            save_summary_csv(rows, os.path.join(results_dir, "summary.csv"))
            print_summary(rows)
        return

    # Load preset and apply overrides
    preset = PRESETS[args.preset]
    seq_lengths = args.seq_lengths or preset["seq_lengths"]
    warmup = args.warmup if args.warmup is not None else preset["warmup"]
    iterations = args.iterations if args.iterations is not None else preset["iterations"]
    model_sizes = args.model_sizes or preset.get("model_sizes")
    format_types = args.format_types or preset.get("format_types")
    bit_widths = args.bits or preset.get("bit_widths")
    run_unfused = not args.no_unfused and preset.get("run_unfused", True)

    # Discover and filter models
    print(f"Discovering models in: {args.model_dir}")
    all_models = discover_models(args.model_dir)
    print(f"Found {len(all_models)} models total")

    models = filter_models(all_models, model_sizes, format_types, bit_widths)
    print(f"After filtering: {len(models)} models")

    if not models:
        print("No models matched the filter criteria.")
        sys.exit(1)

    # Build run matrix
    runs = build_run_matrix(
        models, seq_lengths, warmup, iterations,
        args.batch_size, args.seed, results_dir, run_unfused,
    )
    print(f"Total runs: {len(runs)}")

    # Filter to only failed models if --retry-failed
    if args.retry_failed:
        failed_path = os.path.join(results_dir, "failed_models.json")
        if not os.path.isfile(failed_path):
            print(f"Error: {failed_path} not found. Run the full experiment first.")
            sys.exit(1)
        with open(failed_path) as f:
            failed_entries = json.load(f)
        failed_keys = {(e["model"], e["scenario"]) for e in failed_entries}
        runs = [r for r in runs if (r.model.dir_name, r.scenario) in failed_keys]
        print(f"Retrying {len(runs)} failed runs (from {len(failed_entries)} failures)")

    # Create output directories
    per_model_dir = os.path.join(results_dir, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)

    # Save experiment config (includes system info) — skip in dry-run mode
    if not args.dry_run:
        config_data = {
            "preset": args.preset,
            "model_dir": args.model_dir,
            "seq_lengths": seq_lengths,
            "warmup": warmup,
            "iterations": iterations,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "run_unfused": run_unfused,
            "model_sizes_filter": model_sizes,
            "format_types_filter": format_types,
            "bit_widths_filter": bit_widths,
            "total_models": len(models),
            "total_runs": len(runs),
            "system": get_system_info(),
        }
        config_path = os.path.join(results_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    # Execute runs
    print(f"\n{'#'*60}")
    print(f"# Experiment: {args.preset}")
    print(f"# Models: {len(models)}, Runs: {len(runs)}")
    print(f"# Seq lengths: {seq_lengths}")
    print(f"# Warmup: {warmup}, Iterations: {iterations}")
    print(f"{'#'*60}\n")

    succeeded = 0
    failed = 0
    failed_runs = []  # Track failures for writing to file
    total_elapsed = 0.0
    failed_path = os.path.join(results_dir, "failed_models.json")

    # When retrying, start with all existing failures; entries are removed as they succeed
    if args.retry_failed:
        failed_runs = list(failed_entries)

    for idx, run in enumerate(runs, 1):
        label = f"{run.model.dir_name} ({run.scenario})"
        print(f"[{idx}/{len(runs)}] {label}", end=" ... ", flush=True)

        if args.dry_run:
            cmd = build_command(run, benchmark_script)
            print(f"\n  [DRY RUN] {' '.join(cmd)}")
            continue

        result = run_single_benchmark(run, benchmark_script, timeout=args.timeout)

        # Remove old entry for this run (no-op on fresh runs)
        failed_runs = [e for e in failed_runs
                       if not (e["model"] == run.model.dir_name and e["scenario"] == run.scenario)]

        if result["success"]:
            print(f"Done ({result['elapsed_s']:.1f}s)")
            succeeded += 1
        else:
            failed += 1
            failed_runs.append({
                "model": run.model.dir_name,
                "scenario": run.scenario,
                "error": result["error"],
            })

        total_elapsed += result["elapsed_s"]

        # Write failed_models.json after each run so progress is saved
        # Always write (even if empty) so succeeded retries are removed from the file
        with open(failed_path, "w") as f:
            json.dump(failed_runs, f, indent=2)

    if args.dry_run:
        print(f"\nDry run complete. {len(runs)} runs would be executed.")
        return

    # Aggregate results and print summary
    rows = aggregate_results(results_dir)
    if rows:
        save_aggregate_csv(rows, os.path.join(results_dir, "benchmark_results.csv"))
        save_summary_csv(rows, os.path.join(results_dir, "summary.csv"))

    print(f"\n{'='*110}")
    print(f"Experiment Complete  |  Succeeded: {succeeded}  Failed: {failed}  "
          f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*110}")

    if rows:
        print_summary(rows)

    if failed_runs:
        print(f"\nFailed models ({len(failed_runs)}):")
        for f_run in failed_runs:
            print(f"  - {f_run['model']} ({f_run['scenario']}): {f_run['error'][:100]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
This script is used as a developer aid for feature validation and is not intended for
for production gating or official measurement.

Benchmark script for memory-mapped .ort model loading. Used to validate session construction time and
memory usage implications from memory-mapped loading.
Measures session construction time, inference latency, and memory usage across multiple
loading configurations.

Usage:
    python benchmark_mmap_ort.py --perf_test <path_to_onnxruntime_perf_test> --model <path_to_model.ort>

    # Multi-process mode (measures shared memory benefits):
    python benchmark_mmap_ort.py --perf_test <path> --model <path> --multi-process --num-processes 4

    # Convert .onnx to .ort first:
    python benchmark_mmap_ort.py --perf_test <path> --model <path_to_model.onnx> --convert

Requirements:
    - Built onnxruntime_perf_test executable
    - .ort model file (or .onnx with --convert flag)
    - Windows: psutil package for memory measurement in multi-process mode
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def parse_perf_test_output(output: str) -> dict:
    """Parse onnxruntime_perf_test stdout into a dict of metrics."""
    metrics = {}

    patterns = {
        "session_creation_time_s": r"Session creation time cost:\s+([\d.]+)\s+s",
        "first_inference_time_ms": r"First inference time cost:\s+([\d.]+)\s+ms",
        "total_inference_time_s": r"Total inference time cost:\s+([\d.]+)\s+s",
        "total_inference_requests": r"Total inference requests:\s+(\d+)",
        "avg_inference_time_ms": r"Average inference time cost total:\s+([\d.]+)\s+ms",
        "inferences_per_second": r"Number of inferences per second:\s+([\d.]+)",
        "peak_working_set_bytes": r"Peak working set size:\s+(\d+)\s+bytes",
        "avg_cpu_usage_pct": r"Avg CPU usage:\s+(\d+)\s+%",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)

    return metrics


def get_process_memory_info() -> dict:
    """Get current process memory info (Windows)."""
    if not HAS_PSUTIL:
        return {}
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "private_bytes": mem.private,
        "working_set_bytes": mem.wset,
        "peak_working_set_bytes": mem.peak_wset,
    }


def run_perf_test(
    perf_test_exe: str,
    model_path: str,
    session_configs: dict | None = None,
    session_only: bool = False,
    num_runs: int = 10,
    duration_seconds: int = 0,
) -> dict:
    """Run onnxruntime_perf_test and return parsed metrics.

    For session-only runs, also captures peak memory via psutil polling
    during the same execution (no separate run).
    """
    cmd = [perf_test_exe]

    # Session config entries
    if session_configs:
        config_str = " ".join(f"{k}|{v}" for k, v in session_configs.items())
        cmd.extend(["-C", config_str])

    if session_only:
        cmd.append("-n")
    else:
        if duration_seconds > 0:
            cmd.extend(["-d", str(duration_seconds)])
        else:
            cmd.extend(["-r", str(num_runs)])

    cmd.append(model_path)

    if session_only and HAS_PSUTIL:
        # Single run: capture both timing (from stdout) and memory (from psutil) simultaneously
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ps = psutil.Process(proc.pid)
        peak_private = 0
        peak_ws = 0
        try:
            while proc.poll() is None:
                try:
                    mem = ps.memory_info()
                    private = getattr(mem, "private", mem.rss)
                    ws = getattr(mem, "wset", mem.rss)
                    peak_private = max(peak_private, private)
                    peak_ws = max(peak_ws, ws)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.001)  # 1ms polling for better peak capture
        except psutil.NoSuchProcess:
            pass  # process exited during polling, peak already captured
        stdout, _ = proc.communicate(timeout=30)
        if proc.returncode != 0:
            print(f"  ERROR: perf_test failed (exit code {proc.returncode})")
            return {}
        metrics = parse_perf_test_output(stdout.decode() if isinstance(stdout, bytes) else stdout)
        if peak_private > 0:
            metrics["peak_private_bytes"] = peak_private
            metrics["peak_working_set_bytes"] = peak_ws
        return metrics

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  ERROR: perf_test failed (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return {}

    return parse_perf_test_output(result.stdout)


def run_configuration(
    perf_test_exe: str,
    model_path: str,
    config_name: str,
    session_configs: dict,
    num_iterations: int = 5,
    num_inference_runs: int = 10,
    warmup_iterations: int = 2,
) -> dict:
    """Run a single configuration multiple times and aggregate results."""
    print(f"\n{'=' * 60}")
    print(f"Configuration: {config_name}")
    print(f"  Session configs: {session_configs}")
    print(f"  Warmup: {warmup_iterations}, Iterations: {num_iterations}")
    print(f"{'=' * 60}")

    # Warmup runs (not included in results)
    for i in range(warmup_iterations):
        run_perf_test(perf_test_exe, model_path, session_configs=session_configs, session_only=True)
        print(f"  Warmup {i + 1}: done")

    # Phase 1: Session creation time (run with -n flag, multiple iterations)
    session_times = []
    peak_ws_session = []
    peak_private_session = []
    for i in range(num_iterations):
        metrics = run_perf_test(
            perf_test_exe,
            model_path,
            session_configs=session_configs,
            session_only=True,
        )
        if metrics:
            session_times.append(metrics.get("session_creation_time_s", 0) * 1000)  # convert to ms
            peak_ws_session.append(metrics.get("peak_working_set_bytes", 0))
            peak_private_session.append(metrics.get("peak_private_bytes", 0))
            ws_mb = peak_ws_session[-1] / 1024 / 1024
            priv_mb = peak_private_session[-1] / 1024 / 1024
            print(f"  Run {i + 1}: session={session_times[-1]:.2f}ms, peak_ws={ws_mb:.1f}MB, private={priv_mb:.1f}MB")

    # Phase 2: Inference performance (run with inference)
    inference_metrics_list = []
    for i in range(min(num_iterations, 3)):  # fewer inference runs
        metrics = run_perf_test(
            perf_test_exe,
            model_path,
            session_configs=session_configs,
            session_only=False,
            num_runs=num_inference_runs,
        )
        if metrics:
            inference_metrics_list.append(metrics)
            print(
                f"  Inference run {i + 1}: avg={metrics.get('avg_inference_time_ms', 0):.2f}ms, "
                f"peak_ws={metrics.get('peak_working_set_bytes', 0) / 1024 / 1024:.1f}MB"
            )

    # Aggregate
    result = {
        "config_name": config_name,
        "session_configs": session_configs,
    }

    if session_times:
        result["session_creation_time_ms"] = {
            "mean": statistics.mean(session_times),
            "median": statistics.median(session_times),
            "stdev": statistics.stdev(session_times) if len(session_times) > 1 else 0,
            "min": min(session_times),
            "max": max(session_times),
            "count": len(session_times),
        }

    if peak_ws_session:
        result["peak_working_set_session_mb"] = {
            "mean": statistics.mean(peak_ws_session) / 1024 / 1024,
            "min": min(peak_ws_session) / 1024 / 1024,
            "max": max(peak_ws_session) / 1024 / 1024,
        }

    non_zero_private = [p for p in peak_private_session if p > 0]
    if non_zero_private:
        result["peak_private_session_mb"] = {
            "mean": statistics.mean(non_zero_private) / 1024 / 1024,
            "min": min(non_zero_private) / 1024 / 1024,
            "max": max(non_zero_private) / 1024 / 1024,
        }

    if inference_metrics_list:
        avg_times = [m.get("avg_inference_time_ms", 0) for m in inference_metrics_list]
        peak_ws_inf = [m.get("peak_working_set_bytes", 0) for m in inference_metrics_list]
        first_inf = [m.get("first_inference_time_ms", 0) for m in inference_metrics_list]

        result["avg_inference_time_ms"] = {
            "mean": statistics.mean(avg_times),
            "median": statistics.median(avg_times),
        }
        result["first_inference_time_ms"] = {
            "mean": statistics.mean(first_inf),
        }
        result["peak_working_set_inference_mb"] = {
            "mean": statistics.mean(peak_ws_inf) / 1024 / 1024,
        }

    return result


def run_multi_process_benchmark(
    perf_test_exe: str,
    model_path: str,
    session_configs: dict,
    num_processes: int = 4,
    config_name: str = "",
) -> dict:
    """Launch multiple processes loading the same model and measure total system memory."""
    if not HAS_PSUTIL:
        print("  WARNING: psutil not installed, skipping multi-process benchmark")
        print("  Install with: pip install psutil")
        return {}

    print(f"\n{'=' * 60}")
    print(f"Multi-Process Benchmark: {config_name}")
    print(f"  Processes: {num_processes}")
    print(f"{'=' * 60}")

    # Build the perf_test command
    cmd = [perf_test_exe]
    if session_configs:
        config_str = " ".join(f"{k}|{v}" for k, v in session_configs.items())
        cmd.extend(["-C", config_str])
    cmd.extend(["-n", model_path])

    # Use a Python wrapper that runs perf_test, then waits for stdin before exiting.
    # This keeps the process alive after session creation so we can measure memory.
    wrapper_code = (
        "import subprocess, sys;"
        f"p = subprocess.run({cmd!r}, capture_output=True);"
        "sys.stdin.readline()"  # block until parent signals
    )
    wrapper_cmd = [sys.executable, "-c", wrapper_code]

    processes = []
    ps_processes = []
    for i in range(num_processes):
        proc = subprocess.Popen(wrapper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(proc)
        try:
            ps = psutil.Process(proc.pid)
            ps_processes.append((i, proc, ps))
        except psutil.NoSuchProcess:
            ps_processes.append((i, proc, None))
        print(f"  Started process {i + 1} (PID={proc.pid})")

    # Wait for all wrapper processes to finish session creation and stabilize
    time.sleep(3)

    # Measure memory for each process (they're all alive, blocked on stdin)
    total_private = 0
    total_working_set = 0
    per_process = []
    for i, proc, ps in ps_processes:
        if ps and proc.poll() is None:
            try:
                # Get memory of the wrapper process and its children (the perf_test subprocess)
                mem = ps.memory_info()
                private = getattr(mem, "private", mem.rss)
                ws = getattr(mem, "wset", mem.rss)
                # Also check children
                for child in ps.children(recursive=True):
                    try:
                        cmem = child.memory_info()
                        private += getattr(cmem, "private", cmem.rss)
                        ws += getattr(cmem, "wset", cmem.rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # child process exited or inaccessible, skip it
                total_private += private
                total_working_set += ws
                per_process.append(
                    {
                        "pid": proc.pid,
                        "private_mb": private / 1024 / 1024,
                        "working_set_mb": ws / 1024 / 1024,
                    }
                )
                print(
                    f"  Process {i + 1} (PID={proc.pid}): private={private / 1024 / 1024:.1f}MB, ws={ws / 1024 / 1024:.1f}MB"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  Process {i + 1}: could not read memory ({e})")
        else:
            print(f"  Process {i + 1}: not running")

    # Signal all wrappers to exit
    for _, proc, _ in ps_processes:
        try:
            proc.stdin.write(b"\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # wrapper already exited, nothing to signal
    for _, proc, _ in ps_processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()

    result = {
        "config_name": config_name,
        "num_processes": num_processes,
        "total_private_mb": total_private / 1024 / 1024,
        "total_working_set_mb": total_working_set / 1024 / 1024,
        "per_process": per_process,
    }

    if per_process:
        avg_private = statistics.mean(p["private_mb"] for p in per_process)
        result["avg_private_per_process_mb"] = avg_private
        result["theoretical_total_without_sharing_mb"] = avg_private * num_processes

    return result


def convert_onnx_to_ort(onnx_path: str) -> str:
    """Convert .onnx model to .ort format."""
    ort_path = onnx_path + ".ort"
    if os.path.exists(ort_path):
        print(f"  .ort file already exists: {ort_path}")
        return ort_path

    print(f"  Converting {onnx_path} to .ort format...")
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python", "util", "convert_onnx_models_to_ort.py"
    )

    if not os.path.exists(script):
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parents[2]
        script = str(repo_root / "tools" / "python" / "util" / "convert_onnx_models_to_ort.py")

    cmd = [sys.executable, script, os.path.dirname(onnx_path)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Conversion failed: {result.stderr[:500]}")
        raise RuntimeError(f"Failed to convert {onnx_path} to .ort")

    # The converter creates .ort files next to the .onnx files
    expected = onnx_path.replace(".onnx", ".ort")
    if os.path.exists(expected):
        return expected
    if os.path.exists(ort_path):
        return ort_path
    raise RuntimeError(f"Could not find converted .ort file. Expected: {expected} or {ort_path}")


def print_summary_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 100}")

    # Header
    header = f"{'Configuration':<45} {'Session (ms)':<15} {'Peak WS (MB)':<15} {'Private (MB)':<15}"
    print(header)
    print("-" * len(header))

    for r in results:
        name = r.get("config_name", "?")
        session_ms = r.get("session_creation_time_ms", {}).get("mean", 0)
        peak_ws = r.get("peak_working_set_session_mb", {}).get("mean", 0)
        peak_priv = r.get("peak_private_session_mb", {}).get("mean", 0)

        print(f"{name:<45} {session_ms:<15.2f} {peak_ws:<15.1f} {peak_priv:<15.1f}")

    # Relative comparison — use .ort standard load as baseline
    if len(results) >= 2:
        # Find the .ort standard baseline (config "1.")
        baseline = next((r for r in results if r.get("config_name", "").startswith("1.")), results[0])
        print(f"\nRelative to baseline ({baseline['config_name']}):")
        print("-" * 80)
        for r in results:
            if r is baseline:
                continue
            name = r.get("config_name", "?")
            b_session = baseline.get("session_creation_time_ms", {}).get("mean", 0)
            r_session = r.get("session_creation_time_ms", {}).get("mean", 0)
            b_ws = baseline.get("peak_working_set_session_mb", {}).get("mean", 0)
            r_ws = r.get("peak_working_set_session_mb", {}).get("mean", 0)
            b_priv = baseline.get("peak_private_session_mb", {}).get("mean", 0)
            r_priv = r.get("peak_private_session_mb", {}).get("mean", 0)

            print(f"  {name}:")
            if b_session > 0:
                session_pct = (r_session - b_session) / b_session * 100
                print(f"    Session time: {session_pct:+.1f}%")
            if b_ws > 0:
                ws_pct = (r_ws - b_ws) / b_ws * 100
                print(f"    Peak WS:      {ws_pct:+.1f}%")
            if b_priv > 0:
                priv_pct = (r_priv - b_priv) / b_priv * 100
                print(f"    Private:      {priv_pct:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory-mapped .ort model loading")
    parser.add_argument("--perf-test", required=True, help="Path to onnxruntime_perf_test executable")
    parser.add_argument("--model", required=True, help="Path to .ort or .onnx model file")
    parser.add_argument("--convert", action="store_true", help="Convert .onnx to .ort before benchmarking")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per configuration")
    parser.add_argument("--inference-runs", type=int, default=20, help="Number of inference runs per iteration")
    parser.add_argument("--multi-process", action="store_true", help="Run multi-process sharing benchmark")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of processes for multi-process test")
    parser.add_argument("--output", help="Path to save JSON results")
    args = parser.parse_args()

    perf_test = os.path.abspath(args.perf_test)
    model_path = os.path.abspath(args.model)

    if not os.path.exists(perf_test):
        print(f"ERROR: perf_test executable not found: {perf_test}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    # Convert if needed
    ort_model_path = model_path
    if args.convert or model_path.endswith(".onnx"):
        if model_path.endswith(".onnx"):
            ort_model_path = convert_onnx_to_ort(model_path)
        else:
            print("WARNING: --convert specified but model doesn't end with .onnx")

    model_size_mb = os.path.getsize(ort_model_path) / 1024 / 1024
    print(f"\nModel: {os.path.basename(ort_model_path)} ({model_size_mb:.1f} MB)")
    print(f"Perf test: {perf_test}")
    print(f"Iterations: {args.iterations}")

    # Define configurations to benchmark
    configs = [
        ("1. .ort standard load (baseline)", {}),
        (
            "2. .ort memory-mapped load",
            {
                "session.use_memory_mapped_ort_model": "1",
            },
        ),
        (
            "3. .ort mmap + direct initializers",
            {
                "session.use_memory_mapped_ort_model": "1",
                "session.use_ort_model_bytes_for_initializers": "1",
            },
        ),
    ]

    # Also test .onnx baseline if the .onnx file exists
    onnx_path = model_path if model_path.endswith(".onnx") else model_path.replace(".ort", ".onnx")
    if os.path.exists(onnx_path) and onnx_path != ort_model_path:
        configs.insert(0, ("0. .onnx standard load", {}))
        onnx_configs = [configs.pop(0)]
    else:
        onnx_configs = []

    # Run benchmarks
    all_results = []

    # Run .onnx baseline if available
    for config_name, session_configs in onnx_configs:
        result = run_configuration(
            perf_test,
            onnx_path,
            config_name,
            session_configs,
            num_iterations=args.iterations,
            num_inference_runs=args.inference_runs,
        )
        all_results.append(result)

    # Run .ort configurations
    for config_name, session_configs in configs:
        result = run_configuration(
            perf_test,
            ort_model_path,
            config_name,
            session_configs,
            num_iterations=args.iterations,
            num_inference_runs=args.inference_runs,
        )
        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Multi-process benchmark
    mp_results = []
    if args.multi_process:
        for config_name, session_configs in configs:
            mp_result = run_multi_process_benchmark(
                perf_test,
                ort_model_path,
                session_configs,
                num_processes=args.num_processes,
                config_name=f"MP: {config_name}",
            )
            mp_results.append(mp_result)

        if mp_results:
            print(f"\n{'=' * 80}")
            print("MULTI-PROCESS MEMORY SHARING RESULTS")
            print(f"{'=' * 80}")
            for r in mp_results:
                if r:
                    print(f"\n  {r.get('config_name', '?')} ({r.get('num_processes', 0)} processes):")
                    print(f"    Total private memory:  {r.get('total_private_mb', 0):.1f} MB")
                    print(f"    Total working set:     {r.get('total_working_set_mb', 0):.1f} MB")
                    print(f"    Theoretical w/o sharing: {r.get('theoretical_total_without_sharing_mb', 0):.1f} MB")

    # Save JSON results
    if args.output:
        output_data = {
            "model": os.path.basename(ort_model_path),
            "model_size_mb": model_size_mb,
            "iterations": args.iterations,
            "single_process_results": all_results,
            "multi_process_results": mp_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

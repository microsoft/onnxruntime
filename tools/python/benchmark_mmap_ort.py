#!/usr/bin/env python3
"""
Developer benchmark for memory-mapped .ort model loading.

Compares session construction time and process memory across loading configurations:
  1. Standard .ort load (file read into heap buffer)
  2. Memory-mapped .ort load (session.use_memory_mapped_ort_model)
  3. Memory-mapped + direct initializers (+ session.use_ort_model_bytes_for_initializers)

Not intended for CI gating or official performance measurement.

Usage:
    python benchmark_mmap_ort.py --perf-test <path_to_onnxruntime_perf_test> --model <model.ort>
    python benchmark_mmap_ort.py --perf-test <path> --model <model.ort> --multi-process

Requirements:
    - Built onnxruntime_perf_test executable (with --hold_ms_after_session_creation support for --multi-process)
    - .ort model file
    - psutil package (pip install psutil) for memory measurements
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

IS_WINDOWS = sys.platform == "win32"


def _get_private_and_ws(ps: "psutil.Process") -> tuple[int, int]:
    """Get private memory and working set for a process.

    On Windows, memory_info() exposes 'private' and 'wset' directly.
    On POSIX, use memory_full_info().uss for true private (unique set size),
    falling back to RSS if memory_full_info() is unavailable.
    """
    if IS_WINDOWS:
        mem = ps.memory_info()
        return getattr(mem, "private", mem.rss), getattr(mem, "wset", mem.rss)
    # POSIX: prefer USS (unique set size) for accurate private memory
    try:
        mem_full = ps.memory_full_info()
        return mem_full.uss, mem_full.rss
    except (psutil.AccessDenied, AttributeError):
        mem = ps.memory_info()
        return mem.rss, mem.rss


# -- Helpers --


def parse_perf_test_output(output: str) -> dict:
    """Parse onnxruntime_perf_test stdout for session creation time."""
    metrics = {}
    for key, pattern in {
        "session_creation_time_s": r"Session creation time cost:\s+([\d.]+)\s+s",
        "peak_working_set_bytes": r"Peak working set size:\s+(\d+)\s+bytes",
    }.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)
    return metrics


def build_perf_test_cmd(perf_test_exe: str, model_path: str, session_configs: dict) -> list[str]:
    """Build the onnxruntime_perf_test command line for session-only mode."""
    cmd = [perf_test_exe]
    if session_configs:
        config_str = " ".join(f"{k}|{v}" for k, v in session_configs.items())
        cmd.extend(["-C", config_str])
    cmd.append("-n")
    cmd.append(model_path)
    return cmd


def run_session_benchmark(perf_test_exe: str, model_path: str, session_configs: dict) -> dict:
    """Run a single session-creation benchmark, capturing timing and memory."""
    cmd = build_perf_test_cmd(perf_test_exe, model_path, session_configs)

    if HAS_PSUTIL:
        # Launch and poll memory during execution
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ps = psutil.Process(proc.pid)
        peak_private = 0
        peak_ws = 0
        try:
            while proc.poll() is None:
                try:
                    private, ws = _get_private_and_ws(ps)
                    peak_private = max(peak_private, private)
                    peak_ws = max(peak_ws, ws)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.005)
        except psutil.NoSuchProcess:
            pass  # process exited during polling, peak already captured
        try:
            stdout, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {}
        if proc.returncode != 0:
            return {}
        metrics = parse_perf_test_output(stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout)
        if peak_private > 0:
            metrics["peak_private_bytes"] = peak_private
            metrics["peak_working_set_bytes"] = peak_ws
        return metrics

    # Fallback without psutil: timing only
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    return parse_perf_test_output(result.stdout) if result.returncode == 0 else {}


# -- Single-process benchmark --


def run_configuration(
    perf_test_exe: str,
    model_path: str,
    config_name: str,
    session_configs: dict,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
) -> dict:
    """Run a configuration multiple times and return aggregated results."""
    print(f"\n{'=' * 60}")
    print(f"  {config_name}")
    print(f"  Warmup: {warmup_iterations}, Iterations: {num_iterations}")
    print(f"{'=' * 60}")

    for i in range(warmup_iterations):
        run_session_benchmark(perf_test_exe, model_path, session_configs)
        print(f"  Warmup {i + 1}: done")

    session_times = []
    private_samples = []
    ws_samples = []

    for i in range(num_iterations):
        metrics = run_session_benchmark(perf_test_exe, model_path, session_configs)
        if not metrics:
            print(f"  Run {i + 1}: FAILED")
            continue
        t = metrics.get("session_creation_time_s", 0) * 1000
        p = metrics.get("peak_private_bytes", 0) / 1024 / 1024
        w = metrics.get("peak_working_set_bytes", 0) / 1024 / 1024
        session_times.append(t)
        private_samples.append(p)
        ws_samples.append(w)
        print(f"  Run {i + 1}: session={t:.2f}ms, private={p:.1f}MB, ws={w:.1f}MB")

    result = {"config_name": config_name}
    if session_times:
        result["session_ms"] = {
            "mean": statistics.mean(session_times),
            "stdev": statistics.stdev(session_times) if len(session_times) > 1 else 0,
        }
    if any(p > 0 for p in private_samples):
        result["private_mb"] = {"mean": statistics.mean(private_samples)}
    if any(w > 0 for w in ws_samples):
        result["ws_mb"] = {"mean": statistics.mean(ws_samples)}
    return result


# -- Multi-process benchmark --


def run_multi_process_benchmark(
    perf_test_exe: str,
    model_path: str,
    session_configs: dict,
    num_processes: int = 4,
    config_name: str = "",
) -> dict:
    """Launch N processes with live ORT sessions and measure concurrent memory.

    Requires onnxruntime_perf_test built with --hold_ms_after_session_creation support
    and psutil for memory measurement.
    """
    if not HAS_PSUTIL:
        print("  WARNING: psutil not installed, skipping multi-process benchmark")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  {config_name} ({num_processes} processes)")
    print(f"{'=' * 60}")

    cmd = build_perf_test_cmd(perf_test_exe, model_path, session_configs)
    cmd.insert(-1, "--hold_ms_after_session_creation=30000")  # insert before model_path

    # Launch all processes
    ps_processes = []
    try:
        for i in range(num_processes):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                ps = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                ps = None
            ps_processes.append((i, proc, ps))
            print(f"  Started process {i + 1} (PID={proc.pid})")

        # Wait for each process to signal SESSION_READY
        for i, proc, _ps in ps_processes:
            for line in proc.stdout:
                if b"SESSION_READY" in line:
                    print(f"  Process {i + 1}: ready")
                    break

        time.sleep(0.5)  # stabilization

        # Measure memory (all processes alive with loaded sessions)
        total_private = 0
        total_ws = 0
        per_process = []
        for i, proc, ps in ps_processes:
            if ps and proc.poll() is None:
                try:
                    private, ws = _get_private_and_ws(ps)
                    private_mb = private / 1024 / 1024
                    ws_mb = ws / 1024 / 1024
                    total_private += private_mb
                    total_ws += ws_mb
                    per_process.append({"pid": proc.pid, "private_mb": private_mb, "ws_mb": ws_mb})
                    print(f"  Process {i + 1} (PID={proc.pid}): private={private_mb:.1f}MB, ws={ws_mb:.1f}MB")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"  Process {i + 1}: could not read memory ({e})")
            else:
                print(f"  Process {i + 1}: not running")
    finally:
        # Cleanup: ensure all child processes are terminated
        for _, proc, _ in ps_processes:
            proc.terminate()
        for _, proc, _ in ps_processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    return {
        "config_name": config_name,
        "num_processes": num_processes,
        "total_private_mb": total_private,
        "total_ws_mb": total_ws,
        "per_process": per_process,
    }


# -- Output --


def print_summary(results: list[dict]):
    """Print results table with relative comparison."""
    print(f"\n{'=' * 90}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 90}")

    header = f"{'Configuration':<45} {'Session (ms)':<15} {'Private (MB)':<15} {'WS (MB)':<15}"
    print(header)
    print("-" * len(header))

    for r in results:
        name = r.get("config_name", "?")
        t = r.get("session_ms", {}).get("mean", 0)
        p = r.get("private_mb", {}).get("mean", 0)
        w = r.get("ws_mb", {}).get("mean", 0)
        print(f"{name:<45} {t:<15.2f} {p:<15.1f} {w:<15.1f}")

    # Relative to .ort standard baseline
    if len(results) >= 2:
        baseline = next((r for r in results if r.get("config_name", "").startswith("1.")), results[0])
        bt = baseline.get("session_ms", {}).get("mean", 0)
        bp = baseline.get("private_mb", {}).get("mean", 0)
        print(f"\nRelative to {baseline['config_name']}:")
        print("-" * 60)
        for r in results:
            if r is baseline:
                continue
            name = r.get("config_name", "?")
            rt = r.get("session_ms", {}).get("mean", 0)
            rp = r.get("private_mb", {}).get("mean", 0)
            parts = [f"  {name}:"]
            if bt > 0:
                parts.append(f"    Session: {(rt - bt) / bt * 100:+.1f}%")
            if bp > 0:
                parts.append(f"    Private: {(rp - bp) / bp * 100:+.1f}%")
            print("\n".join(parts))


# -- Main --


CONFIGS = [
    ("1. .ort standard load (baseline)", {}),
    ("2. .ort memory-mapped load", {"session.use_memory_mapped_ort_model": "1"}),
    (
        "3. .ort mmap + direct initializers",
        {"session.use_memory_mapped_ort_model": "1", "session.use_ort_model_bytes_for_initializers": "1"},
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory-mapped .ort model loading")
    parser.add_argument("--perf-test", required=True, help="Path to onnxruntime_perf_test executable")
    parser.add_argument("--model", required=True, help="Path to .ort model file")
    parser.add_argument("--iterations", type=int, default=10, help="Number of measured iterations per config")
    parser.add_argument("--multi-process", action="store_true", help="Run multi-process memory sharing benchmark")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of processes for --multi-process")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    perf_test = os.path.abspath(args.perf_test)
    model_path = os.path.abspath(args.model)

    for path, label in [(perf_test, "perf_test"), (model_path, "model")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"\nModel: {os.path.basename(model_path)} ({model_size_mb:.1f} MB)")
    print(f"Perf test: {perf_test}")
    print(f"Iterations: {args.iterations}")
    if not HAS_PSUTIL:
        print("WARNING: psutil not installed — memory metrics will not be collected")

    # Single-process benchmarks
    results = []
    for config_name, session_configs in CONFIGS:
        results.append(
            run_configuration(perf_test, model_path, config_name, session_configs, num_iterations=args.iterations)
        )
    print_summary(results)

    # Multi-process benchmarks
    mp_results = []
    if args.multi_process:
        for config_name, session_configs in CONFIGS:
            mp_results.append(
                run_multi_process_benchmark(
                    perf_test, model_path, session_configs, num_processes=args.num_processes, config_name=config_name
                )
            )
        if mp_results:
            print(f"\n{'=' * 60}")
            print("MULTI-PROCESS RESULTS")
            print(f"{'=' * 60}")
            for r in mp_results:
                if r:
                    print(f"  {r['config_name']} ({r['num_processes']} processes):")
                    print(f"    Total private: {r['total_private_mb']:.1f} MB")
                    print(f"    Total WS:      {r['total_ws_mb']:.1f} MB")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": os.path.basename(model_path),
                    "model_size_mb": model_size_mb,
                    "single": results,
                    "multi": mp_results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Benchmark thread-pool spin settings for ONNX Runtime.

This script wraps ``onnxruntime_perf_test`` and runs the same model under several
combinations of the spin configuration introduced by PR 27916 (configurable
``spin_duration_us``) and the exponential backoff combined with that from
PR 23278 (new ``spin_backoff_max``). It then prints a table summarizing average
latency, throughput, and (on Linux) CPU utilization for each configuration.

Examples
--------
Run all presets against a model, 1 intra-op thread:

    python tools/perftest/benchmark_spin_settings.py \\
        --perf_test build/.../onnxruntime_perf_test \\
        --model path/to/model.onnx \\
        --intra_op 1 \\
        --repeats 3 \\
        --duration 10

Run only two configurations:

    python tools/perftest/benchmark_spin_settings.py \\
        --perf_test build/.../onnxruntime_perf_test \\
        --model path/to/model.onnx \\
        --configs default no_spin spin_1000_backoff_8

Notes
-----
The script relies on psutil (optional) for CPU usage measurement. If psutil is
not installed, the "CPU %" column will be empty.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    psutil: Any | None = importlib.import_module("psutil")
except ImportError:
    psutil = None

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinConfig:
    """A single spin-wait configuration.

    ``spin_duration_us`` maps to ``--spin_duration_us``:
        -1 = legacy iteration-count spinning (PR 27916 default)
         0 = pass the flag and disable spinning
        >0 = time-based spinning window

    ``spin_backoff_max`` maps to ``--spin_backoff_max``:
         1 = one SpinPause() per iteration (legacy)
        >1 = exponential backoff cap

    ``disable_spinning`` passes ``-D`` which forces ``allow_spinning = false``.
    """

    name: str
    spin_duration_us: int | None = None
    spin_backoff_max: int | None = None
    disable_spinning: bool = False

    def to_cli_args(self) -> list[str]:
        args: list[str] = []
        if self.disable_spinning:
            args.append("-D")
        if self.spin_duration_us is not None:
            args += ["--spin_duration_us", str(self.spin_duration_us)]
        if self.spin_backoff_max is not None:
            args += ["--spin_backoff_max", str(self.spin_backoff_max)]
        return args


PRESETS: dict[str, SpinConfig] = {
    # Baselines
    "default": SpinConfig("default"),
    "no_spin": SpinConfig("no_spin", disable_spinning=True),
    "spin_0": SpinConfig("spin_0", spin_duration_us=0),
    # PR 27916 knob alone (time-bounded spin, no backoff)
    "spin_500": SpinConfig("spin_500", spin_duration_us=500),
    "spin_1000": SpinConfig("spin_1000", spin_duration_us=1000),
    "spin_2000": SpinConfig("spin_2000", spin_duration_us=2000),
    # PR 23278 style (backoff, default duration)
    "backoff_4": SpinConfig("backoff_4", spin_backoff_max=4),
    "backoff_8": SpinConfig("backoff_8", spin_backoff_max=8),
    # Combined: time-bounded + exp backoff
    "spin_1000_backoff_4": SpinConfig("spin_1000_backoff_4", spin_duration_us=1000, spin_backoff_max=4),
    "spin_1000_backoff_8": SpinConfig("spin_1000_backoff_8", spin_duration_us=1000, spin_backoff_max=8),
    "spin_2000_backoff_8": SpinConfig("spin_2000_backoff_8", spin_duration_us=2000, spin_backoff_max=8),
}


# ---------------------------------------------------------------------------
# CPU monitor (optional, psutil-based)
# ---------------------------------------------------------------------------


class CpuMonitor:
    """Sample CPU% of a child process while it runs.

    Returns the mean over samples, or ``None`` if psutil is unavailable or
    sampling failed.
    """

    def __init__(self, interval: float = 0.2) -> None:
        self._interval = interval
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._proc_pid: int | None = None

    def start(self, pid: int) -> None:
        if psutil is None:
            return
        self._samples = []
        self._stop.clear()
        self._proc_pid = pid
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if psutil is None:
            return
        try:
            p = psutil.Process(self._proc_pid)
            p.cpu_percent(None)  # prime
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        while not self._stop.is_set():
            try:
                self._samples.append(p.cpu_percent(self._interval))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                return

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self._samples:
            return None
        # Drop the priming sample (first value is often 0).
        useful = self._samples[1:] if len(self._samples) > 1 else self._samples
        return statistics.mean(useful) if useful else None


# ---------------------------------------------------------------------------
# perf_test invocation + parsing
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config_name: str
    avg_latency_ms: float | None = None
    throughput_ips: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p99_ms: float | None = None
    cpu_percent: float | None = None
    wall_time_s: float = 0.0
    stdout_tail: str = ""


# Regexes cover the output lines emitted by ``onnxruntime_perf_test``.
_RE_AVG = re.compile(r"Average inference time cost(?:\s+\w+)?:\s+([\d.]+)\s*(\S+)", re.IGNORECASE)
_RE_THROUGHPUT = re.compile(r"(?:Number of inferences per second|Throughput):\s+([\d.]+)", re.IGNORECASE)
_RE_PCT = re.compile(
    r"P(?P<pct>\d+)\s+Latency(?:\s+is)?:?\s+(?P<val>[\d.]+)\s*(?P<unit>\S+)",
    re.IGNORECASE,
)


def _to_ms(value: float, unit: str) -> float:
    u = unit.lower().rstrip(".")
    if u in ("us", "µs", "microseconds"):
        return value / 1000.0
    if u in ("ns", "nanoseconds"):
        return value / 1_000_000.0
    if u in ("s", "sec", "seconds"):
        return value * 1000.0
    # default: assume milliseconds
    return value


def _parse_output(text: str, result: RunResult) -> None:
    m = _RE_AVG.search(text)
    if m:
        result.avg_latency_ms = _to_ms(float(m.group(1)), m.group(2))
    m = _RE_THROUGHPUT.search(text)
    if m:
        result.throughput_ips = float(m.group(1))
    for m in _RE_PCT.finditer(text):
        pct = int(m.group("pct"))
        val = _to_ms(float(m.group("val")), m.group("unit"))
        if pct == 50:
            result.p50_ms = val
        elif pct == 90:
            result.p90_ms = val
        elif pct == 99:
            result.p99_ms = val


def _run_once(
    perf_test: str,
    model: str,
    config: SpinConfig,
    intra_op: int,
    duration: int,
    extra_args: list[str],
    sample_cpu: bool,
) -> RunResult:
    cmd = [
        perf_test,
        "-t",
        str(duration),
        "-x",
        str(intra_op),
        "-I",  # generate model input binding
    ]
    cmd += config.to_cli_args()
    cmd += extra_args
    cmd += [model, os.devnull if os.name != "nt" else "NUL"]

    def _run_command(command: list[str]) -> tuple[int, str, float, float | None]:
        monitor = CpuMonitor() if sample_cpu else None
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            print(f"error: perf_test binary not found: {perf_test}", file=sys.stderr)
            sys.exit(2)

        if monitor is not None:
            monitor.start(proc.pid)

        stdout, _ = proc.communicate()
        wall_time_s = time.monotonic() - start
        cpu_percent = monitor.stop() if monitor is not None else None
        return proc.returncode, stdout, wall_time_s, cpu_percent

    result = RunResult(config_name=config.name)
    return_code, stdout, result.wall_time_s, result.cpu_percent = _run_command(cmd)

    if return_code != 0:
        result.stdout_tail = stdout[-2000:]
        return result

    _parse_output(stdout, result)
    result.stdout_tail = stdout[-2000:]
    return result


def _aggregate(results: list[RunResult]) -> RunResult:
    """Aggregate multiple repetitions by taking medians for latency and means for CPU/throughput."""
    if not results:
        return RunResult(config_name="(empty)")

    def _median(f: Callable[[RunResult], float | None]) -> float | None:
        vals = [value for r in results if (value := f(r)) is not None]
        return statistics.median(vals) if vals else None

    def _mean(f: Callable[[RunResult], float | None]) -> float | None:
        vals = [value for r in results if (value := f(r)) is not None]
        return statistics.mean(vals) if vals else None

    agg = RunResult(config_name=results[0].config_name)
    agg.avg_latency_ms = _median(lambda r: r.avg_latency_ms)
    agg.throughput_ips = _mean(lambda r: r.throughput_ips)
    agg.p50_ms = _median(lambda r: r.p50_ms)
    agg.p90_ms = _median(lambda r: r.p90_ms)
    agg.p99_ms = _median(lambda r: r.p99_ms)
    agg.cpu_percent = _mean(lambda r: r.cpu_percent)
    agg.wall_time_s = statistics.mean(r.wall_time_s for r in results)
    return agg


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "-"


def _print_table(rows: list[RunResult]) -> None:
    header = f"{'config':<26} {'avg ms':>9} {'p50 ms':>9} {'p90 ms':>9} {'p99 ms':>9} {'tput IPS':>10} {'cpu %':>8}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.config_name:<26} "
            f"{_fmt(r.avg_latency_ms):>9} "
            f"{_fmt(r.p50_ms):>9} "
            f"{_fmt(r.p90_ms):>9} "
            f"{_fmt(r.p99_ms):>9} "
            f"{_fmt(r.throughput_ips, '.1f'):>10} "
            f"{_fmt(r.cpu_percent, '.1f'):>8}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf_test", required=True, help="Path to the onnxruntime_perf_test binary.")
    ap.add_argument("--model", required=True, help="Path to the ONNX model to benchmark.")
    ap.add_argument("--intra_op", type=int, default=0, help="Intra-op thread count (0 = ORT default).")
    ap.add_argument("--duration", type=int, default=10, help="Per-run duration in seconds (-t).")
    ap.add_argument("--repeats", type=int, default=3, help="Runs per configuration (median reported).")
    ap.add_argument(
        "--configs",
        nargs="+",
        choices=[*list(PRESETS.keys()), "all"],
        default=["all"],
        help="Which presets to run.",
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any remaining args are forwarded verbatim to onnxruntime_perf_test (place after --).",
    )
    ap.add_argument(
        "--no_cpu",
        action="store_true",
        help="Skip CPU utilization sampling (faster, no psutil required).",
    )
    args = ap.parse_args()

    if not shutil.which(args.perf_test) and not os.path.exists(args.perf_test):
        print(f"error: --perf_test not found: {args.perf_test}", file=sys.stderr)
        return 2
    if not os.path.exists(args.model):
        print(f"error: --model not found: {args.model}", file=sys.stderr)
        return 2

    config_names = list(PRESETS.keys()) if "all" in args.configs else args.configs
    configs = [PRESETS[n] for n in config_names]

    # Strip leading '--' separator if user added one before --extra tokens.
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra

    print(f"perf_test      : {args.perf_test}")
    print(f"model          : {args.model}")
    print(f"intra_op       : {args.intra_op}")
    print(f"duration / run : {args.duration}s")
    print(f"repeats / cfg  : {args.repeats}")
    print(f"configurations : {', '.join(c.name for c in configs)}")

    aggregated: list[RunResult] = []
    for cfg in configs:
        runs: list[RunResult] = []
        print(f"\n=== {cfg.name} === (args: {' '.join(cfg.to_cli_args()) or '<none>'})")
        for i in range(args.repeats):
            print(f"  run {i + 1}/{args.repeats} ...", end="", flush=True)
            r = _run_once(
                args.perf_test,
                args.model,
                cfg,
                args.intra_op,
                args.duration,
                extra,
                sample_cpu=not args.no_cpu,
            )
            runs.append(r)
            print(
                f" avg={_fmt(r.avg_latency_ms)}ms "
                f"tput={_fmt(r.throughput_ips, '.1f')}IPS "
                f"cpu={_fmt(r.cpu_percent, '.1f')}%"
            )
        aggregated.append(_aggregate(runs))

    _print_table(aggregated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

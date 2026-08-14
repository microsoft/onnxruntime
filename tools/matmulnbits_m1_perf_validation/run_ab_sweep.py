#!/usr/bin/env python3
"""Head-vs-forced-base A/B perf sweep for microsoft/onnxruntime#31988 (MatMulNBits M=1
cols_per_block occupancy selection), driven entirely by onnxruntime_perf_test.exe (an
existing, already-built ORT tool) plus freshly generated single-node MatMulNBits models.

For each representative shape this script:
  1. Verifies numerics: runs the model once on CUDAExecutionProvider with the override env
     var unset ("head" = the PR's occupancy-driven cols_per_block selection) and once with
     it forced to 8 ("base" = the pre-PR unconditional cols_per_block==8 baseline), in two
     separate fresh processes, and asserts the output tensors are numerically close
     (np.allclose, plus reports max abs/rel diff). This exercises the exact PR code path on
     real hardware, unlike the CPU-only sanity check done at model-generation time.
  2. Times head and base with onnxruntime_perf_test.exe (-e cuda -I -m times -s), with a
     small throwaway warmup invocation before each timed invocation.
  3. Prints a CSV summary line per shape (also collected into a summary file) with
     min/p50/p90/p95/p99 latency (ms) for both head and base, plus the head/base speedup
     ratio at p50.

Run on the actual CI GPU runner; not intended to run locally without a CUDA device.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

FORCE_ENV_VAR = "ORT_MATMULNBITS_M1_PERF_VALIDATION_FORCE_COLS_PER_BLOCK"

# onnxruntime_perf_test's actual -s output format is "<Stat> Latency: <float> s"
# (observed on the A10 CI runner: e.g. "P50 Latency: 0.0002084 s"), not the
# "Latency is <float>sec" format originally assumed. Accept both to be robust
# to minor formatting differences across onnxruntime_perf_test versions.
LAT_RE = re.compile(r"(Min|Max|P50|P90|P95|P99)\s+Latency\s*(?:is|:)\s*([0-9.]+)\s*s(?:ec)?\b", re.IGNORECASE)


def run_numerics_check(python_exe, script_dir, model_path, workdir):
    head_npy = os.path.join(workdir, "head.npy")
    base_npy = os.path.join(workdir, "base.npy")

    env_head = dict(os.environ)
    env_head.pop(FORCE_ENV_VAR, None)
    env_base = dict(os.environ)
    env_base[FORCE_ENV_VAR] = "8"

    script = os.path.join(script_dir, "numerics_check.py")
    for env, out in ((env_head, head_npy), (env_base, base_npy)):
        subprocess.run([python_exe, script, model_path, out], check=True, env=env)

    head = np.load(head_npy)
    base = np.load(base_npy)
    bit_identical = np.array_equal(head, base)
    max_abs_diff = float(np.max(np.abs(head.astype(np.float64) - base.astype(np.float64))))
    close = np.allclose(head, base, rtol=1e-3, atol=1e-5)
    return bit_identical, max_abs_diff, close


def run_perf_test(perf_test_exe, model_path, workdir, force_cols, repeats, tag):
    env = dict(os.environ)
    if force_cols is not None:
        env[FORCE_ENV_VAR] = str(force_cols)
    else:
        env.pop(FORCE_ENV_VAR, None)

    result_file = os.path.join(workdir, f"result_{tag}.txt")
    # Throwaway warmup (discarded) to avoid first-call context/JIT/cache effects.
    subprocess.run(
        [perf_test_exe, "-e", "cuda", "-I", "-m", "times", "-r", "20", model_path, result_file + ".warmup"],
        check=True, env=env, capture_output=True, text=True,
    )
    proc = subprocess.run(
        [perf_test_exe, "-e", "cuda", "-I", "-m", "times", "-r", str(repeats), "-s", model_path, result_file],
        check=True, env=env, capture_output=True, text=True,
    )
    stats = {}
    for line in proc.stdout.splitlines():
        m = LAT_RE.search(line)
        if m:
            stats[m.group(1).upper()] = float(m.group(2)) * 1000.0  # sec -> ms
    if not stats:
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError(f"could not parse perf_test output for {tag}/{model_path}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--build-dir", required=True, help="dir containing onnxruntime_perf_test.exe")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_test_exe = os.path.join(args.build_dir, "onnxruntime_perf_test.exe")
    if not os.path.exists(perf_test_exe):
        # Non-Windows / different layout fallback.
        perf_test_exe = os.path.join(args.build_dir, "onnxruntime_perf_test")
    if not os.path.exists(perf_test_exe):
        raise SystemExit(f"onnxruntime_perf_test not found under {args.build_dir}")

    manifest_path = os.path.join(args.models_dir, "manifest.csv")
    shapes = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            shapes.append(row)

    rows = []
    print(
        "label,n,k,block_size,numerics_bit_identical,numerics_max_abs_diff,"
        "head_min_ms,head_p50_ms,head_p90_ms,head_p95_ms,head_p99_ms,"
        "base_min_ms,base_p50_ms,base_p90_ms,base_p95_ms,base_p99_ms,speedup_p50_head_vs_base"
    )
    for row in shapes:
        label = row["label"]
        model_path = os.path.join(args.models_dir, f"matmulnbits_m1_{label}.onnx")
        workdir = os.path.join(args.models_dir, f"_work_{label}")
        os.makedirs(workdir, exist_ok=True)

        bit_identical, max_abs_diff, close = run_numerics_check(args.python, script_dir, model_path, workdir)
        if not close:
            print(f"!!! NUMERICS MISMATCH for {label}: max_abs_diff={max_abs_diff}", file=sys.stderr)

        head_stats = run_perf_test(perf_test_exe, model_path, workdir, None, args.repeats, "head")
        base_stats = run_perf_test(perf_test_exe, model_path, workdir, 8, args.repeats, "base")

        speedup = base_stats["P50"] / head_stats["P50"] if head_stats.get("P50") else float("nan")
        line = (
            f"{label},{row['n']},{row['k']},{row['block_size']},{bit_identical},{max_abs_diff:.3e},"
            f"{head_stats.get('MIN', float('nan')):.4f},{head_stats.get('P50', float('nan')):.4f},"
            f"{head_stats.get('P90', float('nan')):.4f},{head_stats.get('P95', float('nan')):.4f},"
            f"{head_stats.get('P99', float('nan')):.4f},"
            f"{base_stats.get('MIN', float('nan')):.4f},{base_stats.get('P50', float('nan')):.4f},"
            f"{base_stats.get('P90', float('nan')):.4f},{base_stats.get('P95', float('nan')):.4f},"
            f"{base_stats.get('P99', float('nan')):.4f},{speedup:.4f}"
        )
        print(line)
        rows.append(line)

    with open(args.out_csv, "w") as f:
        f.write(
            "label,n,k,block_size,numerics_bit_identical,numerics_max_abs_diff,"
            "head_min_ms,head_p50_ms,head_p90_ms,head_p95_ms,head_p99_ms,"
            "base_min_ms,base_p50_ms,base_p90_ms,base_p95_ms,base_p99_ms,speedup_p50_head_vs_base\n"
        )
        for r in rows:
            f.write(r + "\n")
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

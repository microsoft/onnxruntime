#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Run a QMoE GEMV profiling sweep by launching profile_qmoe_gemv.py in a fresh
process for every candidate.

Examples:
  python profile_qmoe_gemv_sweep.py \
      --cases gpt_oss_20b_m1_top4_fp16_2880x2880_e32,qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256 \
      --block-sizes 0,32,64 --warmup 5 --repeat 100 --output /tmp/qmoe_sweep

  python profile_qmoe_gemv_sweep.py --mode gemv --cases all --block-sizes 64
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from test_qmoe_cuda import _QMOE_GEMV_BENCHMARK_RESULT_PREFIX, _qmoe_gemv_benchmark_cases


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _case_names(value: str) -> list[str]:
    if value == "all":
        return [case["name"] for case in _qmoe_gemv_benchmark_cases()]
    return _parse_csv_strings(value)


def _run_profile(command: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    result = None
    for line in completed.stdout.splitlines():
        if line.startswith(_QMOE_GEMV_BENCHMARK_RESULT_PREFIX):
            result = json.loads(line[len(_QMOE_GEMV_BENCHMARK_RESULT_PREFIX) :])
            break

    if completed.returncode != 0 or result is None:
        raise RuntimeError(
            "QMoE GEMV profiling command failed:\n"
            f"command: {' '.join(command)}\n"
            f"return code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    return result


def _result_without_streams(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CUDA QMoE GEMV benchmark cases")
    parser.add_argument("--cases", default="all", help="Comma-separated case names, or 'all'")
    parser.add_argument("--block-sizes", default="0,32,64,128", help="Comma-separated block sizes")
    parser.add_argument("--quant-bits", default="", help="Optional comma-separated quant bits overrides")
    parser.add_argument("--dtypes", default="", help="Optional comma-separated dtype overrides: FLOAT16,BFLOAT16")
    parser.add_argument("--mode", choices=["both", "gemv", "gemm"], default="both", help="Which route modes to run")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Measured iterations")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child profiler processes")
    parser.add_argument("--output", type=Path, help="Optional output path stem; writes .jsonl and .tsv")
    parser.add_argument("--keep-going", action="store_true", help="Continue after an unsupported or failed candidate")
    args = parser.parse_args()

    script = Path(__file__).with_name("profile_qmoe_gemv.py")
    modes = ["gemv", "gemm"] if args.mode == "both" else [args.mode]
    block_sizes = _parse_csv_ints(args.block_sizes)
    quant_bits = _parse_csv_ints(args.quant_bits) if args.quant_bits else [None]
    dtypes = _parse_csv_strings(args.dtypes) if args.dtypes else [None]

    results = []
    errors = []
    env = os.environ.copy()
    for case_name in _case_names(args.cases):
        for block_size in block_sizes:
            for quant_bit in quant_bits:
                for dtype in dtypes:
                    for mode in modes:
                        command = [
                            args.python,
                            str(script),
                            "--case",
                            case_name,
                            "--block-size",
                            str(block_size),
                            "--warmup",
                            str(args.warmup),
                            "--repeat",
                            str(args.repeat),
                        ]
                        if quant_bit is not None:
                            command.extend(["--quant-bits", str(quant_bit)])
                        if dtype is not None:
                            command.extend(["--dtype", dtype])
                        if mode == "gemm":
                            command.append("--disable-gemv")

                        try:
                            result = _run_profile(command, env)
                        except RuntimeError as exception:
                            if not args.keep_going:
                                raise

                            error = {
                                "block_size": block_size,
                                "case": case_name,
                                "dtype": dtype or "",
                                "error": str(exception),
                                "mode": mode,
                                "quant_bits": quant_bit or "",
                            }
                            errors.append(error)
                            print(json.dumps({"status": "error", **error}, sort_keys=True), file=sys.stderr)
                            continue

                        result["benchmark_case"] = case_name
                        result["mode"] = mode
                        results.append(result)
                        print(json.dumps(_result_without_streams(result), sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = args.output.with_suffix(".jsonl")
        tsv_path = args.output.with_suffix(".tsv")
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for result in results:
                jsonl_file.write(json.dumps(_result_without_streams(result), sort_keys=True))
                jsonl_file.write("\n")

            columns = [
                "benchmark_case",
                "case",
                "mode",
                "block_size",
                "quant_bits",
                "expanded_num_rows",
                "sm",
                "latency_ms",
                "has_invalid_output",
            ]
        with tsv_path.open("w", encoding="utf-8") as tsv_file:
            tsv_file.write("\t".join(columns) + "\n")
            for result in results:
                tsv_file.write("\t".join(str(result.get(column, "")) for column in columns) + "\n")

        if errors:
            errors_path = args.output.with_suffix(".errors.jsonl")
            with errors_path.open("w", encoding="utf-8") as errors_file:
                for error in errors:
                    errors_file.write(json.dumps(error, sort_keys=True))
                    errors_file.write("\n")


if __name__ == "__main__":
    main()

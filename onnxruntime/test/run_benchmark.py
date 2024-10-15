#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile


def warn(message: str):
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark (https://github.com/google/benchmark) program runner. "
        "Runs a benchmark program until the benchmark measurements are within the desired coefficient of variation "
        "(CV) (stddev / mean) tolerance and outputs those measurements."
    )

    parser.add_argument(
        "--program",
        required=True,
        type=pathlib.Path,
        help="Path to the benchmark program to run.",
    )

    parser.add_argument(
        "--pattern",
        required=True,
        dest="patterns",
        action="extend",
        nargs="+",
        help="Benchmark test name pattern to specify which benchmark tests to run. "
        "Each pattern value will have its own invocation of the benchmark program (passed to the benchmark program "
        "with the --benchmark_filter option). "
        "To list the benchmark test names, run the benchmark program with the --benchmark_list_tests option.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of benchmark run repetitions (passed to the benchmark program with the "
        "--benchmark_repetitions option).",
    )

    parser.add_argument(
        "--max-cv",
        type=float,
        default=0.05,
        help="Maximum allowed CV (stddev / mean) value. "
        "The CV value is a number, not a percentage. E.g., a value of 0.05 corresponds to 5%%.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of times to attempt running the benchmark program.",
    )

    parser.add_argument(
        "--show-program-output",
        action="store_true",
        help="Display the output from the benchmark program.",
    )

    return parser.parse_args()


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    median_real_time: float
    median_cpu_time: float
    time_unit: str


def run_benchmark(
    program: pathlib.Path,
    output_file: pathlib.Path,
    show_output: bool,
    pattern: str,
    repetitions: int,
    max_cv: float,
    max_attempts: int,
) -> list[BenchmarkResult]:
    benchmark_cmd = [
        f"{program}",
        f"--benchmark_filter={pattern}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_report_aggregates_only",
        f"--benchmark_out={output_file}",
        "--benchmark_out_format=json",
    ]

    def check_cv(entries):
        valid = True

        for entry in entries:
            if entry.get("aggregate_name") != "cv":
                continue

            run_name = entry["run_name"]

            real_time_cv = float(entry["real_time"])
            if real_time_cv > max_cv:
                warn(f"real_time CV exceeds limit for run '{run_name}': {real_time_cv} > {max_cv}")
                valid = False

            cpu_time_cv = float(entry["cpu_time"])
            if cpu_time_cv > max_cv:
                warn(f"cpu_time CV exceeds limit for run '{run_name}': {cpu_time_cv} > {max_cv}")
                valid = False

        return valid

    def process_entries(entries) -> list[BenchmarkResult]:
        results = []

        for entry in entries:
            if entry.get("aggregate_name") != "median":
                continue

            result = BenchmarkResult(
                name=entry["run_name"],
                median_real_time=float(entry["real_time"]),
                median_cpu_time=float(entry["cpu_time"]),
                time_unit=entry["time_unit"],
            )

            results.append(result)

        return results

    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        output_handle = None if show_output else subprocess.DEVNULL
        subprocess.run(
            benchmark_cmd,
            check=True,
            stdout=output_handle,
            stderr=output_handle,
            creationflags=subprocess.HIGH_PRIORITY_CLASS,
        )

        with open(output_file) as output:
            output_json = json.load(output)
            entries = output_json["benchmarks"]

        if not check_cv(entries):
            warn("Discarding benchmark run.")
            continue

        return process_entries(entries)

    raise RuntimeError("Failed to get measurements within the CV limit.")


def main():
    args = parse_args()

    program = args.program.resolve(strict=True)

    benchmark_results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = pathlib.Path(temp_dir_name)
        output_file = temp_dir / "benchmark.out.json"

        for pattern in args.patterns:
            benchmark_results += run_benchmark(
                program=program,
                output_file=output_file,
                show_output=args.show_program_output,
                pattern=pattern,
                repetitions=args.repetitions,
                max_cv=args.max_cv,
                max_attempts=args.max_attempts,
            )

    print("name|median_real_time|median_cpu_time")
    print("-|-|-")
    for result in benchmark_results:
        print(
            f"{result.name}|"
            f"{round(result.median_real_time)} {result.time_unit}|"
            f"{round(result.median_cpu_time)} {result.time_unit}"
        )


if __name__ == "__main__":
    main()

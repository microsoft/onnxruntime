# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compare release payload size and repeated inference measurements."""

import argparse
import gzip
import hashlib
import json
import math
import random
import shutil
import stat
import statistics
import subprocess
import tarfile
import tempfile
from pathlib import Path


def _payload_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(
            f"Payload directory {root} does not exist or is not a directory; provide an extracted payload."
        )
    files = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() or path.is_symlink()
    }
    if not files:
        raise ValueError(f"Payload directory {root} is empty; extract the native release payload before comparing.")
    return files


def _file_size(path: Path) -> int:
    if path.is_symlink():
        return len(path.readlink().as_posix().encode())
    return path.stat().st_size


def _is_elf(path: Path) -> bool:
    if path.is_symlink():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def _elf_section_size(path: Path, size_tool: str) -> int:
    output = subprocess.run([size_tool, "-A", path], check=True, capture_output=True, text=True).stdout
    for line in reversed(output.splitlines()):
        columns = line.split()
        if len(columns) == 2 and columns[0] == "Total":
            return int(columns[1])
    raise ValueError(f"{size_tool} did not report an ELF section total for {path}.")


def _stripped_measurements(root: Path, strip_tool: str, size_tool: str) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        stripped_root = Path(temporary_directory) / "payload"
        shutil.copytree(root, stripped_root, symlinks=True)
        for path in _payload_files(stripped_root).values():
            if _is_elf(path):
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
                subprocess.run([strip_tool, "--strip-unneeded", path], check=True)
        files = list(_payload_files(stripped_root).values())
        file_bytes = sum(_file_size(path) for path in files)
        section_bytes = sum(_elf_section_size(path, size_tool) for path in files if _is_elf(path))
        return file_bytes, section_bytes


def _compressed_size(root: Path) -> int:
    with tempfile.NamedTemporaryFile() as output:
        with (
            gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for relative, path in _payload_files(root).items():
                info = archive.gettarinfo(path, arcname=relative)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                if path.is_symlink():
                    archive.addfile(info)
                else:
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        output.flush()
        return output.tell()


def _payload_measurements(root: Path, strip_tool: str, size_tool: str) -> dict[str, object]:
    files = _payload_files(root)
    stripped_bytes, stripped_elf_section_bytes = _stripped_measurements(root, strip_tool, size_tool)
    return {
        "files": sorted(files),
        "raw_bytes": sum(_file_size(path) for path in files.values()),
        "stripped_bytes": stripped_bytes,
        "stripped_elf_section_bytes": stripped_elf_section_bytes,
        "compressed_bytes": _compressed_size(root),
        "sha256": {
            relative: hashlib.sha256(path.read_bytes()).hexdigest()
            for relative, path in files.items()
            if not path.is_symlink()
        },
    }


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_median_ratio(
    baseline: list[float], candidate: list[float], iterations: int = 20_000
) -> tuple[float, float]:
    random_generator = random.Random(0)
    ratios = []
    paired_ratios = [
        candidate_value / baseline_value for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    ]
    for _ in range(iterations):
        ratios.append(statistics.median(random_generator.choices(paired_ratios, k=len(paired_ratios))))
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def _load_benchmarks(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    workloads = data.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError(f"{path} must contain a non-empty 'workloads' object.")
    result = {}
    for name, samples in workloads.items():
        if not isinstance(samples, list) or len(samples) < 5:
            raise ValueError(f"{path}: workload '{name}' must contain at least five samples.")
        try:
            values = [float(sample) for sample in samples]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: workload '{name}' samples must be numbers.") from error
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{path}: workload '{name}' samples must be finite.")
        if any(value <= 0 for value in values):
            raise ValueError(f"{path}: workload '{name}' samples must be positive.")
        result[name] = values
    return result


def _compare_benchmarks(
    baseline_path: Path, candidate_path: Path, maximum_regression_percent: float
) -> tuple[dict[str, object], list[str]]:
    baseline = _load_benchmarks(baseline_path)
    candidate = _load_benchmarks(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate benchmark workload names must match.")
    for name in baseline:
        if len(baseline[name]) != len(candidate[name]):
            raise ValueError(f"Workload '{name}' must have the same number of baseline and candidate samples.")

    maximum_ratio = 1 + maximum_regression_percent / 100
    failures = []
    results = {}
    for name in sorted(baseline):
        baseline_median = statistics.median(baseline[name])
        candidate_median = statistics.median(candidate[name])
        paired_ratios = [
            candidate_value / baseline_value
            for baseline_value, candidate_value in zip(baseline[name], candidate[name], strict=True)
        ]
        ratio = statistics.median(paired_ratios)
        confidence_low, confidence_high = _bootstrap_median_ratio(baseline[name], candidate[name])
        if confidence_high <= maximum_ratio:
            status = "pass"
        elif confidence_low > maximum_ratio:
            status = "regression"
            failures.append(
                f"{name}: the 95% confidence interval is entirely above the {maximum_regression_percent:g}% limit"
            )
        else:
            status = "inconclusive"
            failures.append(f"{name}: the 95% confidence interval crosses the {maximum_regression_percent:g}% limit")
        results[name] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "candidate_to_baseline_ratio_of_medians": candidate_median / baseline_median,
            "paired_candidate_to_baseline_ratio_median": ratio,
            "paired_ratio_95_percent_confidence_interval": [confidence_low, confidence_high],
            "paired_ratios": paired_ratios,
            "status": status,
        }
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-payload", type=Path, required=True)
    parser.add_argument("--candidate-payload", type=Path, required=True)
    parser.add_argument("--strip-tool", required=True)
    parser.add_argument("--size-tool", required=True)
    parser.add_argument("--baseline-benchmarks", type=Path)
    parser.add_argument("--candidate-benchmarks", type=Path)
    parser.add_argument("--max-performance-regression-percent", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    benchmark_arguments = (
        args.baseline_benchmarks,
        args.candidate_benchmarks,
        args.max_performance_regression_percent,
    )
    if any(value is not None for value in benchmark_arguments) and not all(
        value is not None for value in benchmark_arguments
    ):
        parser.error("Provide both benchmark files and --max-performance-regression-percent, or none of them.")
    if args.max_performance_regression_percent is not None:
        if not math.isfinite(args.max_performance_regression_percent):
            parser.error("--max-performance-regression-percent must be finite.")
        if args.max_performance_regression_percent < 0:
            parser.error("--max-performance-regression-percent cannot be negative.")

    baseline_payload = _payload_measurements(args.baseline_payload, args.strip_tool, args.size_tool)
    candidate_payload = _payload_measurements(args.candidate_payload, args.strip_tool, args.size_tool)
    failures = []
    if baseline_payload["files"] != candidate_payload["files"]:
        failures.append("Baseline and candidate payload file sets differ.")
    for metric in ("raw_bytes", "stripped_bytes", "stripped_elf_section_bytes", "compressed_bytes"):
        if candidate_payload[metric] > baseline_payload[metric]:
            failures.append(
                f"{metric}: candidate {candidate_payload[metric]} exceeds baseline {baseline_payload[metric]}"
            )

    report: dict[str, object] = {
        "size_policy": (
            "Candidate raw, stripped, stripped ELF section, and deterministic tar.gz bytes must not exceed baseline."
        ),
        "payload": {
            "baseline": baseline_payload,
            "candidate": candidate_payload,
        },
    }
    if args.baseline_benchmarks is not None:
        benchmark_results, benchmark_failures = _compare_benchmarks(
            args.baseline_benchmarks,
            args.candidate_benchmarks,
            args.max_performance_regression_percent,
        )
        failures.extend(benchmark_failures)
        report["performance_policy"] = (
            "The upper bound of the bootstrapped 95% median-ratio confidence interval must not exceed "
            f"{args.max_performance_regression_percent:g}% regression."
        )
        report["benchmarks"] = benchmark_results

    report["failures"] = failures
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

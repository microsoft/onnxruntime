# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Collect Linux perf FP counts and generate scoped MatMulNBits comparison reports."""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .run_matmul_nbits_experiments import load_manifest
else:
    from run_matmul_nbits_experiments import load_manifest


EVENT = "fp_ret_sse_avx_ops.all:u"
LIMITATIONS = (
    "Counts measure recurring process FP work, not isolated operator work. Independent-process subtraction "
    "reduces but does not guarantee cancellation of startup and warm-up work. Estimates cover modeled components "
    "only; INT8 activation quantization, auxiliary reductions and tile repetition may be excluded. Signed differences "
    "are not operator accuracy percentages. Repetition assesses variability, not systematic counter bias. "
    "Elapsed process time is not isolated inference timing."
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def positive(value: int, name: str):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def counter_value(path: Path, event: str) -> int:
    with path.open(encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.reader((line for line in stream if line.strip() and not line.startswith("#")), delimiter=";")
            if len(row) >= 5 and row[2].strip() == event
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {event} counter in {path}")
    try:
        count = int(rows[0][0])
        running = float(rows[0][4])
    except ValueError as error:
        raise ValueError(f"Unsupported/uncounted event or malformed counter: {path}") from error
    if count <= 0 or not math.isfinite(running) or running != 100.0:
        raise ValueError(f"Counter must be positive and report 100% running time: {path}")
    return count


def validate_results(results: dict) -> list[dict]:
    if results.get("complete") is not True:
        raise ValueError("Estimator results must be complete")
    cases = results.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Estimator results must contain cases")
    ids = set()
    for case in cases:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or Path(case_id).name != case_id or case_id in ("", ".", ".."):
            raise ValueError("Invalid case ID")
        if case_id in ids or case.get("status") != "pass":
            raise ValueError(f"Duplicate or non-pass case: {case_id}")
        ids.add(case_id)
        estimate = case["expected_flops"].get("expected_flops")
        if not isinstance(estimate, str) or not estimate.isascii() or not estimate.isdecimal():
            raise ValueError(f"Missing available expected-FLOP count: {case_id}")
    return cases


def workload(model: Path, inputs: Path, repeats: int, warmup: int, threads: int):
    import onnxruntime as ort  # noqa: PLC0415

    positive(repeats, "repeats")
    positive(warmup, "warmup")
    positive(threads, "threads")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(inputs, allow_pickle=False) as data:
        feeds = {"A": data["A"]}
    for _ in range(warmup):
        session.run(["Y"], feeds)
    for _ in range(repeats):
        session.run(["Y"], feeds)


def collect(manifest: Path, results_path: Path, output: Path, repeats: int, trials: int, warmup: int):
    import onnxruntime as ort  # noqa: PLC0415

    for name, value in (("repeats", repeats), ("trials", trials), ("warmup", warmup)):
        positive(value, name)
    manifest = manifest.resolve()
    results = read_json(results_path)
    cases = validate_results(results)
    records = {record["case_id"]: record for record, _ in load_manifest(manifest)}
    if set(records) != {case["case_id"] for case in cases}:
        raise ValueError("Manifest and estimator result case IDs differ")
    for case in cases:
        if case["settings"] != records[case["case_id"]]["settings"]:
            raise ValueError(f"Manifest/result settings mismatch: {case['case_id']}")
    runtime = results["runtime"]
    if ort.get_build_info() != runtime["onnxruntime_build_info"] or ort.__version__ != runtime["onnxruntime_version"]:
        raise ValueError("Installed ORT build differs from estimator run; rerun estimator checks first")
    execution = results["execution"]
    if (execution["provider"], execution["mode"], execution["graph_optimization"]) != (
        "CPUExecutionProvider",
        "sequential",
        "disabled",
    ):
        raise ValueError("Expected sequential CPU results with graph optimizations disabled")
    threads = execution["threads"]
    positive(threads, "threads")
    perf_version = subprocess.check_output(["perf", "--version"], text=True).strip()
    script = Path(__file__).resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "complete": False,
        "trials": trials,
        "repetitions": [repeats, 2 * repeats],
        "warmup_runs": warmup,
        "threads": threads,
        "event": EVENT,
        "runtime_file": ort.__file__,
        "runtime_build": ort.get_build_info(),
        "python": sys.executable,
        "platform": platform.platform(),
        "cpu_info": subprocess.check_output(["lscpu"], text=True).strip(),
        "perf_version": perf_version,
        "source_manifest": str(manifest),
        "source_results": str(results_path.resolve()),
        "limitations": LIMITATIONS,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "estimates.json", results)
    write_json(output / "manifest.json", read_json(manifest))
    index = []
    for trial in range(1, trials + 1):
        for case in cases:
            case_id = case["case_id"]
            directory = output / "hardware" / case_id
            directory.mkdir(parents=True, exist_ok=True)
            for count in (repeats, 2 * repeats):
                stem = directory / f"trial-{trial}-repeats-{count}"
                command = [
                    "perf",
                    "stat",
                    "--no-big-num",
                    "-x",
                    ";",
                    "-o",
                    str(stem.with_suffix(".csv")),
                    "-e",
                    EVENT,
                    "--",
                    sys.executable,
                    str(script),
                    "workload",
                    "--model",
                    str(manifest.parent / records[case_id]["model"]),
                    "--inputs",
                    str(manifest.parent / records[case_id]["inputs"]),
                    "--repeats",
                    str(count),
                    "--warmup",
                    str(warmup),
                    "--threads",
                    str(threads),
                ]
                run = subprocess.run(command, capture_output=True, text=True, check=False)
                stem.with_suffix(".log").write_text(run.stdout + run.stderr, encoding="utf-8")
                index.append(
                    {
                        "case_id": case_id,
                        "trial": trial,
                        "repeats": count,
                        "returncode": run.returncode,
                        "counter_file": str(stem.with_suffix(".csv").relative_to(output)),
                        "command": command,
                    }
                )
                write_json(output / "index.json", index)
                if run.returncode:
                    raise ValueError(f"perf/workload failed; inspect {stem.with_suffix('.log')}")
                counter_value(stem.with_suffix(".csv"), EVENT)
            print(f"Collected trial {trial}/{trials}: {case_id}", flush=True)
    write_reports(output, results, output)
    metadata.update(complete=True, measurements=len(index))
    write_json(output / "metadata.json", metadata)


def write_reports(collection: Path, results: dict, output: Path):
    """Recompute every trial from raw counters, including historical inline collections."""
    collection = collection.resolve()
    metadata = read_json(collection / "metadata.json")
    cases = validate_results(results)
    expected_ids = {case["case_id"] for case in cases}
    positive(metadata["trials"], "trials")
    low, high = metadata["repetitions"]
    positive(low, "lower repetitions")
    if type(high) is not int or high != low * 2:
        raise ValueError("Expected R and 2R repetitions")
    event = metadata["event"]
    if event != EVENT:
        raise ValueError(f"Unsupported event semantics: {event}")
    counts = {}
    for entry in read_json(collection / "index.json"):
        key = (entry["case_id"], entry["trial"], entry["repeats"])
        if (
            key in counts
            or key[0] not in expected_ids
            or type(key[1]) is not int
            or not 1 <= key[1] <= metadata["trials"]
            or key[2] not in (low, high)
            or entry["returncode"] != 0
        ):
            raise ValueError(f"Unexpected, duplicate or failed measurement: {key}")
        path = (collection / entry["counter_file"]).resolve()
        if not path.is_relative_to(collection):
            raise ValueError("Counter path is outside collection")
        counts[key] = counter_value(path, event)
    if len(counts) != len(cases) * metadata["trials"] * 2:
        raise ValueError("Incomplete trial matrix")
    trial_rows = []
    summary = []
    for case in cases:
        values = []
        for trial in range(1, metadata["trials"] + 1):
            first = counts[(case["case_id"], trial, low)]
            second = counts[(case["case_id"], trial, high)]
            measured = (second - first) / (high - low)
            if measured <= 0:
                raise ValueError(f"Nonpositive differential: {case['case_id']}, trial {trial}")
            values.append(measured)
            trial_rows.append(
                {
                    "case_id": case["case_id"],
                    "trial": trial,
                    "count_low": first,
                    "count_high": second,
                    "measured_process_fp_ops_per_invocation": measured,
                }
            )
        settings = case["settings"]
        estimate = int(case["expected_flops"]["expected_flops"])
        mean = statistics.mean(values)
        summary.append(
            {
                "case_id": case["case_id"],
                "dtype": settings["dtype"],
                "accuracy_level": settings["accuracy_level"],
                "m": math.prod(settings["a_shape"][:-1]),
                "k": settings["a_shape"][-1],
                "n": settings["n"],
                "route": case["route"],
                "compute_mode": case["compute_mode"],
                "estimated_modeled_fp_ops": estimate,
                "measured_mean": mean,
                "measured_min": min(values),
                "measured_max": max(values),
                "measured_sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
                "signed_difference_percent": 100 * (mean - estimate) / estimate if estimate else None,
            }
        )
    for name in ("trials.csv", "summary.csv", "report.md", "trial-results.json"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / name}")
    for name, rows in (("trials.csv", trial_rows), ("summary.csv", summary)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "trial-results.json", trial_rows)
    modes = defaultdict(int)
    for case in cases:
        modes[case["compute_mode"]] += 1
    lines = [
        "# MatMulNBits CPU profiling report",
        "",
        f"- Runtime: `{metadata.get('runtime_build', 'not recorded')}`",
        f"- Event: `{event}`; accepted counters report 100% running time.",
        f"- {len(cases)} cases; {metadata['trials']} trial pairs per case; repetitions {low}/{high}.",
        f"- Warm-up runs: {metadata['warmup_runs']}; intra-op threads: {metadata['threads']}.",
        "- CPU sequential execution; graph optimizations and estimator reporting disabled during profiling.",
        "- INT4/other weight settings and bias/zero points are recorded per case in the estimator results.",
        f"- Observed compute-mode coverage: `{dict(modes)}`.",
        "",
        "## Scope and limitations",
        "",
        LIMITATIONS,
        "",
        "Sample standard deviation describes the derived per-trial counts, not latency. "
        "Identical trial values yield zero standard deviation; a single trial has no sample standard deviation.",
        "",
        "## Accuracy request key",
        "",
        "| Request | Meaning |",
        "|---:|---|",
        "| 0 | Default/unset |",
        "| 1 | FP32 accuracy |",
        "| 2 | Permit FP16-level computation |",
        "| 3 | Permit BF16-level computation |",
        "| 4 | Permit INT8 activation quantization/computation |",
        "",
        "Requests permit choices; they do not force a mode or change weight bit width. "
        "The Mode column is observed, not inferred from the request.",
        "",
        "## Results",
        "",
        "| Mode | Input dtype | Accuracy request | M | K | N | Estimated FP ops | Measured mean | "
        "Sample stddev | Min | Max | Difference |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        stddev = row["measured_sample_stddev"]
        diff = row["signed_difference_percent"]
        stddev_text = f"{stddev:,.3f}" if stddev is not None else "N/A"
        diff_text = f"{diff:+.2f}%" if diff is not None else "N/A"
        lines.append(
            f"| {row['compute_mode']} | {row['dtype']} | {row['accuracy_level']} | {row['m']} | {row['k']} "
            f"| {row['n']} | {row['estimated_modeled_fp_ops']:,} | {row['measured_mean']:,.3f} "
            f"| {stddev_text} | {row['measured_min']:,.3f} | {row['measured_max']:,.3f} | {diff_text} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    collect_parser = commands.add_parser("collect", help="Run R/2R processes for each case and generate reports")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--results", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--repeats", type=int, default=1000)
    collect_parser.add_argument("--trials", type=int, default=3)
    collect_parser.add_argument("--warmup", type=int, default=100)
    report_parser = commands.add_parser("report", help="Regenerate documentation from existing raw counters")
    report_parser.add_argument("--collection", type=Path, required=True)
    report_parser.add_argument("--results", type=Path, help="Defaults to collection/estimates.json")
    report_parser.add_argument("--output", type=Path, required=True, help="New report directory")
    worker = commands.add_parser("workload", help="Internal single-case subprocess launched by perf")
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--inputs", type=Path, required=True)
    worker.add_argument("--repeats", type=int, required=True)
    worker.add_argument("--warmup", type=int, default=100)
    worker.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.action == "collect":
            collect(args.manifest, args.results, args.output, args.repeats, args.trials, args.warmup)
        elif args.action == "report":
            results = read_json(args.results or args.collection / "estimates.json")
            args.output.mkdir(parents=True, exist_ok=False)
            write_reports(args.collection, results, args.output)
        else:
            workload(args.model, args.inputs, args.repeats, args.warmup, args.threads)
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Profiling/report failed: {error}\n")


if __name__ == "__main__":
    main()

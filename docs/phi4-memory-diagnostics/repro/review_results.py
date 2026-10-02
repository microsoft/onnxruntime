# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Offline compact evidence review; optional complete historical capture comparison.

Median grouping is adapted from the original evidence assembly's reporting
logic. Raw-capture checks retain the historical test_phi4_logits_parity checks.
Neither mode imports a runtime, initializes NVML, nor accesses the network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
WORKLOADS = {2048: 1801, 32768: 28747}
VARIANTS = ("baseline", "patched")
EARLIER_INPUTS = {2048: 1801, 4096: 3596, 8192: 7189, 16384: 14376, 32768: 28747}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def read_csv(path: Path, text_columns: set[str]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(bool(reader.fieldnames), f"Missing CSV header: {path.name}")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate CSV header: {path.name}")
        require(text_columns <= set(reader.fieldnames), f"Missing text columns: {path.name}")
        rows = []
        for number, row in enumerate(reader, 2):
            require(None not in row and None not in row.values(), f"Malformed CSV row {number}: {path.name}")
            for name, value in row.items():
                if name not in text_columns:
                    numeric = float(value)
                    require(math.isfinite(numeric) and numeric >= 0, f"Invalid {name} at row {number}: {path.name}")
                    row[name] = numeric
            rows.append(row)
    require(bool(rows), f"Empty CSV: {path.name}")
    return rows


def check_grid(rows: list[dict], group_keys: tuple[str, ...], groups: set[tuple]) -> None:
    actual = {tuple(row[key] for key in group_keys) for row in rows}
    require(actual == groups, "Unexpected or missing context/runtime groups")
    for group in groups:
        repetitions = sorted(row["repetition"] for row in rows if tuple(row[key] for key in group_keys) == group)
        require(repetitions == [1, 2, 3], f"Missing or duplicate repetitions in {group}")


def verify_integrity(package: Path) -> int:
    """Check a complete package checksum manifest, excluding the manifest itself."""
    package = package.resolve()
    seen = set()
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        require(match is not None, "Malformed SHA256SUMS entry")
        expected, name = match.groups()
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe checksum path")
        require(name not in seen and name != "SHA256SUMS", "Duplicate/self checksum entry")
        target = (package / relative).resolve()
        require(target.is_relative_to(package) and target.is_file(), f"Missing or external checksum file: {name}")
        require(digest(target.read_bytes()) == expected, f"Checksum mismatch: {name}")
        seen.add(name)
    actual = {
        file.relative_to(package).as_posix()
        for file in package.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts and file.name != "SHA256SUMS"
    }
    require(seen == actual, f"Checksum coverage differs: {sorted(seen ^ actual)}")
    return len(seen)


def equal_bytes(left: bytes, right: bytes, size: int, expected_hash: str) -> None:
    require(len(left) == len(right) == size, "Unexpected capture byte count")
    require(left == right, "Baseline and patched bytes differ")
    require(digest(left) == expected_hash, "Capture hash differs from pinned evidence")


def check_raw_context(root: Path, context: int, manifest: dict, parity: dict) -> None:
    """Retain full-vector, input/configuration and recorded-identity checks."""
    spec = manifest["workloads"][str(context)]
    baseline, patched = (root / f"{variant}-{context}" for variant in VARIANTS)
    for variant, directory in zip(VARIANTS, (baseline, patched), strict=True):
        ids = (directory / "input-ids.i32le").read_bytes()
        require(len(ids) == spec["input_tokens"] * 4, "Wrong original Phi input length")
        require(digest(ids) == spec["input_ids_sha256"], "Wrong original Phi input hash")
        require(ids == (directory / "loaded-input-ids.i32le").read_bytes(), "Loaded inputs differ")
        config = read_json(directory / "effective-config.json")
        require(
            config == spec["effective_config"] == read_json(directory / "applied-config.json"),
            "Effective/applied config differs",
        )
        require(read_json(directory / "search-readback.json") == config["search"], "Search options differ")
        metadata = read_json(directory / "logits.json")
        require(metadata["shape"] == [1, 1, 200064] and metadata["dtype"] == "float32", "Wrong logits shape/dtype")
        require(
            metadata["finite_count"] == 200064
            and metadata["nan_count"]
            == metadata["positive_infinity_count"]
            == metadata["negative_infinity_count"]
            == 0,
            "Nonfinite logits metadata",
        )
        values = struct.unpack("<200064f", (directory / "logits.bin").read_bytes())
        require(all(math.isfinite(value) for value in values), "Nonfinite raw logits")
        tokens = json.loads((directory / "generated-ids.json").read_text(encoding="utf-8"))
        require(
            isinstance(tokens, list) and len(tokens) == 64 and all(type(t) is int and 0 <= t < 200064 for t in tokens),
            "Expected 64 valid generated IDs",
        )
        require(
            struct.pack("<64i", *tokens) == (directory / "generated-ids.i32le").read_bytes(),
            "Token JSON/binary disagreement",
        )
        state = read_json(directory / "capture-state.json")
        require(
            state["input_tokens"] == spec["input_tokens"]
            and state["generated_tokens"] == 64
            and state["max_length"] == spec["input_tokens"] + 64
            and state["chunk_size"] == state["shrink_calls"] == 0,
            "Workload differs",
        )
        for when in ("before", "after"):
            loaded = read_json(directory / f"loaded-hashes-{when}.json")
            for name, expected in manifest["runtimes"]["phase_a"][variant].items():
                require(
                    [value for file, value in loaded.items() if Path(file).name == name] == [expected],
                    f"Missing/mixed recorded loaded runtime: {name}",
                )
    equal_bytes(
        (baseline / "logits.bin").read_bytes(),
        (patched / "logits.bin").read_bytes(),
        800256,
        parity["logits"]["contexts"][str(context)]["baseline_sha256"],
    )
    equal_bytes(
        (baseline / "generated-ids.i32le").read_bytes(),
        (patched / "generated-ids.i32le").read_bytes(),
        256,
        parity["generated_tokens"]["contexts"][str(context)]["baseline_sha256"],
    )


def check_compact(package: Path) -> dict:
    evidence = package / "evidence"
    manifest = read_json(evidence / "model-manifest.json")
    parity = read_json(evidence / "parity-checks.json")
    memory = read_csv(evidence / "memory-before-after.csv", {"variant"})
    earlier = read_csv(evidence / "earlier-runtime-comparison.csv", {"runtime", "ort_allocator_mode"})
    check_grid(memory, ("context", "variant"), {(context, variant) for context in WORKLOADS for variant in VARIANTS})
    early_groups = {
        (context, runtime, allocator)
        for context in EARLIER_INPUTS
        for runtime, allocator in (("ort", "baseline"), ("ort", "device"), ("llama_cpp", "n/a"))
    }
    check_grid(earlier, ("context_tokens", "runtime", "ort_allocator_mode"), early_groups)
    for row in earlier:
        require(row["prompt_tokens"] == EARLIER_INPUTS[row["context_tokens"]], "Earlier input count differs")
        require(row["completion_tokens"].is_integer() and row["completion_tokens"] > 0, "Invalid earlier output count")
        require(row["sampling_interval_ms"] == 10, "Earlier requested sampling interval differs")
    require(set(manifest["workloads"]) == {"2048", "32768"}, "Unexpected manifest workload set")
    require(set(parity["logits"]["contexts"]) == set(manifest["workloads"]), "Logits context set differs")
    require(set(parity["generated_tokens"]["contexts"]) == set(manifest["workloads"]), "Token context set differs")
    for context, input_count in WORKLOADS.items():
        key = str(context)
        spec = manifest["workloads"][key]
        search = spec["effective_config"]["search"]
        require(
            spec["input_tokens"] == input_count
            and spec["output_tokens"] == 64
            and search["max_length"] == input_count + 64
            and search["chunk_size"] == 0
            and search["batch_size"] == search["num_beams"] == 1
            and search["do_sample"] is False,
            f"Historical settings differ: {key}",
        )
        require(re.fullmatch(r"[a-f0-9]{64}", spec["input_ids_sha256"]) is not None, "Malformed input hash")
        logits = parity["logits"]["contexts"][key]
        tokens = parity["generated_tokens"]["contexts"][key]
        for result in (logits, tokens):
            require(re.fullmatch(r"[a-f0-9]{64}", result["baseline_sha256"]) is not None, "Malformed result hash")
            require(result["baseline_sha256"] == result["patched_sha256"], "Result hashes differ")
        require(
            logits["byte_identical"] is True
            and logits["shape"] == [1, 1, 200064]
            and logits["dtype"] == "float32"
            and logits["bytes"] == 800256
            and logits["finite_values"] == 200064,
            "Unexpected logits metadata",
        )
        require(tokens["matching_token_count"] == tokens["generated_token_count"] == 64, "Token count differs")
    token_runs = parity["generated_tokens"]["phase_b_runs"]
    check_grid(token_runs, ("context", "variant"), {(c, v) for c in WORKLOADS for v in VARIANTS})
    for row in token_runs:
        require(
            row["matches_phase_a"] is True
            and row["matching_token_count"] == 64
            and row["sha256"] == parity["generated_tokens"]["contexts"][str(row["context"])]["baseline_sha256"],
            "Phase B output reference differs",
        )
    for row in memory:
        require(
            row["input_tokens"] == WORKLOADS[row["context"]] and row["output_tokens"] == 64, "CSV token count differs"
        )
        require(
            row["whole_device_inference_peak_mib"] >= row["process_inference_peak_mib"], "Process exceeds device peak"
        )
        for checkpoint in ("post_initialize", "post_generation", "post_generator_cleanup"):
            for location in ("device", "pinned_host"):
                prefix = f"{checkpoint}_{location}"
                require(
                    math.isclose(
                        row[f"{prefix}_capacity_mib"],
                        row[f"{prefix}_live_mib"] + row[f"{prefix}_slack_mib"],
                        abs_tol=1e-6,
                        rel_tol=0,
                    ),
                    f"Arena capacity/live/slack disagree: {prefix}",
                )
    allocation_summary = []
    for context, input_count in WORKLOADS.items():
        for variant in VARIANTS:
            rows = [row for row in memory if row["context"] == context and row["variant"] == variant]
            allocation_summary.append(
                {
                    "context": context,
                    "variant": variant,
                    "repetitions": len(rows),
                    "input_tokens": input_count,
                    "output_tokens": 64,
                    **{
                        column: statistics.median(row[column] for row in rows)
                        for column in (
                            "process_inference_peak_mib",
                            "whole_device_inference_peak_mib",
                            "post_generator_cleanup_device_capacity_mib",
                        )
                    },
                }
            )
    earlier_summary = []
    for context, runtime, allocator in sorted(early_groups):
        rows = [
            r
            for r in earlier
            if (r["context_tokens"], r["runtime"], r["ort_allocator_mode"]) == (context, runtime, allocator)
        ]
        require(all(row["completion_tokens"] <= 64 for row in rows), "Earlier output exceeds original cap")
        if runtime == "ort":
            require(all(row["completion_tokens"] == 64 for row in rows), "Historical ORT output count differs")
        earlier_summary.append(
            {
                "context": context,
                "runtime": runtime,
                "allocator": allocator,
                "repetitions": len(rows),
                **{
                    column: statistics.median(row[column] for row in rows)
                    for column in (
                        "prompt_tokens",
                        "completion_tokens",
                        "peak_inference_vram_mib",
                        "ttft_ms",
                        "decode_tokens_per_second",
                    )
                },
            }
        )
    readme = (package / "README.md").read_text(encoding="utf-8")
    for row in allocation_summary:
        line = (
            f"| {row['context']} | {row['variant'].capitalize()} | {row['process_inference_peak_mib']:g} | "
            f"{row['whole_device_inference_peak_mib']:.3f} | {row['post_generator_cleanup_device_capacity_mib']:g} |"
        )
        require(line in readme, f"Published allocation table differs: {row['context']}/{row['variant']}")
    for row in earlier_summary:
        label = f"ORT / {row['allocator']}" if row["runtime"] == "ort" else "llama.cpp"
        line = (
            f"| {row['context']} | {row['prompt_tokens']:g} | {label} | {row['completion_tokens']:g} | "
            f"{row['peak_inference_vram_mib']:.2f} | {row['ttft_ms']:.3f} | {row['decode_tokens_per_second']:.3f} |"
        )
        require(line in readme, f"Published earlier table differs: {row['context']}/{label}")
    return {
        "compact_consistency": "passed",
        "allocation_runs": len(memory),
        "earlier_runs": len(earlier),
        "phase_b_token_records": len(token_runs),
        "allocation_medians": allocation_summary,
        "earlier_medians": earlier_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE, help="Package directory with evidence and README")
    parser.add_argument("--check-integrity", action="store_true", help="Also verify complete SHA256SUMS coverage")
    parser.add_argument("--raw-captures", type=Path, help="Approved external historical Phase A capture directory")
    parser.add_argument("--output", type=Path, help="Write summary JSON to a NEW file; default: stdout")
    args = parser.parse_args()
    try:
        package = args.package.resolve(strict=True)
        if args.output:
            require(not args.output.resolve().is_relative_to(package), "Write results outside the immutable package")
            require(not args.output.exists(), "Output already exists; refusing to overwrite")
        count = verify_integrity(package) if args.check_integrity else None
        result = check_compact(package)
        result["package_integrity"] = (
            {"status": "passed", "files": count} if count is not None else {"status": "not_run"}
        )
        result["raw_vector_comparison"] = "not_run: external captures not supplied"
        if args.raw_captures is not None:
            require(args.raw_captures.is_dir(), "External raw capture directory is missing")
            manifest = read_json(package / "evidence/model-manifest.json")
            parity = read_json(package / "evidence/parity-checks.json")
            for context in WORKLOADS:
                check_raw_context(args.raw_captures, context, manifest, parity)
            result["raw_vector_comparison"] = "passed: both historical full logits and token pairs"
        result["fresh_gpu_reproduction"] = "not_run"
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"Evidence review failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

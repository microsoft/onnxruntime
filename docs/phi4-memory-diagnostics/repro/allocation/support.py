# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Host-only identity, correctness, measurement and report contracts."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "../../evidence/model-manifest.json"
GENAI_COMMIT = "ed5f4e87147731e5b07810f9f5c90103b3603cdf"
ORT_COMMIT = "94e76459dc9a759888707aa430d51a95d878da8f"
COUNTS = {2048: 1801, 32768: 28747}
INPUT_HASHES = {
    2048: "a7100ece1396cf13e81616381b11fed562406cadf539c72a7cae8c955b0877f4",
    32768: "fbe6fe0929666ac5a91dd152aea07bc2bf7675c765f31ad57e701604ccaa3831",
}
VARIANTS = ("baseline", "patched")
PLAN = [(context, repetition, variant) for context in COUNTS for repetition in (1, 2, 3) for variant in VARIANTS]
CHECKPOINTS = ("post_initialize", "post_generation", "post_generator_cleanup")
ARENA_BYTE_FIELDS = (
    "total_allocated_bytes",
    "reserved_bytes",
    "bfc_region_bytes",
    "bytes_in_use",
    "bytes_requested_in_use",
    "arena_slack_bytes",
    "internal_fragmentation_bytes",
    "max_bytes_in_use",
    "max_alloc_size",
)
ARENA_COUNT_FIELDS = (
    "num_allocs",
    "num_reserves",
    "num_arena_extensions",
    "num_arena_shrinkages",
)
ARENA_FIELDS = ARENA_BYTE_FIELDS + ARENA_COUNT_FIELDS
METRICS = (
    "process_inference_peak_mib",
    "whole_device_inference_peak_mib",
    "whole_device_peak_minus_pre_run_mib",
    "pre_run_whole_device_mib",
    "ttft_ms",
    "decode_tokens_per_second",
)
MIB = 1048576
RUNTIME_PREFIXES = ("libonnxruntime-genai", "libonnxruntime.so", "libonnxruntime_providers_")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_input(raw, context):
    require(context in COUNTS, "Unsupported nominal context")
    require(len(raw) == COUNTS[context] * 4, "Saved input byte count mismatch")
    require(hashlib.sha256(raw).hexdigest() == INPUT_HASHES[context], "Saved input SHA-256 mismatch")
    tokens = struct.unpack(f"<{COUNTS[context]}i", raw)
    require(all(0 <= token < 200064 for token in tokens), "Saved token outside vocabulary")
    return tokens


def verify_config(config, context):
    search = config["search"]
    require(
        search["chunk_size"] == 0
        and search["batch_size"] == 1
        and search["num_beams"] == 1
        and search["do_sample"] is False
        and search["max_length"] == COUNTS[context] + 64,
        "Workload configuration changed",
    )
    options = config["model"]["decoder"]["session_options"]
    require("enable_profiling" not in options, "Profiling must be absent")
    require(options.get("log_severity_level", 3) >= 2, "Verbose/info session logging is not allowed")
    require(
        options["provider_options"] == [{"CUDA": {"device_id": "0"}}]
        and options["session.use_device_allocator_for_initializers"] == "0",
        "Provider/allocator changed",
    )


def verify_maps(path, expected):
    loaded = set()
    for line in Path(path).read_text().splitlines():
        index = line.find("/")
        if index >= 0:
            loaded.add(str(Path(line[index:]).resolve()))
    runtimes = {p for p in loaded if Path(p).name.startswith(RUNTIME_PREFIXES)}
    require(runtimes == set(expected), f"Missing/mixed runtime paths: {runtimes ^ set(expected)}")
    for name, digest in expected.items():
        require(sha256(name) == digest, f"Changed runtime: {name}")
    additional = {p for p in loaded if Path(p).name.startswith(("libcudart.", "libcudnn", "libcublas"))}
    return {p: sha256(p) for p in sorted(runtimes | additional)}


def verify_tokens(directory):
    directory = Path(directory)
    tokens = load(directory / "generated-ids.json")
    require(
        len(tokens) == 64 and all(type(t) is int and 0 <= t < 200064 for t in tokens),
        "Invalid generated token sequence",
    )
    raw = struct.pack("<64i", *tokens)
    require(raw == (directory / "generated-ids.i32le").read_bytes(), "Output binary mismatch")
    return raw


def verify_logits(directory):
    directory = Path(directory)
    metadata = load(directory / "logits.json")
    raw = (directory / "logits.bin").read_bytes()
    require(
        metadata["dtype"] == "float32"
        and metadata["shape"] == [1, 1, 200064]
        and metadata["byte_count"] == 800256
        and metadata["nan_count"] == 0
        and len(raw) == 800256,
        "Invalid or incomplete full prefill logits",
    )
    values = struct.unpack("<200064f", raw)
    require(
        not any(math.isnan(x) for x in values) and sum(map(math.isfinite, values)) >= 2,
        "NaN or insufficient finite full logits",
    )
    return raw


def compare_phase_a(left, right):
    logits = [verify_logits(directory) for directory in (left, right)]
    tokens = [verify_tokens(directory) for directory in (left, right)]
    require(logits[0] == logits[1], "Full prefill logits are not byte-identical; Phase B prohibited")
    require(tokens[0] == tokens[1], "64-token sequences differ; Phase B prohibited")
    require(
        load(Path(left) / "effective-config.json") == load(Path(right) / "effective-config.json"),
        "Phase A effective configs differ",
    )
    return {
        "logits_exact_bytes_equal": True,
        "generated_ids_exact_equal": True,
        "logits_sha256": hashlib.sha256(logits[0]).hexdigest(),
        "generated_ids_sha256": hashlib.sha256(tokens[0]).hexdigest(),
    }


def test_result(path, code, expected_count, expected_failure=False):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    require(not list(root.iter("skipped")) and not list(root.iter("error")), "Regression skips/errors")
    failures = [(case.attrib["name"], failure.text or "") for case in cases for failure in case.findall("failure")]
    require(len(cases) == expected_count, "Missing selected regression tests")
    if expected_failure:
        require(
            code == 1
            and len(failures) == 6
            and {name for name, _ in failures} == {f"NoPromptSizedFp32/{i}" for i in range(6)}
            and all("UNWANTED_PROMPT_SIZED_FP32" in text for _, text in failures),
            "Baseline failed for a reason other than oversized-allocation assertion",
        )
    else:
        require(code == 0 and not failures, "Unexpected regression failure")
    return {
        "tests": len(cases),
        "failures": len(failures),
        "exit_code": code,
        "expected_failure_confirmed": expected_failure,
    }


def check_interference(sample):
    require(not sample["process_query_errors"], "Cannot monitor GPU interference")
    require(not sample["foreign_pids"], f"GPU interference: foreign PIDs {sample['foreign_pids']}")


def timing_metrics(timing):
    start, first, end = (timing[key] for key in ("start_ns", "first_token_ns", "end_ns"))
    require(
        timing["clock"] == "CLOCK_MONOTONIC" and timing["generation_steps"] == 64,
        "Unexpected timing clock or output length",
    )
    require(
        start < timing["first_sync_start_ns"] <= first < timing["final_sync_start_ns"] <= end,
        "Invalid timing boundaries",
    )
    return {
        "ttft_ms": (first - start) / 1e6,
        "decode_tokens_per_second": 63 * 1e9 / (end - first),
        "decode_ms": (end - first) / 1e6,
        "total_inference_ms": (end - start) / 1e6,
        "first_completion_sync_ms": (first - timing["first_sync_start_ns"]) / 1e6,
        "final_completion_sync_ms": (end - timing["final_sync_start_ns"]) / 1e6,
    }


def inference_samples(samples, timing):
    selected = [
        s for s in samples if timing["start_ns"] <= s["memory_query_start_ns"] and s["query_end_ns"] <= timing["end_ns"]
    ]
    require(len(selected) >= 2, "Too few complete NVML queries inside inference")
    require(
        all(s["process_present"] and s["process_used_bytes"] is not None for s in selected),
        "NVML process attribution unavailable during inference",
    )
    for sample in selected:
        check_interference(sample)
    return selected


def interval_stats(samples):
    gaps = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e6 for a, b in itertools.pairwise(samples)]
    require(bool(gaps), "Insufficient NVML samples")
    return {
        "count": len(samples),
        "requested_ms": 10,
        "min_ms": min(gaps),
        "median_ms": statistics.median(gaps),
        "max_ms": max(gaps),
        "p95_ms": sorted(gaps)[int((len(gaps) - 1) * 0.95)],
        "intervals_over_15_ms": sum(gap > 15 for gap in gaps),
    }


def parse_arenas(text):
    result = {checkpoint: {"Cuda": [], "CudaPinned": []} for checkpoint in CHECKPOINTS}
    completions = {}
    for line in text.splitlines():
        if not line.startswith(("[ ARENA CHECKPOINT ]", "[ ARENA SHRINK ]")):
            continue
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        checkpoint = fields["checkpoint"]
        require(checkpoint in result, f"Unexpected checkpoint: {checkpoint}")
        if line.startswith("[ ARENA SHRINK ]"):
            require(checkpoint not in completions, "Duplicate snapshot completion")
            require(fields["requested"] == "0" and fields["reclaimed_bytes"] == "0", "Unexpected shrinking")
            completions[checkpoint] = fields
            continue
        name = fields["allocator"]
        require(name in ("Cuda", "CudaPinned"), f"Unexpected allocator: {name}")
        require(fields["device_id"] == "0" and fields["phase"] == "snapshot", "Unexpected arena device/phase")
        record = {key: int(fields[key]) for key in ARENA_FIELDS}
        require(all(value >= 0 for value in record.values()), "Negative arena field")
        require(record["num_arena_shrinkages"] == 0, "Arena shrinkage observed")
        require(
            record["total_allocated_bytes"] == record["bfc_region_bytes"] + record["reserved_bytes"],
            "Arena accounting mismatch",
        )
        require(
            record["arena_slack_bytes"] == record["total_allocated_bytes"] - record["bytes_in_use"], "Slack mismatch"
        )
        require(
            record["internal_fragmentation_bytes"] == record["bytes_in_use"] - record["bytes_requested_in_use"],
            "Fragmentation mismatch",
        )
        result[checkpoint][name].append(record)
    for checkpoint, groups in result.items():
        require(any(a["total_allocated_bytes"] > 0 for a in groups["Cuda"]), f"No CUDA arena: {checkpoint}")
        require(checkpoint in completions, f"Missing snapshot completion: {checkpoint}")
        require(
            int(completions[checkpoint]["arena_count"]) == len(groups["Cuda"]) + len(groups["CudaPinned"]),
            "Arena count mismatch",
        )
        for name, group in (("Cuda", "device_sums"), ("CudaPinned", "pinned_sums")):
            groups[group] = {field: sum(a[field] for a in groups[name]) for field in ARENA_FIELDS}
    return result


def distribution(values):
    require(bool(values) and all(math.isfinite(v) for v in values), "Missing/nonfinite report values")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(records):
    require(len(records) == 12 and len({r["pid"] for r in records}) == 12, "Incomplete/duplicate process matrix")
    require([(r["context"], r["repetition"], r["variant"]) for r in records] == PLAN, "Run ordering changed")
    require(
        all(
            r["status"] == "complete"
            and r["phase_a_output_exact_match"] is True
            and r["interference_detected"] is False
            for r in records
        ),
        "Unverified measurement records",
    )
    groups, differences = {}, {}
    for context, input_count in COUNTS.items():
        rows = {}
        for variant in VARIANTS:
            rows[variant] = [r for r in records if r["context"] == context and r["variant"] == variant]
            result = {key: distribution([r[key] for r in rows[variant]]) for key in METRICS}
            result["arenas_mib"] = {
                checkpoint: {
                    group: {
                        field: distribution([r["arenas"][checkpoint][group][field] / MIB for r in rows[variant]])
                        for field in ARENA_BYTE_FIELDS
                    }
                    for group in ("device_sums", "pinned_sums")
                }
                for checkpoint in CHECKPOINTS
            }
            result["arena_counts"] = {
                checkpoint: {
                    group: {
                        field: distribution([r["arenas"][checkpoint][group][field] for r in rows[variant]])
                        for field in ARENA_COUNT_FIELDS
                    }
                    for group in ("device_sums", "pinned_sums")
                }
                for checkpoint in CHECKPOINTS
            }
            groups[f"{context}-{variant}"] = result
        pairs = list(zip(rows["baseline"], rows["patched"], strict=True))
        delta = {key: distribution([a[key] - b[key] for a, b in pairs]) for key in METRICS}
        delta["device_arena_mib"] = {
            checkpoint: {
                field: distribution(
                    [
                        (a["arenas"][checkpoint]["device_sums"][field] - b["arenas"][checkpoint]["device_sums"][field])
                        / MIB
                        for a, b in pairs
                    ]
                )
                for field in ("total_allocated_bytes", "bytes_in_use", "arena_slack_bytes")
            }
            for checkpoint in CHECKPOINTS
        }
        delta["removed_tensor_request_bytes"] = input_count * 200064 * 4
        differences[str(context)] = delta
    return {"completed": 12, "groups": groups, "paired_baseline_minus_patched": differences}

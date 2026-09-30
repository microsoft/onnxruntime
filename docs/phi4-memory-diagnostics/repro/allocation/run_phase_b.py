# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Twelve fresh-process measurements after a revalidated Phase A gate."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import monitor
import run_validation as phase_a
import support as s


def analyze_capture(directory, reference, context, repetition, variant):
    """Use actual native outputs and NVML samples, never historical summary replay."""
    state = s.load(directory / "status.json")
    s.require(state["exit_code"] == 0 and state["status"] in ("exited", "complete"), "Measurement child failed")
    output = s.verify_tokens(directory)
    s.require(output == s.verify_tokens(reference), "Output differs from matching Phase A reference")
    text = (directory / "stdout.log").read_text()
    stderr = (directory / "stderr.log").read_text()
    s.require("[ SUCCESS ] Phase B measurement" in text, "Missing native success marker")
    s.require(
        not any(
            marker in text + stderr
            for marker in (
                "[ LOGITS ALLOCATION ]",
                "Extending BFCArena",
                "Extended allocation by",
                "Creating BFCArena",
                "Writing profiler data",
            )
        ),
        "Forbidden trace/profile output",
    )
    arenas = s.parse_arenas(text)
    timing = s.load(directory / "timing.json")
    metrics = s.timing_metrics(timing)
    protocol = s.load(directory / "protocol.json")
    s.require(
        set(protocol) == {"go_ns", "ack_ns"}
        and protocol["go_ns"] <= timing["start_ns"] < timing["end_ns"] <= protocol["ack_ns"],
        "Cross-process timing boundaries disagree",
    )
    calls = [json.loads(line) for line in (directory / "snapshot-calls.jsonl").read_text().splitlines()]
    s.require([call["checkpoint"] for call in calls] == list(s.CHECKPOINTS), "Missing/reordered snapshots")
    s.require(
        all(
            call["shrink"] is False
            and call["reclaimed_bytes"] == 0
            and call["arena_count"] > 0
            and call["start_ns"] <= call["end_ns"]
            for call in calls
        ),
        "Invalid or shrinking snapshot",
    )
    s.require(
        calls[0]["end_ns"] < timing["start_ns"]
        and timing["end_ns"] < calls[1]["start_ns"] <= calls[1]["end_ns"] < calls[2]["start_ns"],
        "Snapshots overlap timed inference",
    )
    phase_a.verify_capture_state(directory, context, True)
    samples = [json.loads(line) for line in (directory / "nvml.jsonl").read_text().splitlines()]
    for sample in samples:
        s.check_interference(sample)
    selected = s.inference_samples(samples, timing)
    for name in ("input-ids.i32le", "loaded-input-ids.i32le"):
        s.validate_input((directory / name).read_bytes(), context)
    effective = s.load(directory / "effective-config.json")
    s.verify_config(effective, context)
    s.require(
        effective == s.load(reference / "effective-config.json") == s.load(directory / "applied-config.json")
        and effective["search"] == s.load(directory / "search-readback.json"),
        "Measurement config mismatch",
    )
    expected = s.load(reference / "identity.json")["runtime_sha256"]
    s.require(s.load(directory / "identity.json")["runtime_sha256"] == expected, "Measurement runtime mismatch")
    for when in ("before", "after"):
        loaded = s.load(directory / f"loaded-hashes-{when}.json")
        s.require(
            {path: digest for path, digest in loaded.items() if Path(path).name.startswith(s.RUNTIME_PREFIXES)}
            == expected,
            "Loaded runtime mismatch",
        )
    before = [json.loads(line) for line in (directory / "pre-run-nvml.jsonl").read_text().splitlines()]
    s.require(len(before) == 50, "Incomplete idle baseline")
    for sample in before:
        s.check_interference(sample)
        s.require(sample["gpu_utilization_percent"] == 0, "GPU was not idle")
    baseline = statistics.median(sample["whole_device_used_bytes"] for sample in before) / s.MIB
    peak = max(sample["whole_device_used_bytes"] for sample in selected) / s.MIB
    return {
        "context": context,
        "repetition": repetition,
        "variant": variant,
        "pid": state["pid"],
        "status": "complete",
        "exit_code": 0,
        "generated_tokens": 64,
        "output_ids_sha256": hashlib.sha256(output).hexdigest(),
        "phase_a_output_exact_match": True,
        "interference_detected": False,
        "process_inference_peak_mib": max(sample["process_used_bytes"] for sample in selected) / s.MIB,
        "whole_device_inference_peak_mib": peak,
        "pre_run_whole_device_mib": baseline,
        "whole_device_peak_minus_pre_run_mib": peak - baseline,
        "inference_sample_count": len(selected),
        "sampling_intervals": s.interval_stats(samples),
        "inference_sampling_intervals": s.interval_stats(selected),
        "inference_start_to_first_query_ms": (selected[0]["memory_query_start_ns"] - timing["start_ns"]) / 1e6,
        "inference_last_query_to_end_ms": (timing["end_ns"] - selected[-1]["query_end_ns"]) / 1e6,
        "arenas": arenas,
        "timing": timing,
        **metrics,
    }


def capture(config, root, context, repetition, variant, nv, handle, identities):
    phase_a.verify_phase_a(root.parent / "phase-a", identities)
    directory = root / f"context-{context}-rep-{repetition}-{variant}"
    directory.mkdir()
    identity, command, ready = phase_a.prepare(config, directory, context, variant, "b")
    s.require(identity["runtime_sha256"] == identities[variant], "Runtime changed after Phase A")
    state, _ = monitor.launch(config, directory, command, variant, nv, handle, "b", ready)
    s.require(state["exit_code"] == 0, f"Phase B failed: {directory}")
    s.save(
        directory / "loaded-hashes-after.json", s.verify_maps(directory / "loaded-maps-after.txt", identities[variant])
    )
    reference = root.parent / "phase-a" / f"{variant}-{context}"
    record = analyze_capture(directory, reference, context, repetition, variant)
    s.save(directory / "record.json", record)
    return record


def write_reports(root, records):
    summary = s.summarize(records)
    s.save(root / "summary.json", summary)
    with (root / "runs.csv").open("x", newline="") as stream:
        fields = ("context", "repetition", "variant", "pid", *s.METRICS, "inference_sample_count")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (root / "arenas.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("context", "repetition", "variant", "checkpoint", "group", *s.ARENA_FIELDS)
        )
        writer.writeheader()
        for row in records:
            for checkpoint in s.CHECKPOINTS:
                for group in ("device_sums", "pinned_sums"):
                    writer.writerow(
                        {key: row[key] for key in ("context", "repetition", "variant")}
                        | {"checkpoint": checkpoint, "group": group}
                        | row["arenas"][checkpoint][group]
                    )
    return summary


def report(root):
    root = Path(root).resolve()
    s.require(s.load(root / "status.json")["status"] == "complete", "Fresh run is not complete")
    manifest = s.load(root / "manifest.json")
    identities = manifest["host_check"]["runtime_sha256"]
    phase_a.verify_phase_a(root / "phase-a", identities)
    records = []
    for context, repetition, variant in s.PLAN:
        directory = root / "measurement-results" / f"context-{context}-rep-{repetition}-{variant}"
        reference = root / "phase-a" / f"{variant}-{context}"
        record = analyze_capture(directory, reference, context, repetition, variant)
        s.require(record == s.load(directory / "record.json"), "Stored measurement differs from raw capture")
        records.append(record)
    s.require(records == s.load(root / "measurement-results/records.json"), "Recorded matrix changed")
    summary = s.summarize(records)
    s.require(summary == s.load(root / "measurement-results/summary.json"), "Summary differs from raw capture")
    return summary

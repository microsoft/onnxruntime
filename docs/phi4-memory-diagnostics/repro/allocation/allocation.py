# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Allocation-fix reproduction. Help/config/check/report are host-only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config as settings
import run_phase_b
import run_validation
import support as s


def artifact_check():
    s.require(
        s.sha256(s.HERE / "production.diff") == "34723df6327bb3430ff00a05d8329d3c69660ceaf67a7efd3d2b268314124e38",
        "Historical production.diff changed",
    )
    s.require(
        s.sha256(s.HERE / "historical-tests-normalized.patch")
        == "1485beda53090cc53f4c1cfc0a19c3938d2f4f720a881632e7398d25eb7f038b",
        "Historical regression patch changed",
    )
    manifest = s.load(s.MANIFEST)
    for context in s.COUNTS:
        workload = manifest["workloads"][str(context)]
        s.require(
            workload["input_tokens"] == s.COUNTS[context] and workload["input_ids_sha256"] == s.INPUT_HASHES[context],
            "Manifest input mismatch",
        )
        s.verify_config(workload["effective_config"], context)
    source = (s.HERE / "native/phase_b.cpp").read_text()
    start = source.index("const int64_t start = MonotonicNs();")
    end = source.index("const int64_t end = MonotonicNs();")
    timed = source[start:end]
    s.require(
        "const int64_t first_token_ns = MonotonicNs();" in timed and timed.count("Synchronize();") == 2,
        "Phase B completion barriers changed",
    )
    s.require(
        not any(word in timed for word in ("Snapshot(", "Write(", "GetSequenceData", "GetLogits")),
        "Diagnostic work inside inference timing",
    )
    s.require(
        "GetLogits" not in source and "diagnostic(checkpoint, false," in source,
        "Phase B must not inspect logits or shrink",
    )
    return {
        "artifact_check_passed": True,
        "gpu_execution_validated": False,
        "manifest": str(s.MANIFEST.resolve()),
        "source_pins": {"genai": s.GENAI_COMMIT, "ort": s.ORT_COMMIT},
    }


def execute(config):
    artifact_check()
    checked = settings.check(config)
    root = Path(config["output_dir"]).resolve()
    s.require(not root.exists(), f"Refusing existing output directory: {root}; no resume/retry mode")
    root.mkdir(parents=True)
    state = {"status": "preflight", "phase_b_completed": 0, "retry_attempted": False}
    s.save(root / "status.json", state)
    s.save(root / "config.json", config)
    artifacts = [
        path
        for path in s.HERE.rglob("*")
        if path.is_file()
        and (path.suffix in (".py", ".cpp", ".h", ".sh", ".diff", ".patch") or path.name == "CMakeLists.txt")
    ]
    manifest = {
        "host_check": checked,
        "source_pins": config["source_pins"],
        "runner_sha256": {str(path.relative_to(s.HERE)): s.sha256(path) for path in artifacts},
        "plan": s.PLAN,
        "sampling_requested_ms": 10,
        "timing_definition": "TTFT: AppendTokens start through first-token completion sync; decode: 63/(end-first)",
        "snapshots_outside_timing": list(s.CHECKPOINTS),
        "synchronization": "init/generator/cleanup outside; first and 64th token inside timing",
        "fresh_gpu_replay_validated_by_packaging": False,
    }
    s.save(root / "manifest.json", manifest)
    initialized = False
    try:
        # Only this command imports NVML; all prerequisite checks above are CPU-only.
        import pynvml as nv  # noqa: PLC0415 - GPU dependencies must never load during host-only commands.

        try:
            nv.nvmlInit()
            initialized = True
            handle = nv.nvmlDeviceGetHandleByIndex(config["gpu_index"])
            uuid = nv.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            s.require(uuid == config["gpu_uuid"], "Selected physical GPU does not match configured UUID")
            manifest["gpu"] = {
                "uuid": uuid,
                "physical_index": config["gpu_index"],
                "name": str(nv.nvmlDeviceGetName(handle)),
                "driver": str(nv.nvmlSystemGetDriverVersion()),
            }
            s.save(root / "manifest.json", manifest)
            identities = checked["runtime_sha256"]
            state["status"] = "regression-tests"
            s.save(root / "status.json", state)
            run_validation.tests(config, root / "test-results", nv, handle, identities)
            state["status"] = "phase-a"
            s.save(root / "status.json", state)
            run_validation.phase_a(config, root / "phase-a", nv, handle, identities)
            run_validation.verify_phase_a(root / "phase-a", identities)
            s.require(settings.check(config)["runtime_sha256"] == identities, "Prerequisites changed after Phase A")
            measurement_root = root / "measurement-results"
            measurement_root.mkdir()
            records = []
            for context, repetition, variant in s.PLAN:
                state.update(status="phase-b", context=context, repetition=repetition, variant=variant)
                s.save(root / "status.json", state)
                records.append(
                    run_phase_b.capture(config, measurement_root, context, repetition, variant, nv, handle, identities)
                )
                s.save(measurement_root / "records.json", records)
                state["phase_b_completed"] = len(records)
            run_phase_b.write_reports(measurement_root, records)
            s.require(settings.check(config)["runtime_sha256"] == identities, "Prerequisites changed during replay")
            state["status"] = "complete"
        except nv.NVMLError as error:
            raise RuntimeError(f"NVML prerequisite/monitor failed: {error}") from error
        finally:
            if initialized:
                nv.nvmlShutdown()
    finally:
        if state["status"] != "complete":
            state["status"] = "failed"
        s.save(root / "status.json", state)
    return run_phase_b.report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="Print an explicit external-prerequisite JSON template; no imports/GPU")
    check_parser = commands.add_parser(
        "check", help="Host-only artifact check; with config, check external prerequisites"
    )
    check_parser.add_argument("--config", type=Path)
    fixtures_parser = commands.add_parser("check-fixtures", help="Host-only validation of pinned external tiny models")
    fixtures_parser.add_argument("--models-dir", type=Path, required=True)
    run_parser = commands.add_parser("run", help="GPU execution: regressions -> full-logits Phase A gate -> Phase B")
    run_parser.add_argument("--config", type=Path, required=True)
    report_parser = commands.add_parser("report", help="Host-only revalidation of an actual complete fresh capture")
    report_parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            print((s.HERE / "config.example.json").read_text(), end="")
            return 0
        if args.command == "check":
            result = artifact_check()
            if args.config:
                result["prerequisites"] = settings.check(s.load(args.config))
        elif args.command == "check-fixtures":
            result = settings.verify_fixtures(args.models_dir)
        elif args.command == "run":
            result = execute(s.load(args.config))
        else:
            result = run_phase_b.report(args.results)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

import collections
import json
import math
from pathlib import Path
import re
import subprocess

from build_qmoe_prefill256 import FLAG, ROWS
from prepare_qmoe_extended_prefill import DIRECTORY
from qualify_quickgelu_mul_long_context import ROOT, sha


def gate_rows(event):
    match = re.search(r"inputs\[0\] = \{([^}]+)\}", event["args"]["shapes"])
    assert match, event
    shape = [int(d.strip()) for d in match[1].split(",")]
    assert len(shape) >= 2 and shape[-1] == 512 and all(d > 0 for d in shape), shape
    return math.prod(shape[:-1])


def verify_gate_coverage(events, expected_shapes=None):
    shapes = collections.Counter()
    for event in events:
        if event.get("cat") == "Kernel" and event["name"].endswith("&QmoeGate1Token"):
            shapes[gate_rows(event)] += 1
    expected_shapes = {16: 208, 128: 8, 256: 8} if expected_shapes is None else expected_shapes
    assert shapes == expected_shapes, shapes
    return dict(shapes)


def verify_additional_dispatches(events, expected):
    counts = collections.Counter()
    for event in events:
        if event.get("cat") == "Kernel":
            name = event["name"].rsplit("&", 1)[-1]
            if name in expected:
                counts[name] += 1
    assert dict(counts) == expected, (dict(counts), expected)
    return dict(counts)


def main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1"),
         prefix=ROOT / "qmoe-prefill256-v1-native-gpu7-20261029",
         parent_path=ROOT / "qsa-indexer-stack-v1-contexts-20261029.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=(129, 255, 257), expected_shapes=None,
         report_status="prefill256_retained_chunk16_finite_bitwise_and_dispatch_qualified", extra_flags="",
         additional_dispatches=None, verify_existing=False):
    repos = Path("/home/sayanshaw/repos")
    if prefix.with_suffix(".qualification.json").exists():
        raise FileExistsError(prefix)
    if verify_existing:
        for suffix in (".guard.json", ".log"):
            if not prefix.with_suffix(suffix).is_file():
                raise FileNotFoundError(prefix.with_suffix(suffix))
    elif prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    fixtures = sorted(DIRECTORY.glob("*.onnx"))
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    assert len(fixtures) == 6 and Path(manifest["source_weights"]).samefile(DIRECTORY / "model.onnx.data")
    for fixture in fixtures:
        evidence = manifest["fixtures"][fixture.name]
        assert sha(fixture) == evidence["fixture_sha256"]
        assert sha(Path(evidence["source"])) == evidence["source_sha256"]
    base = json.loads((ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029.guard.json").read_text())
    command = base["command"].copy()
    old = "/workspace/qwen38-int4-qmoe-extended-prefill/build-v1"
    new = str(directory).replace(str(repos), "/workspace")
    assert command[-1].count(old) == 2
    exports = "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1" + (f" {extra_flags}" if extra_flags else "")
    command[-1] = f"export {exports}; " + command[-1].replace(old, new)
    if not verify_existing:
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
            "--output-prefix", str(prefix), "--", *command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    assert guard["command"] == command and "FAIL" not in log and "GetCapability" not in log
    assert log.count("retained_chunk16_reference=1 invalid_flag_rejected=1") == len(fixtures)
    comparisons = len(fixtures) * 2560 * sum(r * (2 if r in fallback_rows else 8) for r in rows)
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    profiles = [Path(line.removeprefix("PROFILE ").replace("/evidence/webgpu-perf-oct20", str(ROOT)))
                for line in log.splitlines() if line.startswith("PROFILE ")]
    assert len(profiles) == len(fixtures)
    coverage = {p.name: verify_gate_coverage(json.loads(p.read_text()), expected_shapes) for p in profiles}
    report = dict(status=report_status,
        original_layers=len(fixtures), rows_tested=list(rows), rounds_per_indirect_shape=8,
        fallback_rows=list(fallback_rows), fallback_rounds=2, pending_windows_tested=[1, 64],
        invalid_flag_explicitly_rejected=True, selector=selector, default_off=True,
        finite_bitwise_comparisons=comparisons, gate_shape_dispatches=coverage,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_manifest=manifest,
        source_sha256={p.name: sha(p) for p in [*profiles, prefix.with_suffix(".guard.json"),
                                               prefix.with_suffix(".log")]},
        reference_arithmetic="retained_chunk16_not_generic_large_batch",
        quality_qualified=False, headline_eligible=False, verified_existing_evidence=verify_existing)
    if additional_dispatches is not None:
        report["additional_program_dispatches"] = {
            p.name: verify_additional_dispatches(json.loads(p.read_text()), additional_dispatches)
            for p in profiles}
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "original_layers", "finite_bitwise_comparisons")}),
          flush=True)


if __name__ == "__main__":
    main()

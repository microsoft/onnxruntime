import argparse
import json
from pathlib import Path
import subprocess

from build_qmoe_prefill1024 import FALLBACK_ROWS, ROWS
from build_qmoe_row4_static_weights import FLAG, PROGRAM
from build_qmoe_node_replay import replace_once
from prepare_qmoe_extended_prefill import DIRECTORY as FIXTURES
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_grouped_prefill_native import PROGRAMS
from run_qmoe_prefill256_native import main as run_mode, verify_additional_dispatches, verify_gate_coverage


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-row4-static-weights/build-v1")
VARIANT = "qmoe-row4-static-weights-v1"


def invalid_command(qualified, comparator, provider, fixture, target):
    assert qualified[:2] == ["bash", "-lc"]
    assert qualified[-1].count("; exec ") == 1
    head = qualified[-1].split("; exec ", 1)[0]
    head = replace_once(head, f"{FLAG}=1 ", f"{FLAG}=2 ")
    return head + (
        f"; unset {FLAG}_STATS_FILE; if {comparator} {provider} {fixture} > {target} 2>&1; then "
        "echo 'FAIL: invalid row4 static-weight selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' {target}")


def verify_completed_mode(prefix, mode, build, additional):
    proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert proof["status"] == "row4_static_weights_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert proof["provider_sha256"] == build["provider_sha256"]
    assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"]
    assert proof["comparator_sha256"] == build["comparator_sha256"]
    assert proof["original_layers"] == 6 and proof["finite_bitwise_comparisons"] == 300810240
    assert proof["rows_tested"] == list(ROWS) and proof["fallback_rows"] == list(FALLBACK_ROWS)
    assert proof["pending_windows_tested"] == [1, 64] and proof["invalid_flag_explicitly_rejected"]
    assert proof["fixture_manifest"] == json.loads((FIXTURES / "manifest.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    assert f"{FLAG}={mode} " in guard["command"][-1]
    profiles = [name for name in proof["source_sha256"]
                if name.endswith(".json") and not name.endswith(".guard.json")]
    assert len(profiles) == 6
    for name, digest in proof["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name in profiles:
        events = json.loads((ROOT / name).read_text())
        shapes = verify_gate_coverage(events, {16: 976, 128: 8, 256: 8, 512: 8, 1024: 8})
        assert {str(key): value for key, value in shapes.items()} == proof["gate_shape_dispatches"][name]
        assert verify_additional_dispatches(events, additional) == proof["additional_program_dispatches"][name]
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-completed-modes", action="store_true")
    args = parser.parse_args()
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["all_152_original_source_objects_byte_identical"] and build["no_object_layout_changes"]
    assert build["baseline_provider_sha256"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    assert build["comparator_profile_prefix_only_changed"]
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    output = ROOT / f"{VARIANT}-native-gpu7-20261101.qualification.json"
    assert not output.exists()
    evidence = [build_path, Path(__file__), FIXTURES / "manifest.json",
                ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
                ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029.guard.json"]
    evidence.extend(ROOT / name for name in (
        "build_qmoe_row4_static_weights.py", "build_scoped_webgpu_source.py", "build_qmoe_node_replay.py",
        "run_qmoe_prefill256_native.py", "run_qmoe_grouped_prefill_native.py",
        "build_qmoe_prefill1024.py", "prepare_qmoe_extended_prefill.py",
        "qualify_quickgelu_mul_long_context.py"))
    profile_names = set()
    comparisons = 0
    for mode in (0, 1):
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                        "--format=csv,noheader"], check=True)
        prefix = ROOT / f"{VARIANT}-native-mode{mode}-gpu7-20261101"
        stats = ROOT / (prefix.name + "-shader-stats.jsonl")
        additional = {**{name: 16 for name in PROGRAMS},
                      PROGRAM if mode == 1 else "MatMulNBitsGroupedRows4F16": 32}
        if args.recover_completed_modes:
            proof = verify_completed_mode(prefix, mode, build, additional)
        else:
            assert not stats.exists()
            run_mode(directory=DIRECTORY, prefix=prefix,
            parent_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
            selector="ORT_WEBGPU_QMOE_GROUPED_PREFILL", rows=ROWS, fallback_rows=FALLBACK_ROWS,
            expected_shapes={16: 976, 128: 8, 256: 8, 512: 8, 1024: 8},
            report_status="row4_static_weights_retained_chunk16_finite_bitwise_and_dispatch_qualified",
            extra_flags=("ORT_WEBGPU_QMOE_PREFILL256=1 ORT_WEBGPU_QMOE_PREFILL512=1 "
                         "ORT_WEBGPU_QMOE_PREFILL1024=1 "
                         f"{FLAG}={mode} {FLAG}_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}"),
                additional_dispatches=additional)
        proof_path = prefix.with_suffix(".qualification.json")
        proof = json.loads(proof_path.read_text())
        assert proof["finite_bitwise_comparisons"] == 300810240
        comparisons += proof["finite_bitwise_comparisons"]
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        assert rows and all(row["operator"] == "QmoeRow4StaticWeightsShader" and
                            row["mode"] == mode and row["words"] == 4 for row in rows)
        evidence.extend((proof_path, stats))
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
            evidence.append(ROOT / name)
            if name.endswith(".json") and not name.endswith(".guard.json"):
                assert name not in profile_names, "Reused native profile"
                profile_names.add(name)
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    fixtures = sorted(FIXTURES.glob("*.onnx"))
    evidence.extend(fixtures)
    evidence.extend(Path(manifest["fixtures"][path.name]["source"]) for path in fixtures)
    if args.recover_completed_modes:
        failed = ROOT / f"{VARIANT}-invalid-gpu7-20261101"
        failed_guard = json.loads(failed.with_suffix(".guard.json").read_text())
        assert failed_guard["status"] == "failed" and failed_guard["exit_code"] == 1
        evidence.extend((failed.with_suffix(".guard.json"), failed.with_suffix(".log"),
                         ROOT / (failed.name + "-invalid.log")))
    prefix = ROOT / f"{VARIANT}-invalid-gpu7-20261101{'-attempt1' if args.recover_completed_modes else ''}"
    invalid = ROOT / (prefix.name + "-invalid.log")
    assert not invalid.exists() and not prefix.with_suffix(".guard.json").exists()
    comparator, provider, fixture = [str(path).replace("/home/sayanshaw/repos", "/workspace", 1)
        for path in (DIRECTORY / "native_compare", DIRECTORY / "libonnxruntime_providers_webgpu.so", fixtures[0])]
    target = "/evidence/webgpu-perf-oct20/" + invalid.name
    qualified = json.loads((ROOT / f"{VARIANT}-native-mode1-gpu7-20261101.guard.json").read_text())["command"]
    command = invalid_command(qualified, comparator, provider, fixture, target)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                    "--format=csv,noheader"], check=True)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "300",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    evidence.extend((invalid, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")))
    report = dict(status="row4_static_weights_process_modes_and_invalid_selector_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], build_manifest_sha256=sha(build_path),
        finite_bitwise_comparisons=comparisons, process_modes_tested=[0, 1],
        invalid_selector_rejected_in_fresh_process=True, all_152_original_objects_byte_identical=True,
        private_kernel_layout_only=True, prefill_only=True, native_profile_names_disjoint=True,
        completed_modes_reverified_without_gpu_rerun=args.recover_completed_modes,
        failed_invalid_probe_preserved=args.recover_completed_modes,
        source_sha256={str(path): sha(path) for path in evidence},
        quality_qualified=False, headline_eligible=False)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}), flush=True)


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import shlex
import subprocess

from build_adapter_inline_capacity17 import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_dense_bitcast_native import main as run_dense


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-adapter-inline-tensors/build-v3")


def main():
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["all_152_original_source_objects_byte_identical"]
    assert build["complete_sdk_copy_for_relative_include_wiring"]
    assert not build["no_object_layout_changes"]
    assert sha(DIRECTORY / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    evidence = []
    for mode in (0, 1):
        prefix = ROOT / f"adapter-inline-tensors-cap17-v3-native-mode{mode}-gpu7-20261101"
        stats = ROOT / (prefix.name + "-adapter-stats.jsonl")
        assert not stats.exists()
        run_dense(directory=DIRECTORY, prefix=prefix,
            baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
            report_status="adapter_inline_tensors_dense_changing_inputs_finite_bitwise_qualified",
            extra_flags=f"{FLAG}={mode} "
                f"ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name} "
                "ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
            fp32_accumulation=False)
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        assert rows and all(row["operator"] == "AdapterInlineTensors" and
            row["contexts"] > 0 and row["enabled_contexts"] == row["contexts"] * mode and
            row["input_capacity"] == 17 and row["output_capacity"] == 5 and
            0 <= row["input_heap_fallbacks"] <= row["enabled_contexts"] and
            0 <= row["output_heap_fallbacks"] <= row["enabled_contexts"] for row in rows), rows
        evidence.append(stats)
        evidence.append(prefix.with_suffix(".qualification.json"))

        prefix = ROOT / f"adapter-inline-tensors-cap17-v3-gdn-mode{mode}-gpu7-20261101"
        assert not prefix.with_suffix(".guard.json").exists()
        comparator = "/workspace/onnxruntime-webgpu-h200-perf/build-int4-gdn-prep/gdn_prep_compare"
        fixtures = [ROOT / "gdn-prep-fixtures" / f"gdn_prep_b{batch}_l{length}.onnx"
                    for batch, length in ((1, 1), (2, 1), (1, 2), (1, 3), (1, 4), (1, 16), (1, 0))]
        args = [comparator, str(DIRECTORY / "libonnxruntime_providers_webgpu.so").replace(
            "/home/sayanshaw/repos", "/workspace", 1),
            *["/evidence/webgpu-perf-oct20/gdn-prep-fixtures/" + path.name for path in fixtures]]
        command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
            "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
            f"export {FLAG}={mode}; exec " + shlex.join(args))
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "300",
            "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        log = prefix.with_suffix(".log").read_text()
        assert guard["status"] == "completed" and guard["exit_code"] == 0
        assert log.count("PASS exact finite outputs ") == 7
        assert "total_exact_elements=2121600" in log and "FAIL" not in log
        evidence.extend((prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")))
        evidence.extend(fixtures)
        evidence.extend(path.with_name(path.stem + "_reference.onnx") for path in fixtures)

    prefix = ROOT / "adapter-inline-tensors-cap17-v3-invalid-gpu7-20261101"
    invalid_log = ROOT / (prefix.name + "-invalid.log")
    assert not prefix.with_suffix(".guard.json").exists() and not invalid_log.exists()
    fixture = "/evidence/webgpu-perf-oct20/gdn-prep-fixtures/gdn_prep_b1_l1.onnx"
    provider = str(DIRECTORY / "libonnxruntime_providers_webgpu.so").replace("/home/sayanshaw/repos", "/workspace", 1)
    target = "/evidence/webgpu-perf-oct20/" + invalid_log.name
    command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
        f"if env {FLAG}=2 {comparator} {provider} {fixture} > {target} 2>&1; then "
        "echo 'FAIL: invalid adapter selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' {target}")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "120",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    evidence.extend((invalid_log, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"), build_path))
    report = dict(status="adapter_inline_tensor_process_modes_dense_gdn_and_invalid_selector_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], build_manifest_sha256=sha(build_path),
        finite_bitwise_comparisons=233165568 + 4243200, process_modes_tested=[0, 1],
        invalid_selector_rejected_in_fresh_process=True, original_expanded_gdn_reference_tested=True,
        all_152_original_objects_byte_identical=True,
        complete_context_layout_rebuild=True, input_inline_capacity=17, output_inline_capacity=5,
        source_sha256={str(path): sha(path) for path in evidence}, quality_qualified=False, headline_eligible=False)
    path = ROOT / "adapter-inline-tensors-cap17-v3-native-gpu7-20261101.qualification.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import shlex
import subprocess

from build_dense_flag_cache import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_dense_bitcast_native import main as run_dense


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-dense-flag-cache/build-v1")


def verify_dense_rows(rows):
    assert len(rows) == 18
    assert all(row["operator"] == "MatMulNBitsExact" and row["requests"] == 128 and
               row["entries"] == row["captures"] == 1 and row["hits"] == 127 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["binding_hits"] > 0 for row in rows), rows


def main():
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["all_152_original_source_objects_byte_identical"]
    assert build["private_kernel_layout_only"]
    assert sha(DIRECTORY / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    evidence = []
    for mode in (0, 1):
        prefix = ROOT / f"dense-flag-cache-v1-native-mode{mode}-gpu7-20261101"
        stats = ROOT / (prefix.name + "-adapter-stats.jsonl")
        assert not stats.exists()
        run_dense(directory=DIRECTORY, prefix=prefix,
            baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
            report_status="dense_flag_caches_dense_changing_inputs_finite_bitwise_qualified",
            extra_flags=f"{FLAG}={mode} "
                f"ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name} "
                "ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
            fp32_accumulation=False, stats_verifier=verify_dense_rows)
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        assert rows and all(row["operator"] == "DenseFrozenFlags" and
            row["enabled"] == mode and row["requests"] >= row["parses"] >= 0 and
            (row["parses"] <= 8 if mode else row["requests"] == row["parses"])
            for row in rows), rows
        if mode:
            assert sum(row["parses"] for row in rows) < sum(row["requests"] for row in rows)
        evidence.append(stats)
        evidence.append(prefix.with_suffix(".qualification.json"))

        prefix = ROOT / f"dense-flag-cache-v1-gdn-mode{mode}-gpu7-20261101"
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

    prefix = ROOT / "dense-flag-cache-v1-invalid-gpu7-20261101"
    invalid_log = ROOT / (prefix.name + "-invalid.log")
    assert not prefix.with_suffix(".guard.json").exists() and not invalid_log.exists()
    fixture_path = sorted(Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))[0]
    fixture = str(fixture_path).replace("/home/sayanshaw/repos", "/workspace", 1)
    comparator = str(DIRECTORY / "native_compare").replace("/home/sayanshaw/repos", "/workspace", 1)
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
    report = dict(status="dense_flag_cache_process_modes_dense_gdn_and_invalid_selector_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], build_manifest_sha256=sha(build_path),
        finite_bitwise_comparisons=233165568 + 4243200, process_modes_tested=[0, 1],
        invalid_selector_rejected_in_fresh_process=True, original_expanded_gdn_reference_tested=True,
        all_152_original_objects_byte_identical=True,
        private_kernel_layout_only=True, cached_selectors_per_kernel=8,
        source_sha256={str(path): sha(path) for path in evidence}, quality_qualified=False, headline_eligible=False)
    path = ROOT / "dense-flag-cache-v1-native-gpu7-20261101.qualification.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

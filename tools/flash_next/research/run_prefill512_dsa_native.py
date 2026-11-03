import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from build_qmoe_node_replay import sha


ROOT = Path(__file__).resolve().parent
PREFIX = "prefill512-v3-dsa-native-gpu7-fixturesv4-20261030"
FLAG = "ORT_WEBGPU_QMOE_PREFILL512"


def inside(directory=None, fixtures=None, prefix=PREFIX, flag=FLAG, chunk=512, extra_flags=None):
    repos = Path("/workspace")
    directory = directory or repos / "qwen38-int4-qmoe-prefill512/build-v3"
    fixtures = fixtures or repos / "qwen38-int4-qmoe-prefill512/fixtures-dsa-v4"
    build = json.loads((directory / "build-manifest.json").read_text())
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"] and build["original_dsa_object_byte_identical"]
    manifest_path = fixtures / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["fixtures"]) == 2 * (len(manifest["positive_cases"]) + 4)
    for name, evidence in manifest["fixtures"].items():
        assert sha(fixtures / name) == evidence["sha256"]
    comparator = repos / "onnxruntime-webgpu-h200-perf/build-int4-dsa-split-kv/dispatch_compare"
    logs = {}
    cases = [fixtures / f"{name}.onnx" for name in manifest["positive_cases"]]
    legacy = [fixtures / f"{name}.onnx" for name in manifest.get("legacy_cases", ["legacy256"])]
    negatives = ((0, f"selected{chunk}", "supported bound 256"), (2, "legacy256", f"{flag} must be 0 or 1"),
                 (1, f"reject{chunk-1}", "supported bound 256"), (1, f"reject{chunk+1}", "supported bound 256"),
                 (1, "reject_batch2", "supported bound 256"), (1, "reject_dispatch", "supported bound 256"))
    for pending in (1, 64):
        environment = dict(os.environ, ORT_TEST_MAX_PENDING_DISPATCHES=str(pending),
                           ORT_TEST_ONLY_MODE="0", ORT_WEBGPU_QMOE_PREFILL512="1",
                           ORT_WEBGPU_DYNAMIC_UNIFORMS="1", ORT_WEBGPU_UNIFORM_ARENA="1",
                           ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES="8192")
        environment.update(extra_flags or {})
        for mode, selected, suffix in ((1, cases, "candidate"), (0, legacy, "legacy")):
            environment[flag] = str(mode)
            path = ROOT / f"{prefix}-pending{pending}-{suffix}.log"
            with path.open("x") as stream:
                subprocess.run([str(comparator), str(provider), *map(str, selected)], env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            text = path.read_text()
            assert "FAIL" not in text and "validationMode=full" in text
            for case in selected:
                assert text.count(f"PASS {case.stem} ") == 1
            assert text.count("samples_per_mode=7 selected_mode=0") == len(selected)
            logs[path.name] = sha(path)
        for index, (mode, name, message) in enumerate(negatives):
            environment[flag] = str(mode)
            path = ROOT / f"{prefix}-pending{pending}-rejection{index}.log"
            with path.open("x") as stream:
                result = subprocess.run([str(comparator), str(provider), str(fixtures / f"{name}.onnx")],
                                        env=environment, stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 1 and message in path.read_text(), path
            logs[path.name] = sha(path)
    comparisons = 18 * sum(manifest["fixtures"][case.name]["output_elements"] for case in cases)
    comparisons += 18 * sum(manifest["fixtures"][case.name]["output_elements"] for case in legacy)
    report = dict(status=f"prefill{chunk}_dsa_cpu_reference_and_admission_qualified",
        provider_sha256=build["provider_sha256"], original_dsa_object_byte_identical=True,
        pending_windows_tested=[1, 64], full_validation=True, shader_arithmetic_unchanged=True,
        finite_cpu_reference_comparisons=comparisons, positive_cases=manifest["positive_cases"],
        expected_rejections=12, default_sequence_bound_preserved=True,
        invalid_selector_rejected=True, cache_outputs_checked=True,
        float16_atol=0.0005, float16_rtol=0.005, float32_atol=0.000002, float32_rtol=0.0005,
        fixture_manifest_sha256=sha(manifest_path), comparator_sha256=sha(comparator),
        build_manifest_sha256=sha(directory / "build-manifest.json"), source_sha256=logs,
        instrumented_not_a_speed_claim=True, quality_qualified=False, headline_eligible=False)
    path = ROOT / f"{prefix}.inner.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_cpu_reference_comparisons", "expected_rejections")}),
          flush=True)


def main(variant="prefill512-v3", fixture_version="v4", chunk=512, selector=FLAG, extra_flags=None,
         fixtures_directory=None, stamp="20261030"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    prefix_name = f"{variant}-dsa-native-gpu7-fixtures{fixture_version}-{stamp}"
    family, version = variant.rsplit("-v", 1)
    if parser.parse_args().inside:
        inside(directory=Path(f"/workspace/qwen38-int4-qmoe-{family}/build-v{version}"),
               fixtures=fixtures_directory or Path(
                   f"/workspace/qwen38-int4-qmoe-{family}/fixtures-dsa-{fixture_version}"),
               prefix=prefix_name, flag=selector, chunk=chunk, extra_flags=extra_flags)
        return
    prefix = ROOT / prefix_name
    if any(prefix.with_suffix(suffix).exists() for suffix in (".guard.json", ".qualification.json", ".inner.json")):
        raise FileExistsError(prefix)
    command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
               "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; exec python3 "
               f"/evidence/webgpu-perf-oct20/{Path(sys.argv[0]).name} --inside")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    report = json.loads(prefix.with_suffix(".inner.json").read_text())
    for name, digest in report["source_sha256"].items():
        assert sha(ROOT / name) == digest
    for suffix in (".guard.json", ".log", ".inner.json"):
        path = prefix.with_suffix(suffix)
        report["source_sha256"][path.name] = sha(path)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_cpu_reference_comparisons", "expected_rejections")}),
          flush=True)


if __name__ == "__main__":
    main()

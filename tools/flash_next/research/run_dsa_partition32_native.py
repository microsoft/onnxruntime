import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import subprocess

from build_dsa_partition32 import FLAG
from build_qmoe_node_replay import sha


ROOT = Path(__file__).resolve().parent
PREFIX = "dsa-partition32-v1-native-gpu7-r3-20261030"
QUALIFICATION = "dsa-partition32-v1-native-gpu7-20261030.qualification.json"


def inside():
    directory = Path("/workspace/qwen38-int4-dsa-partition32/build-v1")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    build = json.loads((directory / "build-manifest.json").read_text())
    assert sha(provider) == build["provider_sha256"] and build["baseline_byte_identical"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    fixtures_dir = ROOT / "dsa-split-kv-fixtures"
    fixtures = sorted(path for path in fixtures_dir.glob("*.onnx") if not path.stem.endswith("_reference"))
    assert len(fixtures) == 27
    fixture_sha256 = {path.name: sha(path) for path in fixtures_dir.glob("*.onnx")}
    assert len(fixture_sha256) == 54
    comparator = Path("/workspace/onnxruntime-webgpu-h200-perf/build-int4-dsa-split-kv/dispatch_compare")
    logs, profiles, counts_by_mode, timings = {}, {}, {}, {}
    for pending in (1, 64):
        for mode in (0, 1):
            prefix = ROOT / f"{PREFIX}-pending{pending}-mode{mode}"
            environment = dict(os.environ, ORT_TEST_MAX_PENDING_DISPATCHES=str(pending),
                               ORT_TEST_ONLY_MODE="1", ORT_TEST_ATTENTION_SPLIT_COMPARE="1",
                               ORT_TEST_PROFILE_PREFIX=str(prefix), ORT_WEBGPU_QMOE_PREFILL512="1",
                               ORT_WEBGPU_DYNAMIC_UNIFORMS="1", ORT_WEBGPU_UNIFORM_ARENA="1",
                               ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES="8192")
            environment[FLAG] = str(mode)
            path = prefix.with_suffix(".log")
            with path.open("x") as stream:
                subprocess.run([str(comparator), str(provider), *map(str, fixtures)], env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            text = path.read_text()
            assert "FAIL" not in text and "validationMode=full" in text
            assert text.count("samples_per_mode=7 selected_mode=1") == 27
            for fixture in fixtures:
                assert text.count(f"PASS {fixture.stem} ") == 1
            paths = [Path(line.removeprefix("PROFILE ")) for line in text.splitlines() if line.startswith("PROFILE ")]
            assert len(paths) == len(fixtures), paths
            events = [event for profile in paths for event in json.loads(profile.read_text())]
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                for event in events if event.get("cat") == "Kernel")
            expected = 207 if mode else 0
            assert counts["DynamicSparseAttentionPartition32"] == expected, counts
            assert counts["DynamicSparseAttention"] == (36 if mode else 243), counts
            model_events = [event for event in events if event.get("cat") == "Kernel" and
                            "outputs[0] = {24," in event.get("args", {}).get("shapes", "") and
                            event["name"].rsplit("&", 1)[-1] in
                            ("DynamicSparseAttention", "DynamicSparseAttentionPartition32")]
            assert len(model_events) == 9
            assert all(f"outputs[0] = {{24,{65 if mode else 33},258}}" in event["args"]["shapes"]
                       for event in model_events)
            merge_events = [event for event in events if event.get("cat") == "Kernel" and
                            event["name"].endswith("&DynamicSparseAttentionMerge") and
                            f"inputs[0] = {{24,{65 if mode else 33},258}}" in event["args"].get("shapes", "")]
            assert len(merge_events) == 9
            timings[f"pending{pending}-mode{mode}"] = dict(
                attention_median_gpu_us=statistics.median(event["dur"] for event in model_events[2:]),
                merge_median_gpu_us=statistics.median(event["dur"] for event in merge_events[2:]),
                instrumented_not_model_tps=True)
            counts_by_mode[f"pending{pending}-mode{mode}"] = dict(counts)
            logs[path.name] = sha(path)
            profiles.update({profile.name: sha(profile) for profile in paths})
    for mode in (0, 1):
        path = ROOT / f"{PREFIX}-capture-mode{mode}.log"
        environment = dict(os.environ, ORT_TEST_ONLY_MODE="1", ORT_TEST_ATTENTION_SPLIT_COMPARE="1",
                           ORT_TEST_ENABLE_GRAPH_CAPTURE="1", ORT_TEST_MAX_PENDING_DISPATCHES="64")
        environment.pop("ORT_TEST_PROFILE_PREFIX", None)
        environment[FLAG] = str(mode)
        with path.open("x") as stream:
            subprocess.run([str(comparator), str(provider), str(fixtures_dir / "model_2k.onnx")],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = path.read_text()
        assert "FAIL" not in text and "PASS model_2k " in text and "samples_per_mode=7 selected_mode=1" in text
        logs[path.name] = sha(path)
    environment[FLAG] = "2"
    environment["ORT_TEST_ENABLE_GRAPH_CAPTURE"] = "0"
    path = ROOT / f"{PREFIX}-invalid.log"
    with path.open("x") as stream:
        rejected = subprocess.run([str(comparator), str(provider), str(fixtures_dir / "model_2k.onnx")],
                                  env=environment, stdout=stream, stderr=subprocess.STDOUT)
    assert rejected.returncode == 1 and f"{FLAG} must be 0 or 1" in path.read_text()
    logs[path.name] = sha(path)
    report = dict(status="dsa_partition32_cpu_reference_and_dispatch_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=27, cpu_reference_executions=972, fixed_flag_capture_executions=18,
        pending_windows_tested=[1, 64], full_validation=True, invalid_selector_rejected=True,
        partition_sizes_tested=[64, 32], program_counts=counts_by_mode,
        model_shaped_gpu_diagnostic=timings, runner_sha256=sha(Path(__file__)),
        float16_atol=0.0005, float16_rtol=0.005, float32_atol=0.000002, float32_rtol=0.0005,
        cache_outputs_checked=True, original_shader_expressions_preserved=True,
        partition_grouping_changes_fp32_softmax_association=True,
        fixture_sha256=fixture_sha256, comparator_sha256=sha(comparator),
        build_manifest_sha256=sha(directory / "build-manifest.json"), source_sha256={**logs, **profiles},
        quality_qualified=False, headline_eligible=False)
    path = ROOT / f"{PREFIX}.inner.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "cpu_reference_executions")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    if parser.parse_args().inside:
        inside()
    else:
        prefix = ROOT / PREFIX
        if any(prefix.with_suffix(suffix).exists() for suffix in (".guard.json", ".qualification.json", ".inner.json")):
            raise FileExistsError(prefix)
        command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
                   "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; exec python3 "
                   "/evidence/webgpu-perf-oct20/run_dsa_partition32_native.py --inside")
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
        proof = ROOT / QUALIFICATION
        if proof.exists():
            raise FileExistsError(proof)
        report["native_guard_prefix"] = PREFIX
        proof.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "cpu_reference_executions")}), flush=True)

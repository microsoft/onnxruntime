import json
from pathlib import Path
import shlex
import subprocess

from build_hc_projection_inner_unroll import FLAG
from functools import partial
import math
import onnx

from qualify_quickgelu_mul_long_context import ROOT, sha


def verify_static_stats(rows, requests=192):
    names = {"HCPostNormProjectionF16", "HCSharedPostNormProjectionF16"}
    assert len(rows) == 12
    assert all(sum(r["operator"] == name for r in rows) == 6 for name in names)
    assert all(r["operator"] in names and r["requests"] == requests and
               r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
               r["hits"] + r["captures"] == r["requests"] and r["evictions"] > 0 and r["captures"] >= 65 and
               0 < r["entries"] <= r["capacity"] == 64 and
               0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows


def run_mode(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-bitcast-unpack/build-v1"),
         prefix=ROOT / "hc-bitcast-unpack-v1-native-gpu7-20261029",
         report_status="hc_bitcast_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-prefill-dense-stack/build-v1/build-manifest.json"),
         stats_verifier=verify_static_stats, compound_candidate_not_replay_cached=False, process_mode=1):
    repos = Path("/home/sayanshaw/repos")
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not stats.exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads(parent_path.read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    assert build["all_152_original_source_objects_byte_identical"] and build["no_object_layout_changes"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest, name
    previous = json.loads((ROOT / "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json").read_text())
    fixtures = sorted((repos / "qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == 36
    assert all(sha(p) == previous["fixture_sha256"][p.name] for p in fixtures)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
        "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
        f"{FLAG}={process_mode} "
        f"ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE=/evidence/webgpu-perf-oct20/{prefix.name}-shader-stats.jsonl "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in [
            directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("changing_rounds=128 invalid_flag_rejected=1") == 36
    expected = 12 * 19 * (2 * 10240 + 324) * 128
    assert f"PASS total_exact_elements={expected}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    stats_verifier(rows)
    shader_stats = ROOT / (prefix.name + "-shader-stats.jsonl")
    shaders = [json.loads(line) for line in shader_stats.read_text().splitlines()]
    assert shaders and all(row["operator"] == "HCProjectionInnerUnrollShader" and
        row["mode"] == process_mode and row["component_steps"] == 4 for row in shaders), shaders
    report = dict(status=report_status,
        fixtures=36, finite_bitwise_comparisons=expected, rounds_per_fixture=128,
        all_four_outputs_compared=True, changing_inputs_and_cpu_splits=True,
        lru_eviction_tested=True, invalid_flag_explicitly_rejected=True,
        fallback_fixtures=24, replay_variant_keys_tested=False, process_static_shader_identity_tested=True,
        compound_candidate_not_replay_cached=compound_candidate_not_replay_cached, stats=rows,
        pending_windows_tested=[1, 64], provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_sha256={p.name: sha(p) for p in fixtures},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


def main():
    directory = Path("/home/sayanshaw/repos/qwen38-int4-hc-projection-inner-unroll/build-v1")
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    evidence = [build_path, Path(__file__), ROOT / "build_hc_projection_inner_unroll.py",
                ROOT / "prepare_hc_projection_inner_unroll_proofs.py"]
    for mode in (0, 1):
        prefix = ROOT / f"hc-projection-inner-unroll-v1-native-mode{mode}-gpu7-20261101"
        run_mode(directory=directory, prefix=prefix,
                 report_status="hc_inner_unroll_four_outputs_finite_bitwise_and_replay_qualified",
                 parent_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
                 stats_verifier=partial(verify_static_stats, requests=96), process_mode=mode)
        evidence.extend(prefix.with_suffix(s) for s in (".qualification.json", ".guard.json", ".log"))
        evidence.extend((ROOT / (prefix.name + "-stats.jsonl"), ROOT / (prefix.name + "-shader-stats.jsonl")))
    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    eligible = []
    for path in fixtures:
        graph = onnx.load(path, load_external_data=False).graph
        tensor = graph.input[0].type.tensor_type
        if tensor.elem_type == onnx.TensorProto.FLOAT16 and math.prod(
                dim.dim_value for dim in tensor.shape.dim) == 10240:
            eligible.append(path)
    assert eligible
    prefix = ROOT / "hc-projection-inner-unroll-v1-invalid-gpu7-20261101"
    assert not prefix.with_suffix(".guard.json").exists()
    invalid = ROOT / (prefix.name + "-invalid.log")
    assert not invalid.exists()
    comparator, provider, fixture = [str(path).replace("/home/sayanshaw/repos", "/workspace", 1)
        for path in (directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", eligible[0])]
    target = "/evidence/webgpu-perf-oct20/" + invalid.name
    command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
        f"if env {FLAG}=2 ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
        f"ORT_WEBGPU_HC_EXACT_DENSE=1 {comparator} {provider} {fixture} > {target} 2>&1; then "
        "echo 'FAIL: invalid HC inner-unroll selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' {target}")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "120",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    evidence.extend((invalid, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"), *fixtures))
    report = dict(status="hc_projection_inner_unroll_process_modes_and_invalid_selector_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], build_manifest_sha256=sha(build_path),
        finite_bitwise_comparisons=1214287872, process_modes_tested=[0, 1],
        invalid_selector_rejected_in_fresh_process=True, all_four_outputs_compared=True,
        all_152_original_objects_byte_identical=True, private_kernel_layout_only=True,
        source_sha256={str(path): sha(path) for path in evidence}, quality_qualified=False, headline_eligible=False)
    path = ROOT / "hc-projection-inner-unroll-v1-native-gpu7-20261101.qualification.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

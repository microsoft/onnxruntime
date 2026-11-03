import json
import os
from pathlib import Path
import shlex
import subprocess

import onnx

from qualify_quickgelu_mul_long_context import ROOT, sha


def extra_fixtures(repos):
    source = repos / "qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx"
    original = onnx.load(source, load_external_data=False)
    directory = repos / "qwen38-int4-qmoe-fc1-activation/fixtures-v3"
    directory.mkdir(exist_ok=False)
    locations = {entry.value for tensor in original.graph.initializer
                 for entry in tensor.external_data if entry.key == "location"}
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unexpected external-data location: {location}")
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source.parent / relative, destination)
        except PermissionError:
            subprocess.run(["sudo", "-n", "ln", str(source.parent / relative), str(destination)], check=True)
    result = []
    for name in ("no_fc1_bias", "automatic_interleaving", "unsupported_concatenated"):
        model = onnx.ModelProto()
        model.CopyFrom(original)
        node = model.graph.node[0]
        if name == "no_fc1_bias":
            assert node.input[4].endswith(".bias") and node.input[3].endswith(".scales")
            node.input[4] = ""
        else:
            attribute = next(a for a in node.attribute if a.name == "swiglu_fusion")
            attribute.i = 0 if name == "automatic_interleaving" else 2
        path = directory / (name + ".onnx")
        onnx.save_model(model, path)
        onnx.checker.check_model(str(path))
        result.append(path)
    return result


def main():
    repos = Path("/home/sayanshaw/repos")
    build_dir = repos / "qwen38-int4-qmoe-fc1-activation/build-v3"
    prefix = ROOT / "qmoe-fc1-activation-v3-native-gpu7-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    manifest = json.loads((repos / "qwen38-int4-qmoe-node-replay/fixtures.json").read_text())
    fixtures = [repos / "qwen38-int4-qmoe-node-replay/fixtures" / item["fixture"] for item in manifest]
    assert len(fixtures) == 18 and sum(item["rows"] == 1 for item in manifest) == 6
    fixtures += extra_fixtures(repos)
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not stats.exists()
    mounted = "/workspace/qwen38-int4-qmoe-fc1-activation/build-v3"
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join([mounted + "/native_compare", mounted + "/libonnxruntime_providers_webgpu.so",
                    *("/workspace/" + str(p.relative_to(repos)) for p in fixtures)]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS changing_inputs_and_routes ") == len(fixtures)
    comparisons = sum(item["rows"] * item["hidden"] * 32 for item in manifest) + 3 * 2560 * 32
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 9
    assert all(row["requests"] == 64 and row["hits"] >= 16 and
               row["captures"] + row["hits"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 8 for row in rows), rows
    report = dict(status="changing_fc1_activation_inputs_routes_qualified",
        fixtures=len(fixtures), eligible_single_token_fixtures=8, fallback_fixtures=13,
        rounds_per_fixture=32, finite_bitwise_comparisons=comparisons,
        both_flags_tested=True, pending_windows_tested=[1, 64],
        variant_cache_keys_tested=True, optional_bias_tested=True,
        automatic_interleaving_and_concatenated_fallback_tested=True,
        four_and_five_command_capture_contracts_tested=True, stats=rows,
        provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=sha(build_dir / "native_compare"),
        fixture_sha256={str(p.relative_to(repos)): sha(p) for p in fixtures},
        actual_model_dispatch_pending=True, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

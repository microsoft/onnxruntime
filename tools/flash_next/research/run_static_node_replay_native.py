import argparse
import json
import math
from pathlib import Path
import subprocess

import onnx

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, choices=(0, 3, 7), required=True)
    parser.add_argument("--revision", choices=("v7", "v8"), default="v8")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    directory = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay")
    build_dir = directory / ("build-" + args.revision)
    prefix = args.output_prefix
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not any(prefix.with_suffix(s).exists() for s in (".guard.json", ".qualification.json"))
    assert not stats.exists()
    gpu_uuid = {0: "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd",
                3: "GPU-38eac315-6ffb-8cb1-c68f-521767e9ba3e",
                7: "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"}[args.gpu_index]
    fixture_dir = "fixtures-v3" if args.revision == "v7" else "fixtures-v8-fixed-shapes"
    fixtures = sorted((directory / fixture_dir).glob("*.onnx"))
    assert len(fixtures) == (20 if args.revision == "v7" else 30)
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    container_build = "/workspace/qwen38-int4-static-node-replay/build-" + args.revision
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        f"{gpu_uuid}; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        f"exec {container_build}/native_compare {container_build}/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-static-node-replay/" + fixture_dir + "/" + p.name for p in fixtures))
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(args.gpu_index),
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "400",
         "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == len(fixtures)
    comparisons = 0
    eligible = 0
    for path in fixtures:
        model = onnx.load(path)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        elements = math.prod(shape)
        if model.graph.node[0].op_type == "MatMulNBits":
            n = next(a.i for a in model.graph.node[0].attribute if a.name == "N")
            elements = elements // shape[-1] * n
        elif model.graph.node[0].op_type.startswith("HyperConnectionPreMix"):
            elements //= 4
        elif model.graph.node[0].op_type in ("HCPostNormF16", "HCSharedPostNormF16"):
            elements *= 2
        comparisons += elements * 128
        logical_row = shape[-1]
        if model.graph.node[0].op_type.startswith("HyperConnectionPreMix"):
            branches = next(a.i for a in model.graph.node[0].attribute if a.name == "num_branches")
            if branches == 0:
                logical_row *= shape[-2]
        eligible += math.prod(shape) == logical_row
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == eligible
    assert all(row["requests"] == (96 if row["operator"] == "MatMulNBitsExact" else 128) and row["hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0 and
               0 < row["binding_entries"] <= row["binding_capacity"] == 64 and
               row["hits"] + row["captures"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 64 for row in rows), rows
    assert any(row["operator"] == "SplitScaledSiLUF16" and row["evictions"] > 0 and
               row["captures"] >= 96 for row in rows), rows
    report = dict(status="changing_gpu_inputs_cpu_values_finite_bitwise_and_replay_qualified",
                  fixtures=len(fixtures), eligible_single_token_fixtures=eligible,
                  fallback_fixtures=len(fixtures) - eligible, rounds_per_fixture=128,
                  finite_bitwise_comparisons=comparisons, stats=rows,
                  pending_windows_tested=[1, 64], invalid_flag_explicitly_rejected=True,
                  changing_cpu_split_values_tested=True, lru_eviction_tested=True,
                  exact_dense_selectors_toggled=True,
                  cached_bindings_checked_against_current_buffers=True,
                  qmoe_replay_disabled=True, comparator_sha256=sha(build_dir / "native_compare"),
                  provider_sha256=build["provider_sha256"], quality_qualified=False, headline_eligible=False)
    report["fixture_sha256"] = {p.name: sha(p) for p in fixtures}
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

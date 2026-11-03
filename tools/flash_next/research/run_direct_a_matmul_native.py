import json
from pathlib import Path
import shlex
import subprocess

from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    repos = Path("/home/sayanshaw/repos")
    build_dir = repos / "qwen38-int4-direct-a-matmul/build-v2"
    prefix = ROOT / "direct-a-matmul-v2-native-gpu7-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    generic = sorted((repos / "qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))
    qmoe_manifest = json.loads((repos / "qwen38-int4-qmoe-node-replay/fixtures.json").read_text())
    qmoe = [repos / "qwen38-int4-qmoe-node-replay/fixtures" / item["fixture"] for item in qmoe_manifest]
    assert len(generic) == len(SHAPES) * 3 and qmoe
    static_stats = ROOT / (prefix.name + "-static-stats.jsonl")
    qmoe_stats = ROOT / (prefix.name + "-qmoe-stats.jsonl")
    assert not static_stats.exists() and not qmoe_stats.exists()
    mounted = "/workspace/qwen38-int4-direct-a-matmul/build-v2"
    provider = mounted + "/libonnxruntime_providers_webgpu.so"
    mount = lambda p: "/workspace/" + str(p.relative_to(repos))
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{static_stats.name} "
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{qmoe_stats.name}; " +
        shlex.join([mounted + "/static_node_replay_compare", provider, *map(mount, generic)]) + "; " +
        shlex.join([mounted + "/qmoe_node_replay_compare", provider, *map(mount, qmoe)]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == len(generic)
    assert log.count("PASS changing_inputs_and_routes ") == len(qmoe)
    generic_comparisons = sum(n for _, n in SHAPES) * (1 + 2 + 16) * 128
    qmoe_comparisons = sum(item["rows"] * item["hidden"] * 32 for item in qmoe_manifest)
    for comparisons in (generic_comparisons, qmoe_comparisons):
        assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    static_rows = [json.loads(line) for line in static_stats.read_text().splitlines()]
    qmoe_rows = [json.loads(line) for line in qmoe_stats.read_text().splitlines()]
    assert len(static_rows) == len(SHAPES)
    assert all(row["requests"] == 256 and row["captures"] == 2 and row["entries"] == 2 and
               row["hits"] + row["captures"] == row["requests"] and
               row["binding_hits"] + row["rebinds"] == row["hits"] and
               row["binding_hits"] > 0 and row["capacity"] == row["binding_capacity"] == 64
               for row in static_rows), static_rows
    assert len(qmoe_rows) == sum(item["rows"] == 1 for item in qmoe_manifest)
    assert all(row["requests"] == 64 and row["hits"] >= 16 and
               row["captures"] + row["hits"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 8 for row in qmoe_rows), qmoe_rows
    report = dict(status="changing_direct_a_inputs_routes_qualified",
        generic_fixtures=len(generic), qmoe_fixtures=len(qmoe),
        finite_bitwise_comparisons=generic_comparisons + qmoe_comparisons,
        both_flags_tested=True, pending_windows_tested=[1, 64],
        variant_cache_keys_tested=True, static_stats=static_rows, qmoe_stats=qmoe_rows,
        provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256={p.name: sha(p) for p in build_dir.glob("*_compare")},
        fixture_sha256={str(p.relative_to(repos)): sha(p) for p in generic + qmoe},
        actual_model_dispatch_pending=True, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import shlex
import subprocess

from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    directory = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay")
    build_dir = directory / "build-v1"
    prefix = ROOT / "generic-matmul-replay-v1-native-gpu0-20261029"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    fixtures = sorted((directory / "fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == len(SHAPES) * 3
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    mounted = "/workspace/qwen38-int4-generic-matmul-replay"
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join([mounted + "/build-v1/native_compare",
                    mounted + "/build-v1/libonnxruntime_providers_webgpu.so",
                    *(mounted + "/fixtures-v1/" + p.name for p in fixtures)]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "0",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "600",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == len(fixtures)
    comparisons = sum(n for _, n in SHAPES) * (1 + 2 + 16) * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == len(SHAPES)
    assert all(r["operator"] == "MatMulNBitsExact" and r["requests"] == 128 and
        r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
        r["hits"] + r["captures"] == r["requests"] and
        0 < r["entries"] <= r["capacity"] == 64 and
        0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows
    report = dict(status="changing_generic_matmul_inputs_qualified", fixtures=len(fixtures),
        single_token_shapes=len(SHAPES), rounds_per_fixture=128, pending_windows_tested=[1, 64],
        both_flags_tested=True, finite_bitwise_comparisons=comparisons, stats=rows,
        provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=sha(build_dir / "native_compare"),
        fixture_sha256={p.name: sha(p) for p in fixtures},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

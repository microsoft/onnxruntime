import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v1", "v2"), default="v2")
    revision = parser.parse_args().revision
    directory = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-premix")
    build_dir = directory / ("build-" + revision)
    prefix = ROOT / f"hc-up-premix-{revision}-native-gpu0-20261029"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    fixtures = sorted((directory / "fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == 9
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    container_build = "/workspace/qwen38-int4-hc-up-premix/build-" + revision
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        f"exec {container_build}/native_compare {container_build}/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-hc-up-premix/fixtures-v1/" + p.name for p in fixtures))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "0",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "500",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == 9
    comparisons = 3 * (1 + 2 + 16) * 2560 * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["operator"] == "HCUpPreMixF16" and r["requests"] == 128 and
        r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
        r["hits"] + r["captures"] == r["requests"] and
        0 < r["entries"] <= r["capacity"] == 64 and
        0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows
    report = dict(status="changing_gpu_inputs_finite_bitwise_and_up_premix_replay_qualified",
        fixtures=9, eligible_single_token_fixtures=3, fallback_fixtures=6, rounds_per_fixture=128,
        finite_bitwise_comparisons=comparisons, stats=rows,
        pending_windows_tested=[1, 64], invalid_flag_explicitly_rejected=True,
        up_premix_fusion_on_off_tested=True, qmoe_replay_disabled=True,
        comparator_sha256=sha(build_dir / "native_compare"),
        provider_sha256=build["provider_sha256"], fixture_sha256={p.name: sha(p) for p in fixtures},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

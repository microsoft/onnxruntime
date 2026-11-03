import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    repos = Path("/home/sayanshaw/repos")
    directory = repos / "qwen38-int4-hc-up-bitcast-unpack/build-v1"
    prefix = ROOT / "hc-up-bitcast-unpack-v1-native-gpu7-20261029"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / "hc-up-premix-v2-native-gpu0-20261029.qualification.json").read_text())
    baseline = json.loads((ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    fixtures = sorted((repos / "qwen38-int4-hc-up-premix/fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == 9
    assert all(sha(p) == parent["fixture_sha256"][p.name] for p in fixtures)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_UP_PREMIX=1 ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 "
        "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in
                   [directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == 9
    comparisons = 3 * 19 * 2560 * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["operator"] == "HCUpPreMixF16" and r["requests"] == 256 and
        r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
        r["hits"] + r["captures"] == r["requests"] and
        r["entries"] == 2 and r["capacity"] == 64 and
        0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows
    report = dict(status="hc_up_bitcast_finite_bitwise_and_replay_qualified",
        fixtures=9, finite_bitwise_comparisons=comparisons, rounds_per_fixture=128,
        parent_fusion_fixed_enabled=True, bitcast_modes_tested=[0, 1],
        pending_windows_tested=[1, 64], variant_cache_keys_tested=True,
        invalid_flag_explicitly_rejected=True, multi_token_fallback_tested=True, stats=rows,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_sha256=parent["fixture_sha256"],
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

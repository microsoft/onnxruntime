import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main(directory=None, prefix=None,
         baseline_name="expert-bitcast-unpack-v2-gpu7-standard2048-20261029",
         report_status="fc2_four_columns_changing_inputs_routes_finite_bitwise_and_replay_qualified",
         variant_field="columns_tested", variant_values=(2, 4)):
    repos = Path("/home/sayanshaw/repos")
    directory = directory or repos / "qwen38-int4-fc2-four-columns/build-v2"
    prefix = prefix or ROOT / "fc2-four-columns-v2-native-gpu7-20261029"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json").read_text())
    baseline = json.loads((ROOT / f"{baseline_name}.qualification.json").read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    fixtures = [repos / path for path in parent["fixture_sha256"]]
    assert len(fixtures) == 23
    assert all(sha(p) == parent["fixture_sha256"][str(p.relative_to(repos))] for p in fixtures)
    identity_data = repos / "qwen38-int4-qmoe-hierarchical-gate/fixtures-v1/identity.data"
    identity = json.loads((ROOT / "qmoe-hierarchical-gate-v2-native-gpu7-20261029.qualification.json").read_text())
    assert sha(identity_data) == identity["identity_manifest"]["data_sha256"]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
        "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 "
        "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
        "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=0 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in
                   [directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS changing_inputs_and_routes ") == 23
    assert "PASS total_exact_elements=9748480; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 11
    assert all(r["requests"] == 64 and r["hits"] > 0 and
        r["captures"] + r["hits"] == r["requests"] and
        0 < r["entries"] <= r["capacity"] == 8 for r in rows), rows
    report = dict(status=report_status,
        fixtures=23, finite_bitwise_comparisons=9748480, rounds_per_fixture=32,
        both_expert_fusions_fixed_enabled=True, expert_bitcast_fixed_enabled=True,
        pending_windows_tested=[1, 64], variant_cache_keys_tested=True,
        optional_bias_and_unsupported_layout_fallbacks_tested=True, stats=rows,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_sha256=parent["fixture_sha256"],
        quality_qualified=False, headline_eligible=False)
    report[variant_field] = list(variant_values)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

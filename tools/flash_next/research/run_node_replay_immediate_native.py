import argparse
import collections
import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


REPOS = Path("/home/sayanshaw/repos")
BUILD = REPOS / "qwen38-int4-node-replay-immediate/build-v1"
PREFIX = "node-replay-immediate-v1-native"
PROOFS = {
    "qmoe": "qmoe-fc1-activation-v3-native-gpu7-20261029",
    "generic": "generic-matmul-replay-v1-native-gpu0-20261029",
    "static": "static-node-replay-v8-native-gpu0-fixedshapes-20261029",
}


def fixtures_for(family):
    if family == "qmoe":
        manifest = json.loads((REPOS / "qwen38-int4-qmoe-node-replay/fixtures.json").read_text())
        paths = [REPOS / "qwen38-int4-qmoe-node-replay/fixtures" / x["fixture"] for x in manifest]
        paths += [REPOS / "qwen38-int4-qmoe-fc1-activation/fixtures-v3" / (name + ".onnx")
                  for name in ("no_fc1_bias", "automatic_interleaving", "unsupported_concatenated")]
        return paths
    directory = ("qwen38-int4-generic-matmul-replay/fixtures-v1" if family == "generic"
                 else "qwen38-int4-static-node-replay/fixtures-v8-fixed-shapes")
    return sorted((REPOS / directory).glob("*.onnx"))


def validate_stats(family, rows, original):
    assert len(rows) == len(original)
    if family == "qmoe":
        assert all(row["requests"] == 65 and row["hits"] > 0 and
                   row["captures"] + row["hits"] == row["requests"] - 1 and
                   0 < row["entries"] <= row["capacity"] == 8 for row in rows), rows
    else:
        expected = collections.Counter((r["operator"], r["requests"] * 2) for r in original)
        actual = collections.Counter((r["operator"], r["requests"] - 1) for r in rows)
        assert actual == expected, (actual, expected)
        assert all(row["rebinds"] + row["binding_hits"] == row["hits"] and
                   row["binding_hits"] > 0 and row["hits"] + row["captures"] == row["requests"] - 1 and
                   0 < row["entries"] <= row["capacity"] == 64 and
                   0 < row["binding_entries"] <= row["binding_capacity"] == 64 for row in rows), rows
        if family == "static":
            assert any(row["operator"] == "SplitScaledSiLUF16" and
                       row["evictions"] > 0 and row["captures"] >= 96 for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    build = json.loads((BUILD / "build-manifest.json").read_text())
    assert sha(BUILD / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    reports = {}
    for family, proof_name in PROOFS.items():
        original = json.loads((ROOT / (proof_name + ".qualification.json")).read_text())
        if family == "qmoe":
            assert build["baseline_byte_identical"]
            assert build["baseline_provider_sha256"] == original["provider_sha256"]
        fixtures = fixtures_for(family)
        assert len(fixtures) == original["fixtures"]
        for path in fixtures:
            key = str(path.relative_to(REPOS)) if family == "qmoe" else path.name
            assert sha(path) == original["fixture_sha256"][key], path
        prefix = ROOT / (PREFIX + "-" + family + "-gpu7-20261029")
        stats = ROOT / (prefix.name + "-stats.jsonl")
        guard_path = prefix.with_suffix(".guard.json")
        resume = args.resume_completed and guard_path.exists()
        if not resume:
            assert not guard_path.exists() and not stats.exists()
        executable = BUILD / (family + "_compare")
        assert sha(executable) == build["comparator_sha256"][family]
        stats_flag = ("ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE" if family == "qmoe"
                      else "ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE")
        command = (
            "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
            "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
            "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
            "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
            "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 "
            "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
            "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
            f"ORT_WEBGPU_GENERIC_MATMUL_REPLAY={0 if family == 'static' else 1} "
            f"{stats_flag}=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
            shlex.join(str(p).replace(str(REPOS), "/workspace") for p in
                       [executable, BUILD / "libonnxruntime_providers_webgpu.so", *fixtures]))
        if not resume:
            subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
        guard = json.loads(guard_path.read_text())
        log = prefix.with_suffix(".log").read_text()
        assert guard["status"] == "completed" and guard["exit_code"] == 0
        assert guard["command"] == ["bash", "-lc", command]
        assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
        assert "FAIL" not in log and "GetCapability" not in log
        marker = ("PASS changing_inputs_and_routes " if family == "qmoe"
                  else " changing_rounds=128 invalid_flag_rejected=1")
        assert log.count(marker) == len(fixtures)
        comparisons = original["finite_bitwise_comparisons"]
        assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        validate_stats(family, rows, original["stats"])
        reports[family] = dict(fixtures=len(fixtures), finite_bitwise_comparisons=comparisons,
            stats=rows, comparator_sha256=sha(executable),
            fixture_sha256={str(p.relative_to(REPOS)): sha(p) for p in fixtures})
        print(family, len(fixtures), comparisons, flush=True)
    report = dict(status="changing_immediate_replay_inputs_routes_bindings_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        families=reports, fixtures=sum(r["fixtures"] for r in reports.values()),
        finite_bitwise_comparisons=sum(r["finite_bitwise_comparisons"] for r in reports.values()),
        pending_windows_tested=[1, 64], both_flags_tested=True, invalid_flag_explicitly_rejected=True,
        changing_cpu_split_values_tested=True, lru_eviction_tested=True,
        cached_bindings_checked_against_current_buffers=True, no_object_layout_changes=True,
        rejected_requests_per_eligible_kernel=1,
        actual_model_dispatch_pending=True, quality_qualified=False, headline_eligible=False)
    (ROOT / (PREFIX + "-gpu7-20261029.qualification.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

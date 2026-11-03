import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    directory = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-node-replay")
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v3", "v4", "v5"), default="v5")
    parser.add_argument("--gpu-index", type=int, choices=(3, 7), default=7)
    parser.add_argument("--compare-gate", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or ROOT / f"qmoe-node-replay-{args.revision}-native-gpu{args.gpu_index}-20261029"
    gpu_uuid = {3: "GPU-38eac315-6ffb-8cb1-c68f-521767e9ba3e",
                7: "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"}[args.gpu_index]
    build_dir = directory / ("build-" + args.revision)
    container_build_dir = "/workspace/qwen38-int4-qmoe-node-replay/build-" + args.revision
    comparator = "native_gate_compare" if args.compare_gate else "native_compare"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not stats.exists()
    fixtures = json.loads((directory / "fixtures.json").read_text())
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        f"{gpu_uuid}; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_UNIFORM_ARENA=1 "
        + ("ORT_TEST_COMPARE_GATE=1 " if args.compare_gate else "") +
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        f"exec {container_build_dir}/{comparator} "
        f"{container_build_dir}/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-qmoe-node-replay/fixtures/" + item["fixture"] for item in fixtures))
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(args.gpu_index),
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "400",
         "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS changing_inputs_and_routes ") == len(fixtures)
    entries = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(entries) == sum(item["rows"] == 1 for item in fixtures)
    assert all(0 < entry.get("entries", entry["captures"]) <= entry["capacity"] == 8 for entry in entries)
    assert all(entry["hits"] >= 16 and entry["requests"] == 32 for entry in entries), entries
    if args.revision == "v5":
        assert all(entry["evictions"] >= 8 and entry["captures"] >= 16 for entry in entries), entries
    comparisons = sum(item["rows"] * item["hidden"] * 32 for item in fixtures)
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    report = dict(status="changing_inputs_routes_finite_bitwise_and_actual_replay_qualified",
                  fixtures=len(fixtures), rounds_per_fixture=32, finite_bitwise_comparisons=comparisons,
                  eligible_single_token_fixtures=len(entries),
                  fallback_fixtures=sum(item["rows"] != 1 for item in fixtures),
                  invalid_flag_explicitly_rejected=True, pending_windows_tested=[1, 64],
                  subgroup_gate_toggles_tested=True, stats=entries,
                  subgroup_gate_reference_control_tested=args.compare_gate,
                  lru_eviction_tested=args.revision == "v5",
                  comparator_sha256=sha(build_dir / comparator),
                  provider_sha256=build["provider_sha256"], quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

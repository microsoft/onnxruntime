import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-gdn-prep-node-replay/build-v2")
FIXTURES = DIRECTORY.parent / "fixtures-v1"
PREFIX = ROOT / "gdn-prep-node-replay-v2-native-gpu7-20261101"


def main():
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_byte_identical"] and build["original_object_byte_identical"]
    assert sha(DIRECTORY / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(DIRECTORY / "native_compare") == build["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        path = DIRECTORY / name
        path = Path(str(path).replace("/workspace", "/home/sayanshaw/repos"))
        assert sha(path) == digest, path
    fixtures_manifest = FIXTURES / "manifest.json"
    fixtures = json.loads(fixtures_manifest.read_text())
    paths = [FIXTURES / row["path"] for row in fixtures["fixtures"]]
    for path, row in zip(paths, fixtures["fixtures"]):
        assert sha(path) == row["sha256"]
    stats = ROOT / (PREFIX.name + "-stats.jsonl")
    assert not stats.exists() and not PREFIX.with_suffix(".guard.json").exists()
    args = [DIRECTORY / "native_compare", DIRECTORY / "libonnxruntime_providers_webgpu.so", *paths]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_TEST_DENSE_FP32_ACCUMULATION=0 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec "
        + shlex.join(str(p).replace("/home/sayanshaw/repos", "/workspace", 1) for p in args))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(PREFIX), "--", "bash", "-lc", command], check=True)
    guard = json.loads(PREFIX.with_suffix(".guard.json").read_text())
    log = PREFIX.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    assert "FP32 accumulation selector=0\n" in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1 pending=") == 14
    assert log.count("distinct_live_gpu_banks=3 inputs_each=5") == 14
    assert f'PASS total_exact_elements={fixtures["total_comparisons"]}; CPU fallback disabled' in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["operator"] == "GdnProjPrepF16Exact" and row["requests"] == 128 and
        row["captures"] == 1 and row["hits"] == 127 and row["rebinds"] > 0 and
        row["rebinds"] + row["binding_hits"] == row["hits"] and
        row["entries"] == 1 and row["capacity"] == 64 for row in rows), rows
    report = dict(status="gdn_prep_replay_changing_gpu_inputs_finite_bitwise_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixtures=7,
        pending_windows_tested=[1, 64], rounds_per_fixture_per_window=128,
        distinct_live_gpu_banks=3, gpu_inputs_each=5, mixed_fp16_fp32_outputs_each=5,
        finite_bitwise_comparisons=fixtures["total_comparisons"],
        constructor_selector_modes=[0, 1], invalid_constructor_selector_rejected=True,
        zero_length_and_multi_row_fallbacks_tested=True, stats=rows,
        source_sha256={str(path): sha(path) for path in (
            build_path, fixtures_manifest, stats, PREFIX.with_suffix(".guard.json"),
            PREFIX.with_suffix(".log"))}, quality_qualified=False, headline_eligible=False)
    path = PREFIX.with_suffix(".qualification.json")
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons", "stats")}))


if __name__ == "__main__":
    main()

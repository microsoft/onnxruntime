import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-immediate-node-replay/build-v1")
FIXTURES = DIRECTORY.parent / "mixed-fixtures-v1"
PREFIX = ROOT / "immediate-node-replay-v1-mixed-native-gpu7-20261031"


def verify_stats(rows):
    assert len(rows) == 4
    assert all(row["operator"] == "MatMulNBitsExact" and row["requests"] == 256 and
               row["captures"] == 1 and row["entries"] == 1 and row["hits"] == 255 and
               row["binding_hits"] > 0 and row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["capacity"] == 64 and 0 < row["binding_entries"] <= row["binding_capacity"] == 64
               for row in rows), rows


def main():
    for suffix in (".guard.json", ".log", ".qualification.json"):
        assert not PREFIX.with_suffix(suffix).exists()
    stats = ROOT / (PREFIX.name + "-stats.jsonl")
    assert not stats.exists()
    build_path, fixture_path = DIRECTORY / "build-manifest.json", FIXTURES / "fixtures-manifest.json"
    build, fixtures = (json.loads(path.read_text()) for path in (build_path, fixture_path))
    parent_path = ROOT / "immediate-node-replay-v1-native-gpu7-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert sha(build_path) == parent["build_manifest_sha256"]
    assert sha(DIRECTORY / "libonnxruntime_providers_webgpu.so") == parent["provider_sha256"]
    assert sha(DIRECTORY / "native_compare") == parent["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    reference = json.loads((ROOT / "generic-matmul-replay-v1-native-gpu0-20261029.qualification.json").read_text())
    for name, digest in fixtures["source_fixture_sha256"].items():
        assert digest == reference["fixture_sha256"][name], name
    assert sha(ROOT / "prepare_immediate_replay_mixed.py") == fixtures["generator_sha256"]
    assert fixtures["run_rows"] == [1, 2, 16, 1]
    paths = [FIXTURES / f"mixed_replay_rows{rows}.onnx" for rows in fixtures["run_rows"]]
    assert all(sha(path) == fixtures["fixture_sha256"][path.name] for path in paths)
    repos = "/home/sayanshaw/repos"
    command = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " \
              "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export " \
              "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 " \
              "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 " \
              "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 " \
              "ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 " \
              "ORT_WEBGPU_DENSE_TILE4=1 ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 " \
              f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " + \
              shlex.join(str(path).replace(repos, "/workspace") for path in
                         [DIRECTORY / "native_compare", DIRECTORY / "libonnxruntime_providers_webgpu.so", *paths])
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                    "--output-prefix", str(PREFIX), "--", "bash", "-lc", command], check=True)
    guard, log = json.loads(PREFIX.with_suffix(".guard.json").read_text()), PREFIX.with_suffix(".log").read_text()
    assert guard["command"] == ["bash", "-lc", command] and guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and log.count(" changing_rounds=128 invalid_flag_rejected=1") == 4
    assert log.count("Experimental immediate WebGPU node replay is active.") == 1
    assert fixtures["finite_bitwise_comparisons"] == 13107200
    assert "PASS total_exact_elements=13107200; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    verify_stats(rows)
    sources = [Path(__file__), fixture_path, build_path, parent_path, stats,
               PREFIX.with_suffix(".guard.json"), PREFIX.with_suffix(".log"), *paths]
    report = dict(status="immediate_node_replay_mixed_prefix_finite_bitwise_qualified",
                  provider_sha256=parent["provider_sha256"], comparator_sha256=parent["comparator_sha256"],
                  finite_bitwise_comparisons=13107200, fixtures=4, rows=fixtures["run_rows"],
                  deferred_prefix_and_between_replays=True, pending_windows_tested=[1, 64],
                  invalid_selector_rejected=True, stats=rows, quality_qualified=False, headline_eligible=False,
                  source_sha256={str(path): sha(path) for path in sources})
    with PREFIX.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

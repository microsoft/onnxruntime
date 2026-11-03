import json
from pathlib import Path
import shlex
import subprocess

from build_qmoe_node_replay import sha
from run_qmoe_prefill_rows4_double_buffer_native import (
    BUILD, DATA, DIRECTORY, FLAG, FLAGS, PROOF, ROOT, run_guard, UUID, workspace,
)
from run_qmoe_fc1_double_buffer_native import FLAGS as REPLAY_FLAGS


if __name__ == "__main__":
    prerequisite = ROOT / "qmoe-prefill-rows4-double-buffer-v2-native-outcome.json"
    native = json.loads(prerequisite.read_text())
    assert native["status"] == "prefill_separate_original_outputs_supported"
    for name, digest in native["source_and_evidence_sha256"].items():
        assert sha(Path(name)) == digest, name
    outcome = ROOT / "qmoe-prefill-rows4-double-buffer-v2-checks-outcome.json"
    if outcome.exists():
        raise FileExistsError(outcome)
    prefix = ROOT / "qmoe-prefill-rows4-double-buffer-v2-invalid-gpu7"
    child_log = ROOT / "qmoe-prefill-rows4-double-buffer-v2-invalid-child.log"
    if child_log.exists() or prefix.with_suffix(".guard.json").exists():
        raise FileExistsError("Existing fresh selector evidence")
    fixture = sorted(DIRECTORY.glob("*.onnx"))[0]
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
        f"if env {FLAGS} {FLAG}=2 " +
        shlex.join([workspace(PROOF / "independent_compare"),
                    workspace(BUILD / "libonnxruntime_providers_webgpu.so"),
                    "compare", workspace(DATA), workspace(fixture)]) +
        f" > /evidence/webgpu-perf-oct20/{child_log.name} 2>&1; then "
        "echo 'FAIL: invalid selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' /evidence/webgpu-perf-oct20/{child_log.name}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    assert code == 0 and f"{FLAG} must be 0 or 1" in log
    evidence = [Path(__file__), prerequisite, child_log, prefix.with_suffix(".log"),
                prefix.with_suffix(".guard.json")]
    original_path = ROOT / "qmoe-fc1-double-buffer-v1-native-outcome.json"
    original = json.loads(original_path.read_text())
    assert original["status"] == "qmoe_fc1_separate_original_outputs_supported"
    assert original["records"][0]["phase"] == "capture" and original["records"][0]["exit_code"] == 0
    for name, digest in original["captured_sha256"].items():
        assert sha(Path(name)) == digest, name
        evidence.append(Path(name))
    replay_proof = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc1-double-buffer/native-proof-v1")
    comparator_path = replay_proof / "build-manifest.json"
    comparator = json.loads(comparator_path.read_text())
    binary = replay_proof / "independent_compare"
    assert sha(binary) == comparator["source_sha256"][workspace(binary)]
    fixture_path = ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json"
    fixtures_proof = json.loads(fixture_path.read_text())
    fixtures = [Path("/home/sayanshaw/repos") / name for name in fixtures_proof["fixture_sha256"]]
    assert len(fixtures) == 23
    for fixture in fixtures:
        assert sha(fixture) == fixtures_proof["fixture_sha256"][str(fixture.relative_to("/home/sayanshaw/repos"))]
    prefix = ROOT / "qmoe-prefill-rows4-double-buffer-v2-replay-fallback-gpu7"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    if stats.exists() or prefix.with_suffix(".guard.json").exists():
        raise FileExistsError("Existing replay/fallback evidence")
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
        f"export {REPLAY_FLAGS} {FLAG}=1 "
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; " +
        shlex.join([workspace(binary), workspace(BUILD / "libonnxruntime_providers_webgpu.so"),
                    "compare", workspace(replay_proof / "original-references"),
                    *[workspace(path) for path in fixtures]]))
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    assert code == 0 and "FAIL" not in log
    assert "PASS total_exact_elements=9748480; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == 11
    assert all(row["requests"] == 64 and row["hits"] > 0 and
               row["captures"] + row["hits"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 8 for row in rows), rows
    evidence.extend([original_path, comparator_path, binary, fixture_path, *fixtures, stats,
                     prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
    outcome.write_text(json.dumps(dict(
        status="fresh_selector_and_original_replay_fallback_supported",
        fresh_invalid_selector_rejected=True, fallback_exact_comparisons=9748480,
        original_capture_reused_not_fresh=True, original_capture_provider_sha256=
        json.loads((BUILD / "build-manifest.json").read_text())["baseline_provider_sha256"],
        replay_instances=len(rows), replay_requests=sum(row["requests"] for row in rows),
        replay_hits=sum(row["hits"] for row in rows),
        source_and_evidence_sha256={str(path): sha(path) for path in evidence}), indent=2) + "\n")
    print("fresh_selector_and_original_replay_fallback_supported", flush=True)

import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_dense_bitcast_native import main as run_native


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-ready-replay-key-trim/build-v2")
FLAG = "ORT_WEBGPU_READY_REPLAY_KEY_TRIM"


def verify_rows(rows):
    assert len(rows) == 9
    assert all(row["operator"] == "MatMulNBitsExact" and row["requests"] >= 256 and
               row["captures"] >= 1 and row["hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["binding_hits"] > 0 and row["capacity"] == 64 and
               0 < row["binding_entries"] <= row["binding_capacity"] == 64
               for row in rows), rows


def main():
    reports = []
    for mode in (0, 1):
        prefix = ROOT / f"ready-replay-key-trim-v2-native-mode{mode}-gpu7-20261031"
        key_stats = ROOT / (prefix.name + "-key-stats.jsonl")
        assert not key_stats.exists()
        run_native(
            directory=DIRECTORY, prefix=prefix,
            baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
            report_status="ready_replay_key_trim_dense_changing_inputs_finite_bitwise_qualified",
            extra_flags=f"{FLAG}={mode} "
                        f"ORT_WEBGPU_READY_REPLAY_KEY_STATS_FILE=/evidence/webgpu-perf-oct20/{key_stats.name} "
                        "ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                        "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
            fp32_accumulation=False, stats_verifier=verify_rows)
        path = prefix.with_suffix(".qualification.json")
        proof = json.loads(path.read_text())
        assert proof["finite_bitwise_comparisons"] == 116582784
        if mode:
            rows = [json.loads(line) for line in key_stats.read_text().splitlines()]
            assert rows and all(row["operator"] == "ReadyReplayWithoutKey" and
                                row["commands"] > 0 and row["key_bytes_avoided"] > 0
                                for row in rows), rows
            reports.append((key_stats, sha(key_stats)))
        else:
            assert not key_stats.exists()
        reports.append((path, sha(path)))

    prefix = ROOT / "ready-replay-key-trim-v2-invalid-gpu7-20261031"
    invalid_log = ROOT / (prefix.name + "-invalid.log")
    assert not invalid_log.exists() and not prefix.with_suffix(".guard.json").exists()
    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))
    assert fixtures
    fixture = Path(str(fixtures[0]).replace("/home/sayanshaw/repos", "/workspace", 1))
    args = [str(p).replace("/home/sayanshaw/repos", "/workspace", 1) for p in (
        DIRECTORY / "native_compare", DIRECTORY / "libonnxruntime_providers_webgpu.so", fixture)]
    target = "/evidence/webgpu-perf-oct20/" + invalid_log.name
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
        f"if env {FLAG}=2 {shlex.join(args)} > {shlex.quote(target)} 2>&1; then "
        "echo 'FAIL: invalid ready replay key selector was accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' {shlex.quote(target)}")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "120",
                    "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert f"{FLAG} must be 0 or 1" in invalid_log.read_text()
    reports.extend((p, sha(p)) for p in (invalid_log, prefix.with_suffix(".guard.json"),
                                       prefix.with_suffix(".log")))
    build = DIRECTORY / "build-manifest.json"
    manifest = json.loads(build.read_text())
    result = dict(status="ready_replay_key_trim_process_modes_and_invalid_selector_qualified",
                  provider_sha256=manifest["provider_sha256"],
                  baseline_provider_sha256=manifest["baseline_provider_sha256"],
                  comparator_sha256=manifest["comparator_sha256"],
                  finite_bitwise_comparisons=233165568,
                  process_modes_tested=[0, 1], invalid_selector_rejected_in_fresh_process=True,
                  source_sha256={str(p): digest for p, digest in reports},
                  build_manifest_sha256=sha(build), quality_qualified=False, headline_eligible=False)
    path = ROOT / "ready-replay-key-trim-v2-native-gpu7-20261031.qualification.json"
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

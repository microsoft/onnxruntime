import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    directory = Path("/home/sayanshaw/repos/qwen38-int4-large-binding-cache")
    prefix = ROOT / "large-binding-cache-native-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not stats.exists()
    fixtures = json.loads(Path("/home/sayanshaw/repos/qwen38-int4-exact-expert/fixtures.json").read_text())
    fixtures = [f for f in fixtures if f["rows"] == 1]
    build = json.loads((directory / "build/build-manifest.json").read_text())
    assert sha(directory / "build/libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 "
        f"ORT_WEBGPU_BIND_GROUP_CACHE_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        "exec /workspace/qwen38-int4-large-binding-cache/build/native_compare "
        "/workspace/qwen38-int4-large-binding-cache/build/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-exact-expert/fixtures/" + f["fixture"] for f in fixtures))
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "0",
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "180",
         "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS bitwise ") == len(fixtures)
    samples = [json.loads(line) for line in stats.read_text().splitlines()]
    assert samples and sum(s["hits"] for s in samples) > 0
    peak = max(s["peak_referenced_bytes"] for s in samples)
    assert 256 * 1024**2 < peak <= 96 * 1024**3, peak
    report = dict(status="real_weight_finite_bitwise_and_large_cache_qualified",
                  fixtures=len(fixtures), samples_per_mode=6,
                  provider_sha256=build["provider_sha256"], stats=samples,
                  bounded_peak_referenced_bytes=peak,
                  quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

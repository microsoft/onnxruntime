import collections
import json
from pathlib import Path
import shlex
import subprocess

from prepare_qmoe_extended_prefill import DIRECTORY
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    repos = Path("/home/sayanshaw/repos")
    directory = repos / "qwen38-int4-qmoe-extended-prefill/build-v1"
    prefix = ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    build = json.loads((directory / "build-manifest.json").read_text())
    baseline = json.loads((ROOT / "expert-bitcast-unpack-v2-gpu7-standard2048-20261029.qualification.json").read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    fixtures = sorted(DIRECTORY.glob("*.onnx"))
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    assert len(fixtures) == 6
    assert Path(manifest["source_weights"]).samefile(DIRECTORY / "model.onnx.data")
    for fixture in fixtures:
        evidence = manifest["fixtures"][fixture.name]
        assert sha(fixture) == evidence["fixture_sha256"]
        assert sha(Path(evidence["source"])) == evidence["source_sha256"]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
        "ORT_WEBGPU_STATIC_NODE_REPLAY=1 ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 "
        "ORT_WEBGPU_QMOE_INT4_PREFILL=1 ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 "
        "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 "
        "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in
                   [directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("retained_chunk16_reference=1 invalid_flag_rejected=1") == len(fixtures)
    comparisons = len(fixtures) * 2560 * (8 * (16 + 32 + 64 + 128) + 2 * 129)
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    profiles = [Path(line.removeprefix("PROFILE ").replace("/evidence/webgpu-perf-oct20", str(ROOT)))
                for line in log.splitlines() if line.startswith("PROFILE ")]
    assert len(profiles) == len(fixtures)
    coverage = {}
    for path in profiles:
        events = json.loads(path.read_text())
        counts = collections.Counter(event["name"].split("&")[-1] for event in events
                                     if event.get("cat") == "Kernel")
        assert counts["QmoeGate1Token"] == 152, (path, counts)
        coverage[path.name] = dict(counts)
    report = dict(status="extended_prefill_retained_chunk16_finite_bitwise_qualified",
        original_layers=len(fixtures), rows_tested=[16, 32, 64, 128, 129],
        rounds_per_extended_shape=8, fallback_rows129_rounds=2,
        pending_windows_tested=[1, 64], invalid_flag_explicitly_rejected=True,
        finite_bitwise_comparisons=comparisons, dispatched_programs=coverage,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_manifest=manifest,
        reference_arithmetic="retained_chunk16_not_generic_large_batch",
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "original_layers", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

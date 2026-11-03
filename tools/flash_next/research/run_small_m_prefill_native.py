import argparse
import json
from pathlib import Path
import shlex
import subprocess

from prepare_small_m_prefill import DIRECTORY, ROWS, SHAPES
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v3", "v4"), default="v4")
    args = parser.parse_args()
    repos = Path("/home/sayanshaw/repos")
    directory = repos / f"qwen38-int4-small-m-prefill/build-{args.version}"
    prefix = ROOT / f"small-m-prefill-{args.version}-native-tail-gpu7-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    build = json.loads((directory / "build-manifest.json").read_text())
    baseline = json.loads((ROOT / "expert-bitcast-unpack-v2-gpu7-standard2048-20261029.qualification.json").read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    fixtures = sorted(DIRECTORY.glob("*.onnx"))
    assert len(fixtures) == len(SHAPES) * len(ROWS) == 52
    hashes = {p.name: sha(p) for p in fixtures}
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL=1 "
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
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == len(fixtures)
    comparisons = sum(n for _, n in SHAPES) * sum(ROWS) * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    profiles = [Path(line.removeprefix("PROFILE ").replace("/evidence/webgpu-perf-oct20", str(ROOT)))
                for line in log.splitlines() if line.startswith("PROFILE ")]
    assert len(profiles) == len(fixtures)
    coverage = {}
    for path in profiles:
        events = json.loads(path.read_text())
        names = {event["name"] for event in events}
        custom = sorted(name for name in names if "SmallMPrefillMatMulNBits" in name)
        coverage[path.name] = custom
        is_m16 = "_m16_" in path.name
        is_n1 = "_n1_" in path.name
        assert bool(custom) == (is_m16 and not is_n1), (path, custom)
        if "k10240_n324_m16_" in path.name:
            assert any("F32" in name for name in custom), custom
    report = dict(status="small_m_prefill_changing_inputs_finite_bitwise_qualified",
        fixtures=len(fixtures), shapes=len(SHAPES), true_matrix_rows=list(ROWS),
        rounds_per_fixture=128, pending_windows_tested=[1, 64],
        both_flags_tested=True, invalid_flag_explicitly_rejected=True,
        finite_bitwise_comparisons=comparisons, dispatched_programs=coverage,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixture_sha256=hashes,
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import shlex
import subprocess

from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT, sha


def main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/build-v1"),
         prefix=ROOT / "dense-bitcast-unpack-v1-native-gpu7-20261029",
         baseline_path=ROOT / "expert-bitcast-unpack-v2-gpu7-standard2048-20261029.qualification.json",
         report_status="dense_bitcast_changing_inputs_finite_bitwise_and_replay_qualified",
         extra_flags="", fp32_accumulation=None, expect_replay=True, verify_existing=False,
         stats_verifier=None):
    repos = Path("/home/sayanshaw/repos")
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".qualification.json").exists()
    if not verify_existing:
        assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / "generic-matmul-replay-v1-native-gpu0-20261029.qualification.json").read_text())
    baseline = json.loads(baseline_path.read_text())
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    fixtures = sorted((repos / "qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == len(SHAPES) * 3 == 27
    assert all(sha(p) == parent["fixture_sha256"][p.name] for p in fixtures)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 "
        "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        + extra_flags +
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in
                   [directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    if not verify_existing:
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
            "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    if fp32_accumulation is not None:
        assert f"FP32 accumulation selector={int(fp32_accumulation)}\n" in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == len(fixtures)
    comparisons = sum(n for _, n in SHAPES) * (1 + 2 + 16) * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    if expect_replay:
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        if stats_verifier is not None:
            stats_verifier(rows)
        else:
            assert len(rows) == len(SHAPES)
            assert all(r["operator"] == "MatMulNBitsExact" and r["requests"] == 256 and
                r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
                r["hits"] + r["captures"] == r["requests"] and
                r["entries"] == 2 and r["capacity"] == 64 and
                0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows
    else:
        assert fp32_accumulation is True and not stats.exists()
        source = directory / "matmul_nbits.cc"
        assert sha(source) == build["source_sha256"][source.name]
        route = source.read_text().split("const bool dense_bitcast_route =", 1)[1].split("auto compute =", 1)[0]
        assert "!context.EnableMatmulFp32Accumulation()" in route
        rows = []
    report = dict(status=report_status,
        fixtures=len(fixtures), single_token_shapes=len(SHAPES), rounds_per_fixture=128,
        pending_windows_tested=[1, 64], both_flags_tested=True,
        invalid_flag_explicitly_rejected=True, replay_variant_keys_tested=expect_replay,
        multi_token_fallback_tested=True,
        finite_bitwise_comparisons=comparisons, stats=rows,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"],
        build_manifest_sha256=sha(directory / "build-manifest.json"),
        fixture_sha256={p.name: sha(p) for p in fixtures},
        quality_qualified=False, headline_eligible=False)
    if fp32_accumulation is not None:
        report["fp32_accumulation_enabled"] = fp32_accumulation
        report["fp32_accumulation_fallback_only"] = not expect_replay
    report["source_sha256"] = {p.name: sha(p) for p in
        (prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"), directory / "build-manifest.json")}
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

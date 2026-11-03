import argparse
import collections
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("projection", "up"), default="projection")
    parser.add_argument("--revision", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    up = args.stage == "up"
    revision = args.revision
    name = "hc-up-premix" if up else "hc-projection-silu"
    directory = Path("/home/sayanshaw/repos/qwen38-int4-" + name)
    prefix = ROOT / f"{name}-{revision}-real-weights-gpu0-20261029"
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    fixtures = sorted(p for p in (directory / "real-weight-fixtures-v2").glob("projection_*.onnx")
                      if not p.stem.endswith("_reference"))
    cases = 291 if up else 288
    assert len(fixtures) == cases
    library = directory / ("build-" + revision) / "libonnxruntime_providers_webgpu.so"
    build = json.loads((library.parent / "build-manifest.json").read_text())
    assert sha(library) == build["provider_sha256"]
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_HC_EXACT_DENSE=1 ORT_WEBGPU_EXACT_DENSE_MID=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_PROJECTION_SILU=1 ORT_WEBGPU_HC_UP_PREMIX=1 "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        f"exec /workspace/qwen38-int4-{name}/real-weight-build-v1/native_compare "
        f"/workspace/qwen38-int4-{name}/build-{revision}/libonnxruntime_providers_webgpu.so " +
        " ".join(f"/workspace/qwen38-int4-{name}/real-weight-fixtures-v2/" + p.name for p in fixtures))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "0",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS exact finite outputs ") == cases
    comparisons = cases * 3 * (2560 if up else 324)
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    counts = collections.Counter(r["operator"] for r in rows)
    expected = ({"HCUpPreMixF16": cases, "HyperConnectionPreMix": cases} if up else
                {"HCProjectionSiLUF16": cases, "MatMulNBitsExact": cases, "SplitScaledSiLUF16": cases})
    assert counts == expected, counts
    assert all(r["requests"] == 3 and r["captures"] == 1 and r["hits"] == 2 and
               r["rebinds"] + r["binding_hits"] == r["hits"] for r in rows)
    report = dict(status="real_weights_finite_bitwise_and_actual_fused_dispatch_qualified",
        fixture_pairs=cases, distinct_model_projections=97 if up else 96, activation_stress_scales=[0.125, 1.0, 4.0],
        finite_bitwise_comparisons=comparisons, actual_node_counts=dict(counts),
        provider_sha256=build["provider_sha256"],
        comparator_sha256=sha(directory / "real-weight-build-v1/native_compare"),
        comparator_source_sha256=sha(directory / "real-weight-build-v1/native_compare.cpp"),
        fixture_sha256={p.name: sha(p) for p in fixtures},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

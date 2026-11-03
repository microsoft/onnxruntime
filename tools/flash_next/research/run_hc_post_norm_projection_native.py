import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-columns", action="store_true")
    args = parser.parse_args()
    directory = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection")
    build_dir = directory / ("build-v3" if args.four_columns else "build-v2")
    prefix = ROOT / ("hc-post-norm-four-columns-v3-native-gpu7-20261029" if args.four_columns
                     else "hc-post-norm-projection-v2-native-gpu7-20261029")
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not any(prefix.with_suffix(s).exists() for s in (".guard.json", ".qualification.json"))
    assert not stats.exists()
    fixtures = sorted((directory / "fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == 36
    hashes = {p.name: sha(p) for p in fixtures}
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    container_build = str(build_dir).replace("/home/sayanshaw/repos", "/workspace")
    gpu_uuid = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        f"{gpu_uuid}; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_PROJECTION_SILU=1 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
        + ("ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS=1 " if args.four_columns else "") +
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; "
        f"exec {container_build}/native_compare {container_build}/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-hc-post-norm-projection/fixtures-v1/" + p.name for p in fixtures))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == 36
    comparisons = 12 * (1 + 2 + 16) * (2 * 10240 + 324) * 128
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    names = ({"HCPostNormProjectionFourColumnsF16", "HCSharedPostNormProjectionFourColumnsF16"}
             if args.four_columns else {"HCPostNormProjectionF16", "HCSharedPostNormProjectionF16"})
    assert len(rows) == 12, rows
    assert all(sum(r["operator"] == name for r in rows) == 6 for name in names)
    assert all(r["operator"] in names and r["requests"] == 96 and
        r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
        r["hits"] + r["captures"] == r["requests"] and r["evictions"] > 0 and r["captures"] >= 65 and
        0 < r["entries"] <= r["capacity"] == 64 and
        0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows
    assert all(sha(p) == hashes[p.name] for p in fixtures)
    report = dict(status="four_outputs_changing_inputs_cpu_values_finite_bitwise_and_replay_qualified",
        fixtures=36, fixture_weights="immutable_random_not_real_model_weights",
        eligible_single_token_fixtures=12, fallback_fixtures=24, rounds_per_fixture=128,
        finite_bitwise_comparisons=comparisons, stats=rows,
        pending_windows_tested=[1, 64], invalid_flag_explicitly_rejected=True,
        changing_cpu_split_values_tested=True, lru_eviction_tested=True,
        exact_dense_selectors_toggled=True, norm_scale_layouts=[[2560], [4, 2560]],
        all_four_outputs_compared=True, compound_fallback_not_fair_window_baseline=True,
        four_columns=args.four_columns,
        invalid_column_selector_rejected=args.four_columns,
        qmoe_replay_disabled=True, comparator_sha256=sha(build_dir / "native_compare"),
        provider_sha256=build["provider_sha256"], fixture_sha256=hashes,
        baseline_provider_sha256=build["baseline_provider_sha256"],
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

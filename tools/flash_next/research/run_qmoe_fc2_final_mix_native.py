import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import onnx

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_node_replay_immediate_native import fixtures_for


def no_fc2_bias_fixture(repos):
    source = repos / "qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx"
    model = onnx.load(source, load_external_data=False)
    assert model.graph.node[0].input[7].endswith(".bias")
    model.graph.node[0].input[7] = ""
    directory = repos / "qwen38-int4-qmoe-fc2-final-mix/fixtures-v2"
    directory.mkdir(exist_ok=False)
    locations = {entry.value for tensor in model.graph.initializer
                 for entry in tensor.external_data if entry.key == "location"}
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unexpected external-data location: {location}")
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source.parent / relative, target)
        except PermissionError:
            subprocess.run(["sudo", "-n", "ln", str(source.parent / relative), str(target)], check=True)
        assert target.samefile(source.parent / relative)
    path = directory / "no_fc2_bias.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    return path


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--combined-map", action="store_true")
    modes.add_argument("--padded-partials", action="store_true")
    modes.add_argument("--padded-activations", action="store_true")
    modes.add_argument("--hierarchical-gate", action="store_true")
    parser.add_argument("--compare-gate", action="store_true")
    args = parser.parse_args()
    if args.compare_gate and not args.combined_map:
        parser.error("--compare-gate requires the combined provider")
    repos = Path("/home/sayanshaw/repos")
    reuses_fixtures = args.combined_map or args.padded_partials or args.padded_activations or args.hierarchical_gate
    directory = repos / ("qwen38-int4-qmoe-hierarchical-gate/build-v2" if args.hierarchical_gate else
                         "qwen38-int4-qmoe-activation-padding/build-v1" if args.padded_activations else
                         "qwen38-int4-qmoe-fc2-padding/build-v1" if args.padded_partials else
                         "qwen38-int4-qmoe-fusions-map/build-v2" if args.combined_map else
                         "qwen38-int4-qmoe-fc2-final-mix/build-v2")
    prefix = ROOT / ("qmoe-hierarchical-gate-v2-native-gpu7-20261029" if args.hierarchical_gate else
                     "qmoe-activation-padding-v1-native-gpu7-20261029" if args.padded_activations else
                     "qmoe-fusions-map-v2-native-gate-isolated-gpu7-20261029" if args.compare_gate else
                     "qmoe-fc2-padding-v1-native-gpu7-20261029" if args.padded_partials else
                     "qmoe-fusions-map-v2-native-qmoe-gpu7-20261029" if args.combined_map else
                     "qmoe-fc2-final-mix-v2-native-gpu7-20261029")
    stats = ROOT / (prefix.name + "-stats.jsonl")
    assert not prefix.with_suffix(".guard.json").exists() and not stats.exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    proof = ("qmoe-fusions-map-v2-native-qmoe-gpu7-20261029" if args.padded_activations or args.hierarchical_gate else
             "qmoe-fc2-final-mix-v2-native-gpu7-20261029" if reuses_fixtures
             else "qmoe-fc1-activation-v3-native-gpu7-20261029")
    parent = json.loads((ROOT / (proof + ".qualification.json")).read_text())
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    baseline = (json.loads((ROOT / "hc-post-norm-projection-v2-gpu7-standard2048-20261029.qualification.json").read_text())
                if args.hierarchical_gate else parent)
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == baseline["provider_sha256"]
    comparator = (directory / "native_compare" if args.padded_partials or args.padded_activations or args.hierarchical_gate else
                  repos / "qwen38-int4-qmoe-fc2-final-mix/build-v2/native_compare")
    expected_comparator = parent["comparator_sha256"] if args.combined_map else build["comparator_sha256"]
    if args.compare_gate:
        gate_build_dir = repos / "qwen38-int4-qmoe-subgroup-gate-compare/build-v1"
        gate_build = json.loads((gate_build_dir / "build-manifest.json").read_text())
        assert gate_build["fc2_fusion_fixed_enabled"] and gate_build["requires_ort_test_compare_gate"]
        assert gate_build["original_source_sha256"] == sha(comparator.with_suffix(".cpp"))
        comparator = gate_build_dir / "native_compare"
        expected_comparator = gate_build["comparator_sha256"]
    assert sha(comparator) == expected_comparator
    fixtures = ([repos / path for path in parent["fixture_sha256"]] if reuses_fixtures
                else fixtures_for("qmoe"))
    assert len(fixtures) == (22 if reuses_fixtures else 21)
    for path in fixtures:
        assert sha(path) == parent["fixture_sha256"][str(path.relative_to(repos))]
    if not reuses_fixtures:
        fixtures.append(no_fc2_bias_fixture(repos))
    identity_manifest = None
    if args.hierarchical_gate:
        identity = repos / "qwen38-int4-qmoe-hierarchical-gate/fixtures-v1/gate_identity_rows1.onnx"
        identity_manifest = json.loads((identity.parent / "manifest.json").read_text())
        assert identity_manifest["synthetic_sparse_identity_fixture"]
        assert sha(identity.parent / "identity.data") == identity_manifest["data_sha256"]
        fixtures.append(identity)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
        "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 "
        + ("ORT_TEST_COMPARE_GATE=1 " if args.compare_gate else "") +
        ("ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 " if args.padded_partials or args.padded_activations or args.hierarchical_gate else "") +
        f"ORT_WEBGPU_QSA_PARALLEL_MAP={1 if args.combined_map or args.padded_activations or args.hierarchical_gate else 0} "
        "ORT_WEBGPU_QMOE_INT4_PREFILL=1 ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 "
        "ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
        f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
        shlex.join(str(p).replace(str(repos), "/workspace") for p in
                   [comparator, directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS changing_inputs_and_routes ") == len(fixtures)
    comparisons = parent["finite_bitwise_comparisons"] + (2560 * 32 if args.hierarchical_gate or not reuses_fixtures else 0)
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert len(rows) == (11 if args.hierarchical_gate else 10)
    assert all(row["requests"] == 64 and row["hits"] > 0 and
               row["captures"] + row["hits"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 8 for row in rows), rows
    report = dict(status="changing_fc2_final_mix_inputs_routes_qualified",
        fixtures=len(fixtures), eligible_single_token_fixtures=10 if args.hierarchical_gate else 9, fallback_fixtures=13,
        rounds_per_fixture=32, finite_bitwise_comparisons=comparisons,
        both_flags_tested=True, fc1_fusion_fixed_enabled=True, pending_windows_tested=[1, 64],
        variant_cache_keys_tested=True, optional_fc1_and_fc2_bias_tested=True,
        automatic_interleaving_and_concatenated_fallback_tested=True,
        three_four_five_command_capture_contracts_tested=True, stats=rows,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=sha(comparator), combined_parallel_map=args.combined_map or args.padded_activations or args.hierarchical_gate,
        fc2_fusion_fixed_enabled=args.padded_partials or args.compare_gate or args.padded_activations or args.hierarchical_gate,
        padded_partials_toggled=args.padded_partials,
        gate_compared_against_original=args.compare_gate,
        activation_padding_toggled=args.padded_activations,
        hierarchical_gate_toggled=args.hierarchical_gate,
        direct_router_selection_and_weights_tested=args.hierarchical_gate,
        identity_manifest=identity_manifest,
        fixture_sha256={str(p.relative_to(repos)): sha(p) for p in fixtures},
        actual_model_dispatch_pending=True, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()

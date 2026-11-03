import argparse
import collections
import json
import math
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha, snapshot
from model_cache_preflight import prepare_model_cache, snapshot_model_cache

REPOS = Path("/home/sayanshaw/repos")
GPU_UUIDS = {0: "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd",
             3: "GPU-38eac315-6ffb-8cb1-c68f-521767e9ba3e",
             4: "GPU-68170cb3-c9ea-a588-7558-ae82635d9173",
             7: "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"}


def projection_replay_counts(tiled, phase):
    if not tiled:
        return {"HCProjectionSiLUF16": 96}
    counts = {"HCProjectionSiLUTiledF16": 96}
    if phase == "windows":
        counts["HCProjectionSiLUF16"] = 96
    return counts


def validate_completed_guard(guard, command, gpu_index, gpu_uuid):
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["command"] == command
    assert guard["gpu_index"] == gpu_index and guard["gpu_uuid"] == gpu_uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("large-cache", "post-norm", "gdn-view", "hc-gdn",
                                         "node-replay", "hc-node-replay", "static-replay", "hc-projection",
                                         "hc-up-premix", "qsa-map", "generic-replay", "direct-a",
                                         "fc1-activation"), required=True)
    parser.add_argument("--phase", choices=("windows", "standard", "profile"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--revision", choices=("v3", "v4", "v5", "v6", "v7", "v8"), default="v8")
    parser.add_argument("--static-revision", choices=("v7", "v8"), default="v7")
    parser.add_argument("--up-revision", choices=("v1", "v2"), default="v2")
    parser.add_argument("--gpu-index", type=int, choices=(0, 3, 4, 7), default=7)
    parser.add_argument("--native-qualification", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--replay-immediate", action="store_true")
    variants.add_argument("--fc2-final-mix", action="store_true")
    combinations = parser.add_mutually_exclusive_group()
    combinations.add_argument("--with-parallel-map", action="store_true")
    combinations.add_argument("--padded-partials", action="store_true")
    accelerators = parser.add_mutually_exclusive_group()
    accelerators.add_argument("--tiled-projection", action="store_true")
    accelerators.add_argument("--subgroup-gate", action="store_true")
    parser.add_argument("--padded-activations", action="store_true")
    parser.add_argument("--post-norm-projection", action="store_true")
    parser.add_argument("--post-norm-four-columns", action="store_true")
    parser.add_argument("--hierarchical-gate", action="store_true")
    parser.add_argument("--mlp-input-projections", action="store_true")
    parser.add_argument("--mlp-bitcast-unpack", action="store_true")
    parser.add_argument("--expert-bitcast-unpack", action="store_true")
    parser.add_argument("--dense-bitcast-unpack", action="store_true")
    parser.add_argument("--fc2-four-columns", action="store_true")
    parser.add_argument("--hc-up-tiled", action="store_true")
    parser.add_argument("--packed-gate", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--pending", type=int, choices=(32, 48, 64, 96, 128, 192, 256, 512), default=64)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if (args.replay_immediate or args.fc2_final_mix) and args.kind != "fc1-activation":
        parser.error("Replay-immediate and FC2-final-mix variants require their qualified fc1-activation parent")
    if (args.with_parallel_map or args.padded_partials) and not args.fc2_final_mix:
        parser.error("Map and padding variants require the qualified FC1/FC2 fusion parent")
    if (args.tiled_projection or args.subgroup_gate) and not args.with_parallel_map:
        parser.error("Projection/gate accelerators require the qualified combined-map provider")
    if args.padded_activations and not args.subgroup_gate:
        parser.error("--padded-activations requires the qualified subgroup-gate stack")
    if args.post_norm_projection:
        if not args.subgroup_gate or args.padded_activations:
            parser.error("--post-norm-projection requires the unpadded qualified subgroup-gate stack")
        if args.phase == "windows" and not (args.post_norm_four_columns or args.hierarchical_gate):
            parser.error("Post-normalization graph fusion requires separate graph runs; its uncached fallback is not a fair window baseline")
    if args.post_norm_four_columns and not args.post_norm_projection:
        parser.error("--post-norm-four-columns requires the qualified post-normalization graph")
    if args.hierarchical_gate and (not args.post_norm_projection or args.post_norm_four_columns):
        parser.error("--hierarchical-gate requires the unchanged qualified eight-column post-normalization stack")
    if args.mlp_input_projections:
        if not args.hierarchical_gate:
            parser.error("--mlp-input-projections requires the qualified hierarchical-gate parent")
        if args.phase == "windows" and not (args.mlp_bitcast_unpack or args.expert_bitcast_unpack):
            parser.error("MLP graph fusion requires separate graph runs; its compound fallback is not a fair window baseline")
    if args.mlp_bitcast_unpack and not args.mlp_input_projections:
        parser.error("--mlp-bitcast-unpack requires the qualified MLP projection graph")
    if args.expert_bitcast_unpack and (not args.mlp_input_projections or args.mlp_bitcast_unpack):
        parser.error("--expert-bitcast-unpack requires the unchanged qualified MLP projection parent")
    if args.dense_bitcast_unpack and not args.expert_bitcast_unpack:
        parser.error("--dense-bitcast-unpack requires the qualified expert-bitcast parent")
    if args.fc2_four_columns and (not args.expert_bitcast_unpack or args.dense_bitcast_unpack):
        parser.error("--fc2-four-columns requires the unchanged qualified expert-bitcast parent")
    if args.hc_up_tiled and (not args.expert_bitcast_unpack or args.dense_bitcast_unpack or args.fc2_four_columns):
        parser.error("--hc-up-tiled requires the unchanged qualified expert-bitcast parent")
    if args.packed_gate and (not args.expert_bitcast_unpack or args.dense_bitcast_unpack or
                             args.fc2_four_columns or args.hc_up_tiled):
        parser.error("--packed-gate requires the unchanged qualified expert-bitcast parent")
    if args.with_parallel_map and not (args.tiled_projection or args.subgroup_gate) and args.phase == "windows" and args.context < 8192:
        parser.error("Combined map windows require an eligible context of at least 8192")
    is_up = args.kind in ("hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation")
    gpu_uuid = GPU_UUIDS[args.gpu_index]
    prefix = args.output_prefix
    if args.resume_completed:
        if prefix.with_suffix(".qualification.json").exists():
            raise FileExistsError(prefix)
        for suffix in (".json", ".guard.json", ".log"):
            if not prefix.with_suffix(suffix).is_file():
                raise FileNotFoundError(prefix.with_suffix(suffix))
    elif any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
        raise FileExistsError(prefix)
    if args.kind == "large-cache":
        build_dir = REPOS / "qwen38-int4-large-binding-cache/build"
        native_path = ROOT / "large-binding-cache-native-20261029.qualification.json"
        native_status = "real_weight_finite_bitwise_and_large_cache_qualified"
        model = REPOS / "qwen38-int4-quickgelu-next/model"
    elif args.kind == "fc1-activation":
        if args.up_revision != "v2":
            parser.error("fc1-activation requires its qualified hc-up-premix v2 ancestor")
        build_dir = REPOS / "qwen38-int4-qmoe-fc1-activation/build-v3"
        native_path = ROOT / "qmoe-fc1-activation-v3-native-gpu7-20261029.qualification.json"
        native_status = "changing_fc1_activation_inputs_routes_qualified"
        model = REPOS / "qwen38-int4-hc-up-premix/model"
    elif args.kind == "direct-a":
        if args.up_revision != "v2":
            parser.error("direct-a requires its qualified hc-up-premix v2 ancestor")
        build_dir = REPOS / "qwen38-int4-direct-a-matmul/build-v2"
        native_path = ROOT / "direct-a-matmul-v2-native-gpu7-20261029.qualification.json"
        native_status = "changing_direct_a_inputs_routes_qualified"
        model = REPOS / "qwen38-int4-hc-up-premix/model"
    elif args.kind == "generic-replay":
        if args.up_revision != "v2":
            parser.error("generic-replay requires its qualified hc-up-premix v2 parent")
        build_dir = REPOS / "qwen38-int4-generic-matmul-replay/build-v1"
        native_path = ROOT / "generic-matmul-replay-v1-native-gpu0-20261029.qualification.json"
        native_status = "changing_generic_matmul_inputs_qualified"
        model = REPOS / "qwen38-int4-hc-up-premix/model"
    elif args.kind == "qsa-map":
        if args.up_revision != "v2":
            parser.error("qsa-map requires its qualified hc-up-premix v2 parent")
        build_dir = REPOS / "qwen38-int4-qsa-parallel-map/build-v1"
        native_path = ROOT / "qsa-parallel-map-v1-native-original-harness-gpu0-20261029.qualification.json"
        native_status = "exact_reference_and_parallel_map_dispatch_qualified"
        model = REPOS / "qwen38-int4-hc-up-premix/model"
    elif args.kind == "hc-up-premix":
        build_dir = REPOS / ("qwen38-int4-hc-up-premix/build-" + args.up_revision)
        native_path = ROOT / f"hc-up-premix-{args.up_revision}-native-gpu0-20261029.qualification.json"
        native_status = "changing_gpu_inputs_finite_bitwise_and_up_premix_replay_qualified"
        model = REPOS / "qwen38-int4-hc-up-premix/model"
    elif args.kind == "hc-projection":
        build_dir = REPOS / "qwen38-int4-hc-projection-silu/build-v1"
        native_path = ROOT / "hc-projection-silu-v1-native-gpu0-20261029.qualification.json"
        native_status = "changing_gpu_inputs_cpu_values_finite_bitwise_and_replay_qualified"
        model = REPOS / "qwen38-int4-hc-projection-silu/model"
    elif args.kind == "static-replay":
        build_dir = REPOS / ("qwen38-int4-static-node-replay/build-" + args.static_revision)
        native_gpu = 7 if args.static_revision == "v7" else 0
        fixture_tag = "" if args.static_revision == "v7" else "-fixedshapes"
        native_path = ROOT / f"static-node-replay-{args.static_revision}-native-gpu{native_gpu}{fixture_tag}-20261029.qualification.json"
        native_status = "changing_gpu_inputs_cpu_values_finite_bitwise_and_replay_qualified"
        model = REPOS / "qwen38-int4-hc-post-norm/model"
    elif args.kind in ("node-replay", "hc-node-replay"):
        build_dir = REPOS / ("qwen38-int4-qmoe-node-replay/build-v5" if args.kind == "node-replay"
                            else "qwen38-int4-hc-node-replay-combined/build-v1")
        native_path = ROOT / "qmoe-node-replay-v5-native-gpu3-20261029.qualification.json"
        native_status = "changing_inputs_routes_finite_bitwise_and_actual_replay_qualified"
        model = REPOS / ("qwen38-int4-quickgelu-next/model" if args.kind == "node-replay"
                        else "qwen38-int4-hc-post-norm/model")
    elif args.kind == "gdn-view":
        build_dir = REPOS / "qwen38-int4-gdn-decode-view/build-original"
        native_path = ROOT / "gdn-decode-view-native-20261029.qualification.json"
        native_status = "native_finite_bitwise_view_and_fallback_qualified"
        model = REPOS / "qwen38-int4-quickgelu-next/model"
    elif args.kind == "hc-gdn":
        build_dir = REPOS / "qwen38-int4-hc-gdn-combined/build-v2"
        native_path = ROOT / "hc-post-norm-v8-gpu3-native-20261029.qualification.json"
        native_status = "native_finite_bitwise_and_actual_dispatch_qualified"
        model = REPOS / "qwen38-int4-hc-post-norm/model"
        if args.phase == "windows":
            parser.error("HC graph changes require separate original/candidate requests")
    else:
        build_dir = REPOS / ("qwen38-int4-hc-post-norm/build-" + args.revision)
        native_path = ROOT / ("hc-post-norm-" + args.revision + "-native-20261029.qualification.json")
        native_status = "native_finite_bitwise_and_actual_dispatch_qualified"
        model = REPOS / "qwen38-int4-hc-post-norm/model"
        if args.phase == "windows":
            parser.error("Post/norm graph changes require separate original/candidate requests")
    native = json.loads((args.native_qualification or native_path).read_text())
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert native["status"] == native_status
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    if is_up:
        up_build = build
        up_native = native
        if args.kind in ("qsa-map", "generic-replay", "direct-a", "fc1-activation"):
            up_build = json.loads((REPOS / "qwen38-int4-hc-up-premix/build-v2/build-manifest.json").read_text())
            up_native = json.loads((ROOT / "hc-up-premix-v2-native-gpu0-20261029.qualification.json").read_text())
            assert native["provider_sha256"] == build["provider_sha256"]
            assert build["baseline_byte_identical"]
            baseline_sha = up_build["provider_sha256"]
            if args.kind in ("direct-a", "fc1-activation"):
                generic_build = json.loads((REPOS / "qwen38-int4-generic-matmul-replay/build-v1/build-manifest.json").read_text())
                generic_native = json.loads((ROOT / "generic-matmul-replay-v1-native-gpu0-20261029.qualification.json").read_text())
                assert generic_native["status"] == "changing_generic_matmul_inputs_qualified"
                assert generic_native["provider_sha256"] == generic_build["provider_sha256"]
                assert generic_native["baseline_provider_sha256"] == generic_build["baseline_provider_sha256"] == baseline_sha
                assert generic_build["baseline_byte_identical"]
                baseline_sha = generic_build["provider_sha256"]
                assert native["variant_cache_keys_tested"]
                if args.kind == "direct-a":
                    assert native["generic_fixtures"] == 27 and native["qmoe_fixtures"] == 18
                else:
                    assert native["fixtures"] == 21 and native["eligible_single_token_fixtures"] == 8
                    assert native["four_and_five_command_capture_contracts_tested"]
            assert native["baseline_provider_sha256"] == build["baseline_provider_sha256"] == baseline_sha
            assert native["both_flags_tested"]
            if args.kind == "qsa-map":
                assert native["reference_executions"] == 2484
            elif args.kind == "generic-replay":
                assert native["single_token_shapes"] == 9 and native["fixtures"] == 27
            assert up_native["status"] == "changing_gpu_inputs_finite_bitwise_and_up_premix_replay_qualified"
        prior = json.loads((REPOS / "qwen38-int4-hc-projection-silu/build-v1/build-manifest.json").read_text())
        assert up_native["up_premix_fusion_on_off_tested"]
        assert up_native["provider_sha256"] == up_build["provider_sha256"]
        assert up_build["baseline_byte_identical"] and up_build["baseline_provider_sha256"] == prior["provider_sha256"]
        proof = json.loads((ROOT / "hc-projection-silu-v1-real-weights-gpu0-20261029.qualification.json").read_text())
        assert proof["provider_sha256"] == prior["provider_sha256"]
        assert proof["status"] == "real_weights_finite_bitwise_and_actual_fused_dispatch_qualified"
        if args.phase == "standard":
            real = json.loads((ROOT / f"hc-up-premix-{args.up_revision}-real-weights-gpu0-20261029.qualification.json").read_text())
            assert real["status"] == "real_weights_finite_bitwise_and_actual_fused_dispatch_qualified"
            assert real["provider_sha256"] == up_build["provider_sha256"] and real["distinct_model_projections"] == 97
    elif args.kind == "hc-projection":
        prior = json.loads((REPOS / "qwen38-int4-static-node-replay/build-v8/build-manifest.json").read_text())
        assert native["projection_fusion_on_off_tested"]
        assert native["provider_sha256"] == build["provider_sha256"]
        assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == prior["provider_sha256"]
        proof = json.loads((ROOT / "static-node-replay-v8-native-gpu0-fixedshapes-20261029.qualification.json").read_text())
        assert proof["provider_sha256"] == prior["provider_sha256"] and proof["status"] == native_status
        if args.phase == "standard":
            real = json.loads((ROOT / "hc-projection-silu-v1-real-weights-gpu0-20261029.qualification.json").read_text())
            assert real["status"] == "real_weights_finite_bitwise_and_actual_fused_dispatch_qualified"
            assert real["provider_sha256"] == build["provider_sha256"] and real["distinct_model_projections"] == 96
    elif args.kind == "static-replay":
        prior = json.loads((REPOS / "qwen38-int4-hc-node-replay-combined/build-v1/build-manifest.json").read_text())
        assert native["provider_sha256"] == build["provider_sha256"]
        assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == prior["provider_sha256"]
        for phase in ("profile", "windows"):
            proof = json.loads((ROOT / f"hc-node-replay-gpu3-{phase}-20261029.qualification.json").read_text())
            assert proof["status"] == "original_sequences_qualified"
            assert proof["provider_sha256"] == prior["provider_sha256"]
    elif args.kind == "hc-gdn":
        boundary = json.loads((ROOT / "hc-post-norm-v8-gpu3-boundaries-20261029.qualification.json").read_text())
        gdn = json.loads((ROOT / "gdn-decode-view-native-20261029.qualification.json").read_text())
        assert boundary["status"] == native_status and gdn["status"] == "native_finite_bitwise_view_and_fallback_qualified"
        assert native["provider_sha256"] == boundary["provider_sha256"] == build["component_provider_sha256"]["hc"]
        assert gdn["provider_sha256"] == build["component_provider_sha256"]["gdn"]
        assert build["component_relinks_byte_identical"]
    elif args.kind == "hc-node-replay":
        hc = json.loads((REPOS / "qwen38-int4-hc-gdn-combined/build-v2/build-manifest.json").read_text())
        replay = json.loads((REPOS / "qwen38-int4-qmoe-node-replay/build-v5/build-manifest.json").read_text())
        assert build["component_relinks_byte_identical"]
        assert build["component_provider_sha256"] == dict(hc=hc["provider_sha256"], replay=replay["provider_sha256"])
        assert build["gdn_object_sha256"] == hc["gdn_object_sha256"]
        assert native["provider_sha256"] == replay["provider_sha256"]
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"hc-gdn-combined-gpu3-standard{context // 1024}k-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == hc["provider_sha256"]
    else:
        assert native["provider_sha256"] == build["provider_sha256"]
    if args.kind == "node-replay":
        gdn = json.loads((ROOT / "gdn-decode-view-native-20261029.qualification.json").read_text())
        assert gdn["status"] == "native_finite_bitwise_view_and_fallback_qualified"
        assert gdn["provider_sha256"] == build["baseline_provider_sha256"] and build["baseline_byte_identical"]
    if args.replay_immediate:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-node-replay-immediate/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        immediate_native = json.loads((ROOT / "node-replay-immediate-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert immediate_native["status"] == "changing_immediate_replay_inputs_routes_bindings_qualified"
        assert immediate_native["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert immediate_native["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert immediate_native["fixtures"] == 78 and immediate_native["finite_bitwise_comparisons"] == 255578240
        assert immediate_native["both_flags_tested"] and immediate_native["invalid_flag_explicitly_rejected"]
        assert immediate_native["cached_bindings_checked_against_current_buffers"]
    if args.fc2_final_mix:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-fc2-final-mix/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        fc2_native = json.loads((ROOT / "qmoe-fc2-final-mix-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert fc2_native["status"] == "changing_fc2_final_mix_inputs_routes_qualified"
        assert fc2_native["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert fc2_native["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert fc2_native["fixtures"] == 22 and fc2_native["finite_bitwise_comparisons"] == 9666560
        assert fc2_native["eligible_single_token_fixtures"] == 9
        assert fc2_native["both_flags_tested"] and fc2_native["variant_cache_keys_tested"]
        assert fc2_native["three_four_five_command_capture_contracts_tested"]
    if args.padded_partials:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-fc2-padding/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "qmoe-fc2-padding-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "changing_fc2_final_mix_inputs_routes_qualified"
        assert proof["padded_partials_toggled"] and proof["fc2_fusion_fixed_enabled"]
        assert proof["both_flags_tested"] and proof["variant_cache_keys_tested"]
        assert proof["fixtures"] == 22 and proof["finite_bitwise_comparisons"] == 9666560
    if args.with_parallel_map:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-fusions-map/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        assert build["baseline_byte_identical"] and build["map_component_byte_identical"]
        assert build["no_object_layout_changes"] and build["baseline_provider_sha256"] == parent_sha
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"qmoe-fc2-final-mix-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["fc2_final_mix"]
        for family, status in (
                ("qmoe", "changing_fc2_final_mix_inputs_routes_qualified"),
                ("qsa", "exact_reference_and_parallel_map_dispatch_qualified")):
            proof = json.loads((ROOT / f"qmoe-fusions-map-v2-native-{family}-gpu7-20261029.qualification.json").read_text())
            assert proof["status"] == status and proof["both_flags_tested"]
            assert proof["provider_sha256"] == build["provider_sha256"]
            assert proof["baseline_provider_sha256"] == parent_sha
            if family == "qmoe":
                assert proof["fixtures"] == 22 and proof["finite_bitwise_comparisons"] == 9666560
                assert proof["combined_parallel_map"]
            else:
                assert proof["fixtures"] == 69 and proof["reference_executions"] == 2484
                assert proof["combined_qmoe_fusions"]
        original_map = json.loads((ROOT / "qsa-parallel-map-v1-native-original-harness-gpu0-20261029.qualification.json").read_text())
        assert original_map["provider_sha256"] == build["map_component_provider_sha256"]
    if args.subgroup_gate:
        proof = json.loads((ROOT / "qmoe-fusions-map-v2-native-gate-isolated-gpu7-20261029.qualification.json").read_text())
        assert proof["status"] == "changing_fc2_final_mix_inputs_routes_qualified"
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert proof["gate_compared_against_original"] and proof["fc2_fusion_fixed_enabled"]
        assert proof["fc1_fusion_fixed_enabled"] and proof["combined_parallel_map"]
        assert proof["fixtures"] == 22 and proof["finite_bitwise_comparisons"] == 9666560
        assert proof["fixture_sha256"] == fc2_native["fixture_sha256"]
    if args.padded_activations:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-activation-padding/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "qmoe-activation-padding-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "changing_fc2_final_mix_inputs_routes_qualified"
        assert proof["activation_padding_toggled"] and proof["fc2_fusion_fixed_enabled"]
        assert proof["fc1_fusion_fixed_enabled"] and proof["combined_parallel_map"]
        assert proof["fixtures"] == 22 and proof["finite_bitwise_comparisons"] == 9666560
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"qmoe-fusions-map-subgroup-gate-v1-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["subgroup_gate"]
    if args.tiled_projection:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-hc-projection-tiled/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "hc-projection-tiled-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_runtime_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "changing_gpu_inputs_cpu_values_finite_bitwise_and_replay_qualified"
        assert proof["tiled_projection_on_off_tested"] and proof["invalid_flag_explicitly_rejected"]
        assert proof["fixtures"] == 9 and proof["finite_bitwise_comparisons"] == 2363904
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"qmoe-fusions-map-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["combined_parallel_map"]
    if args.post_norm_projection:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-hc-post-norm-projection/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_runtime_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "four_outputs_changing_inputs_cpu_values_finite_bitwise_and_replay_qualified"
        assert proof["all_four_outputs_compared"] and proof["invalid_flag_explicitly_rejected"]
        assert proof["compound_fallback_not_fair_window_baseline"]
        assert proof["fixtures"] == 36 and proof["finite_bitwise_comparisons"] == 607143936
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"qmoe-fusions-map-subgroup-gate-v1-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["subgroup_gate"]
        model = REPOS / "qwen38-int4-hc-post-norm-projection/model"
        if args.phase == "standard":
            profile = json.loads((ROOT / "hc-post-norm-projection-v2-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["post_norm_projection"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.post_norm_four_columns:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-hc-post-norm-projection/build-v3"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "hc-post-norm-four-columns-v3-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_runtime_object_layout_changes"]
        assert build["independent_eight_and_four_column_replay_caches"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "four_outputs_changing_inputs_cpu_values_finite_bitwise_and_replay_qualified"
        assert proof["four_columns"] and proof["invalid_column_selector_rejected"]
        assert proof["all_four_outputs_compared"]
        assert proof["fixtures"] == 36 and proof["finite_bitwise_comparisons"] == 607143936
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"hc-post-norm-projection-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["post_norm_projection"]
    if args.hierarchical_gate:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-hierarchical-gate/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "qmoe-hierarchical-gate-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "changing_fc2_final_mix_inputs_routes_qualified"
        assert proof["hierarchical_gate_toggled"] and proof["direct_router_selection_and_weights_tested"]
        assert proof["fc1_fusion_fixed_enabled"] and proof["fc2_fusion_fixed_enabled"]
        assert proof["combined_parallel_map"] and proof["variant_cache_keys_tested"]
        assert proof["fixtures"] == 23 and proof["finite_bitwise_comparisons"] == 9748480
        assert fc2_native["fixture_sha256"].items() <= proof["fixture_sha256"].items()
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"hc-post-norm-projection-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["post_norm_projection"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "qmoe-hierarchical-gate-v2-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["hierarchical_gate"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.mlp_input_projections:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-mlp-input-projections/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "mlp-input-projections-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_runtime_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "three_outputs_changing_inputs_finite_bitwise_and_replay_qualified"
        assert proof["all_three_outputs_compared"] and proof["invalid_flag_explicitly_rejected"]
        assert proof["compound_fallback_not_fair_window_baseline"]
        assert proof["fixtures"] == 8 and proof["finite_bitwise_comparisons"] == 5608192
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"qmoe-hierarchical-gate-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["hierarchical_gate"]
        model = REPOS / "qwen38-int4-mlp-input-projections/model"
        if args.phase == "standard":
            profile = json.loads((ROOT / "mlp-input-projections-v2-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["mlp_input_projections"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.mlp_bitcast_unpack:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-mlp-bitcast-unpack/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "mlp-bitcast-unpack-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_object_layout_changes"]
        assert build["independent_unpack_replay_caches"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "three_outputs_changing_inputs_finite_bitwise_and_replay_qualified"
        assert proof["bitcast_unpack_toggled"] and proof["both_paths_keep_projection_fusion_enabled"]
        assert proof["all_three_outputs_compared"] and proof["invalid_flag_explicitly_rejected"]
        assert proof["fixtures"] == 8 and proof["finite_bitwise_comparisons"] == 5608192
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"mlp-input-projections-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["mlp_input_projections"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "mlp-bitcast-unpack-v1-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["mlp_bitcast_unpack"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.expert_bitcast_unpack:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-expert-bitcast-unpack/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "expert_bitcast_changing_inputs_routes_finite_bitwise_and_replay_qualified"
        assert proof["bitcast_conversion_toggled"] and proof["both_expert_fusions_fixed_enabled"]
        assert proof["direct_router_selection_and_weights_tested"] and proof["variant_cache_keys_tested"]
        assert proof["fixtures"] == 23 and proof["finite_bitwise_comparisons"] == 9748480
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"mlp-input-projections-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["mlp_input_projections"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "expert-bitcast-unpack-v2-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["expert_bitcast_unpack"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.dense_bitcast_unpack:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-dense-bitcast-unpack/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "dense-bitcast-unpack-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_object_layout_changes"]
        assert build["standalone_matmul_only"] and build["replay_variant_keyed"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "dense_bitcast_changing_inputs_finite_bitwise_and_replay_qualified"
        assert proof["both_flags_tested"] and proof["invalid_flag_explicitly_rejected"]
        assert proof["replay_variant_keys_tested"] and proof["multi_token_fallback_tested"]
        assert proof["fixtures"] == 27 and proof["finite_bitwise_comparisons"] == 116582784
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"expert-bitcast-unpack-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["expert_bitcast_unpack"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "dense-bitcast-unpack-v1-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["dense_bitcast_unpack"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.fc2_four_columns:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-fc2-four-columns/build-v2"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "fc2-four-columns-v2-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "fc2_four_columns_changing_inputs_routes_finite_bitwise_and_replay_qualified"
        assert proof["both_expert_fusions_fixed_enabled"] and proof["expert_bitcast_fixed_enabled"]
        assert proof["variant_cache_keys_tested"] and proof["columns_tested"] == [2, 4]
        assert proof["fixtures"] == 23 and proof["finite_bitwise_comparisons"] == 9748480
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"expert-bitcast-unpack-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["expert_bitcast_unpack"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "fc2-four-columns-v2-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["fc2_four_columns"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.hc_up_tiled:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-hc-up-tiled/build-v1"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "hc-up-tiled-v1-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_object_layout_changes"]
        assert build["replay_variant_keyed"] and build["workgroups"] == [2560, 320]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "hc_up_tiled_changing_inputs_finite_bitwise_and_replay_qualified"
        assert proof["parent_fusion_fixed_enabled"] and proof["columns_tested"] == [1, 8]
        assert proof["variant_cache_keys_tested"] and proof["multi_token_fallback_tested"]
        assert proof["fixtures"] == 9 and proof["finite_bitwise_comparisons"] == 18677760
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"expert-bitcast-unpack-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["expert_bitcast_unpack"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "hc-up-tiled-v1-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["hc_up_tiled"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    if args.packed_gate:
        parent_sha = build["provider_sha256"]
        build_dir = REPOS / "qwen38-int4-qmoe-packed-gate/build-v3"
        build = json.loads((build_dir / "build-manifest.json").read_text())
        proof = json.loads((ROOT / "qmoe-packed-gate-v3-native-gpu7-20261029.qualification.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_object_layout_changes"]
        assert proof["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent_sha
        assert proof["provider_sha256"] == build["provider_sha256"]
        assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
        assert proof["status"] == "packed_gate_changing_inputs_routes_finite_bitwise_and_replay_qualified"
        assert proof["both_expert_fusions_fixed_enabled"] and proof["expert_bitcast_fixed_enabled"]
        assert proof["hierarchical_gate_fixed_enabled"] and proof["variant_cache_keys_tested"]
        assert proof["direct_router_selection_and_weights_tested"]
        assert proof["fixtures"] == 23 and proof["finite_bitwise_comparisons"] == 9748480
        for context in (2048, 8192, 32768):
            qualified = json.loads((ROOT / f"expert-bitcast-unpack-v2-gpu7-standard{context}-20261029.qualification.json").read_text())
            assert qualified["status"] == "original_sequences_qualified"
            assert qualified["provider_sha256"] == parent_sha and qualified["expert_bitcast_unpack"]
        if args.phase == "standard":
            profile = json.loads((ROOT / "qmoe-packed-gate-v3-gpu7-profile-20261029.qualification.json").read_text())
            assert profile["status"] == "original_sequences_qualified" and profile["packed_gate"]
            assert profile["provider_sha256"] == build["provider_sha256"]
            assert profile["graph_sha256"] == sha(model / "model.onnx")
    control = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    base = json.loads((ROOT / "quickgelu-mul-candidate-32k-20261027.guard.json").read_text())["command"].copy()
    command = base[-1].replace(
        GPU_UUIDS[0], gpu_uuid
    ).replace("quickgelu-mul-candidate-32k-20261027", prefix.name).replace(
        "--contexts 32768", f"--contexts {args.context}").replace(
        "/workspace/qwen38-int4-quickgelu-next/build/libonnxruntime_providers_webgpu.so",
        str(build_dir / "libonnxruntime_providers_webgpu.so").replace(str(REPOS), "/workspace"))
    command = command.replace("--model /workspace/qwen38-int4-quickgelu-next/model",
                              "--model " + str(model).replace(str(REPOS), "/workspace"))
    assert "--webgpu-max-pending-dispatches 64" in command
    command = command.replace("--webgpu-max-pending-dispatches 64",
                              f"--webgpu-max-pending-dispatches {args.pending}")
    if args.kind == "large-cache":
        command = command.replace("exec python3", "export ORT_WEBGPU_BIND_GROUP_CACHE_LARGE=1 "
                                  "ORT_WEBGPU_BIND_GROUP_CACHE_STATS_FILE=/evidence/webgpu-perf-oct20/" +
                                  prefix.name + "-stats.jsonl; exec python3")
    if args.kind in ("gdn-view", "hc-gdn", "node-replay", "hc-node-replay", "static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        command = command.replace("exec python3", "export ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1; exec python3")
    if args.kind in ("node-replay", "hc-node-replay", "static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        command = command.replace(
            "exec python3", "export ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
            "ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/" +
            prefix.name + "-stats.jsonl; exec python3")
    if args.kind in ("static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        command = command.replace(
            "exec python3", "export ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
            "ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/" +
            prefix.name + "-static-stats.jsonl; exec python3")
    if args.kind in ("hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_PROJECTION_SILU=1; exec python3")
    if is_up:
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_UP_PREMIX=1; exec python3")
    if args.kind == "qsa-map" or args.with_parallel_map:
        command = command.replace("exec python3", "export ORT_WEBGPU_QSA_PARALLEL_MAP=1; exec python3")
    if args.kind in ("generic-replay", "direct-a", "fc1-activation"):
        command = command.replace("exec python3", "export ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1; exec python3")
    if args.kind == "direct-a":
        command = command.replace("exec python3", "export ORT_WEBGPU_MATMUL_NBITS_DIRECT_A=1; exec python3")
    if args.kind == "fc1-activation":
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_FC1_ACTIVATION=1; exec python3")
    if args.replay_immediate:
        command = command.replace("exec python3", "export ORT_WEBGPU_NODE_REPLAY_IMMEDIATE=1; exec python3")
    if args.fc2_final_mix:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1; exec python3")
    if args.padded_partials:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS=1; exec python3")
    if args.tiled_projection:
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_PROJECTION_TILED=1; exec python3")
    if args.subgroup_gate:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_SUBGROUP_GATE=1; exec python3")
    if args.padded_activations:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS=1; exec python3")
    if args.post_norm_projection:
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_POST_NORM_PROJECTION=1; exec python3")
    if args.post_norm_four_columns:
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS=1; exec python3")
    if args.hierarchical_gate:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1; exec python3")
    if args.mlp_input_projections:
        command = command.replace("exec python3", "export ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1; exec python3")
    if args.mlp_bitcast_unpack:
        command = command.replace("exec python3", "export ORT_WEBGPU_MLP_BITCAST_UNPACK=1; exec python3")
    if args.expert_bitcast_unpack:
        command = command.replace("exec python3", "export ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1; exec python3")
    if args.dense_bitcast_unpack:
        command = command.replace("exec python3", "export ORT_WEBGPU_DENSE_BITCAST_UNPACK=1; exec python3")
    if args.fc2_four_columns:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS=1; exec python3")
    if args.hc_up_tiled:
        command = command.replace("exec python3", "export ORT_WEBGPU_HC_UP_TILED=1; exec python3")
    if args.packed_gate:
        command = command.replace("exec python3", "export ORT_WEBGPU_QMOE_PACKED_GATE=1; exec python3")
    if args.phase == "windows":
        command = command.replace("--tokens 256 --repetitions 3",
                                  "--tokens 288 --repetitions 4 --compare-" +
                                  ("packed-gate" if args.packed_gate else
                                   "hc-up-tiled" if args.hc_up_tiled else
                                   "fc2-four-columns" if args.fc2_four_columns else
                                   "dense-bitcast-unpack" if args.dense_bitcast_unpack else
                                   "expert-bitcast-unpack" if args.expert_bitcast_unpack else
                                   "mlp-bitcast-unpack" if args.mlp_bitcast_unpack else
                                   "hierarchical-gate" if args.hierarchical_gate else
                                   "hc-post-norm-four-columns" if args.post_norm_four_columns else
                                   "qmoe-activation-padding" if args.padded_activations else
                                   "subgroup-gate" if args.subgroup_gate else
                                   "hc-projection-tiled" if args.tiled_projection else
                                   "fc2-padded-partials" if args.padded_partials else
                                   "qsa-parallel-map" if args.with_parallel_map else
                                   "fc2-final-mix" if args.fc2_final_mix else
                                   "replay-immediate" if args.replay_immediate else
                                   "fc1-activation" if args.kind == "fc1-activation" else
                                   "direct-a" if args.kind == "direct-a" else
                                   "generic-matmul" if args.kind == "generic-replay" else
                                   "qsa-parallel-map" if args.kind == "qsa-map" else
                                   "hc-up-premix" if is_up else
                                   "hc-projection" if args.kind == "hc-projection" else
                                   "static-replay" if args.kind == "static-replay" else
                                   "gdn-view" if args.kind == "gdn-view" else
                                   "node-replay" if args.kind in ("node-replay", "hc-node-replay")
                                   else "large-cache") + "-windows")
    elif args.phase == "profile":
        command = command.replace("--tokens 256 --repetitions 3",
                                  "--tokens 32 --repetitions 1 --webgpu-validation-mode full "
                                  "--profile-decode-after 31 --profile-prefix /evidence/webgpu-perf-oct20/" +
                                  prefix.name + "-profile")
    run_command = [*base[:-1], command]
    cache_preflight = None if args.resume_completed else prepare_model_cache(model, prefix)
    initial = None if args.resume_completed else snapshot()
    if not args.resume_completed:
        subprocess.run(
            ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(args.gpu_index),
             "--idle-seconds", "3", "--wait-seconds", "15",
             "--run-seconds", "3000" if args.context == 32768 else "1400",
             "--output-prefix", str(prefix), "--", *run_command], check=True)
        cache_preflight["after_run"] = snapshot_model_cache(model)
        Path(cache_preflight["evidence_path"]).write_text(json.dumps(cache_preflight, indent=2) + "\n")
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    data = json.loads(prefix.with_suffix(".json").read_text())
    validate_completed_guard(guard, run_command, args.gpu_index, gpu_uuid)
    assert data["status"] == "completed"
    assert data["plugin_sha256"] == build["provider_sha256"]
    assert data["graph_sha256"] == sha(model / "model.onnx")
    assert data["config_sha256"] == control["config_sha256"]
    assert data["webgpu_max_pending_dispatches_override"] == args.pending
    assert data["chunk"] == 16 and data["provider"] == "webgpu"
    log = prefix.with_suffix(".log").read_text()
    assert "GetCapability" not in log and "FAIL" not in log
    references = [s for s in control["samples"] if s["context"] == args.context]
    samples = data["samples"]
    assert len(samples) == (2 if args.phase == "profile" else 4)
    for index, sample in enumerate(samples):
        reference = references[1] if args.phase == "windows" else references[index]
        count = 32 if args.phase == "profile" else 256
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_token_ids"][:count] == reference["output_token_ids"][:count]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
        assert math.isclose(sample["decode_tps"], (sample["output_tokens"] - 1) /
                            sample["decode_seconds"], rel_tol=1e-12)
    report = dict(status="original_sequences_qualified", kind=args.kind, phase=args.phase,
                  replay_immediate=args.replay_immediate,
                  fc2_final_mix=args.fc2_final_mix,
                  fc2_padded_partials=args.padded_partials,
                  tiled_projection=args.tiled_projection,
                  subgroup_gate=args.subgroup_gate,
                  padded_activations=args.padded_activations,
                  post_norm_projection=args.post_norm_projection,
                  post_norm_four_columns=args.post_norm_four_columns,
                  hierarchical_gate=args.hierarchical_gate,
                  mlp_input_projections=args.mlp_input_projections,
                  mlp_bitcast_unpack=args.mlp_bitcast_unpack,
                  expert_bitcast_unpack=args.expert_bitcast_unpack,
                  dense_bitcast_unpack=args.dense_bitcast_unpack,
                  fc2_four_columns=args.fc2_four_columns,
                  hc_up_tiled=args.hc_up_tiled,
                  packed_gate=args.packed_gate,
                  cache_preflight=cache_preflight,
                  combined_parallel_map=args.with_parallel_map,
                  context=args.context, provider_sha256=build["provider_sha256"],
                  pending_dispatches=args.pending,
                  graph_sha256=data["graph_sha256"], initial_resources=initial,
                  final_resources=None if args.resume_completed else snapshot(),
                  quality_qualified=False, headline_eligible=False)
    if args.resume_completed:
        report.update(resumed_completed_guard=True, initial_and_final_resource_snapshots_unavailable=True,
                      run_gpu_snapshot=guard["gpu_snapshot"], revalidation_resources=snapshot(),
                      completed_evidence_sha256={s: sha(prefix.with_suffix(s))
                                                 for s in (".json", ".guard.json", ".log")})
    if args.phase == "windows":
        assert all(s["same_prompt_output_matches"] for s in samples)
        report["windows"] = data["window_summary"][str(args.context)]
    else:
        report["summary"] = data["summary"][f"{args.context}:configured"]
    if args.phase == "profile":
        profiles = list(ROOT.glob(prefix.name + "-profile-*configured*.json"))
        assert len(profiles) == 2, profiles
        counts = []
        for path in profiles:
            events = json.loads(path.read_text())
            count = collections.Counter(e["name"].split("&")[-1] for e in events
                                        if e.get("cat") == "Kernel")
            if args.subgroup_gate:
                gate_name = ("QmoePackedGate1Token" if args.packed_gate else
                             "QmoeHierarchicalGate1Token" if args.hierarchical_gate else "QmoeGate1Token")
                gates = [e for e in events if e.get("cat") == "Kernel" and
                         e["name"].endswith("&" + gate_name)]
                assert len(gates) == 48
                assert all(e["args"]["cache_key"].startswith(gate_name + "[") and
                           e["args"]["cache_key"].split("]", 1)[0].endswith("|1") for e in gates)
            if args.kind == "post-norm":
                assert count["HCPostNormF16"] == 95 and sum(count.values()) == 1486, count
            elif args.kind in ("gdn-view", "node-replay"):
                assert count["GatedDeltaNetUnpackQkv"] == 0 and sum(count.values()) == 1545, count
            elif args.kind in ("hc-gdn", "hc-node-replay", "static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
                assert count["HCPostNormF16"] == (1 if args.post_norm_projection else 95) and count["GatedDeltaNet"] == 36, count
                expected_total = 1257 if is_up else 1354 if args.kind == "hc-projection" else 1450
                if args.post_norm_projection:
                    expected_total -= 94
                    name = "HCPostNormProjectionFourColumnsF16" if args.post_norm_four_columns else "HCPostNormProjectionF16"
                    assert count[name] == 94, count
                if args.kind == "qsa-map" or args.with_parallel_map:
                    eligible = args.context >= 8192
                    expected_total += 12 if eligible else 0
                    assert count["QsaParallelMapCount"] == count["QsaParallelMapScatter"] == (12 if eligible else 0)
                    assert count["SparseAttentionIndexerQsaMap"] == (0 if eligible else 12)
                if args.kind == "direct-a":
                    assert count["MatMulNBitsDirectA"] == 303 and count["MatMulNBits"] == 1, count
                if args.kind == "fc1-activation":
                    expected_total -= 48
                    fc1_name = ("QmoeFC1SwiGLUBitcastF16" if args.expert_bitcast_unpack else
                                "QmoeFC1SwiGLUActivationPaddedF16" if args.padded_activations else "QmoeFC1SwiGLUF16")
                    assert count[fc1_name] == 48 and count["MoEActivation"] == 0, count
                    assert sum(count[name] for name in ("QmoeFC1SwiGLUF16", "QmoeFC1SwiGLUActivationPaddedF16",
                                                       "QmoeFC1SwiGLUBitcastF16")) == 48, count
                    if args.fc2_final_mix:
                        expected_total -= 48
                        fc2_name = ("QmoeFC2FinalMixFourColumnsBitcastF16" if args.fc2_four_columns else
                                    "QmoeFC2FinalMixBitcastF16" if args.expert_bitcast_unpack else
                                    "QmoeFC2FinalMixActivationPaddedF16" if args.padded_activations else
                                    "QmoeFC2FinalMixF16Padded" if args.padded_partials else "QmoeFC2FinalMixF16")
                        assert count["MatMulNBits"] == (1 if args.dense_bitcast_unpack else
                               160 if args.mlp_input_projections else 208) and count[fc2_name] == 48, count
                        if args.dense_bitcast_unpack:
                            assert count["MatMulNBitsBitcastF16"] == 159, count
                        assert sum(count[name] for name in ("QmoeFC2FinalMixF16", "QmoeFC2FinalMixF16Padded",
                                   "QmoeFC2FinalMixActivationPaddedF16", "QmoeFC2FinalMixBitcastF16",
                                   "QmoeFC2FinalMixFourColumnsBitcastF16")) == 48, count
                        assert count["QmoeFusedFinalMix1Token"] == 0, count
                    else:
                        assert count["MatMulNBits"] == 256, count
                if args.mlp_input_projections:
                    expected_total -= 192
                    name = "MlpInputProjectionsBitcastF16" if args.mlp_bitcast_unpack else "MlpInputProjectionsF16"
                    assert count[name] == 48 and count["QuickGeluMulF16"] == 0, count
                    assert count["ExactDenseMid"] == 12, count
                assert count["GatedDeltaNetUnpackQkv"] == 0 and sum(count.values()) == expected_total, count
                if args.kind in ("hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
                    projection_name = "HCProjectionSiLUTiledF16" if args.tiled_projection else "HCProjectionSiLUF16"
                    projections = 2 if args.post_norm_projection else 96
                    assert count[projection_name] == projections and count["SplitScaledSiLUF16"] == 0, count
                    assert count["HCProjectionSiLUF16"] + count["HCProjectionSiLUTiledF16"] == projections, count
                if is_up:
                    up_name = "HCUpPreMixTiledF16" if args.hc_up_tiled else "HCUpPreMixF16"
                    assert count[up_name] == 97 and count["HyperConnectionPreMix"] == 0, count
            counts.append(dict(count))
        report["actual_kernel_counts"] = counts
    if args.kind == "large-cache":
        stats = [json.loads(line) for line in (ROOT / (prefix.name + "-stats.jsonl")).read_text().splitlines()]
        assert stats and max(s["peak_referenced_bytes"] for s in stats) <= 96 * 1024**3
        assert max(s["peak_referenced_bytes"] for s in stats) > 256 * 1024**2
        report["cache_stats"] = stats
    if args.kind in ("node-replay", "hc-node-replay", "static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_COMMAND_REPLAY"] == "1"
        stats = [json.loads(line) for line in (ROOT / (prefix.name + "-stats.jsonl")).read_text().splitlines()]
        assert len(stats) == 48, len(stats)
        assert all(0 < row["entries"] <= row["capacity"] == 8 for row in stats)
        assert sum(row["hits"] for row in stats) > 0, stats
        requests = sum(row["requests"] for row in stats)
        report["node_replay_stats"] = dict(
            nodes=len(stats), requests=requests, hits=sum(row["hits"] for row in stats),
            captures=sum(row["captures"] for row in stats),
            entries=sum(row["entries"] for row in stats),
            evictions=sum(row["evictions"] for row in stats),
            shape_invalidations=sum(row["shape_invalidations"] for row in stats),
            capacity_fallbacks=sum(row["capacity_fallbacks"] for row in stats),
            hit_fraction=sum(row["hits"] for row in stats) / requests,
            profiling_uses_uncached_path=True)
    if args.kind in ("static-replay", "hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
        assert data["load_experiment_flags"]["ORT_WEBGPU_STATIC_NODE_REPLAY"] == "1"
        stats = [json.loads(line) for line in (ROOT / (prefix.name + "-static-stats.jsonl")).read_text().splitlines()]
        counts = collections.Counter(row["operator"] for row in stats)
        expected = {"QuickGeluMulF16": 48, "SplitScaledSiLUF16": 96, "MatMulNBitsExact": 252}
        if args.static_revision == "v8" or args.kind in ("hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
            expected.update(HyperConnectionPreMix=97, HCPostNormF16=48, HCSharedPostNormF16=47)
        if args.kind in ("hc-projection", "hc-up-premix", "qsa-map", "generic-replay", "direct-a", "fc1-activation"):
            del expected["SplitScaledSiLUF16"]
            expected.update(MatMulNBitsExact=156, HCProjectionSiLUF16=96)
            assert data["load_experiment_flags"]["ORT_WEBGPU_HC_PROJECTION_SILU"] == "1"
        if args.tiled_projection:
            del expected["HCProjectionSiLUF16"]
            expected.update(projection_replay_counts(True, args.phase))
            assert data["load_experiment_flags"]["ORT_WEBGPU_HC_PROJECTION_TILED"] == "1"
        if is_up:
            del expected["HyperConnectionPreMix"]
            expected["HCUpPreMixF16"] = 97
            assert data["load_experiment_flags"]["ORT_WEBGPU_HC_UP_PREMIX"] == "1"
        if args.kind == "qsa-map" or args.with_parallel_map:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QSA_PARALLEL_MAP"] == "1"
        if args.kind in ("generic-replay", "direct-a", "fc1-activation"):
            expected["MatMulNBitsExact"] = 363
            assert data["load_experiment_flags"]["ORT_WEBGPU_GENERIC_MATMUL_REPLAY"] == "1"
        if args.kind == "direct-a":
            assert data["load_experiment_flags"]["ORT_WEBGPU_MATMUL_NBITS_DIRECT_A"] == "1"
        if args.kind == "fc1-activation":
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC1_ACTIVATION"] == "1"
        if args.replay_immediate:
            assert data["load_experiment_flags"]["ORT_WEBGPU_NODE_REPLAY_IMMEDIATE"] == "1"
        if args.fc2_final_mix:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC2_FINAL_MIX"] == "1"
        if args.padded_partials:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS"] == "1"
        if args.subgroup_gate:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_SUBGROUP_GATE"] == "1"
        if args.hierarchical_gate:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"] == "1"
        if args.mlp_input_projections:
            del expected["QuickGeluMulF16"]
            expected.update(MatMulNBitsExact=171, MlpInputProjectionsF16=48)
            assert data["load_experiment_flags"]["ORT_WEBGPU_MLP_INPUT_PROJECTIONS"] == "1"
        if args.mlp_bitcast_unpack:
            if args.phase != "windows":
                del expected["MlpInputProjectionsF16"]
            expected["MlpInputProjectionsBitcastF16"] = 48
            assert data["load_experiment_flags"]["ORT_WEBGPU_MLP_BITCAST_UNPACK"] == "1"
        if args.expert_bitcast_unpack:
            assert data["load_experiment_flags"]["ORT_WEBGPU_EXPERT_BITCAST_UNPACK"] == "1"
        if args.dense_bitcast_unpack:
            assert data["load_experiment_flags"]["ORT_WEBGPU_DENSE_BITCAST_UNPACK"] == "1"
        if args.fc2_four_columns:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS"] == "1"
        if args.hc_up_tiled:
            assert data["load_experiment_flags"]["ORT_WEBGPU_HC_UP_TILED"] == "1"
        if args.packed_gate:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_PACKED_GATE"] == "1"
        if args.padded_activations:
            assert data["load_experiment_flags"]["ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS"] == "1"
        if args.post_norm_projection:
            del expected["HCPostNormF16"]
            expected.update(HCSharedPostNormF16=1, HCProjectionSiLUF16=2)
            if not args.post_norm_four_columns or args.phase == "windows":
                expected.update(HCPostNormProjectionF16=48, HCSharedPostNormProjectionF16=46)
            if args.post_norm_four_columns:
                expected.update(HCPostNormProjectionFourColumnsF16=48, HCSharedPostNormProjectionFourColumnsF16=46)
                assert data["load_experiment_flags"]["ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS"] == "1"
            assert data["load_experiment_flags"]["ORT_WEBGPU_HC_POST_NORM_PROJECTION"] == "1"
        assert counts in (expected, dict(expected, GatedRMSNorm=36)), counts
        assert all(0 < row["entries"] <= row["capacity"] == 64 for row in stats)
        assert all(row["rebinds"] + row["binding_hits"] == row["hits"] for row in stats)
        assert all(0 < row["binding_entries"] <= row["binding_capacity"] == 64 for row in stats)
        requests = sum(row["requests"] for row in stats)
        hits = sum(row["hits"] for row in stats)
        captures = sum(row["captures"] for row in stats)
        assert hits > 0 and hits + captures == requests
        assert sum(row["rebinds"] for row in stats) > 0
        assert sum(row["binding_hits"] for row in stats) > 0
        report["static_replay_stats"] = dict(
            operators=dict(counts), requests=requests, hits=hits, captures=captures,
            rebinds=sum(row["rebinds"] for row in stats),
            binding_hits=sum(row["binding_hits"] for row in stats),
            binding_evictions=sum(row["binding_evictions"] for row in stats),
            evictions=sum(row["evictions"] for row in stats), hit_fraction=hits / requests,
            profiling_uses_uncached_path=True)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

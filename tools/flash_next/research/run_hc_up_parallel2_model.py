import argparse
import collections
import json
import math
from pathlib import Path
import subprocess

from build_hc_up_parallel2 import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_qsa_score16_standard import verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


VARIANT = "hc-up-parallel2-v1"
ROW4_EXPERT_VARIANTS = ("fc2-subgroup-fold-v1", "fc2-compact20-v1", "fc1-experts10-v1",
                       "fc1-full-a-v1", "fc1-columns8-v1", "fc1-paired-dot-v1", "fc1-k16-tile-v1",
                       "fc1-k8-tile-v1", "fc1-u32-partials-v1", "fc1-expert-pairs-v1")
ROW4_DECODE_VARIANTS = (*ROW4_EXPERT_VARIANTS, "dense-qkv-wide16-v1", "dense-qkv-k16-v1",
                       "dense-qkv-predecode8-v1", "dense-qkv-threads64-v1", "dense-qkv-pad34-v1")
ROW4_REPEAT_VARIANTS = ("fc2-compact20-v1", "dense-qkv-wide16-v1", "fc1-experts10-v1",
                       "fc1-full-a-v1", "fc1-columns8-v1", "fc1-paired-dot-v1", "fc1-k16-tile-v1",
                       "fc1-k8-tile-v1", "dense-qkv-k16-v1", "fc1-u32-partials-v1", "fc1-expert-pairs-v1",
                       "dense-qkv-predecode8-v1", "dense-qkv-threads64-v1", "dense-qkv-pad34-v1")


def verify_fc2_subgroup_window_support(result, prefix="fc2_subgroup_fold"):
    assert math.isfinite(result["geometric_mean_matched_window_ratio"])
    assert result["geometric_mean_matched_window_ratio"] > 1
    assert result["matched_window_pairs"] == 16 and result["all_request_output_tokens_match"]
    requests = result["per_request"]
    assert [request["repetition"] for request in requests] == [0, 1, 2, 3]
    for request in requests:
        for mode in (0, 1):
            rate = request[f"{prefix}{mode}_median_window_tps"]
            assert math.isfinite(rate) and rate > 0
    assert sum(request[f"{prefix}1_median_window_tps"] >
               request[f"{prefix}0_median_window_tps"] for request in requests) >= 3


def verify_row4_window_prerequisite(proof, provider, context, full, prefix="fc2_compact20"):
    assert proof["status"] == "original_sequences_qualified" and proof["phase"] == "windows"
    assert proof["context"] == context and proof["full_validation"] == full
    assert proof["provider_sha256"] == provider
    assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
    assert proof["fixed_grouped_prefill_selector"] == 1
    verify_fc2_subgroup_window_support(proof["result"], prefix=prefix)


def verify_gdn_histogram_profile(counts, context):
    assert context in (2048, 8192, 32768)
    expected = 12 if context == 32768 else 0
    histogram_counts = {name: counts[name] for name in
                        ("QsaHistogramUniformRank", "QsaRankHistogram")}
    assert all(count == expected for count in histogram_counts.values())
    assert counts["SparseAttentionIndexerQsaParallelRank"] == (0 if expected else 12)
    return histogram_counts


def main(variant=VARIANT, experiment_flag=FLAG, stamp="20261029", parent_variant="qsa-score16-v5",
         program_name="HCUpPreMixPacked2F16", original_program_name="HCUpPreMixF16", dispatches=97,
         parent_stamp="20261029", chunk=256, parent_standard_prefix=None, parent_provider=None,
         additional_programs=(), initial_context=2048, histogram_qualification=False, original_dispatches=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=initial_context)
    parser.add_argument("--phase", choices=("profile", "windows", "standard"), default="windows")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--attempt", type=int, default=1,
                        help="Use a fresh evidence suffix for a retry; prerequisite gates remain unchanged.")
    args = parser.parse_args()
    if histogram_qualification and (variant != "qsa-hist-rank-v2" or initial_context != 32768):
        raise ValueError("Dedicated histogram qualification requires the native-qualified v2 long-context variant")
    if args.context < initial_context and not histogram_qualification:
        parser.error(f"This experiment first admits context {initial_context}; shorter routes are unchanged")
    if initial_context != 2048 and args.phase == "standard" and not histogram_qualification:
        parser.error("Long-only standard qualification is not yet available; use controlled windows")
    if args.attempt < 1:
        parser.error("--attempt must be a positive integer")
    attempt_suffix = f"-attempt{args.attempt}" if args.attempt > 1 else ""
    if args.phase == "profile" and not args.full_validation:
        parser.error("Active-path profiling requires full validation")
    family, version = variant.rsplit("-v", 1)
    window_prefix = family.replace("-", "_")
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v{version}")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    if variant == "gdn-tree32-hist-v1":
        family, window_prefix = "gdn-tree32", "gdn_tree32"
    native_suffix = "-attempt2" if variant == "gdn-norm32-v1" else ""
    native = json.loads((ROOT / f"{variant}-native-gpu7-{stamp}{native_suffix}.qualification.json").read_text())
    parent = json.loads((ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json").read_text())
    required_flags = (experiment_flag, "ORT_WEBGPU_QSA_SCORE16", "ORT_WEBGPU_HC_UP_PREMIX")
    if variant in (VARIANT, "hc-up-parallel4-v1"):
        assert native["status"] == f"{window_prefix}_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 18677760
        assert native["columns_tested"] == ([1, 2] if variant == VARIANT else [2, 4])
        if variant != VARIANT:
            required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_TILE4",
                               "ORT_WEBGPU_QMOE_PREFILL512")
    elif variant == "hc-post-norm-pad34-v1":
        assert native["status"] == "hc_post_norm_pad34_four_outputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 607143936 and native["fixtures"] == 36
        assert native["all_four_outputs_compared"] and native["invalid_flag_explicitly_rejected"]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_POST_NORM_PROJECTION",
                           "ORT_WEBGPU_HC_PROJECTION_SILU")
    elif variant in ("hc-post-norm-vec4-v1", "hc-post-norm-predecode-v1", "hc-post-norm-predecode-v2",
                     "hc-post-norm-wide16-v1", "hc-quad-fold-v1", "hc-narrow4-v1"):
        assert native["status"] == f"{window_prefix}_four_outputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 607143936 and native["fixtures"] == 36
        assert native["all_four_outputs_compared"] and native["invalid_flag_explicitly_rejected"]
        assert native["changing_inputs_and_cpu_splits"] and native["lru_eviction_tested"]
        assert native["pending_windows_tested"] == [1, 64]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_POST_NORM_PROJECTION",
                           "ORT_WEBGPU_HC_PROJECTION_SILU", "ORT_WEBGPU_DENSE_TILE4",
                           "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024",
                           "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")
        if variant in ("hc-post-norm-wide16-v1", "hc-quad-fold-v1", "hc-narrow4-v1"):
            build = json.loads((directory / "build-manifest.json").read_text())
            assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
        if variant in ("hc-quad-fold-v1", "hc-narrow4-v1"):
            assert build["replay_variant_keyed"]
            required_flags += ("ORT_WEBGPU_QSA_PREFILL_SCORE16",)
    elif variant == "dense-tile4-v1":
        assert native["status"] == "dense_tile4_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 116582784 and native["fixtures"] == 27
        assert native["multi_token_fallback_tested"] and native["invalid_flag_explicitly_rejected"]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_BITCAST_UNPACK")
    elif variant in ("hc-projection-shuffle-v1", "hc-staged-projection-v1"):
        assert native["status"] == f"{window_prefix}_four_outputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 607143936 and native["fixtures"] == 36
        assert native["all_four_outputs_compared"] and native["invalid_flag_explicitly_rejected"]
        assert native["changing_inputs_and_cpu_splits"] and native["lru_eviction_tested"]
        assert native["pending_windows_tested"] == [1, 64]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_POST_NORM_PROJECTION",
                           "ORT_WEBGPU_HC_PROJECTION_SILU", "ORT_WEBGPU_DENSE_TILE4",
                           "ORT_WEBGPU_QMOE_PREFILL512")
    elif variant in ("gdn-value2-v1", "gdn-value8-v1", "gdn-tree32-v1", "gdn-norm32-v1",
                     "gdn-tree32-hist-v1"):
        assert native["status"] == f"{window_prefix}_both_outputs_finite_bitwise_and_dispatch_qualified"
        assert native["finite_bitwise_comparisons"] == 236531952
        assert native["fixtures"] == 36
        assert native["optimized_fixtures"] == (1 if variant == "gdn-norm32-v1" else 2)
        assert native["pending_windows_tested"] == [1, 64]
        assert native["fixed_flag_capture_candidate_executions"] == 9
        assert native["both_output_and_final_state_checked"] and native["invalid_selector_rejected"]
        assert native["original_per_channel_arithmetic_preserved"]
        assert native["multi_token_batch_dtype_and_odd_shape_fallbacks_checked"]
        assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
        capture_directory = directory.parent / "capture-compare-v1"
        assert sha(capture_directory / "build-manifest.json") == native["capture_build_manifest_sha256"]
        assert sha(capture_directory / "native_compare") == native["capture_comparator_sha256"]
        required_flags += ("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW", "ORT_WEBGPU_HC_UP_PARALLEL2",
                           "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_QMOE_PREFILL512",
                           "ORT_WEBGPU_QMOE_PREFILL1024")
        if variant in ("gdn-value8-v1", "gdn-tree32-v1", "gdn-norm32-v1", "gdn-tree32-hist-v1"):
            required_flags += ("ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")
        if variant in ("gdn-tree32-v1", "gdn-norm32-v1", "gdn-tree32-hist-v1"):
            required_flags += ("ORT_WEBGPU_QSA_PREFILL_SCORE16",)
            build = json.loads((directory / "build-manifest.json").read_text())
            assert build["fixed_subgroup_size"] == 32
            assert build["original_128_lane_tree_association_preserved"]
            assert build["only_five_intra_subgroup_tree_levels_replaced"]
            assert build["no_existing_kernel_object_layout_changes"]
            if variant == "gdn-norm32-v1":
                assert build["norm_tree_only"]
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
            if variant == "gdn-tree32-hist-v1":
                assert parent_variant == "qsa-hist-rank-v2" and chunk == 1024
                assert build["gdn_component_object_byte_identical"]
                assert build["histogram_component_object_unchanged"]
                assert build["only_disjoint_gdn_object_replaced"]
                for name, digest in build["qualification_source_sha256"].items():
                    assert sha(ROOT / name) == digest
                required_flags += ("ORT_WEBGPU_QSA_HIST_RANK",)
    elif variant in ("qsa-hist-rank-v1", "qsa-hist-rank-v2"):
        assert native["status"] == "exact_reference_and_histogram_rank_dispatch_qualified"
        assert native["fixtures"] == 69 and native["reference_executions"] == 2484
        assert native["invalid_selector_rejected"] and native["cpu_fallback_disabled"]
        assert native["original_float_scoring_and_nan_tie_tail_semantics_preserved"]
        assert native["candidate_zero_complete_valid_count"] and not native["requires_subgroups"]
        assert native["fixed_subgroup_size"] is None and native["workgroup_size"] == 64
        assert native["auxiliary_programs"] == ["QsaRankHistogram"] and native["exercised"]
        if variant == "qsa-hist-rank-v2":
            assert native["minimum_score_stride"] == 8192
        assert initial_context == (8192 if variant == "qsa-hist-rank-v1" else 32768)
        assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
        required_flags += ("ORT_WEBGPU_QSA_PARALLEL_RANK", "ORT_WEBGPU_HC_UP_PARALLEL2",
                           "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_QMOE_PREFILL512",
                           "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                           "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16")
    elif variant in ("qsa-pruned-rank-v1", "qsa-pruned-rank-v2", "qsa-subgroup-rank-v1"):
        expected_status = ("exact_reference_and_subgroup_rank_dispatch_qualified" if
                           variant == "qsa-subgroup-rank-v1" else "exact_reference_and_pruned_rank_dispatch_qualified")
        assert native["status"] == expected_status
        assert native["fixtures"] == 69 and native["reference_executions"] == 2484
        assert native["invalid_selector_rejected"] and native["cpu_fallback_disabled"]
        assert native["original_float_scoring_and_nan_tie_tail_semantics_preserved"]
        assert native["fixed_subgroup_size"] == 32 and native["candidate_zero_complete_valid_count"]
        assert native["exercised"] and sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
        if variant == "qsa-subgroup-rank-v1":
            assert native["workgroup_size"] == 64 and not native["pruning_enabled"]
        required_flags += ("ORT_WEBGPU_QSA_PARALLEL_RANK", "ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE",
                           "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_TILE4",
                           "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024",
                           "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")
    elif variant in ("dsa-query-cache-v2", "dsa-paired-v1"):
        expected_status = ("dsa_query_cache_finite_bitwise_and_cpu_reference_qualified"
                           if variant == "dsa-query-cache-v2" else
                           "dsa_paired_finite_bitwise_and_cpu_reference_qualified")
        assert native["status"] == expected_status
        assert native["finite_bitwise_comparisons"] == 4870656 and native["fixtures"] == 27
        assert native["cpu_reference_executions"] == 486 and native["fixed_flag_capture_executions"] == 9
        assert native["original_partition_size_preserved"] == 64 and native["invalid_selector_rejected"]
        assert native["original_dot_order_preserved"] and native["original_merge_unchanged"]
        assert native["pending_windows_tested"] == [1, 64]
        assert native["fp16_only"] and native["fp32_original_fallback_preserved"]
        required_flags += ("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV",
                           "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_TILE4",
                           "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024")
        if variant == "dsa-paired-v1":
            build = json.loads((directory / "build-manifest.json").read_text())
            assert build["logical_partitions_per_workgroup"] == 2
            assert build["direct_integer_length_loads_no_cross_subgroup_shared_state"]
            required_flags += ("ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")
    elif variant == "qsa-small-query-cache-v2":
        assert native["status"] == "exact_reference_and_small_query_cache_dispatch_qualified"
        assert native["fixtures"] == 69 and native["reference_executions"] == 2484
        assert native["invalid_selector_rejected"] and native["cpu_fallback_disabled"]
        assert native["exercised"] and native["maximum_score_stride_exclusive"] == 2048
        build = json.loads((directory / "build-manifest.json").read_text())
        assert build["provider_sha256"] == native["provider_sha256"]
        assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
        for name, digest in build["source_sha256"].items():
            assert sha(directory / name) == digest
        required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                           "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16")
    elif variant == "hc-cached-norm-v2":
        assert native["status"] == "hc_cached_norm_four_outputs_finite_bitwise_and_compound_dispatch_qualified"
        assert native["fixtures"] == 36 and native["finite_bitwise_comparisons"] == 607143936
        assert native["all_four_outputs_compared"] and native["invalid_flag_explicitly_rejected"]
        assert native["changing_inputs_and_cpu_splits"] and native["lru_eviction_tested"]
        assert native["pending_windows_tested"] == [1, 64]
        assert native["compound_candidate_not_replay_cached"] and not native["replay_variant_keys_tested"]
        build = json.loads((directory / "build-manifest.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
        assert not build["replay_variant_keyed"]
        for name, digest in build["source_sha256"].items():
            assert sha(directory / name) == digest
        required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                           "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16",
                           "ORT_WEBGPU_HC_POST_NORM_PROJECTION", "ORT_WEBGPU_HC_PROJECTION_SILU")
    elif variant == "dsa-partition32-v1":
        assert native["status"] == "dsa_partition32_cpu_reference_and_dispatch_qualified"
        assert native["cpu_reference_executions"] == 972 and native["fixtures"] == 27
        assert native["fixed_flag_capture_executions"] == 18 and native["invalid_selector_rejected"]
        assert native["pending_windows_tested"] == [1, 64]
        assert native["partition_grouping_changes_fp32_softmax_association"]
        required_flags += ("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV",
                           "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_TILE4",
                           "ORT_WEBGPU_QMOE_PREFILL512")
    elif variant in ("fc1-warp-fold-v1", "fc1-warp-fold-v3", "fc2-paired-dot-v1", *ROW4_EXPERT_VARIANTS):
        assert native["status"] == f"{window_prefix}_changing_inputs_routes_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 9748480 and native["fixtures"] == 23
        assert native["both_expert_fusions_fixed_enabled"] and native["expert_bitcast_fixed_enabled"]
        mode_field = ("expert_pairs_tested" if variant == "fc1-expert-pairs-v1" else
                      "u32_partials_tested" if variant == "fc1-u32-partials-v1" else
                      "k8_tile_tested" if variant == "fc1-k8-tile-v1" else
                      "k16_tile_tested" if variant == "fc1-k16-tile-v1" else
                      "columns_tested" if variant == "fc1-columns8-v1" else
                      "full_activation_tested" if variant == "fc1-full-a-v1" else
                      "multi_expert_tested" if variant == "fc1-experts10-v1" else
                      "compact_lanes_tested" if variant == "fc2-compact20-v1" else
                      "subgroup_fold_tested" if variant == "fc2-subgroup-fold-v1" else
                      "paired_dot_tested" if variant in ("fc2-paired-dot-v1", "fc1-paired-dot-v1")
                      else "warp_fold_tested")
        assert native[mode_field] == [0, 1] and native["variant_cache_keys_tested"]
        assert native["pending_windows_tested"] == [1, 64]
        assert native["optional_bias_and_unsupported_layout_fallbacks_tested"]
        build = json.loads((directory / "build-manifest.json").read_text())
        assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
        assert build["replay_variant_keyed"]
        for name, digest in build["source_sha256"].items():
            assert sha(directory / name) == digest
        required_flags += ("ORT_WEBGPU_QMOE_FC1_ACTIVATION", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX",
                           "ORT_WEBGPU_EXPERT_BITCAST_UNPACK", "ORT_WEBGPU_QMOE_PREFILL1024",
                           "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32",
                           "ORT_WEBGPU_QSA_PREFILL_SCORE16")
    elif variant == "qmoe-fast-flags-v1":
        assert native["status"] == "qmoe_fast_flags_changing_inputs_routes_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 9748480 and native["fixtures"] == 23
        assert native["parser_edge_cases"] == 196 and native["original_parser_errors_preserved"]
        assert native["no_persistent_environment_cache"]
        assert native["pending_windows_tested"] == [1, 64]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_EXPERT_BITCAST_UNPACK",
                           "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_QMOE_PREFILL512")
    elif variant in ("dense-predecode-v2", "dense-qkv-predecode8-v1"):
        scoped = variant == "dense-qkv-predecode8-v1"
        if scoped:
            from run_dense_qkv_predecode8_native import verify_stats as verify_predecoded_stats
        else:
            from run_dense_predecode_native import verify_predecoded_stats
        assert native["status"] == f"{window_prefix}_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 116582784 and native["fixtures"] == 27
        assert native["fp32_accumulation_enabled"] is False
        assert native["multi_token_fallback_tested"] and native["invalid_flag_explicitly_rejected"]
        assert native["pending_windows_tested"] == [1, 64]
        verify_predecoded_stats(native["stats"])
        build = json.loads((directory / "build-manifest.json").read_text())
        assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
        assert build["baseline_byte_identical"] and build["original_kernel_class_unchanged"]
        assert build["constant_weights_and_scales_only"] and build["private_kernel_owns_decoded_weights"]
        assert build["original_dot_order_activation_staging_and_serial_fold_preserved"]
        assert build["no_fp32_accumulation_or_tensorcore_changes"]
        if scoped:
            assert build["logical_shape"] == {"M": 1, "K": 2560, "N": 16384}
            assert build["workgroup_threads"] == 256 and build["output_columns_per_workgroup"] == 8
            assert build["k_lanes_per_column"] == 32 and build["shared_bytes"] == 2560
            assert build["per_initializer_cache_bytes"] == 83886080 and build["row4_prefill_preserved"]
            assert dispatches == 36 and original_dispatches == 123
        for name, digest in build["source_sha256"].items():
            assert sha(directory / name) == digest
        fp32 = json.loads((ROOT / f"{variant}-fp32-native-gpu7-{stamp}.qualification.json").read_text())
        assert fp32["status"] == f"{window_prefix}_fp32_accumulation_fallback_finite_bitwise_qualified"
        assert fp32["provider_sha256"] == native["provider_sha256"]
        assert fp32["finite_bitwise_comparisons"] == 116582784
        assert fp32["fp32_accumulation_enabled"] and fp32["fp32_accumulation_fallback_only"]
        if scoped:
            assert fp32["baseline_provider_sha256"] == native["baseline_provider_sha256"]
            assert fp32["comparator_sha256"] == native["comparator_sha256"]
            assert fp32["fixtures"] == 27 and fp32["multi_token_fallback_tested"]
            assert fp32["invalid_flag_explicitly_rejected"] and fp32["pending_windows_tested"] == [1, 64]
        for proof in (native, fp32):
            for name, digest in proof["source_sha256"].items():
                path = directory / name if name == "build-manifest.json" else ROOT / name
                assert sha(path) == digest
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_BITCAST_UNPACK",
                           "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_QMOE_PREFILL512",
                           "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                           "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16")
    elif variant in ("dense-tile2-v2", "dense-wide8-v1", "dense-preload-v1", "dense-double-buffer-v1",
                     "dense-block32-v1", "dense-block32-v2", "dense-static-shape-v1", "dense-direct-a-v1",
                     "dense-private-partials-v3", "dense-pad-a-v1", "dense-qkv-wide16-v1", "dense-qkv-k16-v1",
                     "dense-qkv-threads64-v1", "dense-qkv-pad34-v1"):
        assert native["status"] == f"{window_prefix}_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 116582784 and native["fixtures"] == 27
        assert native["multi_token_fallback_tested"] and native["invalid_flag_explicitly_rejected"]
        assert native["pending_windows_tested"] == [1, 64]
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_BITCAST_UNPACK",
                           "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_QMOE_PREFILL512")
        if variant in ("dense-block32-v1", "dense-block32-v2"):
            required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024",)
        if variant == "dense-block32-v2":
            build = json.loads((directory / "build-manifest.json").read_text())
            assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
            assert build["dense_component_relink_byte_identical"] and build["query_cache_object_unchanged"]
            assert build["no_new_shader_or_arithmetic_changes"]
            required_flags += ("ORT_WEBGPU_DSA_QUERY_CACHE",)
        if variant in ("dense-static-shape-v1", "dense-direct-a-v1", "dense-private-partials-v3", "dense-pad-a-v1"):
            required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                               "ORT_WEBGPU_DENSE_BLOCK32")
        if variant in ("dense-qkv-wide16-v1", "dense-qkv-k16-v1", "dense-qkv-threads64-v1",
                       "dense-qkv-pad34-v1"):
            from run_dense_qkv_wide16_native import verify_stats
            verify_stats(native["stats"])
            build = json.loads((directory / "build-manifest.json").read_text())
            assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
            assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
            assert build["logical_shape"] == {"M": 1, "K": 2560, "N": 16384}
            k16 = variant == "dense-qkv-k16-v1"
            threads64 = variant == "dense-qkv-threads64-v1"
            pad34 = variant == "dense-qkv-pad34-v1"
            if pad34:
                threads, columns, lanes = 128, 4, 32
            else:
                threads, columns, lanes = (64, 4, 32) if threads64 else ((128, 8, 16) if k16 else (512, 16, 32))
            assert build["workgroup_threads"] == threads and build["output_columns_per_workgroup"] == columns
            assert build["k_lanes_per_column"] == lanes and build["row4_prefill_preserved"]
            preserves_field = ("original_fp16_dot_expressions_lane_updates_and_serial_fold_preserved" if k16 or threads64 or pad34 else
                               "original_fp16_dots_partial_stores_and_serial_fold_preserved")
            assert build[preserves_field]
            if k16 or threads64 or pad34:
                assert build["original_partial_slots"] == 32
            if threads64:
                assert build["physical_column_groups"] == 2 and build["output_columns_per_thread"] == 2
                assert build["compressed_weights_preserved"]
            if pad34:
                assert build["physical_partial_slots"] == 34 and build["shared_bytes"] == 2320
                assert build["compressed_weights_preserved"]
            assert dispatches == 36 and original_dispatches == 123
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
            required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE",
                               "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16")
        if variant == "dense-pad-a-v1":
            build = json.loads((directory / "build-manifest.json").read_text())
            assert build["baseline_byte_identical"] and build["no_existing_kernel_object_layout_changes"]
            assert build["original_dot_order_activation_staging_and_serial_fold_preserved"]
            assert build["activation_padding_vector_block"] == 8 and build["activation_padding_vector_pitch"] == 9
            assert build["fp16_canonical_only"]
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
            required_flags += ("ORT_WEBGPU_QSA_PREFILL_SCORE16",)
        if variant == "dense-direct-a-v1":
            build = json.loads((directory / "build-manifest.json").read_text())
            assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
            assert build["fp16_canonical_direct_activation_reads"]
            assert build["original_shared_partial_rounding_and_serial_fold_preserved"]
            assert build["no_activation_shared_array"] and build["one_final_workgroup_barrier"]
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
        if variant == "dense-private-partials-v3":
            build = json.loads((directory / "build-manifest.json").read_text())
            assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
            assert build["fp16_canonical_private_partials"] and build["typed_fp16_partial_updates_every_k_tile"]
            assert build["original_dot_order_activation_staging_and_serial_fold_preserved"]
            assert build["workgroup_threads"] == 128 and build["k_lanes_per_column"] == 32
            assert build["one_scalar_partial_and_unique_final_shared_slot_per_thread"]
            for name, digest in build["source_sha256"].items():
                assert sha(directory / name) == digest
        if variant in ("dense-double-buffer-v1", "dense-static-shape-v1", "dense-direct-a-v1",
                       "dense-private-partials-v3", "dense-pad-a-v1", "dense-qkv-wide16-v1", "dense-qkv-k16-v1",
                       "dense-qkv-threads64-v1", "dense-qkv-pad34-v1"):
            assert native["fp32_accumulation_enabled"] is False
            fp32 = json.loads((ROOT / f"{variant}-fp32-native-gpu7-{stamp}.qualification.json").read_text())
            assert fp32["status"] == f"{window_prefix}_fp32_accumulation_fallback_finite_bitwise_qualified"
            assert fp32["provider_sha256"] == native["provider_sha256"]
            assert fp32["baseline_provider_sha256"] == native["baseline_provider_sha256"]
            assert fp32["comparator_sha256"] == native["comparator_sha256"]
            assert fp32["fp32_accumulation_enabled"] is True
            assert fp32["finite_bitwise_comparisons"] == 116582784 and fp32["fixtures"] == 27
            assert fp32["multi_token_fallback_tested"] and fp32["invalid_flag_explicitly_rejected"]
            assert fp32["pending_windows_tested"] == [1, 64]
            assert fp32["fp32_accumulation_fallback_only"] and not fp32["replay_variant_keys_tested"]
            required_flags += ("ORT_WEBGPU_QMOE_PREFILL1024",)
    else:
        raise ValueError(f"Unsupported HC experiment: {variant}")
    if variant in ROW4_DECODE_VARIANTS:
        assert parent_variant == "qmoe-prefill-rows4-v2" and chunk == 1024
        assert parent["status"] == "controlled_ttft_screens_all_contexts_qualified_not_standard_decode"
        required_flags += ("ORT_WEBGPU_QMOE_GROUPED_PREFILL",)
        if variant in ROW4_REPEAT_VARIANTS:
            row4_standard_path = ROOT / "qmoe-prefill-rows4-v2-standard-contexts-20261031.qualification.json"
            row4_standard = json.loads(row4_standard_path.read_text())
            assert row4_standard["status"] == "rows4_native_and_standard_original_sequences_qualified"
            assert row4_standard["provider_sha256"] == parent["provider_sha256"]
            assert row4_standard["all_original_sequences_match"] and row4_standard["zero_major_faults"]
            assert row4_standard["original_sequence_comparisons"] == 6144
            for path, digest in row4_standard["source_sha256"].items():
                assert sha(Path(path)) == digest, path
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(provider) == native["provider_sha256"]
    if histogram_qualification:
        from qualify_qsa_hist_rank import verify_prerequisites
        verify_prerequisites(native, args.phase)
    elif args.phase == "standard":
        paired = json.loads((ROOT / f"{variant}-windows-contexts-{stamp}.qualification.json").read_text())
        assert paired["provider_sha256"] == native["provider_sha256"]
        assert paired["all_original_prefixes_match"] and paired["zero_major_faults"]
        assert paired["all_contexts_faster_in_both_validation_modes"]
    if args.context != initial_context and not histogram_qualification:
        if variant not in ROW4_REPEAT_VARIANTS:
            for mode in ("full-", ""):
                proof = json.loads((ROOT / (
                    f"{variant}-chunk{chunk}-gpu7-{mode}windows{initial_context}-{stamp}.qualification.json")).read_text())
                assert proof["provider_sha256"] == native["provider_sha256"] and proof["zero_major_faults"]
                assert proof["result"]["geometric_mean_matched_window_ratio"] > 1
                if variant in ROW4_EXPERT_VARIANTS:
                    verify_fc2_subgroup_window_support(proof["result"], prefix=window_prefix)
        if variant in ROW4_REPEAT_VARIANTS:
            previous_contexts = [initial_context] + ([8192] if args.context == 32768 else [])
            for context in previous_contexts:
                attempts = ("", "-attempt2") if context == initial_context else ("",)
                for attempt in attempts:
                    for mode in ("full-", ""):
                        proof = json.loads((ROOT / (
                            f"{variant}-chunk{chunk}-gpu7-{mode}windows{context}-{stamp}"
                            f"{attempt}.qualification.json")).read_text())
                        verify_row4_window_prerequisite(proof, native["provider_sha256"],
                                                        context, bool(mode), prefix=window_prefix)
                        assert proof["native_proof_sha256"] == sha(ROOT / f"{variant}-native-gpu7-{stamp}.qualification.json")
                        assert proof["parent_proof_sha256"] == sha(ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json")
                        assert proof["row4_standard_proof_sha256"] == sha(row4_standard_path)
                        assert proof["build_manifest_sha256"] == sha(directory / "build-manifest.json")
                        for name, digest in proof["source_sha256"].items():
                            assert sha(ROOT / name) == digest, name
    if args.phase == "windows":
        profile_context = args.context if histogram_qualification else initial_context
        profile_dispatches = 0 if histogram_qualification and args.context < initial_context else dispatches
        profile = json.loads((ROOT / (
            f"{variant}-chunk{chunk}-gpu7-full-profile{profile_context}-{stamp}.qualification.json")).read_text())
        assert profile["provider_sha256"] == native["provider_sha256"]
        assert profile["full_validation"] and profile["zero_major_faults"]
        assert profile["result"]["candidate_program_dispatches"] == {
            name: profile_dispatches for name in (program_name, *additional_programs)}
        if profile_dispatches == 0:
            assert profile["result"]["original_program_dispatches"] == dispatches
        elif original_dispatches:
            assert profile["result"]["original_program_dispatches"] == original_dispatches
        if variant == "gdn-tree32-hist-v1" and args.context == 32768:
            long_profile = json.loads((ROOT / (
                f"{variant}-chunk{chunk}-gpu7-full-profile32768-{stamp}.qualification.json")).read_text())
            assert long_profile["provider_sha256"] == native["provider_sha256"]
            assert long_profile["zero_major_faults"] and long_profile["original_output_prefixes_match"]
            assert long_profile["result"]["histogram_program_dispatches"] == {
                "QsaHistogramUniformRank": 12, "QsaRankHistogram": 12}
    if not args.full_validation:
        validation_phase = "standard" if args.phase == "standard" else "windows"
        proof = json.loads((ROOT / (
            f"{variant}-chunk{chunk}-gpu7-full-{validation_phase}{args.context}-{stamp}"
            f"{attempt_suffix}.qualification.json")).read_text())
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["full_validation"] and proof["zero_major_faults"] and proof["original_output_prefixes_match"]
    prefix = ROOT / (f"{variant}-chunk{chunk}-gpu7-{'full-' if args.full_validation else ''}"
                     f"{args.phase}{args.context}-{stamp}" +
                     attempt_suffix)
    if prefix.with_suffix(".qualification.json").exists() or (
            not args.verify_existing and any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json"))):
        raise FileExistsError(prefix)
    baseline = parent_standard_prefix or f"{parent_variant}-chunk{chunk}-gpu7-full-standard2048-{parent_stamp}"
    command = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    parent_family, parent_version = parent_variant.rsplit("-v", 1)
    old_provider = (str(parent_provider).replace("/home/sayanshaw/repos", "/workspace")
                    if parent_provider is not None else
                    f"/workspace/qwen38-int4-{parent_family}/build-v{parent_version}/libonnxruntime_providers_webgpu.so")
    assert command[-1].count(old_provider) == command[-1].count(f"--chunk {chunk}") == 1
    command[-1] = (f"export {experiment_flag}=1; " + command[-1].replace(
        old_provider, str(provider).replace("/home/sayanshaw/repos", "/workspace")).replace(
        baseline, prefix.name).replace("--contexts 2048", f"--contexts {args.context}"))
    if variant in ROW4_DECODE_VARIANTS:
        command[-1] = "export ORT_WEBGPU_QMOE_GROUPED_PREFILL=1; " + command[-1]
    if variant in ROW4_REPEAT_VARIANTS:
        command[-1] = "export ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0; " + command[-1]
    if variant in ("dense-qkv-wide16-v1", "fc1-experts10-v1", "fc1-full-a-v1", "fc1-columns8-v1",
                   "fc1-paired-dot-v1", "fc1-k16-tile-v1", "fc1-k8-tile-v1", "fc1-u32-partials-v1",
                   "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_QMOE_FC2_COMPACT20=0; " + command[-1]
    if variant in ("fc1-experts10-v1", "fc1-full-a-v1", "fc1-columns8-v1",
                   "fc1-paired-dot-v1", "fc1-k16-tile-v1", "fc1-k8-tile-v1", "fc1-u32-partials-v1",
                   "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_DENSE_QKV_WIDE16=0; " + command[-1]
    if variant in ("fc1-full-a-v1", "fc1-columns8-v1", "fc1-paired-dot-v1", "fc1-k16-tile-v1", "fc1-k8-tile-v1",
                   "fc1-u32-partials-v1", "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_QMOE_FC1_EXPERTS10=0; " + command[-1]
    if variant in ("fc1-columns8-v1", "fc1-paired-dot-v1", "fc1-k16-tile-v1", "fc1-k8-tile-v1",
                   "fc1-u32-partials-v1", "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_QMOE_FC1_FULL_A=0; " + command[-1]
    if variant in ("fc1-paired-dot-v1", "fc1-k16-tile-v1", "fc1-k8-tile-v1", "fc1-u32-partials-v1",
                   "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_QMOE_FC1_COLUMNS8=0 ORT_WEBGPU_QMOE_PREFILL_ROWS8=0; " + command[-1]
        for other_flag in ("ORT_WEBGPU_QMOE_FC1_PAIRED_DOT", "ORT_WEBGPU_QMOE_FC1_K16_TILE",
                           "ORT_WEBGPU_QMOE_FC1_K8_TILE", "ORT_WEBGPU_QMOE_FC1_U32_PARTIALS",
                           "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS"):
            if other_flag != experiment_flag:
                command[-1] = f"export {other_flag}=0; " + command[-1]
    if variant in ("fc1-u32-partials-v1", "fc1-expert-pairs-v1"):
        command[-1] = "export ORT_WEBGPU_DENSE_QKV_K16=0; " + command[-1]
    if variant == "dense-qkv-k16-v1":
        command[-1] = (
            "export ORT_WEBGPU_DENSE_QKV_WIDE16=0 ORT_WEBGPU_QMOE_FC2_COMPACT20=0 "
            "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0 ORT_WEBGPU_QMOE_FC1_EXPERTS10=0 "
            "ORT_WEBGPU_QMOE_FC1_FULL_A=0 ORT_WEBGPU_QMOE_FC1_COLUMNS8=0 "
            "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT=0 ORT_WEBGPU_QMOE_FC1_K16_TILE=0 "
            "ORT_WEBGPU_QMOE_FC1_K8_TILE=0 ORT_WEBGPU_QMOE_PREFILL_ROWS8=0; " + command[-1])
    if variant == "dense-qkv-predecode8-v1":
        command[-1] = (
            "export ORT_WEBGPU_DENSE_PREDECODE=0 ORT_WEBGPU_DENSE_QKV_WIDE16=0 "
            "ORT_WEBGPU_DENSE_QKV_K16=0 ORT_WEBGPU_QMOE_FC2_COMPACT20=0 "
            "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0 ORT_WEBGPU_QMOE_FC1_EXPERTS10=0 "
            "ORT_WEBGPU_QMOE_FC1_FULL_A=0 ORT_WEBGPU_QMOE_FC1_COLUMNS8=0 "
            "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT=0 ORT_WEBGPU_QMOE_FC1_K16_TILE=0 "
            "ORT_WEBGPU_QMOE_FC1_K8_TILE=0 ORT_WEBGPU_QMOE_FC1_U32_PARTIALS=0 "
            "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS=0 ORT_WEBGPU_QMOE_PREFILL_ROWS8=0; " + command[-1])
    if variant == "dense-qkv-threads64-v1":
        command[-1] = (
            "export ORT_WEBGPU_DENSE_PREDECODE=0 ORT_WEBGPU_DENSE_QKV_PREDECODE8=0 "
            "ORT_WEBGPU_DENSE_QKV_WIDE16=0 ORT_WEBGPU_DENSE_QKV_K16=0 "
            "ORT_WEBGPU_QMOE_FC2_COMPACT20=0 ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0 "
            "ORT_WEBGPU_QMOE_FC1_EXPERTS10=0 ORT_WEBGPU_QMOE_FC1_FULL_A=0 "
            "ORT_WEBGPU_QMOE_FC1_COLUMNS8=0 ORT_WEBGPU_QMOE_FC1_PAIRED_DOT=0 "
            "ORT_WEBGPU_QMOE_FC1_K16_TILE=0 ORT_WEBGPU_QMOE_FC1_K8_TILE=0 "
            "ORT_WEBGPU_QMOE_FC1_U32_PARTIALS=0 ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS=0 "
            "ORT_WEBGPU_QMOE_PREFILL_ROWS8=0; " + command[-1])
    if variant == "dense-qkv-pad34-v1":
        command[-1] = (
            "export ORT_WEBGPU_DENSE_PREDECODE=0 ORT_WEBGPU_DENSE_QKV_PREDECODE8=0 "
            "ORT_WEBGPU_DENSE_QKV_THREADS64=0 ORT_WEBGPU_DENSE_QKV_WIDE16=0 "
            "ORT_WEBGPU_DENSE_QKV_K16=0 ORT_WEBGPU_QMOE_FC2_COMPACT20=0 "
            "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0 ORT_WEBGPU_QMOE_FC1_EXPERTS10=0 "
            "ORT_WEBGPU_QMOE_FC1_FULL_A=0 ORT_WEBGPU_QMOE_FC1_COLUMNS8=0 "
            "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT=0 ORT_WEBGPU_QMOE_FC1_K16_TILE=0 "
            "ORT_WEBGPU_QMOE_FC1_K8_TILE=0 ORT_WEBGPU_QMOE_FC1_U32_PARTIALS=0 "
            "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS=0 ORT_WEBGPU_QMOE_PREFILL_ROWS8=0; " + command[-1])
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == 1
    if args.phase == "windows":
        replacement = f"--tokens 288 --repetitions 4 --compare-{family}-windows"
    elif args.phase == "profile":
        replacement = ("--tokens 36 --repetitions 1 --profile-decode-after 35 "
                       f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
    else:
        replacement = marker
    command[-1] = command[-1].replace(marker, replacement)
    if not args.full_validation:
        assert command[-1].count(" --webgpu-validation-mode full") == 1
        command[-1] = command[-1].replace(" --webgpu-validation-mode full", "")
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if args.verify_existing:
        cache = json.loads(prefix.with_name(prefix.name + "-cache.json").read_text())
        assert "after_run" in cache
    else:
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
            "--output-prefix", str(prefix), "--", *command], check=True)
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    verify_metadata(raw, guard, native, parent, args.full_validation, chunk=chunk)
    assert guard["command"] == command
    for flag in required_flags:
        assert raw["load_experiment_flags"][flag] == "1"
    assert raw["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == sha(model / "genai_config.json")
    references = [s for s in json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
                  if s["context"] == args.context]
    profiles = []
    if args.phase == "windows":
        result = verify_paired(raw, references, args.context, prefix=window_prefix, experiment_flag=experiment_flag)
        if histogram_qualification:
            from qualify_qsa_hist_rank import verify_window_result
            verify_window_result(result, long_context=args.context == initial_context)
    elif args.phase == "standard":
        result = verify_standard(raw, guard, native, parent, references, args.context, args.full_validation,
                                 chunk=chunk)
    else:
        verify_tokens(raw, references, args.context, 36)
        assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
        profiles = list(ROOT.glob(prefix.name + f"-profile-ctx{args.context}-rep0-configured_*.json"))
        assert len(profiles) == 1, profiles
        counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                                     for event in json.loads(profiles[0].read_text())
                                     if event.get("cat") == "Kernel")
        inactive_histogram = histogram_qualification and args.context < initial_context
        expected_dispatches = 0 if inactive_histogram else dispatches
        assert all(counts[name] == expected_dispatches for name in (program_name, *additional_programs)), counts
        assert counts[original_program_name] == (dispatches if inactive_histogram else original_dispatches), counts
        if variant in ("dense-block32-v2", "dense-static-shape-v1", "dense-direct-a-v1",
                       "dense-predecode-v2", "dense-qkv-predecode8-v1",
                       "dense-private-partials-v3", "dense-pad-a-v1"):
            assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
            assert counts["DynamicSparseAttention"] == 0, counts
            if variant == "dense-predecode-v2":
                assert counts["MatMulNBitsBlock32F16"] == 0, counts
                assert counts["DenseDecodeWeightsF16"] == 0, counts
            if variant == "dense-qkv-predecode8-v1":
                assert counts["DenseDecodeWeightsF16"] == 0, counts
            if variant in ("dense-static-shape-v1", "dense-direct-a-v1", "dense-private-partials-v3", "dense-pad-a-v1"):
                assert counts["MatMulNBitsTile4F16"] == 0, counts
        if variant in ("hc-post-norm-vec4-v1", "hc-post-norm-predecode-v1",
                       "hc-post-norm-predecode-v2", "gdn-value8-v1", "gdn-tree32-v1", "gdn-norm32-v1",
                       "gdn-tree32-hist-v1",
                       "hc-cached-norm-v2",
                       "hc-quad-fold-v1",
                       "hc-narrow4-v1",
                       "qsa-hist-rank-v1",
                       "qsa-hist-rank-v2",
                       "qsa-pruned-rank-v1", "qsa-pruned-rank-v2", "qsa-subgroup-rank-v1"):
            assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
            assert counts["DynamicSparseAttention"] == 0, counts
            assert counts["MatMulNBitsBlock32F16"] == 159, counts
            assert counts["MatMulNBitsTile4F16"] == 0, counts
        result = dict(candidate_program_dispatches={
                          name: expected_dispatches for name in (program_name, *additional_programs)},
                      instrumented_not_a_speed_claim=True)
        if variant in ("dense-predecode-v2", "dense-qkv-predecode8-v1"):
            result["weight_decode_program_dispatches"] = counts["DenseDecodeWeightsF16"]
        if variant == "gdn-tree32-hist-v1":
            result["histogram_program_dispatches"] = verify_gdn_histogram_profile(counts, args.context)
        if histogram_qualification or original_dispatches:
            result["original_program_dispatches"] = counts[original_program_name]
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        full_validation=args.full_validation, chunk=chunk, zero_major_faults=True,
        original_output_prefixes_match=True, provider_sha256=native["provider_sha256"],
        baseline_provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
        config_sha256=parent["config_sha256"], result=result, cache_preflight=cache,
        source_sha256={path.name: sha(path) for path in [*profiles, prefix.with_suffix(".json"),
                                                       prefix.with_suffix(".guard.json")]},
        quality_qualified=False, headline_eligible=False)
    if variant in ROW4_DECODE_VARIANTS:
        report["fixed_grouped_prefill_selector"] = 1
        report["native_proof_sha256"] = sha(ROOT / f"{variant}-native-gpu7-{stamp}.qualification.json")
        report["parent_proof_sha256"] = sha(ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json")
        report["build_manifest_sha256"] = sha(directory / "build-manifest.json")
        if variant in ROW4_REPEAT_VARIANTS:
            report["row4_standard_proof_sha256"] = sha(row4_standard_path)
    if variant == "hc-cached-norm-v2":
        report["compound_candidate_not_replay_cached"] = True
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "phase", "result")}), flush=True)


if __name__ == "__main__":
    main()

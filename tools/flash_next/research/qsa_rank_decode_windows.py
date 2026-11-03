import math
import os
import statistics
import time

from benchmark_webgpu_int4 import natural_text_prompt, run_request


RANK_FLAG = "ORT_WEBGPU_QSA_PARALLEL_RANK"
DENSE_MAP_FLAG = "ORT_WEBGPU_QSA_DENSE_MAP"
EXACT_MID_FLAG = "ORT_WEBGPU_EXACT_DENSE_MID"
EXACT_WIDE_FLAG = "ORT_WEBGPU_EXACT_DENSE_WIDE"
EXACT_EXPERT_FLAG = "ORT_WEBGPU_EXACT_EXPERT"
BRANCH_NORM_FLAG = "ORT_WEBGPU_EXACT_BRANCH_NORM"
LARGE_CACHE_FLAG = "ORT_WEBGPU_BIND_GROUP_CACHE_LARGE"
GDN_VIEW_FLAG = "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW"
GDN_VALUE2_FLAG = "ORT_WEBGPU_GDN_VALUE2"
GDN_VALUE8_FLAG = "ORT_WEBGPU_GDN_VALUE8"
GDN_TREE32_FLAG = "ORT_WEBGPU_GDN_TREE32"
GDN_NORM32_FLAG = "ORT_WEBGPU_GDN_NORM32"
NODE_REPLAY_FLAG = "ORT_WEBGPU_QMOE_COMMAND_REPLAY"
STATIC_REPLAY_FLAG = "ORT_WEBGPU_STATIC_NODE_REPLAY"
HC_PROJECTION_FLAG = "ORT_WEBGPU_HC_PROJECTION_SILU"
HC_UP_PREMIX_FLAG = "ORT_WEBGPU_HC_UP_PREMIX"
PARALLEL_MAP_FLAG = "ORT_WEBGPU_QSA_PARALLEL_MAP"
GENERIC_MATMUL_FLAG = "ORT_WEBGPU_GENERIC_MATMUL_REPLAY"
DIRECT_A_FLAG = "ORT_WEBGPU_MATMUL_NBITS_DIRECT_A"
FC1_ACTIVATION_FLAG = "ORT_WEBGPU_QMOE_FC1_ACTIVATION"
REPLAY_IMMEDIATE_FLAG = "ORT_WEBGPU_NODE_REPLAY_IMMEDIATE"
FC2_FINAL_MIX_FLAG = "ORT_WEBGPU_QMOE_FC2_FINAL_MIX"
FC2_PADDED_PARTIALS_FLAG = "ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS"
HC_PROJECTION_TILED_FLAG = "ORT_WEBGPU_HC_PROJECTION_TILED"
SUBGROUP_GATE_FLAG = "ORT_WEBGPU_QMOE_SUBGROUP_GATE"
ACTIVATION_PADDING_FLAG = "ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS"
POST_NORM_FOUR_COLUMNS_FLAG = "ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS"
HIERARCHICAL_GATE_FLAG = "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"
MLP_BITCAST_UNPACK_FLAG = "ORT_WEBGPU_MLP_BITCAST_UNPACK"
EXPERT_BITCAST_UNPACK_FLAG = "ORT_WEBGPU_EXPERT_BITCAST_UNPACK"
DENSE_BITCAST_UNPACK_FLAG = "ORT_WEBGPU_DENSE_BITCAST_UNPACK"
HC_POST_NORM_BITCAST_FLAG = "ORT_WEBGPU_HC_POST_NORM_BITCAST"
HC_POST_NORM_FUSED_SUM_FLAG = "ORT_WEBGPU_HC_POST_NORM_FUSED_SUM"
HC_POST_NORM_SUBGROUP_FLAG = "ORT_WEBGPU_HC_POST_NORM_SUBGROUP"
HC_UP_BITCAST_UNPACK_FLAG = "ORT_WEBGPU_HC_UP_BITCAST_UNPACK"
QSA_CONCAT_VECTOR_FLAG = "ORT_WEBGPU_QSA_CONCAT_VECTOR"
QSA_PACKED_SCORE_FLAG = "ORT_WEBGPU_QSA_PACKED_SCORE"
QSA_INDEXER_STACK_FLAG = "ORT_WEBGPU_QSA_INDEXER_STACK"
QSA_SCORE16_FLAG = "ORT_WEBGPU_QSA_SCORE16"
QSA_SMALL_QUERY_CACHE_FLAG = "ORT_WEBGPU_QSA_SMALL_QUERY_CACHE"
FC2_FOUR_COLUMNS_FLAG = "ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS"
HC_UP_TILED_FLAG = "ORT_WEBGPU_HC_UP_TILED"
HC_UP_PARALLEL2_FLAG = "ORT_WEBGPU_HC_UP_PARALLEL2"
HC_UP_PARALLEL4_FLAG = "ORT_WEBGPU_HC_UP_PARALLEL4"
HC_POST_NORM_PAD34_FLAG = "ORT_WEBGPU_HC_POST_NORM_PAD34"
HC_POST_NORM_VEC4_FLAG = "ORT_WEBGPU_HC_POST_NORM_VEC4"
HC_POST_NORM_PREDECODE_FLAG = "ORT_WEBGPU_HC_POST_NORM_PREDECODE"
HC_POST_NORM_WIDE16_FLAG = "ORT_WEBGPU_HC_POST_NORM_WIDE16"
HC_CACHED_NORM_FLAG = "ORT_WEBGPU_HC_CACHED_NORM"
HC_QUAD_FOLD_FLAG = "ORT_WEBGPU_HC_QUAD_FOLD"
HC_NARROW4_FLAG = "ORT_WEBGPU_HC_NARROW4"
FC1_WARP_FOLD_FLAG = "ORT_WEBGPU_QMOE_FC1_WARP_FOLD"
FC2_PAIRED_DOT_FLAG = "ORT_WEBGPU_QMOE_FC2_PAIRED_DOT"
FC2_SUBGROUP_FOLD_FLAG = "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"
FC2_COMPACT20_FLAG = "ORT_WEBGPU_QMOE_FC2_COMPACT20"
FC1_EXPERTS10_FLAG = "ORT_WEBGPU_QMOE_FC1_EXPERTS10"
FC1_FULL_A_FLAG = "ORT_WEBGPU_QMOE_FC1_FULL_A"
FC1_COLUMNS8_FLAG = "ORT_WEBGPU_QMOE_FC1_COLUMNS8"
FC1_PAIRED_DOT_FLAG = "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT"
FC1_K16_TILE_FLAG = "ORT_WEBGPU_QMOE_FC1_K16_TILE"
FC1_K8_TILE_FLAG = "ORT_WEBGPU_QMOE_FC1_K8_TILE"
FC1_U32_PARTIALS_FLAG = "ORT_WEBGPU_QMOE_FC1_U32_PARTIALS"
FC1_EXPERT_PAIRS_FLAG = "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS"
DENSE_QKV_WIDE16_FLAG = "ORT_WEBGPU_DENSE_QKV_WIDE16"
DENSE_QKV_K16_FLAG = "ORT_WEBGPU_DENSE_QKV_K16"
DENSE_QKV_PREDECODE8_FLAG = "ORT_WEBGPU_DENSE_QKV_PREDECODE8"
DENSE_QKV_THREADS64_FLAG = "ORT_WEBGPU_DENSE_QKV_THREADS64"
DENSE_QKV_PAD34_FLAG = "ORT_WEBGPU_DENSE_QKV_PAD34"
QSA_HIST_RANK_FLAG = "ORT_WEBGPU_QSA_HIST_RANK"
HC_PROJECTION_SHUFFLE_FLAG = "ORT_WEBGPU_HC_PROJECTION_SHUFFLE"
HC_STAGED_PROJECTION_FLAG = "ORT_WEBGPU_HC_STAGED_PROJECTION"
DENSE_TILE4_FLAG = "ORT_WEBGPU_DENSE_TILE4"
DENSE_TILE2_FLAG = "ORT_WEBGPU_DENSE_TILE2"
DENSE_WIDE8_FLAG = "ORT_WEBGPU_DENSE_WIDE8"
DENSE_PRELOAD_FLAG = "ORT_WEBGPU_DENSE_PRELOAD"
DENSE_DOUBLE_BUFFER_FLAG = "ORT_WEBGPU_DENSE_DOUBLE_BUFFER"
DENSE_BLOCK32_FLAG = "ORT_WEBGPU_DENSE_BLOCK32"
DENSE_PREDECODE_FLAG = "ORT_WEBGPU_DENSE_PREDECODE"
DENSE_DIRECT_A_FLAG = "ORT_WEBGPU_DENSE_DIRECT_A"
DENSE_PRIVATE_PARTIALS_FLAG = "ORT_WEBGPU_DENSE_PRIVATE_PARTIALS"
DENSE_PAD_A_FLAG = "ORT_WEBGPU_DENSE_PAD_A"
DENSE_STATIC_SHAPE_FLAG = "ORT_WEBGPU_DENSE_STATIC_SHAPE"
QSA_PRUNED_RANK_FLAG = "ORT_WEBGPU_QSA_PRUNED_RANK"
QSA_SUBGROUP_RANK_FLAG = "ORT_WEBGPU_QSA_SUBGROUP_RANK"
QMOE_FAST_FLAGS_FLAG = "ORT_WEBGPU_QMOE_FAST_FLAGS"
DSA_PARTITION32_FLAG = "ORT_WEBGPU_DSA_PARTITION32"
DSA_QUERY_CACHE_FLAG = "ORT_WEBGPU_DSA_QUERY_CACHE"
DSA_PAIRED_FLAG = "ORT_WEBGPU_DSA_PAIRED"
CPU_LOCAL_FLAG = "ORT_WEBGPU_CPU_LOCAL_WINDOWS"
PACKED_GATE_FLAG = "ORT_WEBGPU_QMOE_PACKED_GATE"
WINDOW_PATTERN = (0, 1, 1, 0, 1, 0, 0, 1)
WARMUP_TOKENS = 32


def run_qsa_windows(og, model, tokenizer, pool, args, result, persist, *, experiment_flag=RANK_FLAG):
    prefixes = {RANK_FLAG: "rank", DENSE_MAP_FLAG: "dense_map", EXACT_MID_FLAG: "exact_mid",
                EXACT_WIDE_FLAG: "exact_wide", EXACT_EXPERT_FLAG: "exact_expert",
                BRANCH_NORM_FLAG: "branch_norm", LARGE_CACHE_FLAG: "large_cache",
                GDN_VIEW_FLAG: "gdn_view", GDN_VALUE2_FLAG: "gdn_value2", NODE_REPLAY_FLAG: "node_replay",
                GDN_VALUE8_FLAG: "gdn_value8",
                GDN_TREE32_FLAG: "gdn_tree32",
                GDN_NORM32_FLAG: "gdn_norm32",
                STATIC_REPLAY_FLAG: "static_replay", HC_PROJECTION_FLAG: "hc_projection",
                HC_UP_PREMIX_FLAG: "hc_up_premix", PARALLEL_MAP_FLAG: "parallel_map",
                GENERIC_MATMUL_FLAG: "generic_matmul", DIRECT_A_FLAG: "direct_a",
                FC1_ACTIVATION_FLAG: "fc1_activation", REPLAY_IMMEDIATE_FLAG: "replay_immediate",
                FC2_FINAL_MIX_FLAG: "fc2_final_mix", FC2_PADDED_PARTIALS_FLAG: "fc2_padded_partials",
                HC_PROJECTION_TILED_FLAG: "hc_projection_tiled", SUBGROUP_GATE_FLAG: "subgroup_gate",
                ACTIVATION_PADDING_FLAG: "activation_padding",
                POST_NORM_FOUR_COLUMNS_FLAG: "post_norm_four_columns",
                HIERARCHICAL_GATE_FLAG: "hierarchical_gate",
                MLP_BITCAST_UNPACK_FLAG: "mlp_bitcast_unpack",
                EXPERT_BITCAST_UNPACK_FLAG: "expert_bitcast_unpack",
                DENSE_BITCAST_UNPACK_FLAG: "dense_bitcast_unpack",
                HC_POST_NORM_BITCAST_FLAG: "hc_post_norm_bitcast",
                HC_POST_NORM_FUSED_SUM_FLAG: "hc_post_norm_fused_sum",
                HC_POST_NORM_SUBGROUP_FLAG: "hc_post_norm_subgroup",
                HC_UP_BITCAST_UNPACK_FLAG: "hc_up_bitcast_unpack",
                QSA_CONCAT_VECTOR_FLAG: "qsa_concat_vector",
                QSA_PACKED_SCORE_FLAG: "qsa_packed_score",
                QSA_INDEXER_STACK_FLAG: "qsa_indexer_stack",
                QSA_SCORE16_FLAG: "qsa_score16",
                FC2_FOUR_COLUMNS_FLAG: "fc2_four_columns",
                HC_UP_TILED_FLAG: "hc_up_tiled",
                HC_UP_PARALLEL2_FLAG: "hc_up_parallel2",
                HC_UP_PARALLEL4_FLAG: "hc_up_parallel4",
                HC_POST_NORM_PAD34_FLAG: "hc_post_norm_pad34",
                HC_POST_NORM_VEC4_FLAG: "hc_post_norm_vec4",
                HC_POST_NORM_PREDECODE_FLAG: "hc_post_norm_predecode",
                HC_POST_NORM_WIDE16_FLAG: "hc_post_norm_wide16",
                HC_CACHED_NORM_FLAG: "hc_cached_norm",
                HC_QUAD_FOLD_FLAG: "hc_quad_fold",
                HC_NARROW4_FLAG: "hc_narrow4",
                FC1_WARP_FOLD_FLAG: "fc1_warp_fold",
                FC2_PAIRED_DOT_FLAG: "fc2_paired_dot",
                FC2_SUBGROUP_FOLD_FLAG: "fc2_subgroup_fold",
                FC2_COMPACT20_FLAG: "fc2_compact20",
                FC1_EXPERTS10_FLAG: "fc1_experts10",
                FC1_FULL_A_FLAG: "fc1_full_a",
                FC1_COLUMNS8_FLAG: "fc1_columns8",
                FC1_PAIRED_DOT_FLAG: "fc1_paired_dot",
                FC1_K16_TILE_FLAG: "fc1_k16_tile",
                FC1_K8_TILE_FLAG: "fc1_k8_tile",
                FC1_U32_PARTIALS_FLAG: "fc1_u32_partials",
                FC1_EXPERT_PAIRS_FLAG: "fc1_expert_pairs",
                DENSE_QKV_WIDE16_FLAG: "dense_qkv_wide16",
                DENSE_QKV_K16_FLAG: "dense_qkv_k16",
                DENSE_QKV_PREDECODE8_FLAG: "dense_qkv_predecode8",
                DENSE_QKV_THREADS64_FLAG: "dense_qkv_threads64",
                DENSE_QKV_PAD34_FLAG: "dense_qkv_pad34",
                QSA_HIST_RANK_FLAG: "qsa_hist_rank",
                HC_PROJECTION_SHUFFLE_FLAG: "hc_projection_shuffle",
                HC_STAGED_PROJECTION_FLAG: "hc_staged_projection",
                DENSE_STATIC_SHAPE_FLAG: "dense_static_shape",
                QSA_PRUNED_RANK_FLAG: "qsa_pruned_rank",
                QSA_SUBGROUP_RANK_FLAG: "qsa_subgroup_rank",
                DENSE_TILE4_FLAG: "dense_tile4",
                DENSE_TILE2_FLAG: "dense_tile2",
                DENSE_WIDE8_FLAG: "dense_wide8",
                DENSE_PRELOAD_FLAG: "dense_preload",
                DENSE_DOUBLE_BUFFER_FLAG: "dense_double_buffer",
                DENSE_BLOCK32_FLAG: "dense_block32",
                DENSE_PREDECODE_FLAG: "dense_predecode",
                DENSE_DIRECT_A_FLAG: "dense_direct_a",
                DENSE_PRIVATE_PARTIALS_FLAG: "dense_private_partials",
                DENSE_PAD_A_FLAG: "dense_pad_a",
                QSA_SMALL_QUERY_CACHE_FLAG: "qsa_small_query_cache",
                QMOE_FAST_FLAGS_FLAG: "qmoe_fast_flags",
                DSA_PARTITION32_FLAG: "dsa_partition32",
                DSA_QUERY_CACHE_FLAG: "dsa_query_cache",
                DSA_PAIRED_FLAG: "dsa_paired",
                CPU_LOCAL_FLAG: "cpu_local",
                PACKED_GATE_FLAG: "packed_gate"}
    if experiment_flag not in prefixes:
        raise ValueError(f"Unsupported decode-window experiment: {experiment_flag}")
    prefix = prefixes[experiment_flag]
    enabled_value = 4 if experiment_flag == EXACT_EXPERT_FLAG else 1
    mode_field = prefix + "_flag"
    window_tokens = (args.tokens - WARMUP_TOKENS) // len(WINDOW_PATTERN)
    result.update(
        window_samples=[],
        window_protocol={
            "experiment_flag": experiment_flag,
            "enabled_value": enabled_value,
            "warmup_tokens_per_request": WARMUP_TOKENS,
            "tokens_per_window": window_tokens,
            "pattern_even_requests": list(WINDOW_PATTERN),
            "pattern_odd_requests": [1 - mode for mode in WINDOW_PATTERN],
            "same_prompt_and_generated_tokens_required": True,
            "scope": "Complementary decode windows in repeated requests, not the standard 256-output table",
            "timing": "Window wall time excludes boundary flag changes and evidence writes",
        },
    )
    original_value = os.environ.get(experiment_flag)
    affinity = None
    if experiment_flag == CPU_LOCAL_FLAG:
        from cpu_local_windows import CpuLocalWindows
        affinity = CpuLocalWindows()
        result["window_protocol"]["cpu_affinity"] = affinity.metadata()

    def switch(mode):
        os.environ[experiment_flag] = str(mode)
        if affinity is not None:
            affinity.switch(mode)

    try:
        for context in args.contexts:
            prompt, offset = natural_text_prompt(pool, context, context)
            reference_tokens = None
            for repetition in range(args.repetitions):
                pattern = [mode ^ (repetition % 2) for mode in WINDOW_PATTERN]
                window_index = 0
                window_start = None
                result["active_phase"] = {
                    "context": context, "repetition": repetition, "phase": "prefill",
                }
                switch(0)
                persist()

                def observe(index, timestamp):
                    nonlocal window_index, window_start
                    if index == 0:
                        result["active_phase"]["phase"] = "excluded_decode_warmup"
                        persist()
                    if index == WARMUP_TOKENS // 2 - 1:
                        switch(enabled_value)
                    if index == WARMUP_TOKENS - 1:
                        switch(pattern[0] * enabled_value)
                        result["active_phase"].update(phase="measured_decode", window=0)
                        persist()
                        window_start = time.perf_counter()
                    elif index >= WARMUP_TOKENS and (
                            index + 1 - WARMUP_TOKENS) % window_tokens == 0:
                        if window_start is None:
                            raise RuntimeError("Decode window completed without a start timestamp")
                        seconds = timestamp - window_start
                        if seconds <= 0:
                            raise RuntimeError("Decode window has nonpositive elapsed time")
                        result["window_samples"].append({
                            "context": context, "repetition": repetition, "window": window_index,
                            mode_field: pattern[window_index] * enabled_value, "decode_tokens": window_tokens,
                            "logical_sequence_tokens_before": context + WARMUP_TOKENS +
                            window_index * window_tokens,
                            "decode_seconds": seconds, "decode_tps": window_tokens / seconds,
                        })
                        window_index += 1
                        result["active_phase"]["window"] = window_index
                        persist()
                        if window_index < len(pattern):
                            switch(pattern[window_index] * enabled_value)
                            window_start = time.perf_counter()

                sample = run_request(og, model, tokenizer, prompt, args.tokens, args.chunk,
                                     on_decode_step=observe)
                sample.update(context=context, repetition=repetition, seed=context, offset=offset,
                              mode="complementary_decode_windows", warmup=False)
                result["samples"].append(sample)
                if window_index != len(pattern):
                    raise RuntimeError(f"Expected eight completed windows, found {window_index}")
                if reference_tokens is None:
                    reference_tokens = sample["output_token_ids"]
                    sample["same_prompt_output_matches"] = None
                else:
                    sample["same_prompt_output_matches"] = sample["output_token_ids"] == reference_tokens
                    if not sample["same_prompt_output_matches"]:
                        persist()
                        raise RuntimeError("Complementary decode-window requests generated different tokens")
                    for previous in result["samples"]:
                        if previous["context"] == context:
                            previous["same_prompt_output_matches"] = True
                persist()

        result["window_summary"] = {}
        for context in args.contexts:
            rows = [row for row in result["window_samples"] if row["context"] == context]
            pairs = []
            for repetition in range(0, args.repetitions, 2):
                for window in range(len(WINDOW_PATTERN)):
                    pair = [row for row in rows if row["repetition"] in (repetition, repetition + 1)
                            and row["window"] == window]
                    if len(pair) != 2 or {row[mode_field] for row in pair} != {0, enabled_value}:
                        raise RuntimeError("Missing complementary decode-window pair")
                    control = next(row for row in pair if row[mode_field] == 0)
                    candidate = next(row for row in pair if row[mode_field] == enabled_value)
                    pairs.append(candidate["decode_tps"] / control["decode_tps"])
            result["window_summary"][str(context)] = {
                prefix + "0_median_window_tps": statistics.median(
                    row["decode_tps"] for row in rows if row[mode_field] == 0),
                prefix + str(enabled_value) + "_median_window_tps": statistics.median(
                    row["decode_tps"] for row in rows if row[mode_field] == enabled_value),
                "matched_window_pairs": len(pairs),
                "geometric_mean_matched_window_ratio": math.exp(
                    statistics.mean(math.log(ratio) for ratio in pairs)),
                "all_request_output_tokens_match": True,
                "per_request": [
                    {
                        "repetition": repetition,
                        prefix + "0_median_window_tps": statistics.median(
                            row["decode_tps"] for row in rows
                            if row["repetition"] == repetition and row[mode_field] == 0),
                        prefix + str(enabled_value) + "_median_window_tps": statistics.median(
                            row["decode_tps"] for row in rows
                            if row["repetition"] == repetition and row[mode_field] == enabled_value),
                    }
                    for repetition in range(args.repetitions)
                ],
            }
        result["active_phase"]["phase"] = "completed"
    finally:
        try:
            if affinity is not None:
                affinity.restore()
                result["window_protocol"]["cpu_affinity_restored"] = True
        finally:
            if original_value is None:
                os.environ.pop(experiment_flag, None)
            else:
                os.environ[experiment_flag] = original_value

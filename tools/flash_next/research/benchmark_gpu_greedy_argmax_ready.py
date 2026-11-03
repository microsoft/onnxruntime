import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_webgpu_fp8_checkpoint import natural_text_prompt, summarize, token_hash


def run_request(og, model, tokenizer, prompt, output_tokens, chunk, *,
                decode_profile_prefix=None, profile_after_tokens=0, on_decode_step=None):
    if decode_profile_prefix is not None and not 1 <= profile_after_tokens < output_tokens:
        raise ValueError("Decode profiling must start after at least one token and before completion")
    if decode_profile_prefix is None and profile_after_tokens:
        raise ValueError("A decode profile prefix is required")
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + output_tokens, min_length=len(prompt) + output_tokens,
                              do_sample=False, chunk_size=chunk)
    generator = og.Generator(model, params)
    emitted = []
    times = []
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    usage_first = None
    start = time.perf_counter()
    generator.append_tokens(prompt)
    prefill_finished = None
    for index in range(output_tokens):
        if generator.is_done():
            raise RuntimeError(f"INT4 Generator ended early at {index}/{output_tokens} tokens")
        if decode_profile_prefix is not None and index == profile_after_tokens:
            generator.set_runtime_option("enable_profiling", decode_profile_prefix)
        generator.generate_next_token()
        if index == 0:
            prefill_finished = time.perf_counter()
        emitted.append(int(generator.get_next_tokens()[0]))
        times.append(time.perf_counter())
        if index == 0:
            usage_first = resource.getrusage(resource.RUSAGE_SELF)
        if on_decode_step is not None:
            on_decode_step(index, times[-1])
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    if usage_first is None or prefill_finished is None:
        raise RuntimeError("No generated token resource-usage boundary")
    resource_usage = {}
    for phase, before, after in (("prefill", usage_start, usage_first),
                                 ("decode", usage_first, usage_end)):
        resource_usage[phase] = {
            "major_page_faults": after.ru_majflt - before.ru_majflt,
            "minor_page_faults": after.ru_minflt - before.ru_minflt,
            "user_cpu_seconds": after.ru_utime - before.ru_utime,
            "system_cpu_seconds": after.ru_stime - before.ru_stime,
            "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
            "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
        }
    sequence = list(generator.get_sequence(0))
    if sequence != list(prompt) + emitted:
        raise RuntimeError("Generator sequence does not match prompt plus emitted tokens")
    result = {
        "input_tokens": len(prompt), "output_tokens": len(emitted), "output_token_ids": emitted,
        "prompt_sha256": token_hash(prompt), "output_sha256": token_hash(emitted),
        "output_text": tokenizer.decode(emitted),
        "ttft_seconds": times[0] - start, "request_seconds": times[-1] - start,
        "prefill_seconds": prefill_finished - start,
        "prefill_tps": len(prompt) / (prefill_finished - start),
        "prefill_timing_scope": "append_tokens_through_first_token_selection",
        "decode_seconds": times[-1] - times[0],
        "decode_tps": (len(emitted) - 1) / (times[-1] - times[0]),
        "process_resource_usage": resource_usage,
    }
    if decode_profile_prefix is not None:
        result.update(decode_profile_prefix=decode_profile_prefix,
                      profile_after_tokens=profile_after_tokens,
                      profiled_decode_tokens=output_tokens - profile_after_tokens)
    del generator
    gc.collect()
    return result


def main():
    import onnxruntime_genai as og

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/qwen38-int4-webgpu-perf")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--provider", choices=["webgpu", "cuda"], default="webgpu")
    parser.add_argument("--webgpu-validation-mode", choices=["disabled", "full"])
    parser.add_argument("--webgpu-max-pending-dispatches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inherit-worker-cpu-affinity", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--profile-prefix")
    parser.add_argument("--profile-decode-after", type=int)
    comparisons = parser.add_mutually_exclusive_group()
    comparisons.add_argument("--compare-experiments", action="store_true")
    comparisons.add_argument("--compare-skinny", action="store_true")
    comparisons.add_argument("--compare-hyper-skinny", action="store_true")
    comparisons.add_argument("--compare-uniform-arena", action="store_true")
    comparisons.add_argument("--compare-attention-split", action="store_true")
    comparisons.add_argument("--compare-qsa-rank", action="store_true")
    comparisons.add_argument("--compare-gemv-n96", action="store_true")
    comparisons.add_argument("--compare-subgroup-gate", action="store_true")
    comparisons.add_argument("--compare-bind-group-cache", action="store_true")
    comparisons.add_argument("--compare-dynamic-uniform-cache", action="store_true")
    comparisons.add_argument("--compare-qsa-rank-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-dense-map-windows", action="store_true")
    comparisons.add_argument("--compare-exact-mid-windows", action="store_true")
    comparisons.add_argument("--compare-exact-wide-windows", action="store_true")
    comparisons.add_argument("--compare-exact-expert-windows", action="store_true")
    comparisons.add_argument("--compare-branch-norm-windows", action="store_true")
    comparisons.add_argument("--compare-large-cache-windows", action="store_true")
    comparisons.add_argument("--compare-gdn-view-windows", action="store_true")
    comparisons.add_argument("--compare-node-replay-windows", action="store_true")
    comparisons.add_argument("--compare-static-replay-windows", action="store_true")
    comparisons.add_argument("--compare-hc-projection-windows", action="store_true")
    comparisons.add_argument("--compare-hc-up-premix-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-parallel-map-windows", action="store_true")
    comparisons.add_argument("--compare-generic-matmul-windows", action="store_true")
    comparisons.add_argument("--compare-direct-a-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-activation-windows", action="store_true")
    comparisons.add_argument("--compare-replay-immediate-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-final-mix-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-padded-partials-windows", action="store_true")
    comparisons.add_argument("--compare-hc-projection-tiled-windows", action="store_true")
    comparisons.add_argument("--compare-subgroup-gate-windows", action="store_true")
    comparisons.add_argument("--compare-qmoe-activation-padding-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-four-columns-windows", action="store_true")
    comparisons.add_argument("--compare-hierarchical-gate-windows", action="store_true")
    comparisons.add_argument("--compare-mlp-bitcast-unpack-windows", action="store_true")
    comparisons.add_argument("--compare-expert-bitcast-unpack-windows", action="store_true")
    comparisons.add_argument("--compare-dense-bitcast-unpack-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-bitcast-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-fused-sum-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-subgroup-windows", action="store_true")
    comparisons.add_argument("--compare-hc-up-bitcast-unpack-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-concat-vector-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-packed-score-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-indexer-stack-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-score16-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-small-query-cache-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-four-columns-windows", action="store_true")
    comparisons.add_argument("--compare-hc-up-tiled-windows", action="store_true")
    comparisons.add_argument("--compare-hc-up-parallel2-windows", action="store_true")
    comparisons.add_argument("--compare-hc-up-parallel4-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-pad34-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-vec4-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-predecode-windows", action="store_true")
    comparisons.add_argument("--compare-hc-post-norm-wide16-windows", action="store_true")
    comparisons.add_argument("--compare-hc-cached-norm-windows", action="store_true")
    comparisons.add_argument("--compare-hc-quad-fold-windows", action="store_true")
    comparisons.add_argument("--compare-hc-narrow4-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-warp-fold-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-paired-dot-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-subgroup-fold-windows", action="store_true")
    comparisons.add_argument("--compare-fc2-compact20-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-experts10-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-full-a-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-columns8-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-paired-dot-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-k16-tile-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-k8-tile-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-u32-partials-windows", action="store_true")
    comparisons.add_argument("--compare-fc1-expert-pairs-windows", action="store_true")
    comparisons.add_argument("--compare-dense-static-shape-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-pruned-rank-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-hist-rank-windows", action="store_true")
    comparisons.add_argument("--compare-qsa-subgroup-rank-windows", action="store_true")
    comparisons.add_argument("--compare-hc-projection-shuffle-windows", action="store_true")
    comparisons.add_argument("--compare-hc-staged-projection-windows", action="store_true")
    comparisons.add_argument("--compare-dense-tile4-windows", action="store_true")
    comparisons.add_argument("--compare-dense-tile2-windows", action="store_true")
    comparisons.add_argument("--compare-dense-wide8-windows", action="store_true")
    comparisons.add_argument("--compare-dense-preload-windows", action="store_true")
    comparisons.add_argument("--compare-dense-double-buffer-windows", action="store_true")
    comparisons.add_argument("--compare-dense-block32-windows", action="store_true")
    comparisons.add_argument("--compare-dense-direct-a-windows", action="store_true")
    comparisons.add_argument("--compare-dense-predecode-windows", action="store_true")
    comparisons.add_argument("--compare-dense-private-partials-windows", action="store_true")
    comparisons.add_argument("--compare-dense-pad-a-windows", action="store_true")
    comparisons.add_argument("--compare-dense-qkv-wide16-windows", action="store_true")
    comparisons.add_argument("--compare-dense-qkv-k16-windows", action="store_true")
    comparisons.add_argument("--compare-dense-qkv-predecode8-windows", action="store_true")
    comparisons.add_argument("--compare-dense-qkv-threads64-windows", action="store_true")
    comparisons.add_argument("--compare-dense-qkv-pad34-windows", action="store_true")
    comparisons.add_argument("--compare-gdn-value2-windows", action="store_true")
    comparisons.add_argument("--compare-gdn-value8-windows", action="store_true")
    comparisons.add_argument("--compare-gdn-tree32-windows", action="store_true")
    comparisons.add_argument("--compare-gdn-norm32-windows", action="store_true")
    comparisons.add_argument("--compare-qmoe-fast-flags-windows", action="store_true")
    comparisons.add_argument("--compare-dsa-partition32-windows", action="store_true")
    comparisons.add_argument("--compare-dsa-query-cache-windows", action="store_true")
    comparisons.add_argument("--compare-dsa-paired-windows", action="store_true")
    comparisons.add_argument("--compare-cpu-local-windows", action="store_true")
    comparisons.add_argument("--compare-packed-gate-windows", action="store_true")
    parser.add_argument("--quality-output", type=Path)
    parser.add_argument("--compare-contexts", type=int, nargs="+")
    args = parser.parse_args()
    if args.webgpu_validation_mode is not None and args.provider != "webgpu":
        parser.error("--webgpu-validation-mode requires the WebGPU provider")
    if args.webgpu_max_pending_dispatches is not None:
        if args.provider != "webgpu":
            parser.error("--webgpu-max-pending-dispatches requires the WebGPU provider")
        if not 1 <= args.webgpu_max_pending_dispatches <= 4096:
            parser.error("--webgpu-max-pending-dispatches must be between 1 and 4096")
    window_comparison = (args.compare_qsa_rank_windows or args.compare_qsa_dense_map_windows or
                         args.compare_exact_mid_windows or args.compare_exact_wide_windows or
                         args.compare_exact_expert_windows or args.compare_branch_norm_windows or
                         args.compare_large_cache_windows or args.compare_gdn_view_windows or
                         args.compare_node_replay_windows or args.compare_static_replay_windows or
                         args.compare_hc_projection_windows or args.compare_hc_up_premix_windows or
                         args.compare_qsa_parallel_map_windows or args.compare_generic_matmul_windows or
                         args.compare_direct_a_windows or args.compare_fc1_activation_windows or
                         args.compare_replay_immediate_windows or args.compare_fc2_final_mix_windows or
                         args.compare_fc2_padded_partials_windows or args.compare_hc_projection_tiled_windows or
                         args.compare_subgroup_gate_windows or args.compare_qmoe_activation_padding_windows or
                         args.compare_hc_post_norm_four_columns_windows or args.compare_hierarchical_gate_windows or
                         args.compare_mlp_bitcast_unpack_windows or args.compare_expert_bitcast_unpack_windows or
                         args.compare_dense_bitcast_unpack_windows or args.compare_hc_post_norm_bitcast_windows or
                         args.compare_hc_post_norm_fused_sum_windows or
                         args.compare_hc_post_norm_subgroup_windows or
                         args.compare_hc_up_bitcast_unpack_windows or
                         args.compare_qsa_concat_vector_windows or
                         args.compare_qsa_packed_score_windows or
                         args.compare_qsa_indexer_stack_windows or
                         args.compare_qsa_score16_windows or
                         args.compare_qsa_small_query_cache_windows or
                         args.compare_fc2_four_columns_windows or
                         args.compare_hc_up_tiled_windows or args.compare_hc_up_parallel2_windows or
                         args.compare_hc_up_parallel4_windows or
                         args.compare_hc_post_norm_pad34_windows or
                         args.compare_hc_post_norm_vec4_windows or
                         args.compare_hc_post_norm_predecode_windows or
                         args.compare_hc_post_norm_wide16_windows or
                         args.compare_hc_cached_norm_windows or
                         args.compare_hc_quad_fold_windows or
                         args.compare_hc_narrow4_windows or
                         args.compare_fc1_warp_fold_windows or
                         args.compare_fc2_paired_dot_windows or
                         args.compare_fc2_subgroup_fold_windows or
                         args.compare_fc2_compact20_windows or
                         args.compare_fc1_experts10_windows or
                         args.compare_fc1_full_a_windows or
                         args.compare_fc1_columns8_windows or
                         args.compare_fc1_paired_dot_windows or
                         args.compare_fc1_k16_tile_windows or
                         args.compare_fc1_k8_tile_windows or
                         args.compare_fc1_u32_partials_windows or
                         args.compare_fc1_expert_pairs_windows or
                         args.compare_qsa_hist_rank_windows or
                         args.compare_hc_projection_shuffle_windows or
                         args.compare_hc_staged_projection_windows or
                         args.compare_dense_tile4_windows or
                         args.compare_dense_tile2_windows or
                         args.compare_dense_wide8_windows or
                         args.compare_dense_preload_windows or
                         args.compare_dense_double_buffer_windows or
                         args.compare_dense_block32_windows or
                         args.compare_dense_direct_a_windows or
                         args.compare_dense_predecode_windows or
                         args.compare_dense_private_partials_windows or
                         args.compare_dense_pad_a_windows or
                         args.compare_dense_qkv_wide16_windows or
                         args.compare_dense_qkv_k16_windows or
                         args.compare_dense_qkv_predecode8_windows or
                         args.compare_dense_qkv_threads64_windows or
                         args.compare_dense_qkv_pad34_windows or
                         args.compare_dense_static_shape_windows or
                         args.compare_qsa_pruned_rank_windows or
                         args.compare_qsa_subgroup_rank_windows or
                         args.compare_gdn_value2_windows or
                         args.compare_gdn_value8_windows or
                         args.compare_gdn_tree32_windows or
                         args.compare_gdn_norm32_windows or
                         args.compare_qmoe_fast_flags_windows or
                         args.compare_dsa_partition32_windows or
                         args.compare_dsa_query_cache_windows or
                         args.compare_dsa_paired_windows or
                         args.compare_cpu_local_windows or
                         args.compare_packed_gate_windows)
    single_flag_comparison = (args.compare_skinny or args.compare_hyper_skinny or args.compare_uniform_arena or
                              args.compare_attention_split or args.compare_qsa_rank or args.compare_gemv_n96 or
                              args.compare_subgroup_gate or args.compare_bind_group_cache or
                              args.compare_dynamic_uniform_cache or window_comparison)
    compare = args.compare_experiments or single_flag_comparison
    if args.tokens < 2 or args.repetitions < 1 or args.chunk < 1 or min(args.contexts) < 1:
        parser.error("Positive contexts, chunk and repetitions, and at least two output tokens are required")
    if compare and args.profile_prefix:
        parser.error("Counterbalanced performance comparisons must not enable profiling")
    if args.profile_decode_after is not None and (
            not args.profile_prefix or not 1 <= args.profile_decode_after < args.tokens):
        parser.error("--profile-decode-after requires --profile-prefix and must be at least one and below --tokens")
    if compare and args.provider != "webgpu":
        parser.error("WebGPU experiment comparisons require the WebGPU provider")
    if args.compare_contexts and (not compare or
                                  not set(args.compare_contexts).issubset(args.contexts)):
        parser.error("--compare-contexts must be measured contexts and requires a comparison")
    if window_comparison and (
            args.repetitions % 2 or args.tokens < 40 or (args.tokens - 32) % 8 or args.compare_contexts):
        parser.error("Decode windows require even repetitions, 32 warmup tokens plus eight equal windows, "
                     "and no --compare-contexts")
    comparison_contexts = set(args.compare_contexts or args.contexts)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "natural-pool-manifest.json").read_text())
    text = (root / "alice-natural-pool.txt").read_bytes()
    tokenizer_path = Path(args.model) / "tokenizer.json"
    result = {
        "status": "running", "quantization": "Original INT4 RTN block32 MoE; INT4 dense, INT8 final matmul",
        "runtime": "Generator, no MTP, non-paged attention; explicit CPU embedding layer",
        "provider": args.provider, "webgpu_experiments_applicable": args.provider == "webgpu",
        "webgpu_validation_mode_override": args.webgpu_validation_mode,
        "webgpu_max_pending_dispatches_override": args.webgpu_max_pending_dispatches,
        "provider_registration": "external_plugin" if args.provider == "webgpu" else "builtin_cuda",
        "python_version": sys.version,
        "model": args.model, "chunk": args.chunk, "profiling": bool(args.profile_prefix),
        "profiling_scope": ("decode_only" if args.profile_decode_after is not None else "session")
        if args.profile_prefix else None,
        "profile_decode_after_tokens": args.profile_decode_after,
        "headline_eligible": False,
        "quality_qualified": False, "not_exact_tianlei_reproduction": True, "samples": [],
        "comparison": ("large_cache_decode_windows" if args.compare_large_cache_windows else
                       "exact_expert_decode_windows" if args.compare_exact_expert_windows else
                       "branch_norm_decode_windows" if args.compare_branch_norm_windows else
                       "exact_wide_decode_windows" if args.compare_exact_wide_windows else
                       "exact_mid_decode_windows" if args.compare_exact_mid_windows else
                       "qsa_dense_map_decode_windows" if args.compare_qsa_dense_map_windows else
                       "hyper_skinny_only" if args.compare_hyper_skinny else
                       "dynamic_uniform_cache_bundle" if args.compare_dynamic_uniform_cache else
                       "bind_group_cache_only" if args.compare_bind_group_cache else
                       "qmoe_subgroup_gate_only" if args.compare_subgroup_gate else
                       "gemv_n96_only" if args.compare_gemv_n96 else
                       "qmoe_node_replay_decode_windows" if args.compare_node_replay_windows else
                       "static_node_replay_decode_windows" if args.compare_static_replay_windows else
                       "hc_projection_silu_decode_windows" if args.compare_hc_projection_windows else
                       "hc_up_premix_decode_windows" if args.compare_hc_up_premix_windows else
                       "qsa_parallel_map_decode_windows" if args.compare_qsa_parallel_map_windows else
                       "generic_matmul_replay_decode_windows" if args.compare_generic_matmul_windows else
                       "direct_a_matmul_decode_windows" if args.compare_direct_a_windows else
                       "qmoe_activation_padding_decode_windows" if args.compare_qmoe_activation_padding_windows else
                       "qmoe_packed_gate_decode_windows" if args.compare_packed_gate_windows else
                       "hc_up_tiled_decode_windows" if args.compare_hc_up_tiled_windows else
                       "hc_up_parallel2_decode_windows" if args.compare_hc_up_parallel2_windows else
                       "hc_up_parallel4_decode_windows" if args.compare_hc_up_parallel4_windows else
                       "hc_post_norm_pad34_decode_windows" if args.compare_hc_post_norm_pad34_windows else
                       "hc_post_norm_vec4_decode_windows" if args.compare_hc_post_norm_vec4_windows else
                       "hc_post_norm_predecode_decode_windows" if args.compare_hc_post_norm_predecode_windows else
                       "hc_post_norm_wide16_decode_windows" if args.compare_hc_post_norm_wide16_windows else
                       "hc_cached_norm_decode_windows" if args.compare_hc_cached_norm_windows else
                       "hc_quad_fold_decode_windows" if args.compare_hc_quad_fold_windows else
                       "hc_narrow4_decode_windows" if args.compare_hc_narrow4_windows else
                       "fc1_warp_fold_decode_windows" if args.compare_fc1_warp_fold_windows else
                       "fc2_paired_dot_decode_windows" if args.compare_fc2_paired_dot_windows else
                       "fc2_subgroup_fold_decode_windows" if args.compare_fc2_subgroup_fold_windows else
                       "fc2_compact20_decode_windows" if args.compare_fc2_compact20_windows else
                       "fc1_experts10_decode_windows" if args.compare_fc1_experts10_windows else
                       "fc1_full_a_decode_windows" if args.compare_fc1_full_a_windows else
                       "fc1_columns8_decode_windows" if args.compare_fc1_columns8_windows else
                       "fc1_paired_dot_decode_windows" if args.compare_fc1_paired_dot_windows else
                       "fc1_k16_tile_decode_windows" if args.compare_fc1_k16_tile_windows else
                       "fc1_k8_tile_decode_windows" if args.compare_fc1_k8_tile_windows else
                       "fc1_u32_partials_decode_windows" if args.compare_fc1_u32_partials_windows else
                       "fc1_expert_pairs_decode_windows" if args.compare_fc1_expert_pairs_windows else
                       "qsa_hist_rank_decode_windows" if args.compare_qsa_hist_rank_windows else
                       "hc_projection_shuffle_decode_windows" if args.compare_hc_projection_shuffle_windows else
                       "hc_staged_projection_decode_windows" if args.compare_hc_staged_projection_windows else
                       "dense_tile4_decode_windows" if args.compare_dense_tile4_windows else
                       "dense_tile2_decode_windows" if args.compare_dense_tile2_windows else
                       "dense_wide8_decode_windows" if args.compare_dense_wide8_windows else
                       "dense_preload_decode_windows" if args.compare_dense_preload_windows else
                       "dense_double_buffer_decode_windows" if args.compare_dense_double_buffer_windows else
                       "dense_block32_decode_windows" if args.compare_dense_block32_windows else
                       "dense_direct_a_decode_windows" if args.compare_dense_direct_a_windows else
                       "dense_predecode_decode_windows" if args.compare_dense_predecode_windows else
                       "dense_private_partials_decode_windows" if args.compare_dense_private_partials_windows else
                       "dense_pad_a_decode_windows" if args.compare_dense_pad_a_windows else
                       "dense_qkv_wide16_decode_windows" if args.compare_dense_qkv_wide16_windows else
                       "dense_qkv_k16_decode_windows" if args.compare_dense_qkv_k16_windows else
                       "dense_qkv_predecode8_decode_windows" if args.compare_dense_qkv_predecode8_windows else
                       "dense_qkv_threads64_decode_windows" if args.compare_dense_qkv_threads64_windows else
                       "dense_qkv_pad34_decode_windows" if args.compare_dense_qkv_pad34_windows else
                       "dense_static_shape_decode_windows" if args.compare_dense_static_shape_windows else
                       "qsa_pruned_rank_decode_windows" if args.compare_qsa_pruned_rank_windows else
                       "qsa_subgroup_rank_decode_windows" if args.compare_qsa_subgroup_rank_windows else
                       "gdn_value2_decode_windows" if args.compare_gdn_value2_windows else
                       "gdn_value8_decode_windows" if args.compare_gdn_value8_windows else
                       "gdn_tree32_decode_windows" if args.compare_gdn_tree32_windows else
                       "gdn_norm32_decode_windows" if args.compare_gdn_norm32_windows else
                       "qmoe_fast_flags_decode_windows" if args.compare_qmoe_fast_flags_windows else
                       "dsa_partition32_decode_windows" if args.compare_dsa_partition32_windows else
                       "dsa_query_cache_decode_windows" if args.compare_dsa_query_cache_windows else
                       "dsa_paired_decode_windows" if args.compare_dsa_paired_windows else
                       "cpu_local_decode_windows" if args.compare_cpu_local_windows else
                       "fc2_four_columns_decode_windows" if args.compare_fc2_four_columns_windows else
                       "dense_bitcast_unpack_decode_windows" if args.compare_dense_bitcast_unpack_windows else
                       "hc_post_norm_bitcast_decode_windows" if args.compare_hc_post_norm_bitcast_windows else
                       "hc_post_norm_fused_sum_decode_windows" if args.compare_hc_post_norm_fused_sum_windows else
                       "hc_post_norm_subgroup_decode_windows" if args.compare_hc_post_norm_subgroup_windows else
                       "hc_up_bitcast_unpack_decode_windows" if args.compare_hc_up_bitcast_unpack_windows else
                       "qsa_concat_vector_decode_windows" if args.compare_qsa_concat_vector_windows else
                       "qsa_packed_score_decode_windows" if args.compare_qsa_packed_score_windows else
                       "qsa_indexer_stack_decode_windows" if args.compare_qsa_indexer_stack_windows else
                       "qsa_score16_decode_windows" if args.compare_qsa_score16_windows else
                       "qsa_small_query_cache_decode_windows" if args.compare_qsa_small_query_cache_windows else
                       "expert_bitcast_unpack_decode_windows" if args.compare_expert_bitcast_unpack_windows else
                       "mlp_bitcast_unpack_decode_windows" if args.compare_mlp_bitcast_unpack_windows else
                       "qmoe_hierarchical_gate_decode_windows" if args.compare_hierarchical_gate_windows else
                       "qmoe_subgroup_gate_decode_windows" if args.compare_subgroup_gate_windows else
                       "hc_projection_tiled_decode_windows" if args.compare_hc_projection_tiled_windows else
                       "qmoe_fc2_padded_partials_decode_windows" if args.compare_fc2_padded_partials_windows else
                       "qmoe_fc2_final_mix_decode_windows" if args.compare_fc2_final_mix_windows else
                       "node_replay_immediate_decode_windows" if args.compare_replay_immediate_windows else
                       "qmoe_fc1_activation_decode_windows" if args.compare_fc1_activation_windows else
                       "qsa_parallel_rank_decode_windows" if args.compare_qsa_rank_windows else
                       "qsa_parallel_rank_only" if args.compare_qsa_rank else
                       "uniform_arena_only" if args.compare_uniform_arena else
                       "attention_split_only" if args.compare_attention_split else
                       "skinny_only" if args.compare_skinny else "all_experiments" if compare else None),
        "plugin_sha256": hashlib.sha256(Path(args.plugin).read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256((Path(args.model) / "model.onnx").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((Path(args.model) / "genai_config.json").read_bytes()).hexdigest(),
        "prompt_pool_manifest": manifest,
        "load_experiment_flags": {
            name: os.environ.get(name, "0") for name in (
                "ORT_WEBGPU_QMOE_INT4_PREFILL", "ORT_WEBGPU_QMOE_INT4_COOPERATIVE",
                "ORT_WEBGPU_QMOE_EXTENDED_PREFILL",
                "ORT_WEBGPU_QMOE_PREFILL256",
                "ORT_WEBGPU_QMOE_PREFILL512",
                "ORT_WEBGPU_QMOE_PREFILL1024",
                "ORT_WEBGPU_QMOE_PREFILL1280",
                "ORT_WEBGPU_QMOE_PREFILL2048",
                "ORT_WEBGPU_QMOE_PREFILL2048_ROWS4",
                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",
                "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER",
                "ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8",
                "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512",
                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX",
                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE",
                "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE",
                "ORT_WEBGPU_QMOE_FC1_WARP_FOLD",
                "ORT_WEBGPU_QMOE_FC2_PAIRED_DOT",
                "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD",
                "ORT_WEBGPU_QMOE_FC2_COMPACT20",
                "ORT_WEBGPU_QMOE_FC1_EXPERTS10",
                "ORT_WEBGPU_QMOE_FC1_FULL_A",
                "ORT_WEBGPU_QMOE_FC1_COLUMNS8",
                "ORT_WEBGPU_QMOE_PREFILL_ROWS8",
                "ORT_WEBGPU_QMOE_FC1_PAIRED_DOT",
                "ORT_WEBGPU_QMOE_FC1_K16_TILE",
                "ORT_WEBGPU_QMOE_FC1_K8_TILE",
                "ORT_WEBGPU_QMOE_FC1_U32_PARTIALS",
                "ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS",
                "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL", "ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER",
                "ORT_WEBGPU_SMALL_READBACK_CACHE", "ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE",
                "ORT_WEBGPU_MATMUL_NBITS_SKINNY", "ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", "ORT_WEBGPU_UNIFORM_ARENA",
                "ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP", "ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV",
                "ORT_WEBGPU_DSA_PARTITION32",
                "ORT_WEBGPU_DSA_QUERY_CACHE",
                "ORT_WEBGPU_DSA_PAIRED",
                "ORT_WEBGPU_DENSE_DOUBLE_BUFFER",
                "ORT_WEBGPU_DENSE_BLOCK32",
                "ORT_WEBGPU_DENSE_PREDECODE",
                "ORT_WEBGPU_DENSE_DIRECT_A",
                "ORT_WEBGPU_DENSE_PRIVATE_PARTIALS",
                "ORT_WEBGPU_DENSE_PAD_A",
                "ORT_WEBGPU_DENSE_QKV_WIDE16",
                "ORT_WEBGPU_DENSE_QKV_K16",
                "ORT_WEBGPU_DENSE_QKV_PREDECODE8",
                "ORT_WEBGPU_DENSE_QKV_THREADS64",
                "ORT_WEBGPU_DENSE_QKV_PAD34",
                "ORT_WEBGPU_DENSE_STATIC_SHAPE",
                "ORT_WEBGPU_QSA_PRUNED_RANK",
                "ORT_WEBGPU_QSA_SUBGROUP_RANK",
                "ORT_WEBGPU_QSA_PREFILL_SCORE16",
                "ORT_WEBGPU_QSA_SMALL_QUERY_CACHE",
                "ORT_WEBGPU_HC_CACHED_NORM",
                "ORT_WEBGPU_HC_QUAD_FOLD",
                "ORT_WEBGPU_HC_NARROW4",
                "ORT_WEBGPU_QSA_HIST_RANK",
                "ORT_WEBGPU_GDN_VALUE2",
                "ORT_WEBGPU_GDN_VALUE8",
                "ORT_WEBGPU_GDN_TREE32",
                "ORT_WEBGPU_GDN_NORM32",
                "ORT_WEBGPU_QSA_PARALLEL_RANK", "ORT_WEBGPU_QSA_DENSE_MAP",
                "ORT_WEBGPU_MATMUL_GEMV_N96", "ORT_WEBGPU_QMOE_SUBGROUP_GATE",
                "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "ORT_WEBGPU_DYNAMIC_UNIFORMS",
                "ORT_WEBGPU_HC_EXACT_DENSE", "ORT_WEBGPU_EXACT_DENSE_MID",
                "ORT_WEBGPU_EXACT_DENSE_WIDE", "ORT_WEBGPU_EXACT_EXPERT",
                "ORT_WEBGPU_EXACT_BRANCH_NORM", "ORT_WEBGPU_BIND_GROUP_CACHE_LARGE",
                "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW", "ORT_WEBGPU_QMOE_COMMAND_REPLAY",
                "ORT_WEBGPU_STATIC_NODE_REPLAY", "ORT_WEBGPU_HC_PROJECTION_SILU",
                "ORT_WEBGPU_HC_UP_PREMIX", "ORT_WEBGPU_QSA_PARALLEL_MAP", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY",
                "ORT_WEBGPU_MATMUL_NBITS_DIRECT_A", "ORT_WEBGPU_QMOE_FC1_ACTIVATION",
                "ORT_WEBGPU_NODE_REPLAY_IMMEDIATE", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX",
                "ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS", "ORT_WEBGPU_HC_PROJECTION_TILED",
                "ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS", "ORT_WEBGPU_HC_POST_NORM_PROJECTION",
                "ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE",
                "ORT_WEBGPU_MLP_INPUT_PROJECTIONS", "ORT_WEBGPU_MLP_BITCAST_UNPACK",
                "ORT_WEBGPU_EXPERT_BITCAST_UNPACK", "ORT_WEBGPU_DENSE_BITCAST_UNPACK",
                "ORT_WEBGPU_HC_POST_NORM_BITCAST",
                "ORT_WEBGPU_HC_POST_NORM_FUSED_SUM",
                "ORT_WEBGPU_HC_POST_NORM_SUBGROUP",
                "ORT_WEBGPU_HC_UP_BITCAST_UNPACK",
                "ORT_WEBGPU_QSA_CONCAT_VECTOR",
                "ORT_WEBGPU_QSA_PACKED_SCORE",
                "ORT_WEBGPU_QSA_INDEXER_STACK",
                "ORT_WEBGPU_QSA_SCORE16",
                "ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS", "ORT_WEBGPU_HC_UP_TILED",
                "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_POST_NORM_PAD34",
                "ORT_WEBGPU_HC_POST_NORM_VEC4",
                "ORT_WEBGPU_HC_POST_NORM_PREDECODE",
                "ORT_WEBGPU_HC_POST_NORM_WIDE16",
                "ORT_WEBGPU_HC_UP_PARALLEL4",
                "ORT_WEBGPU_HC_PROJECTION_SHUFFLE",
                "ORT_WEBGPU_HC_STAGED_PROJECTION",
                "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_DENSE_TILE2", "ORT_WEBGPU_DENSE_WIDE8",
                "ORT_WEBGPU_DENSE_PRELOAD",
                "ORT_WEBGPU_QMOE_FAST_FLAGS",
                "ORT_WEBGPU_QMOE_PACKED_GATE")
        },
    }

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        native_paths = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if line.endswith("/libonnxruntime-genai.so")}
        if len(native_paths) != 1:
            raise RuntimeError(f"Expected exactly one mapped GenAI native library, found {sorted(native_paths)}")
        native_path = Path(next(iter(native_paths)))
        result["genai_native"] = {
            "path": str(native_path), "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        }
        ort_paths = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if line.split(maxsplit=5)[-1].rsplit("/", 1)[-1].startswith("libonnxruntime.so")}
        if len(ort_paths) != 1:
            raise RuntimeError(f"Expected exactly one ORT core library, found {sorted(ort_paths)}")
        ort_path = Path(next(iter(ort_paths)))
        result["ort_native"] = {
            "path": str(ort_path), "sha256": hashlib.sha256(ort_path.read_bytes()).hexdigest(),
        }
        if hashlib.sha256(text).hexdigest() != manifest["pool_sha256"]:
            raise ValueError("Natural-text pool changed")
        if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer changed")
        pool = og.Tokenizer(args.model).encode(text.decode("utf-8"))
        if len(pool) != manifest["pool_tokens"] or token_hash(pool) != manifest["tokens_sha256"]:
            raise ValueError("GenAI tokenization differs from frozen pool")
        for context in args.contexts:
            prompts = [natural_text_prompt(pool, context, seed)[0] for seed in (
                [1000 + context] + [context + 7 * i for i in range(args.repetitions)]
            )]
            if len({token_hash(prompt) for prompt in prompts}) != len(prompts):
                raise ValueError(f"Duplicate frozen prompts at context {context}")
        if args.provider == "webgpu":
            og.register_execution_provider_library("webgpu", args.plugin)
        config = og.Config(args.model)
        if args.webgpu_validation_mode is not None:
            config.set_provider_option("webgpu", "validationMode", args.webgpu_validation_mode)
        if args.webgpu_max_pending_dispatches is not None:
            config.set_provider_option("webgpu", "maxNumPendingDispatches",
                                       str(args.webgpu_max_pending_dispatches))
        if args.provider == "cuda":
            config.clear_providers()
            config.append_provider("cuda")
            config.set_provider_option("cuda", "enable_cuda_graph", "0")
            config.set_provider_option("cuda", "use_tf32", "0")
        if args.profile_prefix and args.profile_decode_after is None:
            config.overlay(json.dumps({"model": {"decoder": {
                "session_options": {"enable_profiling": args.profile_prefix},
            }}}))
        persist()
        print(json.dumps({"stage": "model_load", "model": args.model}), flush=True)
        start = time.perf_counter()
        model = og.Model(config)
        result["load_seconds"] = time.perf_counter() - start
        if args.provider == "cuda":
            cuda_paths = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                          if line.endswith("/libonnxruntime-genai-cuda.so")}
            if len(cuda_paths) != 1:
                raise RuntimeError(f"Expected exactly one CUDA GenAI add-on, found {sorted(cuda_paths)}")
            cuda_path = Path(next(iter(cuda_paths)))
            result["cuda_genai_native"] = {
                "path": str(cuda_path), "sha256": hashlib.sha256(cuda_path.read_bytes()).hexdigest(),
            }
            provider_paths = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                              if line.endswith("/libonnxruntime_providers_cuda.so")}
            if len(provider_paths) != 1:
                raise RuntimeError(f"Expected exactly one CUDA provider, found {sorted(provider_paths)}")
            provider_path = Path(next(iter(provider_paths)))
            result["cuda_provider_native"] = {
                "path": str(provider_path), "sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            }
            if result["cuda_provider_native"]["sha256"] != result["plugin_sha256"]:
                raise RuntimeError("Loaded CUDA provider differs from the requested native binary")
        tokenizer = og.Tokenizer(model)
        persist()
        print(json.dumps({"stage": "model_loaded", "seconds": result["load_seconds"]}), flush=True)
        if args.inherit_worker_cpu_affinity:
            if not hasattr(os, "sched_getaffinity"):
                raise RuntimeError("Worker CPU affinity requires Linux sched_getaffinity support")
            from cpu_local_windows import inherit_worker_affinity
            result["worker_affinity_override"] = inherit_worker_affinity()
        if args.quality_output:
            from check_int4_chat_quality import run_suite

            template = (Path(args.model) / "chat_template.jinja").read_text()
            eos_tokens = json.loads((Path(args.model) / "genai_config.json").read_text())["model"]["eos_token_id"]
            if isinstance(eos_tokens, int):
                eos_tokens = [eos_tokens]
            quality_result = {
                key: result[key] for key in (
                    "model", "provider", "plugin_sha256", "graph_sha256", "config_sha256",
                    "genai_native", "ort_native", "load_experiment_flags",
                    "webgpu_validation_mode_override",
                    "webgpu_max_pending_dispatches_override",
                )
            }
            quality_result.update(
                status="running", samples=[], model_reused_for_throughput=True,
                quality_qualified=False, headline_eligible=False,
                chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                scope="Three short deterministic sanity prompts, not a model-quality benchmark",
            )

            def persist_quality():
                args.quality_output.write_text(json.dumps(quality_result, indent=2) + "\n")

            def on_quality_sample(sample):
                quality_result["samples"].append(sample)
                persist_quality()
                print(json.dumps({"stage": "quality", **sample}), flush=True)

            persist_quality()
            try:
                run_suite(og, model, tokenizer, template, eos_tokens, on_quality_sample)
            except Exception as error:
                quality_result.update(status="failed", error=str(error))
                persist_quality()
                raise
            quality_result["status"] = "completed"
            persist_quality()
        if window_comparison:
            from qsa_rank_decode_windows import (
                BRANCH_NORM_FLAG, DENSE_MAP_FLAG, EXACT_EXPERT_FLAG, EXACT_MID_FLAG,
                EXACT_WIDE_FLAG, GDN_VIEW_FLAG, LARGE_CACHE_FLAG, NODE_REPLAY_FLAG, RANK_FLAG,
                STATIC_REPLAY_FLAG, HC_PROJECTION_FLAG, HC_UP_PREMIX_FLAG, PARALLEL_MAP_FLAG,
                GENERIC_MATMUL_FLAG, DIRECT_A_FLAG, FC1_ACTIVATION_FLAG, REPLAY_IMMEDIATE_FLAG,
                FC2_FINAL_MIX_FLAG, FC2_PADDED_PARTIALS_FLAG, HC_PROJECTION_TILED_FLAG,
                SUBGROUP_GATE_FLAG, ACTIVATION_PADDING_FLAG, POST_NORM_FOUR_COLUMNS_FLAG,
                HIERARCHICAL_GATE_FLAG, MLP_BITCAST_UNPACK_FLAG, EXPERT_BITCAST_UNPACK_FLAG,
                DENSE_BITCAST_UNPACK_FLAG, HC_POST_NORM_BITCAST_FLAG, FC2_FOUR_COLUMNS_FLAG,
                HC_POST_NORM_FUSED_SUM_FLAG, HC_POST_NORM_SUBGROUP_FLAG,
                HC_UP_BITCAST_UNPACK_FLAG, HC_UP_TILED_FLAG, HC_UP_PARALLEL2_FLAG,
                HC_UP_PARALLEL4_FLAG,
                HC_POST_NORM_PAD34_FLAG,
                HC_POST_NORM_VEC4_FLAG,
                HC_POST_NORM_PREDECODE_FLAG,
                HC_POST_NORM_WIDE16_FLAG,
                HC_CACHED_NORM_FLAG,
                HC_QUAD_FOLD_FLAG,
                HC_NARROW4_FLAG,
                QSA_HIST_RANK_FLAG,
                HC_PROJECTION_SHUFFLE_FLAG,
                HC_STAGED_PROJECTION_FLAG,
                DENSE_TILE4_FLAG,
                DENSE_TILE2_FLAG,
                DENSE_WIDE8_FLAG,
                DENSE_PRELOAD_FLAG,
                DENSE_DOUBLE_BUFFER_FLAG,
                DENSE_BLOCK32_FLAG,
                DENSE_DIRECT_A_FLAG,
                DENSE_PREDECODE_FLAG,
                DENSE_PRIVATE_PARTIALS_FLAG,
                DENSE_PAD_A_FLAG,
                DENSE_QKV_WIDE16_FLAG,
                DENSE_QKV_K16_FLAG,
                DENSE_QKV_PREDECODE8_FLAG,
                DENSE_QKV_THREADS64_FLAG,
                DENSE_QKV_PAD34_FLAG,
                DENSE_STATIC_SHAPE_FLAG,
                QSA_PRUNED_RANK_FLAG,
                QSA_SUBGROUP_RANK_FLAG,
                GDN_VALUE2_FLAG,
                GDN_VALUE8_FLAG,
                GDN_TREE32_FLAG,
                GDN_NORM32_FLAG,
                QMOE_FAST_FLAGS_FLAG,
                DSA_PARTITION32_FLAG,
                DSA_QUERY_CACHE_FLAG,
                DSA_PAIRED_FLAG,
                CPU_LOCAL_FLAG,
                PACKED_GATE_FLAG, run_qsa_windows,
                QSA_CONCAT_VECTOR_FLAG,
                QSA_PACKED_SCORE_FLAG,
                QSA_INDEXER_STACK_FLAG,
                QSA_SCORE16_FLAG,
                QSA_SMALL_QUERY_CACHE_FLAG,
                FC1_WARP_FOLD_FLAG,
                FC2_PAIRED_DOT_FLAG,
                FC2_SUBGROUP_FOLD_FLAG,
                FC2_COMPACT20_FLAG,
                FC1_EXPERTS10_FLAG,
                FC1_FULL_A_FLAG,
                FC1_COLUMNS8_FLAG,
                FC1_PAIRED_DOT_FLAG,
                FC1_K16_TILE_FLAG,
                FC1_K8_TILE_FLAG,
                FC1_U32_PARTIALS_FLAG,
                FC1_EXPERT_PAIRS_FLAG,
            )

            flag = (PACKED_GATE_FLAG if args.compare_packed_gate_windows else
                    HC_UP_TILED_FLAG if args.compare_hc_up_tiled_windows else
                    HC_UP_PARALLEL2_FLAG if args.compare_hc_up_parallel2_windows else
                    HC_UP_PARALLEL4_FLAG if args.compare_hc_up_parallel4_windows else
                    HC_POST_NORM_PAD34_FLAG if args.compare_hc_post_norm_pad34_windows else
                    HC_POST_NORM_VEC4_FLAG if args.compare_hc_post_norm_vec4_windows else
                    HC_POST_NORM_PREDECODE_FLAG if args.compare_hc_post_norm_predecode_windows else
                    HC_POST_NORM_WIDE16_FLAG if args.compare_hc_post_norm_wide16_windows else
                    HC_CACHED_NORM_FLAG if args.compare_hc_cached_norm_windows else
                    HC_QUAD_FOLD_FLAG if args.compare_hc_quad_fold_windows else
                    HC_NARROW4_FLAG if args.compare_hc_narrow4_windows else
                    FC1_WARP_FOLD_FLAG if args.compare_fc1_warp_fold_windows else
                    FC2_PAIRED_DOT_FLAG if args.compare_fc2_paired_dot_windows else
                    FC2_SUBGROUP_FOLD_FLAG if args.compare_fc2_subgroup_fold_windows else
                    FC2_COMPACT20_FLAG if args.compare_fc2_compact20_windows else
                    FC1_EXPERTS10_FLAG if args.compare_fc1_experts10_windows else
                    FC1_FULL_A_FLAG if args.compare_fc1_full_a_windows else
                    FC1_COLUMNS8_FLAG if args.compare_fc1_columns8_windows else
                    FC1_PAIRED_DOT_FLAG if args.compare_fc1_paired_dot_windows else
                    FC1_K16_TILE_FLAG if args.compare_fc1_k16_tile_windows else
                    FC1_K8_TILE_FLAG if args.compare_fc1_k8_tile_windows else
                    FC1_U32_PARTIALS_FLAG if args.compare_fc1_u32_partials_windows else
                    FC1_EXPERT_PAIRS_FLAG if args.compare_fc1_expert_pairs_windows else
                    QSA_HIST_RANK_FLAG if args.compare_qsa_hist_rank_windows else
                    HC_PROJECTION_SHUFFLE_FLAG if args.compare_hc_projection_shuffle_windows else
                    HC_STAGED_PROJECTION_FLAG if args.compare_hc_staged_projection_windows else
                    DENSE_TILE4_FLAG if args.compare_dense_tile4_windows else
                    DENSE_TILE2_FLAG if args.compare_dense_tile2_windows else
                    DENSE_WIDE8_FLAG if args.compare_dense_wide8_windows else
                    DENSE_PRELOAD_FLAG if args.compare_dense_preload_windows else
                    DENSE_DOUBLE_BUFFER_FLAG if args.compare_dense_double_buffer_windows else
                    DENSE_BLOCK32_FLAG if args.compare_dense_block32_windows else
                    DENSE_DIRECT_A_FLAG if args.compare_dense_direct_a_windows else
                    DENSE_PREDECODE_FLAG if args.compare_dense_predecode_windows else
                    DENSE_PRIVATE_PARTIALS_FLAG if args.compare_dense_private_partials_windows else
                    DENSE_PAD_A_FLAG if args.compare_dense_pad_a_windows else
                    DENSE_QKV_WIDE16_FLAG if args.compare_dense_qkv_wide16_windows else
                    DENSE_QKV_K16_FLAG if args.compare_dense_qkv_k16_windows else
                    DENSE_QKV_PREDECODE8_FLAG if args.compare_dense_qkv_predecode8_windows else
                    DENSE_QKV_THREADS64_FLAG if args.compare_dense_qkv_threads64_windows else
                    DENSE_QKV_PAD34_FLAG if args.compare_dense_qkv_pad34_windows else
                    DENSE_STATIC_SHAPE_FLAG if args.compare_dense_static_shape_windows else
                    QSA_PRUNED_RANK_FLAG if args.compare_qsa_pruned_rank_windows else
                    QSA_SUBGROUP_RANK_FLAG if args.compare_qsa_subgroup_rank_windows else
                    QSA_SMALL_QUERY_CACHE_FLAG if args.compare_qsa_small_query_cache_windows else
                    GDN_VALUE2_FLAG if args.compare_gdn_value2_windows else
                    GDN_VALUE8_FLAG if args.compare_gdn_value8_windows else
                    GDN_TREE32_FLAG if args.compare_gdn_tree32_windows else
                    GDN_NORM32_FLAG if args.compare_gdn_norm32_windows else
                    QMOE_FAST_FLAGS_FLAG if args.compare_qmoe_fast_flags_windows else
                    DSA_PARTITION32_FLAG if args.compare_dsa_partition32_windows else
                    DSA_QUERY_CACHE_FLAG if args.compare_dsa_query_cache_windows else
                    DSA_PAIRED_FLAG if args.compare_dsa_paired_windows else
                    CPU_LOCAL_FLAG if args.compare_cpu_local_windows else
                    FC2_FOUR_COLUMNS_FLAG if args.compare_fc2_four_columns_windows else
                    DENSE_BITCAST_UNPACK_FLAG if args.compare_dense_bitcast_unpack_windows else
                    HC_POST_NORM_BITCAST_FLAG if args.compare_hc_post_norm_bitcast_windows else
                    HC_POST_NORM_FUSED_SUM_FLAG if args.compare_hc_post_norm_fused_sum_windows else
                    HC_POST_NORM_SUBGROUP_FLAG if args.compare_hc_post_norm_subgroup_windows else
                    HC_UP_BITCAST_UNPACK_FLAG if args.compare_hc_up_bitcast_unpack_windows else
                    QSA_CONCAT_VECTOR_FLAG if args.compare_qsa_concat_vector_windows else
                    QSA_PACKED_SCORE_FLAG if args.compare_qsa_packed_score_windows else
                    QSA_INDEXER_STACK_FLAG if args.compare_qsa_indexer_stack_windows else
                    QSA_SCORE16_FLAG if args.compare_qsa_score16_windows else
                    EXPERT_BITCAST_UNPACK_FLAG if args.compare_expert_bitcast_unpack_windows else
                    MLP_BITCAST_UNPACK_FLAG if args.compare_mlp_bitcast_unpack_windows else
                    HIERARCHICAL_GATE_FLAG if args.compare_hierarchical_gate_windows else
                    POST_NORM_FOUR_COLUMNS_FLAG if args.compare_hc_post_norm_four_columns_windows else
                    ACTIVATION_PADDING_FLAG if args.compare_qmoe_activation_padding_windows else
                    SUBGROUP_GATE_FLAG if args.compare_subgroup_gate_windows else
                    HC_PROJECTION_TILED_FLAG if args.compare_hc_projection_tiled_windows else
                    FC2_PADDED_PARTIALS_FLAG if args.compare_fc2_padded_partials_windows else
                    FC2_FINAL_MIX_FLAG if args.compare_fc2_final_mix_windows else
                    REPLAY_IMMEDIATE_FLAG if args.compare_replay_immediate_windows else
                    FC1_ACTIVATION_FLAG if args.compare_fc1_activation_windows else
                    DIRECT_A_FLAG if args.compare_direct_a_windows else
                    GENERIC_MATMUL_FLAG if args.compare_generic_matmul_windows else
                    PARALLEL_MAP_FLAG if args.compare_qsa_parallel_map_windows else
                    HC_UP_PREMIX_FLAG if args.compare_hc_up_premix_windows else
                    HC_PROJECTION_FLAG if args.compare_hc_projection_windows else
                    STATIC_REPLAY_FLAG if args.compare_static_replay_windows else
                    NODE_REPLAY_FLAG if args.compare_node_replay_windows else
                    GDN_VIEW_FLAG if args.compare_gdn_view_windows else
                    LARGE_CACHE_FLAG if args.compare_large_cache_windows else
                    EXACT_EXPERT_FLAG if args.compare_exact_expert_windows else
                    BRANCH_NORM_FLAG if args.compare_branch_norm_windows else
                    EXACT_WIDE_FLAG if args.compare_exact_wide_windows else
                    EXACT_MID_FLAG if args.compare_exact_mid_windows else
                    DENSE_MAP_FLAG if args.compare_qsa_dense_map_windows else RANK_FLAG)
            run_qsa_windows(og, model, tokenizer, pool, args, result, persist, experiment_flag=flag)
            result["status"] = "completed"
            persist()
            print(json.dumps(result["window_summary"]), flush=True)
            return
        modes = ["control", "candidate"] if compare else ["configured"]
        configured_flag = os.environ.get("ORT_WEBGPU_QMOE_INT4_PREFILL", "0")
        configured_cooperative = os.environ.get("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", "0")
        configured_tail = os.environ.get("ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL", "0")
        configured_upload = os.environ.get("ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER", "0")
        configured_readback = os.environ.get("ORT_WEBGPU_SMALL_READBACK_CACHE", "0")
        configured_qsa = os.environ.get("ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE", "0")
        configured_skinny = os.environ.get("ORT_WEBGPU_MATMUL_NBITS_SKINNY", "0")
        configured_hyper_skinny = os.environ.get("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", "0")
        configured_arena = os.environ.get("ORT_WEBGPU_UNIFORM_ARENA", "0")
        configured_subgroup = os.environ.get("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP", "0")
        configured_split = os.environ.get("ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV", "0")
        configured_rank = os.environ.get("ORT_WEBGPU_QSA_PARALLEL_RANK", "0")
        configured_gemv_n96 = os.environ.get("ORT_WEBGPU_MATMUL_GEMV_N96", "0")
        configured_subgroup_gate = os.environ.get("ORT_WEBGPU_QMOE_SUBGROUP_GATE", "0")
        configured_bind_group_cache = os.environ.get("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "0")
        configured_dynamic_uniforms = os.environ.get("ORT_WEBGPU_DYNAMIC_UNIFORMS", "0")
        for context in args.contexts:
            context_modes = modes if context in comparison_contexts else ["configured"]
            for repetition in range(-1, args.repetitions):
                seed = 1000 + context if repetition == -1 else context + 7 * repetition
                prompt, offset = natural_text_prompt(pool, context, seed)
                order = context_modes if repetition % 2 == 0 else list(reversed(context_modes))
                paired = {}
                for mode in order:
                    flag = {"control": "0", "candidate": "1", "configured": configured_flag}[mode]
                    if single_flag_comparison:
                        flag = configured_flag
                    cooperative = {"control": "0", "candidate": configured_cooperative,
                                   "configured": configured_cooperative}[mode]
                    os.environ["ORT_WEBGPU_QMOE_INT4_PREFILL"] = flag
                    os.environ["ORT_WEBGPU_QMOE_INT4_COOPERATIVE"] = cooperative
                    disable_experiments = mode == "control" and not single_flag_comparison
                    if single_flag_comparison:
                        cooperative = configured_cooperative
                    tail = "0" if disable_experiments else configured_tail
                    upload = "0" if disable_experiments else configured_upload
                    readback = "0" if disable_experiments else configured_readback
                    qsa = "0" if disable_experiments else configured_qsa
                    skinny = ({"control": "0", "candidate": "1"}.get(mode, configured_skinny)
                              if args.compare_skinny else ("0" if disable_experiments else configured_skinny))
                    hyper_skinny = ({"control": "0", "candidate": "1"}.get(mode, configured_hyper_skinny)
                                    if args.compare_hyper_skinny else
                                    ("0" if disable_experiments else configured_hyper_skinny))
                    arena = ({"control": "0", "candidate": "1"}.get(mode, configured_arena)
                             if args.compare_uniform_arena else ("0" if disable_experiments else configured_arena))
                    subgroup = "0" if disable_experiments else configured_subgroup
                    split = ({"control": "0", "candidate": "1"}.get(mode, configured_split)
                             if args.compare_attention_split else ("0" if disable_experiments else configured_split))
                    rank = ({"control": "0", "candidate": "1"}.get(mode, configured_rank)
                            if args.compare_qsa_rank else ("0" if disable_experiments else configured_rank))
                    gemv_n96 = ({"control": "0", "candidate": "1"}.get(mode, configured_gemv_n96)
                                if args.compare_gemv_n96 else ("0" if disable_experiments else configured_gemv_n96))
                    subgroup_gate = ({"control": "0", "candidate": "1"}.get(mode, configured_subgroup_gate)
                                     if args.compare_subgroup_gate else
                                     ("0" if disable_experiments else configured_subgroup_gate))
                    bind_group_cache = ({"control": "0", "candidate": "8192"}.get(mode, configured_bind_group_cache)
                                        if args.compare_bind_group_cache or args.compare_dynamic_uniform_cache else
                                        ("0" if disable_experiments else configured_bind_group_cache))
                    dynamic_uniforms = (
                        {"control": "0", "candidate": "1"}.get(mode, configured_dynamic_uniforms)
                        if args.compare_dynamic_uniform_cache else
                        ("0" if disable_experiments else configured_dynamic_uniforms))
                    os.environ["ORT_WEBGPU_QMOE_INT4_COOPERATIVE"] = cooperative
                    os.environ["ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL"] = tail
                    os.environ["ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER"] = upload
                    os.environ["ORT_WEBGPU_SMALL_READBACK_CACHE"] = readback
                    os.environ["ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE"] = qsa
                    os.environ["ORT_WEBGPU_MATMUL_NBITS_SKINNY"] = skinny
                    os.environ["ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY"] = hyper_skinny
                    os.environ["ORT_WEBGPU_UNIFORM_ARENA"] = arena
                    os.environ["ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP"] = subgroup
                    os.environ["ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV"] = split
                    os.environ["ORT_WEBGPU_QSA_PARALLEL_RANK"] = rank
                    os.environ["ORT_WEBGPU_MATMUL_GEMV_N96"] = gemv_n96
                    os.environ["ORT_WEBGPU_QMOE_SUBGROUP_GATE"] = subgroup_gate
                    os.environ["ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES"] = bind_group_cache
                    os.environ["ORT_WEBGPU_DYNAMIC_UNIFORMS"] = dynamic_uniforms
                    print(json.dumps({"stage": "request", "context": context, "repetition": repetition,
                                      "mode": mode, "int4_prefill_flag": flag,
                                      "int4_cooperative_flag": cooperative, "matmul_n_tail_flag": tail,
                                      "small_upload_write_buffer_flag": upload,
                                      "small_readback_cache_flag": readback,
                                      "qsa_global_score_cache_flag": qsa,
                                      "skinny_gemv_flag": skinny, "hyper_skinny_gemv_flag": hyper_skinny,
                                      "uniform_arena_flag": arena,
                                      "attention_subgroup_flag": subgroup, "attention_split_flag": split,
                                      "qsa_parallel_rank_flag": rank,
                                      "matmul_gemv_n96_flag": gemv_n96,
                                      "qmoe_subgroup_gate_flag": subgroup_gate,
                                      "bind_group_cache_entries": bind_group_cache,
                                      "dynamic_uniforms_flag": dynamic_uniforms}), flush=True)
                    profile_options = {}
                    if args.profile_decode_after is not None:
                        profile_prefix_path = Path(
                            f"{args.profile_prefix}-ctx{context}-rep{repetition}-{mode}")
                        profiles_before = set(profile_prefix_path.parent.glob(profile_prefix_path.name + "_*.json"))
                        profile_options = {
                            "decode_profile_prefix": str(profile_prefix_path),
                            "profile_after_tokens": args.profile_decode_after,
                        }
                    sample = run_request(og, model, tokenizer, prompt, args.tokens, args.chunk, **profile_options)
                    if profile_options:
                        profiles = set(profile_prefix_path.parent.glob(profile_prefix_path.name + "_*.json"))
                        gpu_profiles = [path for path in sorted(profiles - profiles_before)
                                        if any(event.get("cat") == "Kernel"
                                               for event in json.loads(path.read_text()))]
                        if len(gpu_profiles) < sample["profiled_decode_tokens"]:
                            raise RuntimeError(
                                f"Expected at least {sample['profiled_decode_tokens']} new GPU decode profiles, "
                                f"found {len(gpu_profiles)}")
                        sample["decode_profile_files"] = [str(path) for path in gpu_profiles]
                    sample.update(mode=mode, context=context, repetition=repetition, warmup=repetition == -1,
                                  seed=seed, offset=offset, int4_prefill_flag=flag,
                                  int4_cooperative_flag=cooperative, matmul_n_tail_flag=tail,
                                  small_upload_write_buffer_flag=upload,
                                  small_readback_cache_flag=readback, qsa_global_score_cache_flag=qsa,
                                  skinny_gemv_flag=skinny, hyper_skinny_gemv_flag=hyper_skinny,
                                  uniform_arena_flag=arena,
                                  attention_subgroup_flag=subgroup, attention_split_flag=split,
                                  qsa_parallel_rank_flag=rank,
                                  matmul_gemv_n96_flag=gemv_n96,
                                  qmoe_subgroup_gate_flag=subgroup_gate,
                                  bind_group_cache_entries=bind_group_cache,
                                  dynamic_uniforms_flag=dynamic_uniforms,
                                  qsa_dense_map_flag=os.environ.get("ORT_WEBGPU_QSA_DENSE_MAP", "0"))
                    paired[mode] = sample
                    result["samples"].append(sample)
                    persist()
                    print(json.dumps({key: value for key, value in sample.items()
                                      if key not in ("output_token_ids", "output_text")}), flush=True)
                if compare and context in comparison_contexts:
                    control = paired["control"]["output_token_ids"]
                    candidate = paired["candidate"]["output_token_ids"]
                    first_difference = next((i for i, (a, b) in enumerate(zip(control, candidate)) if a != b), None)
                    for sample in paired.values():
                        sample["paired_output_matches"] = control == candidate
                        sample["paired_first_different_token"] = first_difference
                    persist()
        result["summary"] = {
            f"{context}:{mode}": summarize([
                row for row in result["samples"]
                if row["context"] == context and row["mode"] == mode and not row["warmup"]
            ]) for context, mode in sorted({(row["context"], row["mode"]) for row in result["samples"]
                                           if not row["warmup"]})
        }
        if hasattr(os, "sched_getaffinity"):
            result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            result["thread_cpu_affinity"] = {
                task.name: sorted(os.sched_getaffinity(int(task.name)))
                for task in Path("/proc/self/task").iterdir()
            }
        result["status"] = "completed"
        persist()
        print(json.dumps(result["summary"]), flush=True)
    except Exception as error:
        result.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()

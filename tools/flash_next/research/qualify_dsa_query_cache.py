import argparse
import collections
import json
from pathlib import Path

from build_dsa_query_cache import FLAG
from build_dsa_paired import FLAG as PAIRED_FLAG
from qualify_qsa_score16_standard import all_contexts_reach_target, collect_standard_cases, verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


VARIANT = "dsa-query-cache-v2"
STAMP = "20261030"
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-dsa-query-cache/build-v2")


def read(name):
    return json.loads((ROOT / name).read_text())


def verify_flags(raw, paired):
    flags = (FLAG, "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DENSE_TILE4",
             "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_UP_PREMIX", "ORT_WEBGPU_QSA_SCORE16")
    if paired:
        flags += (PAIRED_FLAG, "ORT_WEBGPU_DENSE_BLOCK32")
    assert all(raw["load_experiment_flags"][flag] == "1" for flag in flags)


def qualify_windows(variant=VARIANT):
    assert variant in (VARIANT, "dsa-paired-v1")
    paired = variant == "dsa-paired-v1"
    flag = PAIRED_FLAG if paired else FLAG
    family = variant.rsplit("-v", 1)[0].replace("-", "_")
    program_name = "DynamicSparseAttentionPaired" if paired else "DynamicSparseAttentionQueryCache"
    directory = Path("/home/sayanshaw/repos/qwen38-int4-dsa-paired/build-v1") if paired else DIRECTORY
    native_name = f"{variant}-native-gpu7-{STAMP}.qualification.json"
    parent_name = f"{'dense-block32-v2' if paired else 'prefill1024-v2'}-contexts-{STAMP}.qualification.json"
    controls_name = "int4-pending64-standard-contexts.json"
    native, parent, controls = map(read, (native_name, parent_name, controls_name))
    assert native["status"] == f"{family}_finite_bitwise_and_cpu_reference_qualified"
    assert native["finite_bitwise_comparisons"] == 4870656 and native["fixtures"] == 27
    assert native["cpu_reference_executions"] == 486 and native["fixed_flag_capture_executions"] == 9
    assert native["pending_windows_tested"] == [1, 64] and native["invalid_selector_rejected"]
    assert native["original_dot_order_preserved"] and native["original_merge_unchanged"]
    assert native["original_partition_size_preserved"] == 64
    assert native["fp16_only"] and native["fp32_original_fallback_preserved"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    build = json.loads((directory / "build-manifest.json").read_text())
    assert build["baseline_byte_identical"] and build["provider_sha256"] == native["provider_sha256"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    if paired:
        assert build["logical_partitions_per_workgroup"] == 2
        assert build["direct_integer_length_loads_no_cross_subgroup_shared_state"]
        proof = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill1280/build-v2/build-manifest.json")
        assert build["original_attention_object_byte_identical"]
        assert sha(proof) == build["original_attention_object_proof_sha256"]
    header = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.h")
    assert sha(header) == build["original_attention_header_sha256"]
    assert len(native["fixture_sha256"]) == 54
    for name, digest in native["fixture_sha256"].items():
        assert sha(ROOT / "dsa-split-kv-fixtures" / name) == digest
    guard = read(f"{variant}-native-gpu7-{STAMP}.guard.json")
    assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    sources = {name: sha(ROOT / name) for name in (native_name, parent_name, controls_name)}
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    rows = []
    cases = [(2048, True, "profile")]
    cases.extend((context, full, "windows") for context in (2048, 8192, 32768) for full in (True, False))
    for context, full, phase in cases:
        prefix = ROOT / f"{variant}-chunk1024-gpu7-{'full-' if full else ''}{phase}{context}-{STAMP}"
        proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        verify_metadata(raw, guard, native, parent, full, chunk=1024)
        verify_flags(raw, paired)
        assert proof["status"] == "original_sequences_qualified"
        assert proof["context"] == context and proof["phase"] == phase and proof["full_validation"] == full
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        references = [sample for sample in controls["samples"] if sample["context"] == context]
        if phase == "profile":
            verify_tokens(raw, references, context, 36)
            assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
            profiles = [ROOT / name for name in proof["source_sha256"]
                        if f"-profile-ctx{context}-rep0-configured_" in name]
            assert len(profiles) == 1
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                for event in json.loads(profiles[0].read_text()) if event.get("cat") == "Kernel")
            assert counts[program_name] == 12
            assert counts["DynamicSparseAttention"] == 0
            if paired:
                assert counts["DynamicSparseAttentionQueryCache"] == 0
                assert counts["MatMulNBitsBlock32F16"] == 159
                assert counts["MatMulNBitsTile4F16"] == 0
            assert proof["result"]["candidate_program_dispatches"] == {program_name: 12}
        else:
            result = verify_paired(raw, references, context, prefix=family, experiment_flag=flag)
            assert result == proof["result"]
            rows.append(dict(context=context, full_validation=full, result=result))
        sources[prefix.with_suffix(".qualification.json").name] = sha(prefix.with_suffix(".qualification.json"))
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status=f"{family}_native_and_paired_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, chunk=1024, pending_dispatches=64, contexts=[2048, 8192, 32768],
        native_finite_bitwise_comparisons=4870656, window_original_prefix_comparisons=6144,
        all_original_prefixes_match=True, all_repeated_288_tokens_match=True, zero_major_faults=True,
        all_contexts_faster_in_both_validation_modes=all(
            row["result"]["geometric_mean_matched_window_ratio"] > 1 for row in rows),
        comparison_kind="controlled_paired_windows_not_standard_decode_headline",
        default_off=True, quality_qualified=False, headline_eligible=False,
        rows=rows, source_sha256=sources, build_manifest_sha256=sha(directory / "build-manifest.json"),
        frozen_dependencies_sha256={str(header): sha(header)})
    return report, native, parent, controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("windows", "standard"), default="windows")
    parser.add_argument("--variant", choices=(VARIANT, "dsa-paired-v1"), default=VARIANT)
    args = parser.parse_args()
    output = ROOT / f"{args.variant}-{'windows-' if args.phase == 'windows' else ''}contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    paired, native, parent, controls = qualify_windows(args.variant)
    report = paired
    if args.phase == "standard":
        saved = read(f"{args.variant}-windows-contexts-{STAMP}.qualification.json")
        assert paired == saved and paired["all_contexts_faster_in_both_validation_modes"]
        sources = dict(paired["source_sha256"])
        name = f"{args.variant}-windows-contexts-{STAMP}.qualification.json"
        sources[name] = sha(ROOT / name)

        def verify_case(raw, guard, native, parent, references, context, full):
            verify_flags(raw, args.variant == "dsa-paired-v1")
            return verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)

        rows = collect_standard_cases(args.variant, native, parent, controls, paired, sources,
            verifier=verify_case, stamp=STAMP, chunk=1024)
        family = args.variant.rsplit("-v", 1)[0].replace("-", "_")
        report = dict(status=f"{family}_native_and_standard_original_sequences_qualified",
            provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
            graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
            gpu_index=7, contexts=[2048, 8192, 32768], chunk=1024, pending_dispatches=64,
            output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
            original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
            all_contexts_at_least_100_decode_tps=all_contexts_reach_target(rows),
            default_off=True, model_and_weights_unchanged=True, quality_qualified=False,
            headline_eligible=False, rows=rows, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "rows")}), flush=True)


if __name__ == "__main__":
    main()

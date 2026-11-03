import argparse
import collections
import json
from pathlib import Path

from build_dense_block32 import FLAG
from qualify_qsa_score16_standard import all_contexts_reach_target, collect_standard_cases, verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


STAMP = "20261030"


def verify_flags(raw, stacked):
    flags = (FLAG, "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DENSE_TILE4",
             "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16")
    if stacked:
        flags += ("ORT_WEBGPU_DSA_QUERY_CACHE",)
    assert all(raw["load_experiment_flags"][flag] == "1" for flag in flags)


def qualify_windows(variant):
    stacked = variant == "dense-block32-v2"
    parent_variant = "dsa-query-cache-v2" if stacked else "prefill1024-v2"
    directory = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32") / (
        "build-v2" if stacked else "build-v1")
    paths = [ROOT / f"{variant}-native-gpu7-{STAMP}.qualification.json",
             ROOT / f"{parent_variant}-contexts-{STAMP}.qualification.json",
             ROOT / "int4-pending64-standard-contexts.json"]
    native, parent, controls = [json.loads(path.read_text()) for path in paths]
    assert native["status"] == "dense_block32_changing_inputs_finite_bitwise_and_replay_qualified"
    assert native["finite_bitwise_comparisons"] == 116582784 and native["fixtures"] == 27
    assert native["pending_windows_tested"] == [1, 64]
    assert native["multi_token_fallback_tested"] and native["invalid_flag_explicitly_rejected"]
    assert native["replay_variant_keys_tested"] and native["both_flags_tested"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    manifest_path = directory / "build-manifest.json"
    build = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == native["build_manifest_sha256"]
    assert build["baseline_byte_identical"] and build["provider_sha256"] == native["provider_sha256"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    fixtures = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1")
    assert len(native["fixture_sha256"]) == 27
    for name, digest in native["fixture_sha256"].items():
        assert sha(fixtures / name) == digest
    if stacked:
        assert build["dense_component_relink_byte_identical"] and build["query_cache_object_unchanged"]
        assert build["no_new_shader_or_arithmetic_changes"]
        assert sha(directory.parent / "build-v1/matmul_nbits.cc.o") == build["dense_object_sha256"]
        for path, digest in build["component_manifests_sha256"].items():
            assert sha(Path(path.replace("/workspace/", "/home/sayanshaw/repos/"))) == digest
    sources = {path.name: sha(path) for path in paths}
    for name, digest in native["source_sha256"].items():
        path = manifest_path if name == "build-manifest.json" else ROOT / name
        assert sha(path) == digest
        if path != manifest_path:
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
        verify_flags(raw, stacked)
        assert proof["status"] == "original_sequences_qualified"
        assert proof["context"] == context and proof["phase"] == phase and proof["full_validation"] == full
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        references = [sample for sample in controls["samples"] if sample["context"] == context]
        if phase == "profile":
            verify_tokens(raw, references, context, 36)
            assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
            assert proof["result"]["candidate_program_dispatches"] == {"MatMulNBitsBlock32F16": 159}
            profiles = [ROOT / name for name in proof["source_sha256"]
                        if "-profile-ctx2048-rep0-configured_" in name]
            assert len(profiles) == 1
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                for event in json.loads(profiles[0].read_text()) if event.get("cat") == "Kernel")
            assert counts["MatMulNBitsBlock32F16"] == 159 and counts["MatMulNBitsTile4F16"] == 0
            if stacked:
                assert counts["DynamicSparseAttentionQueryCache"] == 12
                assert counts["DynamicSparseAttention"] == 0
        else:
            result = verify_paired(raw, references, context, prefix="dense_block32", experiment_flag=FLAG)
            assert result == proof["result"]
            rows.append(dict(context=context, full_validation=full, result=result))
        sources[prefix.with_suffix(".qualification.json").name] = sha(prefix.with_suffix(".qualification.json"))
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
    report = dict(status="dense_block32_native_and_paired_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, chunk=1024, pending_dispatches=64, contexts=[2048, 8192, 32768],
        native_finite_bitwise_comparisons=116582784, window_original_prefix_comparisons=6144,
        all_original_prefixes_match=True, all_repeated_288_tokens_match=True, zero_major_faults=True,
        all_contexts_faster_in_both_validation_modes=all(
            row["result"]["geometric_mean_matched_window_ratio"] > 1 for row in rows),
        comparison_kind="controlled_paired_windows_not_standard_decode_headline",
        query_cache_stacked=stacked, default_off=True, quality_qualified=False, headline_eligible=False,
        rows=rows, source_sha256=sources, build_manifest_sha256=sha(manifest_path))
    return report, native, parent, controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense-block32-v1", "dense-block32-v2"), default="dense-block32-v1")
    parser.add_argument("--phase", choices=("windows", "standard"), default="windows")
    args = parser.parse_args()
    output = ROOT / f"{args.variant}-{'windows-' if args.phase == 'windows' else ''}contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    paired, native, parent, controls = qualify_windows(args.variant)
    report = paired
    if args.phase == "standard":
        saved_path = ROOT / f"{args.variant}-windows-contexts-{STAMP}.qualification.json"
        assert paired == json.loads(saved_path.read_text())
        assert paired["all_contexts_faster_in_both_validation_modes"]
        sources = dict(paired["source_sha256"])
        sources[saved_path.name] = sha(saved_path)

        def verify_case(raw, guard, native, parent, references, context, full):
            verify_flags(raw, args.variant == "dense-block32-v2")
            return verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)

        rows = collect_standard_cases(args.variant, native, parent, controls, paired, sources,
            verifier=verify_case, stamp=STAMP, chunk=1024)
        report = dict(status="dense_block32_native_and_standard_original_sequences_qualified",
            provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
            graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
            gpu_index=7, contexts=[2048, 8192, 32768], chunk=1024, pending_dispatches=64,
            output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
            original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
            all_contexts_at_least_100_decode_tps=all_contexts_reach_target(rows),
            query_cache_stacked=paired["query_cache_stacked"], default_off=True,
            model_and_weights_unchanged=True, quality_qualified=False,
            headline_eligible=False, rows=rows, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "rows")}), flush=True)


if __name__ == "__main__":
    main()

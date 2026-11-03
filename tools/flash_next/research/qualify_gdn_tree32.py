import argparse
import collections
import json
import math
from pathlib import Path

from build_gdn_tree32 import FLAG, PROGRAM
from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_qsa_score16_standard import collect_standard_cases, verify_standard
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


VARIANT = "gdn-tree32-v1"
STAMP = "20261031"
CONTEXTS = (2048, 8192, 32768)


def controls_supported(rows):
    assert [(row["context"], row["full_validation"]) for row in rows] == [
        (context, full) for context in CONTEXTS for full in (True, False)]
    supported = True
    for row in rows:
        result = row["result"]
        ratio = result["geometric_mean_matched_window_ratio"]
        assert math.isfinite(ratio) and ratio > 0
        assert result["matched_window_pairs"] == 16 and result["all_request_output_tokens_match"]
        requests = result["per_request"]
        assert [request["repetition"] for request in requests] == [0, 1, 2, 3]
        for request in requests:
            for mode in (0, 1):
                value = request[f"gdn_tree32{mode}_median_window_tps"]
                assert math.isfinite(value) and value > 0
        wins = sum(request["gdn_tree321_median_window_tps"] >
                   request["gdn_tree320_median_window_tps"] for request in requests)
        supported = supported and ratio > 1 and wins >= 3
    return supported


def verify_gdn_standard(raw, guard, native, parent, references, context, full):
    assert raw["load_experiment_flags"][FLAG] == "1"
    return verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)


def verify_gdn_histogram_standard(raw, guard, native, parent, references, context, full):
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QSA_HIST_RANK"] == "1"
    return verify_gdn_standard(raw, guard, native, parent, references, context, full)


def main(variant=VARIANT, parent_variant="qsa-prefill-score16-v1"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("windows", "standard"), default="windows")
    args = parser.parse_args()
    assert variant in (VARIANT, "gdn-tree32-hist-v1")
    assert parent_variant == ("qsa-hist-rank-v2" if variant == "gdn-tree32-hist-v1"
                              else "qsa-prefill-score16-v1")
    output = ROOT / (f"{variant}-windows-contexts-{STAMP}.qualification.json" if args.phase == "windows"
                     else f"{variant}-contexts-{STAMP}.qualification.json")
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{variant}-native-gpu7-{STAMP}.qualification.json"
    parent_path = ROOT / f"{parent_variant}-contexts-{STAMP}.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, parent, controls = (json.loads(path.read_text())
                                for path in (native_path, parent_path, controls_path))
    assert native["status"] == "gdn_tree32_both_outputs_finite_bitwise_and_dispatch_qualified"
    assert native["fixtures"] == 36 and native["optimized_fixtures"] == 2
    assert native["finite_bitwise_comparisons"] == 236531952
    assert native["pending_windows_tested"] == [1, 64]
    assert native["fixed_flag_capture_candidate_executions"] == 9
    assert native["both_output_and_final_state_checked"] and native["invalid_selector_rejected"]
    assert native["original_per_channel_arithmetic_preserved"]
    assert native["multi_token_batch_dtype_and_odd_shape_fallbacks_checked"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    family, version = variant.rsplit("-v", 1)
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v{version}")
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert sha(build_path) == native["build_manifest_sha256"]
    assert build["baseline_byte_identical"] and build["original_gdn_object_byte_identical"]
    assert build["no_existing_kernel_object_layout_changes"]
    assert build["fixed_subgroup_size"] == 32 and build["original_128_lane_tree_association_preserved"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    capture = directory.parent / "capture-compare-v1"
    assert sha(capture / "build-manifest.json") == native["capture_build_manifest_sha256"]
    assert sha(capture / "native_compare") == native["capture_comparator_sha256"]
    if variant == "gdn-tree32-hist-v1":
        assert build["gdn_component_object_byte_identical"]
        assert build["histogram_component_object_unchanged"] and build["only_disjoint_gdn_object_replaced"]
        for name, digest in build["qualification_source_sha256"].items():
            assert sha(ROOT / name) == digest
    sources = {path.name: sha(path) for path in (native_path, parent_path, controls_path)}
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    for name, digest in native["fixture_sha256"].items():
        assert sha(Path("/home/sayanshaw/repos/qwen38-int4-gdn-decode-view/fixtures") / name) == digest
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest

    attempts = {}
    for attempt in (1, 2):
        rows = []
        cases = [(2048, True, "profile")] if attempt == 1 else []
        if variant == "gdn-tree32-hist-v1" and attempt == 1:
            cases.append((32768, True, "profile"))
        cases += [(context, full, "windows") for context in CONTEXTS for full in (True, False)]
        for context, full, phase in cases:
            suffix = f"-attempt{attempt}" if attempt > 1 else ""
            prefix = ROOT / (f"{variant}-chunk1024-gpu7-{'full-' if full else ''}"
                             f"{phase}{context}-{STAMP}{suffix}")
            proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            verify_metadata(raw, guard, native, parent, full, chunk=1024)
            assert raw["load_experiment_flags"][FLAG] == "1"
            if variant == "gdn-tree32-hist-v1":
                assert raw["load_experiment_flags"]["ORT_WEBGPU_QSA_HIST_RANK"] == "1"
            assert proof["status"] == "original_sequences_qualified"
            assert proof["provider_sha256"] == native["provider_sha256"]
            assert proof["baseline_provider_sha256"] == parent["provider_sha256"]
            assert proof["context"] == context and proof["phase"] == phase and proof["full_validation"] == full
            assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
            assert proof["graph_sha256"] == parent["graph_sha256"]
            assert proof["config_sha256"] == parent["config_sha256"]
            references = [sample for sample in controls["samples"] if sample["context"] == context]
            if phase == "profile":
                verify_tokens(raw, references, context, 36)
                assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
                profiles = [ROOT / name for name in proof["source_sha256"]
                            if f"-profile-ctx{context}-rep0-configured_" in name]
                assert len(profiles) == 1
                counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                                             for event in json.loads(profiles[0].read_text())
                                             if event.get("cat") == "Kernel")
                assert counts[PROGRAM] == 36 and counts["GatedDeltaNet"] == 0
                assert counts["DynamicSparseAttentionQueryCache"] == 12
                assert counts["MatMulNBitsBlock32F16"] == 159
                assert proof["result"]["candidate_program_dispatches"] == {PROGRAM: 36}
                if variant == "gdn-tree32-hist-v1":
                    from run_hc_up_parallel2_model import verify_gdn_histogram_profile
                    assert proof["result"]["histogram_program_dispatches"] == \
                        verify_gdn_histogram_profile(counts, context)
            else:
                result = verify_paired(raw, references, context, prefix="gdn_tree32", experiment_flag=FLAG)
                assert result == proof["result"]
                rows.append(dict(context=context, full_validation=full, result=result))
            sources[prefix.with_suffix(".qualification.json").name] = sha(prefix.with_suffix(".qualification.json"))
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest
                sources[name] = digest
        attempts[attempt] = rows
    supported = all([controls_supported(attempts[attempt]) for attempt in (1, 2)])
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(
        status="gdn_tree32_native_and_paired_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        build_manifest_sha256=sha(build_path), gpu_index=7, chunk=1024, pending_dispatches=64,
        contexts=list(CONTEXTS), native_finite_bitwise_comparisons=236531952,
        window_original_prefix_comparisons=12288, all_original_prefixes_match=True,
        all_repeated_288_tokens_match=True, zero_major_faults=True,
        all_contexts_faster_in_both_validation_modes=supported,
        repeated_control_attempts=[1, 2], minimum_positive_request_medians_per_case=3,
        initial_rows=attempts[1], rows=attempts[2], default_off=True,
        quality_qualified=False, headline_eligible=False, source_sha256=sources)
    if args.phase == "standard":
        paired_path = ROOT / f"{variant}-windows-contexts-{STAMP}.qualification.json"
        paired = json.loads(paired_path.read_text())
        assert paired == report
        assert supported, "Repeated controls do not support standard escalation"
        sources[paired_path.name] = sha(paired_path)
        verifier = verify_gdn_histogram_standard if variant == "gdn-tree32-hist-v1" else verify_gdn_standard
        rows = collect_standard_cases(variant, native, parent, controls, paired, sources,
                                      verifier=verifier, stamp=STAMP, chunk=1024)
        for row in rows:
            for mode in ("candidate", "full_validation"):
                tps = row[mode]["median_decode_tps"]
                ttft = row[mode]["mean_ttft_seconds"]
                assert math.isfinite(tps) and tps > 0 and math.isfinite(ttft) and ttft > 0
        report.update(status="gdn_tree32_native_and_standard_original_sequences_qualified",
                      rows=rows, output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
                      original_sequence_comparisons=6144, all_original_sequences_match=True,
                      model_and_weights_unchanged=True,
                      all_contexts_above_100_decode_tps=all(
                          row["candidate"]["median_decode_tps"] > 100 for row in rows))
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "all_contexts_faster_in_both_validation_modes", "rows")}), flush=True)


if __name__ == "__main__":
    main()

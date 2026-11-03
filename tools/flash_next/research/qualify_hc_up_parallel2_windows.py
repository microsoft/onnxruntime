import collections
import json
from pathlib import Path

from build_hc_up_parallel2 import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_hc_up_parallel2_model import VARIANT
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


def main(variant=VARIANT, experiment_flag=FLAG, stamp="20261029", parent_variant="qsa-score16-v5",
         program_name="HCUpPreMixPacked2F16", original_program_name="HCUpPreMixF16", dispatches=97):
    family, version = variant.rsplit("-v", 1)
    window_prefix = family.replace("-", "_")
    output = ROOT / f"{variant}-windows-contexts-{stamp}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{variant}-native-gpu7-{stamp}.qualification.json"
    parent_path = ROOT / f"{parent_variant}-contexts-20261029.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, parent, controls = (json.loads(path.read_text())
                                for path in (native_path, parent_path, controls_path))
    required_flags = (experiment_flag, "ORT_WEBGPU_QSA_SCORE16", "ORT_WEBGPU_HC_UP_PREMIX")
    if variant == VARIANT:
        assert native["status"] == "hc_up_parallel2_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 18677760
        assert native["columns_tested"] == [1, 2] and native["variant_cache_keys_tested"]
        assert native["parent_fusion_fixed_enabled"]
        fixture_family, fixture_count, stats_count, operator = "hc-up-premix", 9, 3, "HCUpPreMixF16"
    elif variant == "dense-tile4-v1":
        assert native["status"] == "dense_tile4_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 116582784
        assert native["single_token_shapes"] == 9 and native["replay_variant_keys_tested"]
        assert native["both_flags_tested"]
        fixture_family, fixture_count, stats_count, operator = "generic-matmul-replay", 27, 9, "MatMulNBitsExact"
        required_flags += ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_DENSE_BITCAST_UNPACK")
    else:
        raise ValueError(f"Unsupported HC experiment: {variant}")
    assert native["fixtures"] == fixture_count and native["rounds_per_fixture"] == 128
    assert native["pending_windows_tested"] == [1, 64]
    assert native["invalid_flag_explicitly_rejected"] and native["multi_token_fallback_tested"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v{version}")
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_byte_identical"] and build["provider_sha256"] == native["provider_sha256"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    sources = {path.name: sha(path) for path in (native_path, parent_path, controls_path)}
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    fixtures = sorted(Path(f"/home/sayanshaw/repos/qwen38-int4-{fixture_family}/fixtures-v1").glob("*.onnx"))
    assert len(fixtures) == fixture_count
    assert {path.name: sha(path) for path in fixtures} == native["fixture_sha256"]
    native_prefix = ROOT / f"{variant}-native-gpu7-{stamp}"
    native_guard = json.loads(native_prefix.with_suffix(".guard.json").read_text())
    assert native_guard["status"] == "completed" and native_guard["exit_code"] == 0
    assert native_guard["gpu_index"] == 7
    assert native_guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    native_log = native_prefix.with_suffix(".log").read_text()
    assert "FAIL" not in native_log and "GetCapability" not in native_log
    assert native_log.count(" changing_rounds=128 invalid_flag_rejected=1") == fixture_count
    assert f"PASS total_exact_elements={native['finite_bitwise_comparisons']}; CPU fallback disabled" in native_log
    stats_path = ROOT / f"{native_prefix.name}-stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines()]
    assert stats == native["stats"] and len(stats) == stats_count
    assert all(row["operator"] == operator and row["requests"] == 256 and
               row["hits"] + row["captures"] == row["requests"] and row["entries"] == 2 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0
               for row in stats)
    for path in (native_prefix.with_suffix(".guard.json"), native_prefix.with_suffix(".log"),
                 stats_path):
        sources[path.name] = sha(path)
    rows = []
    cases = [(2048, True, "profile")]
    cases.extend((context, full, "windows") for context in (2048, 8192, 32768) for full in (True, False))
    for context, full, phase in cases:
        prefix = ROOT / f"{variant}-chunk256-gpu7-{'full-' if full else ''}{phase}{context}-{stamp}"
        proof_path = prefix.with_suffix(".qualification.json")
        proof = json.loads(proof_path.read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        verify_metadata(raw, guard, native, parent, full)
        for flag in required_flags:
            assert raw["load_experiment_flags"][flag] == "1"
        assert proof["status"] == "original_sequences_qualified"
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["context"] == context and proof["phase"] == phase and proof["full_validation"] == full
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
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
            assert counts[program_name] == dispatches and counts[original_program_name] == 0
            assert proof["result"]["candidate_program_dispatches"] == {program_name: dispatches}
        else:
            result = verify_paired(raw, references, context, prefix=window_prefix, experiment_flag=experiment_flag)
            assert result == proof["result"]
            rows.append(dict(context=context, full_validation=full, result=result))
        sources[proof_path.name] = sha(proof_path)
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status=f"{window_prefix}_native_and_paired_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, chunk=256, pending_dispatches=64, contexts=[2048, 8192, 32768],
        native_finite_bitwise_comparisons=native["finite_bitwise_comparisons"],
        window_original_prefix_comparisons=256 * 4 * len(rows),
        all_original_prefixes_match=True, all_repeated_288_tokens_match=True, zero_major_faults=True,
        all_contexts_faster_in_both_validation_modes=all(
            row["result"]["geometric_mean_matched_window_ratio"] > 1 for row in rows),
        comparison_kind="controlled_paired_windows_not_standard_decode_headline",
        default_off=True, quality_qualified=False, headline_eligible=False,
        rows=rows, source_sha256=sources, build_manifest_sha256=sha(build_path))
    if variant == "dense-tile4-v1":
        component = Path("/home/sayanshaw/repos/qwen38-int4-dense-bitcast-unpack/build-v1")
        component_build = component / "build-manifest.json"
        template = component / "dense_bitcast_template.inc"
        assert sha(template) == json.loads(component_build.read_text())["source_sha256"][template.name]
        report["frozen_dependencies_sha256"] = {str(path): sha(path) for path in (component_build, template)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "all_original_prefixes_match", "rows")}), flush=True)


if __name__ == "__main__":
    main()

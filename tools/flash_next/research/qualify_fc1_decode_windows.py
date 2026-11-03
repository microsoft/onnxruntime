import argparse
import collections
import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_hc_up_parallel2_model import verify_row4_window_prerequisite
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


VARIANTS = {
    "fc1-paired-dot-v1": ("ORT_WEBGPU_QMOE_FC1_PAIRED_DOT", "QmoeFC1PairedDotBitcastF16", "paired_dot_tested"),
    "fc1-k16-tile-v1": ("ORT_WEBGPU_QMOE_FC1_K16_TILE", "QmoeFC1K16TileBitcastF16", "k16_tile_tested"),
    "fc1-k8-tile-v1": ("ORT_WEBGPU_QMOE_FC1_K8_TILE", "QmoeFC1K8TileBitcastF16", "k8_tile_tested"),
}


def verify_profile_proof(proof, provider, context, full, program):
    assert proof["status"] == "original_sequences_qualified" and proof["phase"] == "profile"
    assert proof["provider_sha256"] == provider
    assert proof["context"] == context and proof["full_validation"] == full
    assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
    assert proof["fixed_grouped_prefill_selector"] == 1
    assert proof["result"]["candidate_program_dispatches"] == {program: 48}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    variant = args.variant
    flag, program, mode_field = VARIANTS[variant]
    family, version = variant.rsplit("-v", 1)
    prefix = family.replace("-", "_")
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v{version}")
    output = ROOT / f"{variant}-windows-contexts-20261031.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{variant}-native-gpu7-20261031.qualification.json"
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    standard_path = ROOT / "qmoe-prefill-rows4-v2-standard-contexts-20261031.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, parent, standard, controls = (json.loads(path.read_text())
                                        for path in (native_path, parent_path, standard_path, controls_path))
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert native["status"] == f"{prefix}_changing_inputs_routes_finite_bitwise_and_replay_qualified"
    assert native["fixtures"] == 23 and native["finite_bitwise_comparisons"] == 9748480
    assert native[mode_field] == [0, 1] and native["variant_cache_keys_tested"]
    assert native["pending_windows_tested"] == [1, 64]
    assert native["optional_bias_and_unsupported_layout_fallbacks_tested"]
    assert native["both_expert_fusions_fixed_enabled"] and native["expert_bitcast_fixed_enabled"]
    assert build["baseline_byte_identical"] and build["replay_variant_keyed"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"] == native["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"] == native["comparator_sha256"]
    assert native["baseline_provider_sha256"] == build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert parent["status"] == "controlled_ttft_screens_all_contexts_qualified_not_standard_decode"
    assert standard["status"] == "rows4_native_and_standard_original_sequences_qualified"
    assert standard["provider_sha256"] == parent["provider_sha256"]
    assert standard["all_original_sequences_match"] and standard["zero_major_faults"]
    assert standard["original_sequence_comparisons"] == 6144
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest, name
    for path, digest in standard["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    for name, digest in native["fixture_sha256"].items():
        assert sha(Path("/home/sayanshaw/repos") / name) == digest, name
    guard_path = ROOT / f"{variant}-native-gpu7-20261031.guard.json"
    log_path = ROOT / f"{variant}-native-gpu7-20261031.log"
    stats_path = ROOT / f"{variant}-native-gpu7-20261031-stats.jsonl"
    guard = json.loads(guard_path.read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    log = log_path.read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS changing_inputs_and_routes ") == 23
    assert "PASS total_exact_elements=9748480; CPU fallback disabled" in log
    assert [json.loads(line) for line in stats_path.read_text().splitlines()] == native["stats"]
    assert len(native["stats"]) == 11
    assert all(row["requests"] == 64 and row["hits"] > 0 and
               row["captures"] + row["hits"] == row["requests"] and
               0 < row["entries"] <= row["capacity"] == 8 for row in native["stats"])
    sources = {str(path): sha(path) for path in
               (native_path, parent_path, standard_path, controls_path, build_path, guard_path, log_path,
                stats_path, Path(__file__), directory / "libonnxruntime_providers_webgpu.so")}
    sources.update({str(directory / name): digest for name, digest in build["source_sha256"].items()})
    sources.update(standard["source_sha256"])
    sources.update({str(Path("/home/sayanshaw/repos") / name): digest
                    for name, digest in native["fixture_sha256"].items()})
    rows, comparisons = [], 0
    cases = [(2048, True, "profile", 1)]
    cases += [(context, full, "windows", attempt) for context in (2048, 8192, 32768)
              for attempt in ((1, 2) if context == 2048 else (1,)) for full in (True, False)]
    for context, full, phase, attempt in cases:
        suffix = f"-attempt{attempt}" if attempt > 1 else ""
        path = ROOT / f"{variant}-chunk1024-gpu7-{'full-' if full else ''}{phase}{context}-20261031{suffix}"
        proof_path = path.with_suffix(".qualification.json")
        proof = json.loads(proof_path.read_text())
        raw = json.loads(path.with_suffix(".json").read_text())
        guard = json.loads(path.with_suffix(".guard.json").read_text())
        verify_metadata(raw, guard, native, parent, full, chunk=1024)
        assert proof["native_proof_sha256"] == sha(native_path)
        assert proof["parent_proof_sha256"] == sha(parent_path)
        assert proof["build_manifest_sha256"] == sha(build_path)
        assert proof["row4_standard_proof_sha256"] == sha(standard_path)
        assert raw["load_experiment_flags"][flag] == "1"
        assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_GROUPED_PREFILL"] == "1"
        references = [sample for sample in controls["samples"] if sample["context"] == context]
        if phase == "profile":
            verify_profile_proof(proof, native["provider_sha256"], context, full, program)
            verify_tokens(raw, references, context, 36)
            assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
            profiles = [ROOT / name for name in proof["source_sha256"] if "-profile-ctx2048-" in name]
            assert len(profiles) == 1
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                                         for event in json.loads(profiles[0].read_text())
                                         if event.get("cat") == "Kernel")
            assert counts[program] == 48 and counts["QmoeFC1SwiGLUBitcastF16"] == 0
        else:
            verify_row4_window_prerequisite(proof, native["provider_sha256"], context, full, prefix=prefix)
            result = verify_paired(raw, references, context, prefix=prefix, experiment_flag=flag)
            assert result == proof["result"]
            rows.append(dict(context=context, full_validation=full, attempt=attempt, result=result))
        comparisons += sum(min(len(sample["output_token_ids"]), 256) for sample in raw["samples"])
        sources[str(proof_path)] = sha(proof_path)
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
            sources[str(ROOT / name)] = digest
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status=f"{prefix}_native_and_repeated_windows_original_sequences_qualified",
                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
                  graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
                  contexts=[2048, 8192, 32768], gpu_index=7, chunk=1024, pending_dispatches=64,
                  native_finite_bitwise_comparisons=9748480, original_prefix_comparisons=comparisons,
                  all_original_prefixes_match=True, all_contexts_faster_in_both_validation_modes=True,
                  fresh_2k_repeat_qualified=True, zero_major_faults=True, cases=rows,
                  window_timing_not_natural_standard=True, quality_qualified=False, headline_eligible=False,
                  source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "original_prefix_comparisons",
                                                 "all_contexts_faster_in_both_validation_modes")}), flush=True)


if __name__ == "__main__":
    main()

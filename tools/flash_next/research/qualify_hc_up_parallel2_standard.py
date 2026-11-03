import json
from pathlib import Path

from build_hc_up_parallel2 import FLAG
from qualify_qsa_score16_standard import all_contexts_reach_target, collect_standard_cases
from qualify_qsa_score16_standard import verify_standard as verify_parent_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_hc_up_parallel2_model import VARIANT


def verify_standard(raw, guard, native, parent, references, context, full, experiment_flag=FLAG):
    result = verify_parent_standard(raw, guard, native, parent, references, context, full)
    assert raw["load_experiment_flags"][experiment_flag] == "1"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_HC_UP_PREMIX"] == "1"
    if experiment_flag == "ORT_WEBGPU_DENSE_TILE4":
        assert raw["load_experiment_flags"][FLAG] == "1"
    return result


def main(variant=VARIANT, experiment_flag=FLAG, stamp="20261029", parent_variant="qsa-score16-v5"):
    family, version = variant.rsplit("-v", 1)
    window_prefix = family.replace("-", "_")
    output = ROOT / f"{variant}-contexts-{stamp}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{variant}-native-gpu7-{stamp}.qualification.json"
    paired_path = ROOT / f"{variant}-windows-contexts-{stamp}.qualification.json"
    parent_path = ROOT / f"{parent_variant}-contexts-20261029.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, paired, parent, controls = (json.loads(path.read_text())
                                       for path in (native_path, paired_path, parent_path, controls_path))
    if variant == VARIANT:
        assert native["status"] == "hc_up_parallel2_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 18677760 and native["columns_tested"] == [1, 2]
    elif variant == "dense-tile4-v1":
        assert native["status"] == "dense_tile4_changing_inputs_finite_bitwise_and_replay_qualified"
        assert native["finite_bitwise_comparisons"] == 116582784 and native["fixtures"] == 27
        assert paired["frozen_dependencies_sha256"]
        for path, digest in paired["frozen_dependencies_sha256"].items():
            assert sha(Path(path)) == digest
    else:
        raise ValueError(f"Unsupported HC experiment: {variant}")
    assert paired["status"] == f"{window_prefix}_native_and_paired_original_sequences_qualified"
    assert paired["all_original_prefixes_match"] and paired["all_repeated_288_tokens_match"]
    assert paired["window_original_prefix_comparisons"] == 6144 and paired["zero_major_faults"]
    assert paired["all_contexts_faster_in_both_validation_modes"]
    assert paired["provider_sha256"] == native["provider_sha256"]
    assert paired["baseline_provider_sha256"] == native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v{version}/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == native["provider_sha256"]
    sources = {path.name: sha(path) for path in (native_path, paired_path, parent_path, controls_path)}
    for name, digest in paired["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    build = provider.parent / "build-manifest.json"
    assert sha(build) == paired["build_manifest_sha256"]
    def verify_case(raw, guard, native, parent, references, context, full):
        return verify_standard(raw, guard, native, parent, references, context, full, experiment_flag)

    rows = collect_standard_cases(variant, native, parent, controls, paired, sources,
                                  verifier=verify_case, stamp=stamp)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"] == paired["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"] == paired["config_sha256"]
    report = dict(status=f"{window_prefix}_native_and_standard_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, contexts=[2048, 8192, 32768], chunk=256, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        all_contexts_at_least_100_decode_tps=all_contexts_reach_target(rows),
        default_off=True, model_and_weights_unchanged=True, quality_qualified=False,
        headline_eligible=False, rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "all_original_sequences_match",
          "all_contexts_at_least_100_decode_tps", "rows")}), flush=True)


if __name__ == "__main__":
    main()

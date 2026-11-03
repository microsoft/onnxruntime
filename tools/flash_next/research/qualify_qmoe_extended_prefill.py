import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    output = ROOT / "qmoe-extended-prefill-v1-contexts-20261029.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    directory = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-extended-prefill/build-v1")
    build = json.loads((directory / "build-manifest.json").read_text())
    native = json.loads((ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029.qualification.json").read_text())
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"] == native["provider_sha256"]
    assert build["baseline_byte_identical"] and native["finite_bitwise_comparisons"] == 33454080
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    rows = []
    sources = {}
    graph_sha = None
    for context in (2048, 8192, 32768):
        baseline_prefix = ROOT / f"expert-bitcast-unpack-v2-gpu7-standard{context}-20261029"
        baseline_proof = json.loads(baseline_prefix.with_suffix(".qualification.json").read_text())
        baseline_raw = json.loads(baseline_prefix.with_suffix(".json").read_text())
        assert baseline_proof["provider_sha256"] == native["baseline_provider_sha256"]
        graph_sha = baseline_proof["graph_sha256"]
        references = [s for s in controls["samples"] if s["context"] == context]
        assert len(references) == 4
        summaries = {}
        for standard in (False, True):
            suffix = "-standard-validation" if standard else ""
            prefix = ROOT / (f"qmoe-extended-prefill-v1-chunk64-gpu7-screen{context}"
                             f"-t256-r3{suffix}-20261029")
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            diagnostic = json.loads(prefix.with_suffix(".diagnostic.json").read_text())
            assert raw["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
            assert guard["gpu_index"] == 7
            assert "GPU-0e70e969-bbfc-80c5-d582-88be134632ea" in guard["command"][-1]
            assert "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1" in guard["command"][-1]
            assert "--webgpu-max-pending-dispatches 64" in guard["command"][-1]
            assert raw["plugin_sha256"] == native["provider_sha256"]
            assert raw["graph_sha256"] == graph_sha and raw["config_sha256"] == controls["config_sha256"]
            assert raw["provider"] == "webgpu" and raw["chunk"] == 64
            assert raw["webgpu_validation_mode_override"] == (None if standard else "full")
            assert len(raw["samples"]) == 4
            for sample, reference in zip(raw["samples"], references):
                assert sample["prompt_sha256"] == reference["prompt_sha256"]
                assert len(sample["output_token_ids"]) == 256
                assert sample["output_token_ids"] == reference["output_token_ids"]
                assert all(value["major_page_faults"] == 0
                           for value in sample["process_resource_usage"].values())
            assert diagnostic["original_output_prefixes_match"] and diagnostic["zero_major_faults"]
            assert diagnostic["output_tokens"] == 256 and diagnostic["repetitions"] == 3
            assert diagnostic["full_validation"] == (not standard)
            summaries["retained_validation" if standard else "full_validation"] = raw["summary"][f"{context}:configured"]
            for suffix in (".json", ".guard.json", ".diagnostic.json"):
                path = prefix.with_suffix(suffix)
                sources[path.name] = sha(path)
        previous = baseline_raw["summary"][f"{context}:configured"]
        current = summaries["retained_validation"]
        rows.append(dict(context=context, previous_chunk=16, candidate_chunk=64,
            previous=previous, candidate=current, full_validation=summaries["full_validation"],
            ttft_reduction_percent=100 * (1 - current["mean_ttft_seconds"] / previous["mean_ttft_seconds"]),
            e2e_output_tps_ratio=current["e2e_tps"] / previous["e2e_tps"],
            comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    assert sha(Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model/model.onnx")) == graph_sha
    report = dict(status="prefill_extension_native_and_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=native["baseline_provider_sha256"],
        graph_sha256=graph_sha, config_sha256=controls["config_sha256"], gpu_index=7,
        contexts=[2048, 8192, 32768], candidate_chunk=64, output_tokens=256,
        measured_repetitions=3, excluded_warmup_requests=1, pending_dispatches=64,
        original_sequence_comparisons=3 * 2 * 4 * 256, all_original_sequences_match=True,
        zero_major_faults=True, native_finite_bitwise_comparisons=native["finite_bitwise_comparisons"],
        model_and_weights_unchanged=True, default_off=True, retained_decode_headline_unchanged=True,
        decode_speedup_claimed=False, quality_qualified=False, headline_eligible=False,
        rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_original_sequences_match", "zero_major_faults", "rows")}))


if __name__ == "__main__":
    main()

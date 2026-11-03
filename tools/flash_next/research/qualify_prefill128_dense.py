import json
import math
from pathlib import Path
import statistics

from qualify_quickgelu_mul_long_context import ROOT, sha


def verify_samples(raw, references, context):
    samples = raw["samples"]
    assert len(samples) == len(references) == 4
    assert [s["warmup"] for s in samples] == [True, False, False, False]
    for sample, reference in zip(samples, references):
        assert sample["context"] == context
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert len(sample["output_token_ids"]) == sample["output_tokens"] == 256
        assert sample["output_token_ids"] == reference["output_token_ids"]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
    measured = samples[1:]
    expected = dict(
        median_decode_tps=statistics.median(s["decode_tps"] for s in measured),
        decode_tps=3 * 255 / sum(s["decode_seconds"] for s in measured),
        e2e_tps=3 * 256 / sum(s["request_seconds"] for s in measured),
        mean_ttft_seconds=statistics.mean(s["ttft_seconds"] for s in measured))
    summary = raw["summary"][f"{context}:configured"]
    assert all(math.isclose(summary[key], value, rel_tol=1e-12) for key, value in expected.items())
    return summary


def main():
    output = ROOT / "prefill128-dense-stack-v1-contexts-20261029.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    directory = Path("/home/sayanshaw/repos/qwen38-int4-prefill-dense-stack/build-v1")
    native = json.loads((ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    assert native["status"] == "combined_dense_and_prefill_native_qualified"
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sum(v["finite_bitwise_comparisons"] for v in native["evidence"].values()) == 150036864
    parent = json.loads((ROOT / "qmoe-extended-prefill-v1-contexts-20261029.qualification.json").read_text())
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    previous = {row["context"]: row["candidate"] for row in parent["rows"]}
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    sources = {}
    rows = []
    for context in (2048, 8192, 32768):
        references = [s for s in controls["samples"] if s["context"] == context]
        summaries = {}
        for standard in (False, True):
            suffix = "-standard-validation" if standard else ""
            prefix = ROOT / (f"prefill-dense-stack-v1-chunk128-gpu7-screen{context}"
                             f"-t256-r3{suffix}-20261029")
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            diagnostic = json.loads(prefix.with_suffix(".diagnostic.json").read_text())
            assert guard["status"] == raw["status"] == "completed" and guard["exit_code"] == 0
            assert guard["gpu_index"] == 7
            assert "GPU-0e70e969-bbfc-80c5-d582-88be134632ea" in guard["command"][-1]
            assert raw["plugin_sha256"] == diagnostic["provider_sha256"] == native["provider_sha256"]
            assert raw["graph_sha256"] == parent["graph_sha256"]
            assert raw["config_sha256"] == controls["config_sha256"] == parent["config_sha256"]
            assert raw["provider"] == "webgpu" and raw["chunk"] == diagnostic["chunk"] == 128
            assert raw["webgpu_max_pending_dispatches_override"] == 64
            assert raw["webgpu_validation_mode_override"] == (None if standard else "full")
            assert raw["load_experiment_flags"]["ORT_WEBGPU_DENSE_BITCAST_UNPACK"] == "1"
            assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_EXTENDED_PREFILL"] == "1"
            assert diagnostic["dense_stack"] and diagnostic["extended_prefill"]
            assert diagnostic["original_prompts_match"] and diagnostic["original_output_prefixes_match"]
            assert diagnostic["zero_major_faults"] and diagnostic["full_validation"] == (not standard)
            assert diagnostic["output_tokens"] == 256 and diagnostic["repetitions"] == 3
            summaries["retained_validation" if standard else "full_validation"] = verify_samples(raw, references, context)
            for suffix in (".json", ".guard.json", ".diagnostic.json"):
                path = prefix.with_suffix(suffix)
                sources[path.name] = sha(path)
        current = summaries["retained_validation"]
        rows.append(dict(context=context, previous_chunk=64, candidate_chunk=128,
            previous=previous[context], candidate=current, full_validation=summaries["full_validation"],
            ttft_reduction_percent=100 * (1 - current["mean_ttft_seconds"] / previous[context]["mean_ttft_seconds"]),
            e2e_output_tps_ratio=current["e2e_tps"] / previous[context]["e2e_tps"],
            comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    assert sha(Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model/model.onnx")) == parent["graph_sha256"]
    report = dict(status="prefill128_dense_native_and_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=native["baseline_provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"], gpu_index=7,
        contexts=[2048, 8192, 32768], candidate_chunk=128, output_tokens=256,
        measured_repetitions=3, excluded_warmup_requests=1, pending_dispatches=64,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        native_finite_bitwise_comparisons=150036864, model_and_weights_unchanged=True, default_off=True,
        retained_decode_headline_unchanged=True, decode_speedup_claimed=False,
        quality_qualified=False, headline_eligible=False, rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_original_sequences_match", "zero_major_faults", "rows")}))


if __name__ == "__main__":
    main()

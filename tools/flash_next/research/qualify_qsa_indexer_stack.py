import json
import math
from pathlib import Path

from qualify_prefill128_dense import verify_samples
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill_dense_model import verify_windows


CONTEXTS = (2048, 8192, 32768)
UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"


def verify_standard(raw, guard, proof, native, parent, controls, context, full):
    assert raw["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == UUID
    assert raw["plugin_sha256"] == proof["provider_sha256"] == native["provider_sha256"]
    assert raw["graph_sha256"] == proof["graph_sha256"] == parent["graph_sha256"]
    assert raw["config_sha256"] == controls["config_sha256"] == parent["config_sha256"]
    assert raw["provider"] == "webgpu" and raw["chunk"] == proof["chunk"] == 128
    assert raw["webgpu_max_pending_dispatches_override"] == 64
    assert raw["webgpu_validation_mode_override"] == ("full" if full else None)
    assert proof["status"] == "original_sequences_qualified"
    assert proof["context"] == context and proof["phase"] == "standard"
    assert proof["candidate"] == "indexer-stack" and proof["full_validation"] == full
    assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
    for flag, value in (
            ("ORT_WEBGPU_QSA_INDEXER_STACK", "1"), ("ORT_WEBGPU_QSA_CONCAT_VECTOR", "0"),
            ("ORT_WEBGPU_QSA_PACKED_SCORE", "0"), ("ORT_WEBGPU_DENSE_BITCAST_UNPACK", "1"),
            ("ORT_WEBGPU_QMOE_EXTENDED_PREFILL", "1")):
        assert raw["load_experiment_flags"][flag] == value, flag
    for sample in raw["samples"]:
        for field in ("ttft_seconds", "request_seconds", "decode_seconds", "decode_tps"):
            assert math.isfinite(sample[field]) and sample[field] > 0, field
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
    references = [s for s in controls["samples"] if s["context"] == context]
    return verify_samples(raw, references, context)


def meets_decode_target(rows):
    assert len(rows) == 3 and {row["context"] for row in rows} == set(CONTEXTS)
    assert all(math.isfinite(row["candidate"]["median_decode_tps"]) for row in rows)
    return all(row["candidate"]["median_decode_tps"] >= 100 for row in rows)


def main():
    output = ROOT / "qsa-indexer-stack-v1-contexts-20261029.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / "qsa-indexer-stack-v1-native-gpu7-20261029.qualification.json"
    native = json.loads(native_path.read_text())
    assert native["status"] == "exact_reference_and_indexer_stack_dispatch_qualified"
    assert native["reference_executions"] == 6210 and native["invalid_selector_rejected"]
    provider = Path("/home/sayanshaw/repos/qwen38-int4-qsa-indexer-stack/build-v1/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == native["provider_sha256"]
    parent_path = ROOT / "prefill128-dense-stack-v1-contexts-20261029.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    controls = json.loads(controls_path.read_text())
    previous = {row["context"]: row["candidate"] for row in parent["rows"]}
    sources = {p.name: sha(p) for p in (native_path, parent_path, controls_path)}
    rows = []
    for context in CONTEXTS:
        window_prefix = ROOT / f"qsa-indexer-stack-v1-chunk128-gpu7-windows{context}-20261029"
        window_raw = json.loads(window_prefix.with_suffix(".json").read_text())
        window_proof = json.loads(window_prefix.with_suffix(".qualification.json").read_text())
        assert window_proof["provider_sha256"] == native["provider_sha256"]
        assert window_proof["original_output_prefixes_match"] and window_proof["zero_major_faults"]
        paired = verify_windows(window_raw, context, prefix="qsa_indexer_stack")
        assert paired == window_proof["result"]
        reference = [s for s in controls["samples"] if s["context"] == context][1]
        for sample in window_raw["samples"]:
            assert sample["output_token_ids"][:256] == reference["output_token_ids"]
            assert sample["output_token_ids"] == window_raw["samples"][0]["output_token_ids"]
            assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
        for suffix in (".json", ".qualification.json"):
            path = window_prefix.with_suffix(suffix)
            sources[path.name] = sha(path)
        summaries = {}
        for full in (True, False):
            prefix = ROOT / (f"qsa-indexer-stack-v1-chunk128-gpu7-{'full-' if full else ''}"
                             f"standard{context}-20261029")
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
            summaries["full_validation" if full else "candidate"] = verify_standard(
                raw, guard, proof, native, parent, controls, context, full)
            for suffix in (".json", ".guard.json", ".qualification.json"):
                path = prefix.with_suffix(suffix)
                sources[path.name] = sha(path)
        candidate = summaries["candidate"]
        rows.append(dict(context=context, previous=previous[context], paired_windows=paired, **summaries,
            ttft_reduction_percent=100 * (1 - candidate["mean_ttft_seconds"] /
                                         previous[context]["mean_ttft_seconds"]),
            standard_decode_tps_ratio=candidate["median_decode_tps"] / previous[context]["median_decode_tps"],
            standard_comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status="indexer_stack_native_and_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, contexts=list(CONTEXTS), chunk=128, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        all_contexts_at_least_100_decode_tps=meets_decode_target(rows),
        default_off=True, model_and_weights_unchanged=True, quality_qualified=False,
        headline_eligible=False, rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "all_original_sequences_match", "all_contexts_at_least_100_decode_tps", "rows")}))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from build_qsa_score16 import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens


def verify_standard(raw, guard, native, parent, references, context, full, chunk=256, pending=64):
    verify_metadata(raw, guard, native, parent, full, chunk=chunk, pending=pending)
    assert raw["load_experiment_flags"][FLAG] == "1"
    assert not raw["profiling"] and raw["profiling_scope"] is None
    assert raw["comparison"] is None and raw["profile_decode_after_tokens"] is None
    return verify_tokens(raw, references, context, 256)


def all_contexts_reach_target(rows):
    assert [r["context"] for r in rows] == [2048, 8192, 32768]
    return all(r["candidate"]["median_decode_tps"] >= 100 for r in rows)


def collect_standard_cases(variant, native, parent, controls, paired, sources, verifier=verify_standard,
                           stamp="20261029", chunk=256):
    paired_rows = {r["context"]: r["result"] for r in paired["rows"] if not r["full_validation"]}
    assert set(paired_rows) == {2048, 8192, 32768}
    previous = {r["context"]: r["candidate"] for r in parent["rows"]}
    rows = []
    for context in (2048, 8192, 32768):
        references = [s for s in controls["samples"] if s["context"] == context]
        results = {}
        for full in (True, False):
            prefix = ROOT / f"{variant}-chunk{chunk}-gpu7-{'full-' if full else ''}standard{context}-{stamp}"
            proof_path = prefix.with_suffix(".qualification.json")
            proof = json.loads(proof_path.read_text())
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            result = verifier(raw, guard, native, parent, references, context, full)
            assert proof["status"] == "original_sequences_qualified"
            assert proof["context"] == context and proof["phase"] == "standard"
            assert proof["full_validation"] == full and proof["zero_major_faults"]
            assert proof["original_output_prefixes_match"] and proof["provider_sha256"] == native["provider_sha256"]
            assert proof["result"] == result
            results["full_validation" if full else "candidate"] = result
            sources[proof_path.name] = sha(proof_path)
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest
                sources[name] = digest
        candidate = results["candidate"]
        rows.append(dict(context=context, previous=previous[context], **results,
            paired_windows=paired_rows[context],
            ttft_reduction_percent=100 * (1 - candidate["mean_ttft_seconds"] / previous[context]["mean_ttft_seconds"]),
            standard_decode_tps_ratio=candidate["median_decode_tps"] / previous[context]["median_decode_tps"],
            standard_comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    return rows


def main():
    output = ROOT / "qsa-score16-v5-contexts-20261029.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / "qsa-score16-v5-native-gpu7-20261029.qualification.json"
    paired_path = ROOT / "qsa-score16-v5-windows-contexts-20261029.qualification.json"
    parent_path = ROOT / "prefill256-v1-contexts-20261029.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, paired, parent, controls = (json.loads(p.read_text()) for p in
                                       (native_path, paired_path, parent_path, controls_path))
    assert native["status"] == "exact_reference_and_score16_dispatch_qualified"
    assert native["program_name"] == "QsaPackedScore16QueryWide"
    assert native["reference_executions"] == 2484 and native["score16_dispatches"] == 198
    assert native["minimum_score_stride"] == 2048
    assert paired["status"] == "score16_native_and_paired_original_sequences_qualified"
    assert paired["all_original_prefixes_match"] and paired["all_repeated_288_tokens_match"]
    assert paired["window_original_prefix_comparisons"] == 5120 and paired["zero_major_faults"]
    assert paired["provider_sha256"] == native["provider_sha256"]
    assert paired["baseline_provider_sha256"] == native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = Path("/home/sayanshaw/repos/qwen38-int4-qsa-score16/build-v5/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == native["provider_sha256"]
    sources = {p.name: sha(p) for p in (native_path, paired_path, parent_path, controls_path)}
    for name, digest in paired["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    rows = collect_standard_cases("qsa-score16-v5", native, parent, controls, paired, sources)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"] == paired["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"] == paired["config_sha256"]
    report = dict(status="score16_native_and_standard_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, contexts=[2048, 8192, 32768], chunk=256, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        all_contexts_at_least_100_decode_tps=all_contexts_reach_target(rows),
        default_off=True, model_and_weights_unchanged=True, quality_qualified=False,
        headline_eligible=False, rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_original_sequences_match",
          "all_contexts_at_least_100_decode_tps", "rows")}), flush=True)


if __name__ == "__main__":
    main()

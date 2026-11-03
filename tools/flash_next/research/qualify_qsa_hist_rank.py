import math
import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from build_qsa_hist_rank import FLAG
from qualify_qsa_score16_standard import collect_standard_cases, verify_standard


VARIANT = "qsa-hist-rank-v2"
STAMP = "20261031"
SHORT_REGRESSION_FLOOR = 0.99


def verify_histogram_standard(raw, guard, native, parent, references, context, full):
    assert raw["load_experiment_flags"][FLAG] == "1"
    return verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)


def read_proof(native, context, phase, full, attempt=1, sources=None):
    suffix = f"-attempt{attempt}" if attempt > 1 else ""
    path = ROOT / (
        f"{VARIANT}-chunk1024-gpu7-{'full-' if full else ''}"
        f"{phase}{context}-{STAMP}{suffix}.qualification.json")
    proof = json.loads(path.read_text())
    assert proof["status"] == "original_sequences_qualified"
    assert proof["context"] == context and proof["phase"] == phase
    assert proof["full_validation"] == full
    assert proof["provider_sha256"] == native["provider_sha256"]
    assert proof["baseline_provider_sha256"] == native["baseline_provider_sha256"]
    assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
    assert proof["graph_sha256"] == "8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9"
    assert proof["config_sha256"] == "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6"
    for name, digest in proof["source_sha256"].items():
        assert sha(ROOT / name) == digest
    if sources is not None:
        sources[path.name] = sha(path)
        sources.update(proof["source_sha256"])
    return proof


def verify_window_result(result, long_context):
    ratio = result["geometric_mean_matched_window_ratio"]
    assert math.isfinite(ratio)
    assert result["matched_window_pairs"] == 16
    assert result["all_request_output_tokens_match"]
    requests = result["per_request"]
    assert [request["repetition"] for request in requests] == [0, 1, 2, 3]
    for request in requests:
        for mode in (0, 1):
            value = request[f"qsa_hist_rank{mode}_median_window_tps"]
            assert math.isfinite(value) and value > 0
    if long_context:
        assert ratio > 1
        assert sum(request["qsa_hist_rank1_median_window_tps"] >
                   request["qsa_hist_rank0_median_window_tps"] for request in requests) >= 3
    else:
        assert ratio >= SHORT_REGRESSION_FLOOR, "Short-context regression exceeds the 1% screen bound"


def verify_prerequisites(native, phase, sources=None):
    for attempt in (1, 2):
        for full in (True, False):
            proof = read_proof(native, 32768, "windows", full, attempt, sources=sources)
            verify_window_result(proof["result"], long_context=True)
    if phase == "standard":
        for context in (2048, 8192):
            profile = read_proof(native, context, "profile", True, sources=sources)
            assert profile["result"]["candidate_program_dispatches"] == {
                "QsaHistogramUniformRank": 0, "QsaRankHistogram": 0}
            assert profile["result"]["original_program_dispatches"] == 12
            for full in (True, False):
                proof = read_proof(native, context, "windows", full, sources=sources)
                verify_window_result(proof["result"], long_context=False)


def main():
    output = ROOT / f"{VARIANT}-contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{VARIANT}-native-gpu7-{STAMP}.qualification.json"
    parent_path = ROOT / f"qsa-prefill-score16-v1-contexts-{STAMP}.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, parent, controls = (json.loads(path.read_text()) for path in
                                (native_path, parent_path, controls_path))
    assert native["status"] == "exact_reference_and_histogram_rank_dispatch_qualified"
    assert native["fixtures"] == 69 and native["reference_executions"] == 2484
    assert native["minimum_score_stride"] == 8192
    assert native["original_float_scoring_and_nan_tie_tail_semantics_preserved"]
    assert native["invalid_selector_rejected"] and native["cpu_fallback_disabled"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    directory = Path("/home/sayanshaw/repos/qwen38-int4-qsa-hist-rank/build-v2")
    assert sha(directory / "build-manifest.json") == native["build_manifest_sha256"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    for name, digest in native["fixture_sha256"].items():
        assert sha(Path(name)) == digest
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
    sources = {path.name: sha(path) for path in (native_path, parent_path, controls_path)}
    verify_prerequisites(native, "standard", sources=sources)
    paired_rows = []
    for context in (2048, 8192, 32768):
        for full in (True, False):
            proof = read_proof(native, context, "windows", full, 2 if context == 32768 else 1,
                               sources=sources)
            paired_rows.append({"context": context, "full_validation": full, "result": proof["result"]})
            sources.update(proof["source_sha256"])
    rows = collect_standard_cases(VARIANT, native, parent, controls,
                                  {"rows": paired_rows}, sources, verifier=verify_histogram_standard,
                                  stamp=STAMP, chunk=1024)
    for row in rows:
        for mode in ("candidate", "full_validation"):
            tps = row[mode]["median_decode_tps"]
            ttft = row[mode]["mean_ttft_seconds"]
            assert math.isfinite(tps) and tps > 0 and math.isfinite(ttft) and ttft > 0
    report = dict(
        status="histogram_rank_native_and_standard_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, contexts=[2048, 8192, 32768], chunk=1024, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        repeated_long_context_controls_qualified=True, short_context_regression_screen_floor=SHORT_REGRESSION_FLOOR,
        all_contexts_above_100_decode_tps=all(row["candidate"]["median_decode_tps"] > 100 for row in rows),
        default_off=True, model_and_weights_unchanged=True, quality_qualified=False,
        headline_eligible=False, rows=rows, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "all_contexts_above_100_decode_tps", "rows")}), flush=True)


if __name__ == "__main__":
    main()

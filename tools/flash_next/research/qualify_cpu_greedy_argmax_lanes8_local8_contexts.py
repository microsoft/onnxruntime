import json
import math
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_row4_cpu_policy_contexts import candidates_above_target
from run_cpu_greedy_argmax_lanes8_local8_standard import RUNTIME, STAMP, VARIANT


def verify_screen(proof, context, repeat):
    expected = ("repeated_" if repeat else "initial_") + "cpu_greedy_decode_gain_both_modes"
    assert proof["context"] == context and proof["repeat"] == repeat
    assert proof["status"] in (expected, "no_supported_decode_gain")
    assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
    assert proof["main_affinity_unchanged"] and proof["manual_affinity_override"]
    assert proof["fresh_reuses_initial_ending_controls"] == repeat
    assert [row["full_validation"] for row in proof["rows"]] == [True, False]
    supported = []
    for row in proof["rows"]:
        comparison = row["comparison"]
        ratios = comparison["request_ratios"]
        assert len(ratios) == 3 and all(math.isfinite(value) and value > 0 for value in ratios)
        assert comparison["positive_requests"] == sum(value > 1 for value in ratios)
        ratio = comparison["matched_request_decode_ratio"]
        ttft = comparison["matched_request_ttft_ratio"]
        assert math.isfinite(ratio) and math.isfinite(ttft) and ratio > 0 and ttft > 0
        assert math.isclose(ratio, math.exp(sum(math.log(value) for value in ratios) / 3), rel_tol=1e-12)
        valid = ratio > 1 and comparison["positive_requests"] == 3 and ttft <= 1.02
        assert comparison["supported"] == valid
        for phase in ("before", "candidate", "after"):
            for field in ("median_decode_tps", "mean_ttft_seconds"):
                assert math.isfinite(row[phase][field]) and row[phase][field] > 0
        supported.append(valid)
    assert (proof["status"] == expected) == all(supported)
    return all(supported)


def main():
    output = ROOT / f"{VARIANT}-all-contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    build_path = RUNTIME / "build-manifest.json"
    build = json.loads(build_path.read_text())
    provider = "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    sources, rows, support = {}, [], []
    for context in (2048, 8192, 32768):
        screens = {}
        for repeat in (False, True):
            path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}screen-{STAMP}.qualification.json"
            proof = json.loads(path.read_text())
            support.append(verify_screen(proof, context, repeat))
            assert proof["provider_sha256"] == provider
            assert proof["runtime_sha256"] == build["runtime_sha256"]
            assert proof["parent_runtime_sha256"] == build["parent_runtime_sha256"]
            for name, digest in proof["source_sha256"].items():
                assert sha(Path(name)) == digest, name
                sources[name] = digest
            sources[str(path)] = sha(path)
            screens["fresh" if repeat else "initial"] = proof
        rows.append(dict(context=context, **screens))
    provider_path = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
    runtime_path = RUNTIME / "libonnxruntime-genai.so"
    assert sha(provider_path) == provider and sha(runtime_path) == build["runtime_sha256"]
    report = dict(status=("repeated_cpu_greedy_local8_supported_all_contexts" if all(support) else
                          "cpu_greedy_local8_held_insufficient_all_context_support"),
        contexts=[2048, 8192, 32768], provider_sha256=provider,
        runtime_sha256=build["runtime_sha256"], parent_runtime_sha256=build["parent_runtime_sha256"],
        rows=rows, all_contexts_both_modes_above_100_decode_tps=candidates_above_target(rows),
        measured_request_comparisons=36,
        positive_request_comparisons=sum(mode["comparison"]["positive_requests"]
            for row in rows for phase in ("initial", "fresh") for mode in row[phase]["rows"]),
        original_output_prefixes_match=True, zero_major_faults=True,
        cpu_placement="local8", matched_cpu_and_worker_policy_in_both_modes=True,
        original_runtime_controls=True, fresh_reuses_initial_ending_controls=True,
        default_enabled=False, quality_qualified=False, headline_eligible=False,
        comparison_kind="repeated_bracketed_original_runtime_net_screen_matched_local8",
        source_sha256={**sources, **{str(path): sha(path) for path in
            (Path(__file__), build_path, provider_path, runtime_path, ROOT / "qualify_row4_cpu_policy_contexts.py")}})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("status", "positive_request_comparisons", "all_contexts_both_modes_above_100_decode_tps")}), flush=True)


if __name__ == "__main__":
    main()

import json
import math
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_row4_default_cpu_policy import STAMP, VARIANT


def verify_screen(proof, context, repeat):
    assert proof["context"] == context and proof["repeat"] == repeat
    expected = ("repeated_" if repeat else "initial_") + "default_policy_decode_gain_both_modes"
    assert proof["status"] in (expected, "no_supported_decode_gain")
    assert proof["default_worker_policy_preserved_in_controls"]
    assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
    assert proof["process_only_affinity"] and proof["memory_policy_unchanged"]
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
        for key in ("before", "candidate", "after"):
            for field in ("median_decode_tps", "mean_ttft_seconds"):
                value = row[key][field]
                assert math.isfinite(value) and value > 0
        supported.append(valid)
    assert (proof["status"] == expected) == all(supported)
    return all(supported)


def candidates_above_target(rows):
    return (len(rows) == 3 and [row["context"] for row in rows] == [2048, 8192, 32768] and
            all([mode["full_validation"] for mode in row["fresh"]["rows"]] == [True, False] for row in rows) and
            all(math.isfinite(mode["candidate"]["median_decode_tps"]) and
                mode["candidate"]["median_decode_tps"] > 100 for row in rows for mode in row["fresh"]["rows"]))


def main():
    output = ROOT / f"{VARIANT}-all-contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    sources, rows, support = {}, [], []
    provider = None
    for context in (2048, 8192, 32768):
        screens = {}
        for repeat in (False, True):
            path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}screen-{STAMP}.qualification.json"
            proof = json.loads(path.read_text())
            support.append(verify_screen(proof, context, repeat))
            provider = proof["provider_sha256"] if provider is None else provider
            assert proof["provider_sha256"] == provider
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest, name
                sources[name] = digest
            sources[path.name] = sha(path)
            screens["fresh" if repeat else "initial"] = proof
        rows.append(dict(context=context, **screens))
    provider_path = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
    assert sha(provider_path) == provider
    report = dict(status=("repeated_default_policy_cpu_locality_supported_all_contexts" if all(support) else
                          "cpu_locality_held_insufficient_all_context_support"),
                  contexts=[2048, 8192, 32768], provider_sha256=provider, rows=rows,
                  all_contexts_both_modes_above_100_decode_tps=candidates_above_target(rows),
                  measured_request_comparisons=36,
                  positive_request_comparisons=sum(mode["comparison"]["positive_requests"]
                      for row in rows for phase in ("initial", "fresh") for mode in row[phase]["rows"]),
                  original_output_prefixes_match=True, zero_major_faults=True,
                  default_worker_policy_preserved_in_controls=True, default_enabled=False,
                  comparison_kind="repeated_bracketed_separate_load_cpu_placement_not_shader_gain",
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={**sources, Path(__file__).name: sha(Path(__file__))})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "positive_request_comparisons", "all_contexts_both_modes_above_100_decode_tps")}))


if __name__ == "__main__":
    main()

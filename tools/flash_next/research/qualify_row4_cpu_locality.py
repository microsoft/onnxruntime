import argparse
import json
import math
import statistics

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_row4_cpu_locality import CPU_MASKS, STAMP, VARIANT, evidence_prefix


def compare_samples(candidate, before, after):
    assert len(candidate) == len(before) == len(after) == 3
    ratios = []
    ttft_ratios = []
    for selected, first, last in zip(candidate, before, after):
        assert selected["prompt_sha256"] == first["prompt_sha256"] == last["prompt_sha256"]
        assert selected["output_token_ids"] == first["output_token_ids"] == last["output_token_ids"]
        for sample in (selected, first, last):
            for field in ("decode_tps", "ttft_seconds"):
                assert math.isfinite(sample[field]) and sample[field] > 0, sample
        ratios.append(selected["decode_tps"] / math.sqrt(first["decode_tps"] * last["decode_tps"]))
        ttft_ratios.append(selected["ttft_seconds"] / math.sqrt(first["ttft_seconds"] * last["ttft_seconds"]))
    ratio = math.exp(statistics.mean(math.log(value) for value in ratios))
    ttft_ratio = math.exp(statistics.mean(math.log(value) for value in ttft_ratios))
    support = sum(value > 1 for value in ratios)
    return dict(matched_request_decode_ratio=ratio, positive_requests=support, request_ratios=ratios,
                matched_request_ttft_ratio=ttft_ratio,
                supported=ratio > 1 and support == 3 and ttft_ratio <= 1.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", choices=("local", "local8"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    args = parser.parse_args()
    output = ROOT / f"{VARIANT}-{args.placement}-context{args.context}-screen-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    sources, rows = {}, []
    provider = None
    for full in (True, False):
        samples = []
        results = []
        times = []
        for placement, trial in (("baseline", 0), (args.placement, 0), ("baseline", 1)):
            prefix = evidence_prefix(placement, args.context, full, trial)
            proof_path = prefix.with_suffix(".qualification.json")
            raw_path = prefix.with_suffix(".json")
            guard_path = prefix.with_suffix(".guard.json")
            proof, raw, guard = (json.loads(path.read_text()) for path in (proof_path, raw_path, guard_path))
            assert proof["status"] == "original_sequences_qualified"
            assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
            assert proof["placement"] == placement and proof["trial"] == trial
            assert proof["context"] == args.context and proof["full_validation"] == full
            assert raw["cpu_affinity"] == proof["cpu_affinity"] == CPU_MASKS[placement]
            provider = proof["provider_sha256"] if provider is None else provider
            assert proof["provider_sha256"] == provider
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest, name
                sources[name] = digest
            sources[proof_path.name] = sha(proof_path)
            measured = [sample for sample in raw["samples"] if not sample["warmup"]]
            assert len(raw["samples"]) == 4 and len(measured) == 3
            samples.append(measured)
            results.append(proof["result"])
            assert guard["status"] == "completed" and guard["exit_code"] == 0
            times.append((proof["started_ns"], proof["completed_ns"]))
        assert times[0][1] <= times[1][0] and times[1][1] <= times[2][0], times
        comparison = compare_samples(samples[1], samples[0], samples[2])
        rows.append(dict(full_validation=full, before=results[0], candidate=results[1],
                         after=results[2], comparison=comparison))
    supported = all(row["comparison"]["supported"] for row in rows)
    report = dict(status="bracketed_decode_gain_both_modes" if supported else "no_supported_decode_gain",
                  placement=args.placement, context=args.context, provider_sha256=provider, rows=rows,
                  original_output_prefixes_match=True, zero_major_faults=True,
                  process_only_affinity=True, memory_policy_unchanged=True,
                  comparison_kind="bracketed_separate_load_screen_not_same_load_kernel_gain",
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={**sources, **{path.name: sha(path) for path in
                                  (ROOT / "run_row4_cpu_locality.py", ROOT / "benchmark_webgpu_int4.py",
                                   ROOT / "qualify_row4_cpu_locality.py")}})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "placement", "context", "rows")}), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
import math
import statistics

from build_qsa_prefill_score16 import FLAG
from build_qmoe_grouped_prefill import FLAG as GROUPED_FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha


def ttft_metrics(values):
    assert len(values) == 4 and all(math.isfinite(v) and v > 0 for v in values)
    pairs = (values[0] / values[1], values[3] / values[2])
    ratio = math.sqrt(pairs[0] * pairs[1])
    return dict(baseline_median_ttft_seconds=statistics.median((values[0], values[3])),
                candidate_median_ttft_seconds=statistics.median((values[1], values[2])),
                matched_pairs=2, baseline_over_candidate_pair_ratios=list(pairs),
                geometric_mean_matched_ttft_ratio=ratio,
                matched_ttft_reduction_percent=100 * (1 - 1 / ratio),
                all_pairs_favor_candidate=all(v > 1 for v in pairs),
                ttft_win_screen_qualified=ratio > 1 and all(v > 1 for v in pairs))


def main(grouped_prefill=False, prefill_rows4=False):
    assert not prefill_rows4 or grouped_prefill
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--attempt", type=int, choices=range(8), default=0)
    args = parser.parse_args()
    variant = "qmoe-grouped-prefill-v1" if grouped_prefill else "qsa-prefill-score16-v1"
    if prefill_rows4:
        variant = "qmoe-prefill-rows4-v2"
    flag = GROUPED_FLAG if grouped_prefill else FLAG
    attempt_suffix = f"-attempt{args.attempt}" if args.attempt else ""
    marker = "full-" if args.full_validation else ""
    output = ROOT / (
        f"{variant}-gpu7-{marker}paired-screen{args.context}-20261031{attempt_suffix}.qualification.json")
    if output.exists():
        raise FileExistsError(output)
    native = json.loads((ROOT / f"{variant}-native-gpu7-20261031.qualification.json").read_text())
    records, sources, references = [], {}, None
    sequences = {True: set(), False: set()}
    for trial, mode in enumerate((0, 1, 1, 0)):
        prefix = ROOT / (
            f"{variant}-mode{mode}-gpu7-{marker}screen{args.context}-20261031-trial{trial}{attempt_suffix}")
        proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        assert proof["status"] == "original_sequences_qualified" and raw["status"] == "completed"
        assert proof["provider_sha256"] == raw["plugin_sha256"] == native["provider_sha256"]
        assert proof["selector_value"] == mode and proof["trial"] == trial
        assert proof["selector"] == flag and proof.get("attempt", 0) == args.attempt
        assert proof["full_validation"] == args.full_validation
        assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
        assert raw["load_experiment_flags"][flag] == str(mode)
        if grouped_prefill:
            assert raw["load_experiment_flags"][FLAG] == "1"
        assert len(raw["samples"]) == 2 and [s["warmup"] for s in raw["samples"]] == [True, False]
        digests = (proof["graph_sha256"], proof["config_sha256"], proof["baseline_provider_sha256"])
        if references is None:
            references = digests
        assert digests == references
        for sample in raw["samples"]:
            assert sample["context"] == args.context and sample["output_tokens"] == 36
            assert len(sample["output_token_ids"]) == 36
            assert all(u["major_page_faults"] == 0 for u in sample["process_resource_usage"].values())
            sequences[sample["warmup"]].add((sample["prompt_sha256"], tuple(sample["output_token_ids"])))
        measured = raw["samples"][1]
        records.append(dict(trial=trial, selector_value=mode, ttft_seconds=measured["ttft_seconds"],
                            short_screen_decode_tps=measured["decode_tps"]))
        for suffix in (".qualification.json", ".json", ".guard.json", ".log"):
            path = prefix.with_suffix(suffix)
            if suffix != ".qualification.json":
                assert sha(path) == proof["source_sha256"][path.name]
            sources[path.name] = sha(path)
    assert all(len(s) == 1 for s in sequences.values())
    metrics = ttft_metrics([r["ttft_seconds"] for r in records])
    report = dict(status="paired_ttft_win_qualified" if metrics["ttft_win_screen_qualified"]
                  else "exact_sequences_qualified_no_paired_ttft_win",
        context=args.context, full_validation=args.full_validation, attempt=args.attempt, trials=records, **metrics,
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=references[2],
        graph_sha256=references[0], config_sha256=references[1], all_original_sequences_match=True,
        repeated_sequences_match=True, original_sequence_comparisons=288, zero_major_faults=True,
        output_tokens=36, excluded_warmup_requests=4, measured_requests=4, order=[0, 1, 1, 0],
        short_screen_decode_not_standard_headline=True, quality_qualified=False, headline_eligible=False,
        source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "full_validation",
          "baseline_median_ttft_seconds", "candidate_median_ttft_seconds", "matched_ttft_reduction_percent")}),
          flush=True)


if __name__ == "__main__":
    main()

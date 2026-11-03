import argparse
import json

from build_qmoe_prefill_rows8 import FLAG
from qualify_qsa_prefill_score16_screens import ttft_metrics
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_prefill_rows8_model import GROUP_FLAG, VARIANT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    marker = "full-" if args.full_validation else ""
    output = ROOT / f"{VARIANT}-gpu7-{marker}paired-screen{args.context}-20261031.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    native = json.loads(native_path.read_text())
    assert native["status"] == "grouped_rows8_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    records, sources, references = [], {}, None
    sequences = {True: set(), False: set()}
    for trial, mode in enumerate((0, 1, 1, 0)):
        prefix = ROOT / f"{VARIANT}-mode{mode}-gpu7-{marker}screen{args.context}-20261031-trial{trial}"
        proof = json.loads(prefix.with_suffix(".qualification.json").read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        assert proof["status"] == "original_sequences_qualified" and raw["status"] == "completed"
        assert proof["provider_sha256"] == raw["plugin_sha256"] == native["provider_sha256"]
        assert proof["context"] == args.context and proof["selector_value"] == mode and proof["trial"] == trial
        assert proof["selector"] == FLAG and proof["fixed_grouped_prefill_selector"] == 1
        assert proof["full_validation"] == args.full_validation
        assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
        assert raw["load_experiment_flags"][FLAG] == str(mode)
        assert raw["load_experiment_flags"][GROUP_FLAG] == "1"
        assert raw["load_experiment_flags"]["ORT_WEBGPU_QSA_PREFILL_SCORE16"] == "1"
        assert proof["native_proof_sha256"] == sha(native_path)
        digests = (proof["graph_sha256"], proof["config_sha256"], proof["baseline_provider_sha256"],
                   proof["build_manifest_sha256"], proof["parent_proof_sha256"])
        if references is None:
            references = digests
        assert references == digests
        assert len(raw["samples"]) == 2 and [s["warmup"] for s in raw["samples"]] == [True, False]
        for sample in raw["samples"]:
            assert sample["context"] == args.context and sample["output_tokens"] == 36
            assert len(sample["output_token_ids"]) == 36
            assert all(u["major_page_faults"] == 0 for u in sample["process_resource_usage"].values())
            sequences[sample["warmup"]].add((sample["prompt_sha256"], tuple(sample["output_token_ids"])))
        measured = raw["samples"][1]
        records.append(dict(trial=trial, selector_value=mode, ttft_seconds=measured["ttft_seconds"]))
        for name, digest in proof["source_sha256"].items():
            path = ROOT / name
            assert sha(path) == digest, name
            sources[name] = digest
        sources[prefix.with_suffix(".qualification.json").name] = sha(prefix.with_suffix(".qualification.json"))
    assert all(len(sequence) == 1 for sequence in sequences.values())
    metrics = ttft_metrics([record["ttft_seconds"] for record in records])
    report = dict(status="paired_ttft_win_qualified" if metrics["ttft_win_screen_qualified"]
                  else "exact_sequences_qualified_no_paired_ttft_win",
                  context=args.context, full_validation=args.full_validation, selector=FLAG,
                  fixed_grouped_prefill_selector=1, trials=records, **metrics,
                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=references[2],
                  graph_sha256=references[0], config_sha256=references[1],
                  build_manifest_sha256=references[3], parent_proof_sha256=references[4],
                  native_proof_sha256=sha(native_path), all_original_sequences_match=True,
                  repeated_sequences_match=True, original_sequence_comparisons=288, zero_major_faults=True,
                  output_tokens=36, excluded_warmup_requests=4, measured_requests=4, order=[0, 1, 1, 0],
                  short_screen_decode_not_standard_headline=True, quality_qualified=False,
                  headline_eligible=False, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "full_validation",
                     "baseline_median_ttft_seconds", "candidate_median_ttft_seconds",
                     "matched_ttft_reduction_percent")}), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_row4_cpu_locality import compare_samples
from run_dense_lane8_model import FLAG, STAMP, VARIANT, prefix_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    output = ROOT / f"{VARIANT}-context{args.context}-{'repeat-' if args.repeat else ''}screen-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    sources, rows = {}, []
    provider = None
    if args.repeat:
        initial_path = ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json"
        initial = json.loads(initial_path.read_text())
        assert initial["status"] == "initial_lane8_decode_gain_both_modes"
        for name, digest in initial["source_sha256"].items():
            assert sha(Path(name)) == digest, name
        provider = initial["provider_sha256"]
        sources[str(initial_path)] = sha(initial_path)
    baseline_trial = int(args.repeat)
    for full in (True, False):
        samples, results, times = [], [], []
        for mode, trial in (("control", baseline_trial), ("lane8", baseline_trial),
                            ("control", baseline_trial + 1)):
            prefix = prefix_for(mode, args.context, full, trial)
            proof_path = prefix.with_suffix(".qualification.json")
            proof, raw, guard = (json.loads(prefix.with_suffix(suffix).read_text())
                                 for suffix in (".qualification.json", ".json", ".guard.json"))
            assert proof["status"] == "original_sequences_qualified"
            assert proof["phase"] == "standard" and proof["mode"] == mode and proof["trial"] == trial
            assert proof["context"] == args.context and proof["full_validation"] == full
            assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
            assert proof["worker_policy_unchanged"] and raw["cpu_affinity"] == list(range(96))
            assert proof["experiment_flag"] == FLAG and proof["experiment_flag_value"] == int(mode == "lane8")
            assert guard["status"] == "completed" and guard["exit_code"] == 0
            assert f"export {FLAG}={int(mode == 'lane8')}; " in guard["command"][-1]
            provider = proof["provider_sha256"] if provider is None else provider
            assert proof["provider_sha256"] == raw["plugin_sha256"] == provider
            for name, digest in proof["source_sha256"].items():
                assert sha(Path(name)) == digest, name
                sources[name] = digest
            sources[str(proof_path)] = sha(proof_path)
            measured = [sample for sample in raw["samples"] if not sample["warmup"]]
            assert len(raw["samples"]) == 4 and len(measured) == 3
            samples.append(measured)
            results.append(proof["result"])
            times.append((proof["started_ns"], proof["completed_ns"]))
        assert times[0][1] <= times[1][0] and times[1][1] <= times[2][0], times
        comparison = compare_samples(samples[1], samples[0], samples[2])
        rows.append(dict(full_validation=full, before=results[0], candidate=results[1],
                         after=results[2], comparison=comparison))
    supported = all(row["comparison"]["supported"] for row in rows)
    report = dict(status=(("repeated_" if args.repeat else "initial_") + "lane8_decode_gain_both_modes"
                          if supported else "no_supported_decode_gain"),
                  context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,
                  all_candidate_modes_above_100_decode_tps=all(
                      row["candidate"]["median_decode_tps"] > 100 for row in rows),
                  original_output_prefixes_match=True, zero_major_faults=True, worker_policy_unchanged=True,
                  comparison_kind="bracketed_separate_load_shader_flag_screen_not_same_load_windows",
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={**sources, str(Path(__file__)): sha(Path(__file__))})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "repeat", "rows")}), flush=True)


if __name__ == "__main__":
    main()
